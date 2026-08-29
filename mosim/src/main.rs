#[macro_use]
mod trace;

mod config;
mod gpu_cluster;
mod gpu_job;
mod gpu_scheduler;
mod lcm_utils;
mod timer;

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::process::ExitCode;
use std::time::Instant;

use csv::ReaderBuilder;

use config::SimConfig;
use gpu_cluster::{CorunProfile, GPUCluster, InterferenceModel, PlacementMethod};
use gpu_job::GpuJob;
use gpu_scheduler::GPUScheduler;
use timer::Timer;

const USAGE: &str = "\
mosim: MoSim GPU Cluster Placement Simulator (Rust core)

This binary is the simulation engine. It does not parse user-facing CLI
flags; instead, it reads a JSON SimConfig from stdin. Use the Python
front-end (simulator-trace-timer-bw.py) for the user-facing CLI.

Usage:
    simulator-trace-timer-bw.py --num_node 8 --schedule k8s-bin-packing ...
    cat config.json | mosim                     # for debugging
";

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let k = (p / 100.0) * (n - 1) as f64;
    let f = k.floor() as usize;
    let c = k.ceil() as usize;
    if f == c {
        sorted[f]
    } else {
        let d0 = sorted[f] * (c as f64 - k);
        let d1 = sorted[c] * (k - f as f64);
        d0 + d1
    }
}

fn main() -> ExitCode {
    // Surface --help / -h locally even though there are no real flags,
    // so accidental invocations are not silently waiting on stdin.
    if let Some(arg) = std::env::args().nth(1) {
        if arg == "--help" || arg == "-h" {
            print!("{}", USAGE);
            return ExitCode::SUCCESS;
        }
        eprintln!("mosim: unexpected argument: {}", arg);
        eprintln!("{}", USAGE);
        return ExitCode::from(2);
    }

    let mut buf = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut buf) {
        eprintln!("mosim: failed to read config from stdin: {}", e);
        return ExitCode::from(1);
    }

    let cfg: SimConfig = match serde_json::from_str(&buf) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("mosim: invalid SimConfig JSON: {}", e);
            return ExitCode::from(1);
        }
    };

    // Echo the parsed config so a user running this manually can confirm
    // what the simulator actually received. Goes to stdout (one line) so
    // it does not pollute stderr-based progress logs.
    println!("[mosim] config: {:?}", cfg);

    // Per-event tracing is opt-in; see mosim/src/trace.rs.
    trace::init(&cfg.trace_log);

    run(cfg);
    trace::flush();
    ExitCode::SUCCESS
}

fn run(cfg: SimConfig) {
    let run_start = Instant::now();
    let servers = cfg.num_node;
    let gpus_per_server = cfg.num_gpus_per_node;
    let cpu_cores_per_server = cfg.num_cpus_per_node;
    let bandwidth_per_server = cfg.bandwidth as f64;
    let intra_bandwidth_per_server = cfg.intra_bandwidth as f64;

    // Load iteration time profile (solo + colocated rows + loading time).
    let iter_profile = load_iteration_time_csv(&cfg.iteration_time_csv_file);
    // Load communication-volume profile -> profiled_network (MB/s).
    let net_profile = load_communication_volume_csv(&cfg.communication_volume_csv_file);

    // Load job trace.
    //
    // The new trace format only carries placement/queueing inputs:
    //   job_id, model, arrival_time, num_iteration, gpu_workers,
    //   gpu_per_worker, cpu_per_gpu_worker
    // (an optional `duration` column is ignored). Iteration timing,
    // loading time, and profiled_network all come from the two profile
    // CSVs above.
    let mut jobs: Vec<GpuJob> = Vec::new();
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(&cfg.jobtrace)
        .expect("Cannot open job trace file");

    let headers = rdr
        .headers()
        .expect("Cannot read job trace header")
        .clone();
    let col = |name: &str| -> Option<usize> {
        headers.iter().position(|h| h.eq_ignore_ascii_case(name))
    };
    let col_required = |name: &str| -> usize {
        col(name).unwrap_or_else(|| {
            panic!(
                "job trace is missing required column '{}'. Trace columns: {:?}",
                name,
                headers.iter().collect::<Vec<_>>()
            )
        })
    };

    let c_job_id = col_required("job_id");
    let c_model = col_required("model");
    let c_arrival = col_required("arrival_time");
    let c_num_iter = col_required("num_iteration");
    let c_gpu_workers = col_required("gpu_workers");
    let c_gpu_per_worker = col_required("gpu_per_worker");
    let c_cpu_per_gpu = col_required("cpu_per_gpu_worker");
    // Optional columns: ps, cpu_per_ps_worker, skewness, tensorsizes.
    let c_ps = col("ps");
    let c_cpu_per_ps = col("cpu_per_ps_worker");
    let c_skewness = col("skewness");
    let c_tensorsizes = col("tensorsizes");

    for result in rdr.records() {
        let record = result.expect("Error reading CSV record");

        let pick_str = |idx: usize| record.get(idx).unwrap_or("");
        let pick_opt_f64 = |idx: Option<usize>| -> f64 {
            idx.and_then(|i| record.get(i))
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0)
        };
        let pick_opt_i32 = |idx: Option<usize>| -> i32 {
            idx.and_then(|i| record.get(i))
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        };

        let model: String = pick_str(c_model).to_string();
        let gpu_workers: i32 = pick_str(c_gpu_workers).parse().unwrap_or_else(|_| {
            panic!("trace: invalid gpu_workers for model={}", model)
        });
        let gpu_per_worker: i32 = pick_str(c_gpu_per_worker).parse().unwrap_or(1);
        let total_gpus = gpu_workers * gpu_per_worker;

        let solo_key = (model.clone(), total_gpus);
        let (iter_compute, iter_network) = iter_profile
            .solo
            .get(&solo_key)
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "iteration_time_csv: missing solo row for (model={}, total_gpus={}). \
                     A solo row has empty colocated_model_name and colocated_gpu_workers.",
                    solo_key.0, solo_key.1
                )
            });
        // Default to the solo loading_time at job creation. If the
        // corun-profile interference model is active, this value is
        // overwritten at `place_job` time with a snapshot that reflects
        // the co-runners present at the moment of resource allocation.
        let loading_time = iter_profile
            .solo_loading
            .get(&solo_key)
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "iteration_time_csv: missing solo loading_time for \
                     (model={}, total_gpus={}). A solo row (empty colocated_* \
                     columns) must define loading_time for every (model, total_gpus) \
                     used in the trace.",
                    solo_key.0, solo_key.1
                )
            });
        // Single-GPU jobs have no inter-server communication, so the
        // network summary CSV is allowed to omit them (profiled_network
        // = 0). For multi-GPU jobs the row must be present.
        let profiled_network = match net_profile.get(&solo_key).copied() {
            Some(v) => v,
            None if solo_key.1 <= 1 => 0.0,
            None => panic!(
                "communication_volume_csv: missing row for (model={}, num_workers={})",
                solo_key.0, solo_key.1
            ),
        };

        let job = GpuJob::new(
            pick_str(c_job_id).parse().unwrap_or(0),         // job_id
            model,
            pick_str(c_arrival).parse().unwrap_or(0.0),      // arrival_time
            pick_str(c_num_iter).parse().unwrap_or(0.0),     // num_iteration
            iter_compute,                                    // iteration_computing_time
            iter_network,                                    // iteration_networking_time
            gpu_workers,
            pick_opt_i32(c_ps),                              // ps (default 0)
            gpu_per_worker,
            pick_str(c_cpu_per_gpu).parse().unwrap_or(0),    // cpu_per_gpu_worker
            pick_opt_i32(c_cpu_per_ps),                      // cpu_per_ps_worker (default 0)
            pick_opt_f64(c_tensorsizes),                     // tensorsizes (default 0)
            pick_opt_f64(c_skewness),                        // skewness (default 0)
            profiled_network,                                // profiled_network (MB/s)
            loading_time,
        );
        jobs.push(job);
    }

    // Ensure jobs are indexed by job_id (arena pattern)
    jobs.sort_by_key(|j| j.job_id);

    // Build a dense arena: jobs[i] should have job_id == i
    let max_id = jobs.iter().map(|j| j.job_id).max().unwrap_or(0);
    if jobs.len() as i32 != max_id + 1 || jobs[0].job_id != 0 {
        for (idx, job) in jobs.iter_mut().enumerate() {
            job.job_id = idx as i32;
        }
    }

    // Write allocation log header
    let placement = PlacementMethod::from_str(&cfg.schedule);
    if let Ok(mut f) = OpenOptions::new()
        .append(true)
        .create(true)
        .open(&cfg.allocationlog)
    {
        let _ = writeln!(
            f,
            "logging... placement: {} current time: now",
            placement.as_str()
        );
    }

    let interference_model = InterferenceModel::from_str(&cfg.interference_model);

    // Forward the corun-profile tables only when the user actually
    // selected `corun-profile`; for other models, an empty profile keeps
    // the cluster lean.
    let corun_profile = if interference_model.is_corun_profile() {
        CorunProfile {
            solo: iter_profile.solo.clone(),
            colocate: iter_profile.colocate.clone(),
            colocate_loading: iter_profile.colocate_loading.clone(),
        }
    } else {
        CorunProfile::default()
    };

    let mut timer = Timer::new();
    let mut cluster = GPUCluster::new(
        servers,
        gpus_per_server,
        bandwidth_per_server,
        intra_bandwidth_per_server,
        cpu_cores_per_server,
        cfg.allocationlog.clone(),
        interference_model,
        cfg.interference_ratio,
        cfg.lcm_time_decimals,
        cfg.enable_lcm_cycle_jump,
        cfg.overlapping_ratio,
        corun_profile,
        cfg.gpu_util_log.clone(),
        cfg.required_bandwidth_cap_factor,
        cfg.comm_contention_fraction,
        cfg.cap_n_exponent,
        cfg.min_guaranteed_bw,
        matches!(cfg.contention_model.as_deref(), Some("mm1")),
        cfg.mm1_epsilon.unwrap_or(0.05),
        cfg.phase_overlap_weighted,
        cfg.hol_efficiency,
    );

    let mut scheduler = GPUScheduler::new(placement, &mut jobs, &mut timer);
    scheduler.run(&mut jobs, &mut cluster, &mut timer);

    // Persist the buffered GPU utilization log (no-op if disabled).
    cluster.flush_gpu_util_log();

    let metrics = scheduler.metrics(&jobs);
    let job_metrics = &metrics.jobs;

    if job_metrics.is_empty() {
        println!("No jobs completed.");
        return;
    }

    let jcts: Vec<f64> = job_metrics.iter().map(|m| m.jct).collect();
    let wait_times: Vec<f64> = job_metrics.iter().map(|m| m.wait_time).collect();
    let queueing_waits: Vec<f64> = job_metrics.iter().map(|m| m.queueing_wait_time).collect();
    let capacity_waits: Vec<f64> = job_metrics.iter().map(|m| m.capacity_wait_time).collect();
    let placement_waits: Vec<f64> = job_metrics.iter().map(|m| m.placement_wait_time).collect();
    let training_times: Vec<f64> = job_metrics.iter().map(|m| m.training_time).collect();
    let loading_times_vec: Vec<f64> = job_metrics.iter().map(|m| m.loading_time).collect();

    let n = job_metrics.len() as f64;
    let avg_jct = jcts.iter().sum::<f64>() / n;
    let avg_waiting = wait_times.iter().sum::<f64>() / n;
    let avg_training = training_times.iter().sum::<f64>() / n;
    let avg_loading = loading_times_vec.iter().sum::<f64>() / n;

    let tail_jct = percentile(&jcts, 99.0);
    let tail_waiting = percentile(&wait_times, 99.0);
    let tail_queueing = percentile(&queueing_waits, 99.0);
    let tail_capacity = percentile(&capacity_waits, 99.0);
    let tail_placement = percentile(&placement_waits, 99.0);
    let tail_training = percentile(&training_times, 99.0);
    let tail_loading = percentile(&loading_times_vec, 99.0);

    let max_end_time = job_metrics
        .iter()
        .map(|m| m.end_time)
        .fold(0.0_f64, f64::max);
    let min_arrival_time = job_metrics
        .iter()
        .map(|m| m.arrival_time)
        .fold(f64::MAX, f64::min);
    let makespan = max_end_time - min_arrival_time;

    // Write CSV output
    let output_csv = cfg.log.replace(".txt", ".csv");
    write_csv_output(&output_csv, placement.as_str(), job_metrics);

    // Write text output
    let sched_idle = metrics.sched_idle_time;
    let sched_cap = metrics.sched_capacity_wait_time;
    let sched_place = metrics.sched_placement_wait_time;

    if let Ok(mut f) = OpenOptions::new()
        .append(true)
        .create(true)
        .open(&cfg.log)
    {
        let _ = writeln!(f, "placement name: {}", placement.as_str());
        let _ = writeln!(
            f,
            "Job ids: {:?}",
            job_metrics.iter().map(|m| m.job_id).collect::<Vec<_>>()
        );
        let _ = writeln!(
            f,
            "Arrival times: {:?}",
            job_metrics
                .iter()
                .map(|m| m.arrival_time)
                .collect::<Vec<_>>()
        );
        let _ = writeln!(
            f,
            "Start times: {:?}",
            job_metrics
                .iter()
                .map(|m| m.start_time)
                .collect::<Vec<_>>()
        );
        let _ = writeln!(
            f,
            "End times: {:?}",
            job_metrics.iter().map(|m| m.end_time).collect::<Vec<_>>()
        );
        let _ = writeln!(
            f,
            "Wait times: {:?}",
            job_metrics
                .iter()
                .map(|m| m.wait_time)
                .collect::<Vec<_>>()
        );
        let _ = writeln!(f, "Queueing wait times: {:?}", queueing_waits);
        let _ = writeln!(f, "Capacity wait times: {:?}", capacity_waits);
        let _ = writeln!(f, "Placement wait times: {:?}", placement_waits);
        let _ = writeln!(f, "Queueing delays: {:?}", queueing_waits);
        let _ = writeln!(f, "Capacity wait times: {:?}", capacity_waits);
        let _ = writeln!(f, "Placement wait times: {:?}", placement_waits);
        let _ = writeln!(f, "Training times: {:?}", training_times);
        let _ = writeln!(f, "Loading times: {:?}", loading_times_vec);
        let _ = writeln!(f, "JCTs: {:?}", jcts);
        let _ = writeln!(
            f,
            "Compute times: {:?}",
            job_metrics
                .iter()
                .map(|m| m.compute_time)
                .collect::<Vec<_>>()
        );
        let _ = writeln!(
            f,
            "Comms times: {:?}",
            job_metrics
                .iter()
                .map(|m| m.comms_time)
                .collect::<Vec<_>>()
        );
        let _ = writeln!(f, "avg_jct: {}", avg_jct);
        let _ = writeln!(f, "avg_waiting: {}", avg_waiting);
        let _ = writeln!(f, "avg_training: {}", avg_training);
        let _ = writeln!(f, "avg_loading: {}", avg_loading);
        let _ = writeln!(f, "tail_jct: {}", tail_jct);
        let _ = writeln!(f, "tail_waiting: {}", tail_waiting);
        let _ = writeln!(f, "tail_queueing_wait_time: {}", tail_queueing);
        let _ = writeln!(f, "tail_capacity_wait_time: {}", tail_capacity);
        let _ = writeln!(f, "tail_placement_wait_time: {}", tail_placement);
        let _ = writeln!(f, "tail_training: {}", tail_training);
        let _ = writeln!(f, "tail_loading: {}", tail_loading);
        let _ = writeln!(f, "Makespan: {}", makespan);
        let _ = writeln!(
            f,
            "sched_idle_time: {}, ratio: {:.2}",
            sched_idle,
            if makespan > 0.0 {
                sched_idle / makespan
            } else {
                0.0
            }
        );
        let _ = writeln!(
            f,
            "sched_capacity_wait_time: {}, ratio: {:.2}",
            sched_cap,
            if makespan > 0.0 {
                sched_cap / makespan
            } else {
                0.0
            }
        );
        let _ = writeln!(
            f,
            "sched_placement_wait_time: {}, ratio: {:.2}",
            sched_place,
            if makespan > 0.0 {
                sched_place / makespan
            } else {
                0.0
            }
        );
        let _ = writeln!(
            f,
            "sim_runtime_seconds: {:.6}",
            run_start.elapsed().as_secs_f64()
        );
    }

    // Print summary
    println!("placement name: {}", placement.as_str());
    println!(
        "Job ids: {:?}",
        job_metrics.iter().map(|m| m.job_id).collect::<Vec<_>>()
    );
    println!("avg_jct: {}", avg_jct);
    println!("avg_waiting: {}", avg_waiting);
    println!("avg_training: {}", avg_training);
    println!("avg_loading: {}", avg_loading);
    println!("tail_jct: {}", tail_jct);
    println!("tail_waiting: {}", tail_waiting);
    println!("tail_training: {}", tail_training);
    println!("tail_loading: {}", tail_loading);
    println!("Makespan: {}", makespan);
    println!(
        "sched_idle_time: {}, ratio: {:.2}",
        sched_idle,
        if makespan > 0.0 {
            sched_idle / makespan
        } else {
            0.0
        }
    );
    println!(
        "sched_capacity_wait_time: {}, ratio: {:.2}",
        sched_cap,
        if makespan > 0.0 {
            sched_cap / makespan
        } else {
            0.0
        }
    );
    println!(
        "sched_placement_wait_time: {}, ratio: {:.2}",
        sched_place,
        if makespan > 0.0 {
            sched_place / makespan
        } else {
            0.0
        }
    );
    println!(
        "sim_runtime_seconds: {:.6}",
        run_start.elapsed().as_secs_f64()
    );

    // Print cycle jump stats if LCM model
    if cluster.cycle_stats.jump_applied > 0 {
        println!("\n--- LCM Cycle Jump Statistics ---");
        println!(
            "observed_sync: {}, cycle_detected: {}, jump_applied: {}, jump_periods: {}, jumped_ticks: {}",
            cluster.cycle_stats.observed_sync,
            cluster.cycle_stats.cycle_detected,
            cluster.cycle_stats.jump_applied,
            cluster.cycle_stats.jump_periods,
            cluster.cycle_stats.jumped_ticks,
        );
    }
}

fn write_csv_output(
    output_path: &str,
    placement_name: &str,
    job_metrics: &[crate::gpu_scheduler::JobMetrics],
) {
    let header = "placement,job_ids,models,arrival_times,start_times,end_times,wait_times,queueing_wait_times,capacity_wait_times,placement_wait_times,training_times,jcts,compute_times,comms_times,loading_times,calculated_gpu_util";

    let mut rows: Vec<String> = Vec::new();
    for m in job_metrics {
        let gpu_util = if m.jct > 0.0 {
            100.0 * m.compute_time / m.jct
        } else {
            0.0
        };
        rows.push(format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            placement_name,
            m.job_id,
            m.model,
            m.arrival_time,
            m.start_time,
            m.end_time,
            m.wait_time,
            m.queueing_wait_time,
            m.capacity_wait_time,
            m.placement_wait_time,
            m.training_time,
            m.jct,
            m.compute_time,
            m.comms_time,
            m.loading_time,
            gpu_util,
        ));
    }

    let content = if std::path::Path::new(output_path).exists() {
        let existing = fs::read_to_string(output_path).unwrap_or_default();
        let mut lines: Vec<String> = existing.lines().map(String::from).collect();
        for row in rows {
            lines.push(row);
        }
        lines.join("\n") + "\n"
    } else {
        let mut lines = vec![header.to_string()];
        lines.extend(rows);
        lines.join("\n") + "\n"
    };

    fs::write(output_path, content).expect("Failed to write CSV output");
}

/// In-memory representation of `iteration_time_csv_file`.
struct IterationTimeProfile {
    /// Solo iteration time per (model, total_gpus): (compute, networking).
    solo: HashMap<(String, i32), (f64, f64)>,
    /// Solo (no colocation) loading time per (model, total_gpus). Taken
    /// strictly from solo rows so it can serve as the "no co-runner"
    /// fallback at placement time.
    solo_loading: HashMap<(String, i32), f64>,
    /// Loading time per (model, total_gpus, co_model, co_total_gpus).
    /// Populated only from colocated rows; consumed by the corun-profile
    /// snapshot at job placement time.
    colocate_loading: HashMap<(String, i32, String, i32), f64>,
    /// Colocated iteration time per (model, total_gpus, co_model,
    /// co_total_gpus): (compute, networking).
    colocate: HashMap<(String, i32, String, i32), (f64, f64)>,
}

fn load_iteration_time_csv(path: &str) -> IterationTimeProfile {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap_or_else(|e| panic!("Cannot open iteration_time_csv_file '{}': {}", path, e));

    let headers = rdr
        .headers()
        .unwrap_or_else(|e| panic!("Cannot read header of '{}': {}", path, e))
        .clone();
    let must = |name: &str| -> usize {
        headers.iter().position(|h| h.eq_ignore_ascii_case(name)).unwrap_or_else(|| {
            panic!(
                "iteration_time_csv: missing required column '{}' in '{}'",
                name, path
            )
        })
    };
    let c_model = must("model_name");
    let c_compute = must("iteration_computing_time");
    let c_networking = must("iteration_networking_time");
    let c_workers = must("gpu_workers");
    let c_loading = must("loading_time");
    let c_co_model = must("colocated_model_name");
    let c_co_workers = must("colocated_gpu_workers");

    let mut solo: HashMap<(String, i32), (f64, f64)> = HashMap::new();
    let mut solo_loading: HashMap<(String, i32), f64> = HashMap::new();
    let mut colocate_loading: HashMap<(String, i32, String, i32), f64> = HashMap::new();
    let mut colocate: HashMap<(String, i32, String, i32), (f64, f64)> = HashMap::new();

    for (lineno, result) in rdr.records().enumerate() {
        let rec = result.unwrap_or_else(|e| {
            panic!("iteration_time_csv: read error at line {}: {}", lineno + 2, e)
        });
        let model = rec.get(c_model).unwrap_or("").to_string();
        let workers: i32 = rec.get(c_workers).unwrap_or("0").parse().unwrap_or_else(|_| {
            panic!(
                "iteration_time_csv: invalid gpu_workers at line {} (value={:?})",
                lineno + 2,
                rec.get(c_workers)
            )
        });
        let compute: f64 = rec.get(c_compute).unwrap_or("0").parse().unwrap_or_else(|_| {
            panic!(
                "iteration_time_csv: invalid iteration_computing_time at line {}",
                lineno + 2
            )
        });
        let networking: f64 = rec.get(c_networking).unwrap_or("0").parse().unwrap_or_else(|_| {
            panic!(
                "iteration_time_csv: invalid iteration_networking_time at line {}",
                lineno + 2
            )
        });
        let lt: f64 = rec.get(c_loading).unwrap_or("0").parse().unwrap_or_else(|_| {
            panic!(
                "iteration_time_csv: invalid loading_time at line {}",
                lineno + 2
            )
        });

        let key = (model.clone(), workers);
        let co_model = rec.get(c_co_model).unwrap_or("").to_string();
        let co_workers_str = rec.get(c_co_workers).unwrap_or("");

        if co_model.is_empty() && co_workers_str.is_empty() {
            // Solo row.
            if let Some(prev) = solo.insert(key.clone(), (compute, networking)) {
                eprintln!(
                    "iteration_time_csv: duplicate solo row for (model={}, gpu_workers={}); \
                     overwriting prior ({:.6},{:.6}) with ({:.6},{:.6})",
                    key.0, key.1, prev.0, prev.1, compute, networking
                );
            }
            if let Some(prev) = solo_loading.insert(key.clone(), lt) {
                if (prev - lt).abs() > 1e-9 {
                    eprintln!(
                        "iteration_time_csv: duplicate solo loading_time for \
                         (model={}, gpu_workers={}); overwriting prior {:.6} with {:.6}",
                        key.0, key.1, prev, lt
                    );
                }
            }
        } else if !co_model.is_empty() && !co_workers_str.is_empty() {
            // Colocated row.
            let co_workers: i32 = co_workers_str.parse().unwrap_or_else(|_| {
                panic!(
                    "iteration_time_csv: invalid colocated_gpu_workers at line {}",
                    lineno + 2
                )
            });
            let ck = (model.clone(), workers, co_model.clone(), co_workers);
            if let Some(prev) = colocate.insert(ck.clone(), (compute, networking)) {
                eprintln!(
                    "iteration_time_csv: duplicate colocated row for \
                     (model={}, gw={}, co_model={}, co_gw={}); \
                     overwriting prior ({:.6},{:.6}) with ({:.6},{:.6})",
                    ck.0, ck.1, ck.2, ck.3, prev.0, prev.1, compute, networking
                );
            }
            if let Some(prev) = colocate_loading.insert(ck.clone(), lt) {
                if (prev - lt).abs() > 1e-9 {
                    eprintln!(
                        "iteration_time_csv: duplicate colocated loading_time for \
                         (model={}, gw={}, co_model={}, co_gw={}); \
                         overwriting prior {:.6} with {:.6}",
                        ck.0, ck.1, ck.2, ck.3, prev, lt
                    );
                }
            }
        } else {
            panic!(
                "iteration_time_csv: line {} has only one of \
                 colocated_model_name / colocated_gpu_workers populated; \
                 both must be set together (colocated row) or both empty (solo row)",
                lineno + 2
            );
        }
    }

    IterationTimeProfile {
        solo,
        solo_loading,
        colocate_loading,
        colocate,
    }
}

fn load_communication_volume_csv(path: &str) -> HashMap<(String, i32), f64> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap_or_else(|e| {
            panic!("Cannot open communication_volume_csv_file '{}': {}", path, e)
        });

    let headers = rdr
        .headers()
        .unwrap_or_else(|e| panic!("Cannot read header of '{}': {}", path, e))
        .clone();
    let must = |name: &str| -> usize {
        headers
            .iter()
            .position(|h| h.eq_ignore_ascii_case(name))
            .unwrap_or_else(|| {
                panic!(
                    "communication_volume_csv: missing required column '{}' in '{}'",
                    name, path
                )
            })
    };
    let c_model = must("Model");
    let c_workers = must("Number of Workers");
    // "Sum of Max TX+RX (MB/s)" is the agreed-upon source for C_j (MB/s).
    let c_mbs = must("Sum of Max TX+RX (MB/s)");

    let mut out: HashMap<(String, i32), f64> = HashMap::new();
    for (lineno, result) in rdr.records().enumerate() {
        let rec = result.unwrap_or_else(|e| {
            panic!(
                "communication_volume_csv: read error at line {}: {}",
                lineno + 2,
                e
            )
        });
        let model = rec.get(c_model).unwrap_or("").to_string();
        let workers: i32 = rec.get(c_workers).unwrap_or("0").parse().unwrap_or_else(|_| {
            panic!(
                "communication_volume_csv: invalid 'Number of Workers' at line {}",
                lineno + 2
            )
        });
        let mbs: f64 = rec.get(c_mbs).unwrap_or("0").parse().unwrap_or_else(|_| {
            panic!(
                "communication_volume_csv: invalid 'Sum of Max TX+RX (MB/s)' at line {}",
                lineno + 2
            )
        });
        let key = (model, workers);
        if let Some(prev) = out.insert(key.clone(), mbs) {
            eprintln!(
                "communication_volume_csv: duplicate (Model={}, Number of Workers={}); \
                 overwriting {:.3} with {:.3}",
                key.0, key.1, prev, mbs
            );
        }
    }
    out
}
