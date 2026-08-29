use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;

use crate::gpu_job::{GpuJob, Phase};
use crate::lcm_utils::lcm_many;
use crate::timer::{EventType, Timer};

/// Per-GPU utilization (percent) when a GPU is allocated to any job.
/// Idle GPUs report `0.0`. Centralized here so swapping in a non-binary
/// model later only needs to touch this constant (or its callers).
pub const GPU_UTIL_WHEN_ALLOCATED: f64 = 100.0;

/// Placement algorithm selector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlacementMethod {
    K8sLoadBalancing,
    K8sBinPacking,
    Colocate,
    Tiresias,
}
impl PlacementMethod {
    pub fn from_str(s: &str) -> Self {
        match s {
            "k8s-load-balancing" => PlacementMethod::K8sLoadBalancing,
            "k8s-bin-packing" => PlacementMethod::K8sBinPacking,
            "colocate" => PlacementMethod::Colocate,
            "tiresias" => PlacementMethod::Tiresias,
            _ => panic!("Unknown placement method: {}", s),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            PlacementMethod::K8sLoadBalancing => "k8s-load-balancing",
            PlacementMethod::K8sBinPacking => "k8s-bin-packing",
            PlacementMethod::Colocate => "colocate",
            PlacementMethod::Tiresias => "tiresias",
        }
    }
}

/// Interference model for bandwidth contention.
#[derive(Debug, Clone, PartialEq)]
pub enum InterferenceModel {
    None,
    Fixed,
    Comms,
    CommsIter,
    CommsIterIntra,
    CommsIterLcm,
    CommsIterIntraLcm,
    /// Co-run profile: iteration_computing_time and iteration_networking_time
    /// are looked up per-(model, gpu_workers) and updated dynamically based
    /// on which jobs share each server. Single-iteration mode (no compute/
    /// comm phase split, no bandwidth contention multiplier).
    CorunProfile,
}

impl InterferenceModel {
    pub fn from_str(s: &str) -> Self {
        match s {
            "none" => InterferenceModel::None,
            "fixed" => InterferenceModel::Fixed,
            "comms" => InterferenceModel::Comms,
            "comms-iter" => InterferenceModel::CommsIter,
            // `mosim` is the canonical name for the configuration used in the
            // MASCOTS 2026 paper; `comms-iter-intra` is kept as an alias that
            // describes the mechanism (comms base + iteration phases + intra).
            "mosim" | "comms-iter-intra" => InterferenceModel::CommsIterIntra,
            "comms-iter-lcm" => InterferenceModel::CommsIterLcm,
            "comms-iter-intra-lcm" => InterferenceModel::CommsIterIntraLcm,
            "corun-profile" => InterferenceModel::CorunProfile,
            _ => panic!("Unknown interference model: {}", s),
        }
    }

    pub fn is_iter_model(&self) -> bool {
        matches!(
            self,
            InterferenceModel::CommsIter
                | InterferenceModel::CommsIterIntra
                | InterferenceModel::CommsIterLcm
                | InterferenceModel::CommsIterIntraLcm
        )
    }

    pub fn is_lcm_model(&self) -> bool {
        matches!(
            self,
            InterferenceModel::CommsIterLcm | InterferenceModel::CommsIterIntraLcm
        )
    }

    pub fn is_intra_model(&self) -> bool {
        matches!(
            self,
            InterferenceModel::CommsIterIntra | InterferenceModel::CommsIterIntraLcm
        )
    }

    pub fn is_corun_profile(&self) -> bool {
        matches!(self, InterferenceModel::CorunProfile)
    }
}

/// Profile lookup tables for the `corun-profile` interference model.
///
/// All tables are keyed off the *target* job (the job whose value we are
/// computing).
#[derive(Default, Clone)]
pub struct CorunProfile {
    /// Solo (no colocation) iteration time: (compute, networking).
    pub solo: HashMap<(String, i32), (f64, f64)>,
    /// Colocated iteration time. Key adds (colocated_model,
    /// colocated_gpu_workers); value is (compute, networking) for the
    /// target job when run alongside that one colocated job.
    pub colocate: HashMap<(String, i32, String, i32), (f64, f64)>,
    /// Colocated loading time, same key shape as `colocate`. Consumed
    /// once per job by the placement-time snapshot in `place_job`; the
    /// recorded value is then frozen for the remainder of the job's
    /// lifetime. Empty for non-corun-profile runs.
    pub colocate_loading: HashMap<(String, i32, String, i32), f64>,
}

/// Cycle jump profile for LCM optimization.
#[derive(Clone)]
struct CycleProfile {
    period_ticks: i64,
    delta_iterations: HashMap<i32, i64>,
    delta_training: HashMap<i32, f64>,
    delta_compute: HashMap<i32, f64>,
    delta_comms: HashMap<i32, f64>,
}

/// Cycle snapshot for observation.
#[derive(Clone)]
struct CycleSnapshot {
    tick: i64,
    iterations: HashMap<i32, i64>,
    training: HashMap<i32, f64>,
    compute: HashMap<i32, f64>,
    comms: HashMap<i32, f64>,
}

/// Cycle key for cache lookups.
type CycleKey = (
    u64,                             // placement_epoch
    String,                          // interference_model name
    i64,                             // base lcm ticks
    Vec<(i32, i64, Vec<i32>, Vec<i32>, Vec<(i32, i64)>)>, // job entries
);

pub struct CycleStats {
    pub observed_sync: u64,
    pub cycle_detected: u64,
    pub jump_applied: u64,
    pub jump_periods: u64,
    pub jumped_ticks: u64,
}

/// GPU cluster state: manages resources, placement, and bandwidth.
pub struct GPUCluster {
    // Immutable config
    pub servers: i32,
    pub gpus_per_server: i32,
    pub bandwidth_per_server: f64,
    pub intra_bandwidth_per_server: f64,
    pub cpu_cores_per_server: i32,
    pub allocation_log_file: String,
    pub interference_model: InterferenceModel,
    pub interference_ratio: Option<f64>,
    pub lcm_time_decimals: i32,
    pub time_scale: f64,
    pub enable_lcm_cycle_jump: bool,
    /// Fraction of compute time (0..=1) that may overlap with communication.
    pub overlapping_ratio: f64,
    /// Multiplicative cap on `required_bandwidth(server)` as a multiple
    /// of `bandwidth_per_server`. `None` disables the cap.
    pub required_bandwidth_cap_factor: Option<f64>,
    /// Fraction of `iteration_networking_time` subject to contention.
    /// `None` or `1.0` = full (legacy). Smaller values dampen the
    /// effective contention factor.
    pub comm_contention_fraction: Option<f64>,
    /// Exponent β for N-scaling of the runtime cap:
    ///   `cap = alpha × (N/2)^β × bandwidth_per_server`
    /// `None` or 0 = uniform cap (legacy).
    pub cap_n_exponent: Option<f64>,
    /// Contention model: "linear" (default) or "mm1".
    pub contention_model_mm1: bool,
    /// Clamping epsilon for the M/M/1 model. Default 0.05.
    pub mm1_epsilon: f64,
    /// Use phase-staggered overlap probability for total demand.
    pub phase_overlap_weighted: bool,
    /// HOL blocking efficiency η ∈ (0,1]. When < 1, contention factor
    /// is scaled by 1/η. None = legacy linear sharing.
    pub hol_efficiency: Option<f64>,

    // Mutable resource state
    pub gpus: Vec<i32>,
    pub bandwidths: Vec<f64>,
    pub cpu_cores: Vec<i32>,
    pub intra_bandwidths: Vec<f64>,
    /// active_jobs[server] = list of job_ids active on that server
    pub active_jobs: Vec<Vec<i32>>,

    // LCM cycle jump state
    placement_epoch: u64,
    cycle_observation_cache: HashMap<CycleKey, CycleSnapshot>,
    cycle_profiles: HashMap<CycleKey, CycleProfile>,
    pub cycle_stats: CycleStats,

    // Minimum guaranteed bandwidth
    pub min_guaranteed_bw: f64,

    /// Profile tables for the `corun-profile` interference model.
    /// Empty for other models.
    pub corun_profile: CorunProfile,

    /// Optional per-event GPU utilization log target. Empty string =
    /// disabled. When enabled, every allocation state change
    /// (place_job success / release_resources) appends a snapshot row,
    /// dedup'd by timestamp (last write wins for the same timestamp).
    pub gpu_util_log_file: String,
    /// Buffered rows for the GPU util log; flushed once at end of run.
    /// Each entry is the fully formatted CSV row (no trailing newline).
    /// The last entry's timestamp is tracked for same-timestamp merging.
    gpu_util_rows: Vec<(f64, String)>,
}

impl GPUCluster {
    pub fn new(
        servers: i32,
        gpus_per_server: i32,
        bandwidth_per_server: f64,
        intra_bandwidth_per_server: f64,
        cpu_cores_per_server: i32,
        allocation_log_file: String,
        interference_model: InterferenceModel,
        interference_ratio: Option<f64>,
        lcm_time_decimals: i32,
        enable_lcm_cycle_jump: bool,
        overlapping_ratio: f64,
        corun_profile: CorunProfile,
        gpu_util_log_file: String,
        required_bandwidth_cap_factor: Option<f64>,
        comm_contention_fraction: Option<f64>,
        cap_n_exponent: Option<f64>,
        min_guaranteed_bw: Option<f64>,
        contention_model_mm1: bool,
        mm1_epsilon: f64,
        phase_overlap_weighted: bool,
        hol_efficiency: Option<f64>,
    ) -> Self {
        let n = servers as usize;
        let mut cluster = GPUCluster {
            servers,
            gpus_per_server,
            bandwidth_per_server,
            intra_bandwidth_per_server,
            cpu_cores_per_server,
            allocation_log_file,
            interference_model,
            interference_ratio,
            lcm_time_decimals,
            time_scale: 10.0_f64.powi(lcm_time_decimals),
            enable_lcm_cycle_jump,
            overlapping_ratio,
            required_bandwidth_cap_factor,
            comm_contention_fraction,
            cap_n_exponent,
            contention_model_mm1,
            mm1_epsilon,
            phase_overlap_weighted,
            hol_efficiency,
            gpus: vec![gpus_per_server; n],
            bandwidths: vec![bandwidth_per_server; n],
            cpu_cores: vec![cpu_cores_per_server; n],
            intra_bandwidths: vec![intra_bandwidth_per_server; n],
            active_jobs: (0..n).map(|_| Vec::new()).collect(),
            placement_epoch: 0,
            cycle_observation_cache: HashMap::new(),
            cycle_profiles: HashMap::new(),
            cycle_stats: CycleStats {
                observed_sync: 0,
                cycle_detected: 0,
                jump_applied: 0,
                jump_periods: 0,
                jumped_ticks: 0,
            },
            min_guaranteed_bw: min_guaranteed_bw.unwrap_or(10.0),
            corun_profile,
            gpu_util_log_file,
            gpu_util_rows: Vec::new(),
        };

        // Seed t=0 row (all GPUs idle) so the log always starts from a
        // well-defined baseline. No-op when logging is disabled.
        cluster.log_gpu_util_snapshot(0.0);
        cluster
    }

    /// Append a GPU utilization snapshot for `current_time`. Same-timestamp
    /// calls overwrite the previous row so a sequence like
    /// release_resources -> place_job at the same instant collapses into a
    /// single row reflecting the final post-event state. No-op when
    /// `gpu_util_log_file` is empty.
    pub fn log_gpu_util_snapshot(&mut self, current_time: f64) {
        if self.gpu_util_log_file.is_empty() {
            return;
        }

        let n = self.servers as usize;
        let gpus_per_server = self.gpus_per_server;
        let total_gpus = (self.servers * gpus_per_server) as f64;

        let mut per_node_avg = Vec::with_capacity(n);
        let mut total_active_gpus: i32 = 0;
        for s in 0..n {
            // gpus[s] is the *remaining* (free) GPU count on server s,
            // so the allocated count is gpus_per_server - gpus[s].
            let allocated = gpus_per_server - self.gpus[s];
            total_active_gpus += allocated;
            let node_avg = if gpus_per_server > 0 {
                allocated as f64 * GPU_UTIL_WHEN_ALLOCATED / gpus_per_server as f64
            } else {
                0.0
            };
            per_node_avg.push(node_avg);
        }

        let cluster_total_util = total_active_gpus as f64 * GPU_UTIL_WHEN_ALLOCATED;
        let cluster_avg_util = if total_gpus > 0.0 {
            cluster_total_util / total_gpus
        } else {
            0.0
        };

        // Format: timestamp_sec with 6 decimals, utils with 6 decimals,
        // active gpu count as integer. Mirrors the testbed CSV style but
        // uses seconds-from-zero instead of HH:MM:SS.
        let mut row = format!(
            "{:.6},{:.6},{:.6},{}",
            current_time, cluster_avg_util, cluster_total_util, total_active_gpus
        );
        for v in &per_node_avg {
            row.push(',');
            row.push_str(&format!("{:.6}", v));
        }

        match self.gpu_util_rows.last_mut() {
            // Same instant -> overwrite (final state wins).
            Some((t, last)) if (*t - current_time).abs() < 1e-12 => {
                *last = row;
            }
            _ => self.gpu_util_rows.push((current_time, row)),
        }
    }

    /// Write the buffered GPU util rows to disk in one shot. Truncates any
    /// existing file at the path. No-op when logging is disabled.
    pub fn flush_gpu_util_log(&self) {
        if self.gpu_util_log_file.is_empty() {
            return;
        }

        let mut header = String::from(
            "timestamp_sec,cluster_avg_util,cluster_total_util,cluster_active_gpus",
        );
        for s in 0..self.servers {
            header.push_str(&format!(",node{}_avg_util", s));
        }

        let mut out = String::with_capacity(header.len() + self.gpu_util_rows.len() * 64);
        out.push_str(&header);
        out.push('\n');
        for (_, row) in &self.gpu_util_rows {
            out.push_str(row);
            out.push('\n');
        }

        if let Err(e) = std::fs::write(&self.gpu_util_log_file, out) {
            eprintln!(
                "mosim: failed to write gpu_util_log '{}': {}",
                self.gpu_util_log_file, e
            );
        }
    }

    /// Total GPUs requested by `job` across all of its workers. This is
    /// the matching key for the `corun-profile` lookup tables and the
    /// `communication_volume_csv_file`.
    fn total_gpus_for(job: &GpuJob) -> i32 {
        job.gpu_workers * job.gpu_per_worker
    }

    /// Lookup colocated iteration time for (model, gpus, co_model, co_gpus).
    /// If the exact key is missing, falls back to the entry with the largest
    /// co_gpus' ≤ co_gpus for the same (model, gpus, co_model) prefix.
    /// Returns None if no matching entry exists at all.
    fn colocate_iter_lookup(
        &self,
        model: &str,
        gpus: i32,
        co_model: &str,
        co_gpus: i32,
    ) -> Option<(f64, f64)> {
        let exact = (model.to_string(), gpus, co_model.to_string(), co_gpus);
        if let Some(&v) = self.corun_profile.colocate.get(&exact) {
            return Some(v);
        }
        self.corun_profile
            .colocate
            .iter()
            .filter(|((m, g, cm, cg), _)| {
                m.as_str() == model && *g == gpus && cm.as_str() == co_model && *cg <= co_gpus
            })
            .max_by_key(|((_, _, _, cg), _)| *cg)
            .map(|(_, &v)| v)
    }

    /// Lookup colocated loading time for (model, gpus, co_model, co_gpus).
    /// If the exact key is missing, falls back to the entry with the largest
    /// co_gpus' ≤ co_gpus for the same (model, gpus, co_model) prefix.
    /// Returns None if no matching entry exists at all.
    fn colocate_loading_lookup(
        &self,
        model: &str,
        gpus: i32,
        co_model: &str,
        co_gpus: i32,
    ) -> Option<f64> {
        let exact = (model.to_string(), gpus, co_model.to_string(), co_gpus);
        if let Some(&v) = self.corun_profile.colocate_loading.get(&exact) {
            return Some(v);
        }
        self.corun_profile
            .colocate_loading
            .iter()
            .filter(|((m, g, cm, cg), _)| {
                m.as_str() == model && *g == gpus && cm.as_str() == co_model && *cg <= co_gpus
            })
            .max_by_key(|((_, _, _, cg), _)| *cg)
            .map(|(_, &v)| v)
    }

    /// Lookup the (compute, networking) iteration time for `job` given
    /// the set of `(model, total_gpus)` of *other* jobs that share at
    /// least one server with it on the server we are evaluating.
    ///
    /// Rules:
    /// - If `colocated` is empty: solo row.
    /// - Else: among rows whose colocated key matches one of the
    ///   colocated jobs, pick the row with the slowest
    ///   iteration_networking_time and return its `(compute, networking)`
    ///   pair as-is (do NOT mix max-of-compute with max-of-networking
    ///   across different rows).
    fn corun_iter_time_for_server(
        &self,
        job: &GpuJob,
        colocated: &[(String, i32)],
    ) -> (f64, f64) {
        let key = (job.model.clone(), Self::total_gpus_for(job));

        if colocated.is_empty() {
            return *self.corun_profile.solo.get(&key).unwrap_or_else(|| {
                panic!(
                    "corun-profile: missing solo iteration time for (model={}, total_gpus={}). \
                     Add a row with empty colocated_* columns to the iteration_time CSV.",
                    key.0, key.1
                )
            });
        }

        let mut best: Option<(f64, f64)> = None;
        for (co_model, co_gpus) in colocated {
            let val = match self.colocate_iter_lookup(&key.0, key.1, co_model, *co_gpus) {
                Some(v) => v,
                None => {
                    eprintln!(
                        "corun-profile: no colocated iteration time entry for \
                         (model={}, total_gpus={}, co_model={}, co_total_gpus={}); \
                         falling back to solo iteration time",
                        key.0, key.1, co_model, co_gpus
                    );
                    *self.corun_profile.solo.get(&key).unwrap_or(&(
                        job.iteration_computing_time,
                        job.iteration_networking_time,
                    ))
                }
            };
            best = Some(match best {
                None => val,
                Some(prev) if val.1 > prev.1 => val,
                Some(prev) => prev,
            });
        }
        best.expect("corun-profile: unreachable, colocated non-empty")
    }

    /// Compute the new iteration (compute, networking) time for `job`
    /// under the current placement. The returned values are bottlenecked
    /// by the slowest server the job uses (largest effective iteration
    /// time across servers).
    fn compute_corun_iteration_times(
        &self,
        job_id: i32,
        jobs: &[GpuJob],
    ) -> (f64, f64) {
        let job = &jobs[job_id as usize];
        let unique_servers: HashSet<i32> = job.allocated.iter().cloned().collect();

        let mut best: Option<(f64, f64, f64)> = None; // (eff_iter, compute, network)
        for &server in &unique_servers {
            // Collect distinct colocated jobs (excluding self) that are
            // currently *training* on this server. Loading-only or
            // completed jobs are not co-runners from a compute/network
            // interference standpoint.
            let mut seen: HashSet<i32> = HashSet::new();
            let mut colocated: Vec<(String, i32)> = Vec::new();
            for &other in &self.active_jobs[server as usize] {
                if other == job_id {
                    continue;
                }
                if !seen.insert(other) {
                    continue;
                }
                let other_job = &jobs[other as usize];
                if other_job.training_time.is_none() {
                    continue;
                }
                if other_job.current_phase == Phase::Completed {
                    continue;
                }
                colocated.push((other_job.model.clone(), Self::total_gpus_for(other_job)));
            }

            let (c, n) = self.corun_iter_time_for_server(job, &colocated);
            let eff = self.effective_iteration_time_with(job, c, n);
            best = Some(match best {
                None => (eff, c, n),
                Some(prev) if eff > prev.0 => (eff, c, n),
                Some(prev) => prev,
            });
        }

        match best {
            Some((_, c, n)) => (c, n),
            // No allocated servers (defensive): fall back to solo.
            None => self.corun_iter_time_for_server(job, &[]),
        }
    }

    /// Variant of `effective_iteration_time` that takes explicit
    /// (compute, networking) values rather than reading them off `job`.
    fn effective_iteration_time_with(
        &self,
        job: &GpuJob,
        compute: f64,
        networking: f64,
    ) -> f64 {
        let overlap_credit = (self.overlapping_ratio * compute).min(networking);
        compute + networking - overlap_credit
    }

    /// Refresh `iteration_computing_time` / `iteration_networking_time`
    /// for `job_id` from the corun profile, returning whether the value
    /// changed.
    fn refresh_corun_iter_times(&self, job_id: i32, jobs: &mut [GpuJob]) -> bool {
        let (new_c, new_n) = self.compute_corun_iteration_times(job_id, jobs);
        let job = &mut jobs[job_id as usize];
        let changed =
            (job.iteration_computing_time - new_c).abs() > 0.0
                || (job.iteration_networking_time - new_n).abs() > 0.0;
        job.iteration_computing_time = new_c;
        job.iteration_networking_time = new_n;
        changed
    }

    /// Compute the placement-time loading_time snapshot for `job_id`
    /// under the `corun-profile` interference model.
    ///
    /// Policy (mirrors the iteration-time lookup):
    /// - For each server on which the job has GPU workers, gather the
    ///   distinct co-runner `(model, total_gpus)` pairs currently sharing
    ///   that server, excluding `job_id` itself and any co-runner whose
    ///   phase is `Completed`. Loading-phase co-runners *are* included on
    ///   purpose: they share I/O at this very moment.
    /// - Per server, take the maximum `colocate_loading[(model, gpus,
    ///   co_model, co_gpus)]` across the co-runners. If a server has no
    ///   co-runners, fall back to `job.loading_time` (the solo value
    ///   stamped at job creation).
    /// - The job-wide snapshot is the maximum across all of its servers.
    ///
    /// The returned value is meant to be written into `job.loading_time`
    /// exactly once and then left untouched for the remainder of the
    /// run.
    fn corun_loading_time_snapshot(&self, job_id: i32, jobs: &[GpuJob]) -> f64 {
        let job = &jobs[job_id as usize];
        let solo = job.loading_time;
        let key_model = job.model.clone();
        let key_gpus = Self::total_gpus_for(job);
        let unique_servers: HashSet<i32> = job.allocated.iter().cloned().collect();

        let mut snapshot = solo;
        for &server in &unique_servers {
            let mut seen: HashSet<i32> = HashSet::new();
            let mut server_best = solo;
            for &other in &self.active_jobs[server as usize] {
                if other == job_id {
                    continue;
                }
                if !seen.insert(other) {
                    continue;
                }
                let other_job = &jobs[other as usize];
                if other_job.current_phase == Phase::Completed {
                    continue;
                }
                let ck = (
                    key_model.clone(),
                    key_gpus,
                    other_job.model.clone(),
                    Self::total_gpus_for(other_job),
                );
                let lt = match self.colocate_loading_lookup(
                    &ck.0, ck.1, &ck.2, ck.3,
                ) {
                    Some(v) => v,
                    None => {
                        eprintln!(
                            "corun-profile: no colocated loading_time entry for \
                             (model={}, total_gpus={}, co_model={}, co_total_gpus={}); \
                             falling back to solo loading time",
                            ck.0, ck.1, ck.2, ck.3
                        );
                        solo
                    }
                };
                if lt > server_best {
                    server_best = lt;
                }
            }
            // If a server has co-runners, server_best is the max over
            // them. Otherwise it stays at `solo`, the no-colocation
            // fallback already stamped on the job.
            if server_best > snapshot {
                snapshot = server_best;
            }
        }
        snapshot
    }

    /// HOL blocking is now baked into the bandwidth allocation step
    /// (`set_stable_bandwidth` caps total demand at `η · C` instead of
    /// `C`), so the per-job `using_bandwidths` map already reflects the
    /// HOL penalty. The `S_J^i = D_J^i / max(A_J^i, MIN_BW)` formula
    /// then naturally produces the HOL-amplified slowdown.
    pub fn apply_hol_blocking(&self, raw_factor: f64) -> f64 {
        // Kept as identity for backward compatibility; HOL is applied
        // upstream in `set_stable_bandwidth` now.
        raw_factor
    }

    /// Apply the `comm_contention_fraction` (r) reshape to a raw factor:
    ///   effective = 1 + r * (raw - 1)
    /// `r < 1` dampens contention; `r > 1` amplifies it; unset preserves
    /// legacy `effective = raw` behaviour.
    pub fn damp_contention_factor(&self, raw_factor: f64) -> f64 {
        match self.comm_contention_fraction {
            Some(r) if r.is_finite() && r > 0.0 => {
                1.0 + r * (raw_factor - 1.0)
            }
            _ => raw_factor,
        }
    }

    /// `job.required_bandwidth(server)` clamped to
    /// `required_bandwidth_cap_factor * bandwidth_per_server` when a cap is
    /// configured. The cap is scaled by `(N/2)^cap_n_exponent` so larger
    /// jobs (which carry more traffic across the boundary) can demand more
    /// bandwidth than smaller jobs. Default exponent 0 = uniform cap.
    pub fn required_bandwidth_capped(&self, job: &GpuJob, server: i32) -> f64 {
        let raw = job.required_bandwidth(server);
        match self.required_bandwidth_cap_factor {
            Some(alpha) if alpha.is_finite() && alpha > 0.0 => {
                let n_scale = match self.cap_n_exponent {
                    Some(beta) if beta.is_finite() && beta != 0.0 && job.gpu_workers > 1 => {
                        (job.gpu_workers as f64 / 2.0).powf(beta)
                    }
                    _ => 1.0,
                };
                raw.min(alpha * n_scale * self.bandwidth_per_server)
            }
            _ => raw,
        }
    }

    /// Wall-clock time per iteration with compute/communication overlap.
    pub fn effective_iteration_time(&self, job: &GpuJob, actual_networking_time: f64) -> f64 {
        let overlap_credit = (self.overlapping_ratio * job.iteration_computing_time)
            .min(actual_networking_time);
        job.iteration_computing_time + actual_networking_time - overlap_credit
    }

    fn to_tick(&self, time_float: f64) -> i64 {
        (time_float * self.time_scale).round() as i64
    }

    fn from_tick(&self, tick: i64) -> f64 {
        tick as f64 / self.time_scale
    }

    fn invalidate_cycle_state(&mut self) {
        self.cycle_observation_cache.clear();
        self.cycle_profiles.clear();
    }

    fn bump_placement_epoch(&mut self) {
        self.placement_epoch += 1;
        self.invalidate_cycle_state();
    }

    fn get_unique_active_training_job_ids(&self, jobs: &[GpuJob]) -> Vec<i32> {
        let mut job_set: HashSet<i32> = HashSet::new();
        for server in 0..self.servers as usize {
            for &job_id in &self.active_jobs[server] {
                let job = &jobs[job_id as usize];
                if job.training_time.is_none() {
                    continue;
                }
                if job.current_phase == Phase::Completed {
                    continue;
                }
                job_set.insert(job_id);
            }
        }
        let mut result: Vec<i32> = job_set.into_iter().collect();
        result.sort();
        result
    }

    fn compute_base_lcm_ticks(&self, job_ids: &[i32], jobs: &[GpuJob]) -> i64 {
        let mut raw_ticks = Vec::new();
        for &jid in job_ids {
            let job = &jobs[jid as usize];
            let base_iter = self.effective_iteration_time(job, job.iteration_networking_time);
            let total_tick = self.to_tick(base_iter);
            if total_tick > 0 {
                raw_ticks.push(total_tick);
            }
        }
        if raw_ticks.is_empty() {
            return 0;
        }
        lcm_many(&raw_ticks)
    }

    fn build_cycle_key(
        &self,
        job_ids: &[i32],
        jobs: &[GpuJob],
        intra_model: bool,
    ) -> CycleKey {
        let min_iteration = job_ids
            .iter()
            .map(|&jid| jobs[jid as usize].current_iteration)
            .min()
            .unwrap_or(0);

        let mut job_entries = Vec::new();
        for &jid in job_ids {
            let job = &jobs[jid as usize];
            let mut alloc_sorted = job.allocated.clone();
            alloc_sorted.sort();
            let mut ps_alloc_sorted = job.ps_allocated.clone();
            ps_alloc_sorted.sort();

            let mut intra_sig = Vec::new();
            if intra_model {
                let mut unique_servers: Vec<i32> = job.allocated.iter().cloned().collect::<HashSet<_>>().into_iter().collect();
                unique_servers.sort();
                for server in unique_servers {
                    let bw = (job.required_intra_bandwidth_per_server(server) * 1e6).round() as i64;
                    intra_sig.push((server, bw));
                }
            }

            job_entries.push((
                jid,
                job.current_iteration - min_iteration,
                alloc_sorted,
                ps_alloc_sorted,
                intra_sig,
            ));
        }

        let model_name = format!("{:?}", self.interference_model);
        let base_lcm = self.compute_base_lcm_ticks(job_ids, jobs);

        (self.placement_epoch, model_name, base_lcm, job_entries)
    }

    fn all_active_jobs_sync_compute_start(
        &self,
        job_ids: &[i32],
        jobs: &[GpuJob],
        current_tick: i64,
    ) -> bool {
        if job_ids.is_empty() {
            return false;
        }
        for &jid in job_ids {
            let job = &jobs[jid as usize];
            if job.current_phase != Phase::Compute {
                return false;
            }
            if job.phase_start_time.is_none() {
                return false;
            }
            if self.to_tick(job.phase_start_time.unwrap()) != current_tick {
                return false;
            }
        }
        true
    }

    fn build_cycle_snapshot(
        &self,
        job_ids: &[i32],
        jobs: &[GpuJob],
        current_tick: i64,
    ) -> CycleSnapshot {
        let mut iterations = HashMap::new();
        let mut training = HashMap::new();
        let mut compute = HashMap::new();
        let mut comms = HashMap::new();
        for &jid in job_ids {
            let job = &jobs[jid as usize];
            iterations.insert(jid, job.current_iteration);
            training.insert(jid, job.training_time.unwrap_or(0.0));
            compute.insert(jid, job.consumed_compute_time.unwrap_or(0.0));
            comms.insert(jid, job.consumed_comms_time.unwrap_or(0.0));
        }
        CycleSnapshot {
            tick: current_tick,
            iterations,
            training,
            compute,
            comms,
        }
    }

    fn compute_jump_count(
        &self,
        profile: &CycleProfile,
        job_ids: &[i32],
        jobs: &[GpuJob],
        current_tick: i64,
        timer: &Timer,
    ) -> i64 {
        let period_ticks = profile.period_ticks;
        if period_ticks <= 0 {
            return 0;
        }

        let mut k_by_external: i64 = i64::MAX;
        if let Some(next_ext) = timer.peek_next_external_state_change_event() {
            let next_external_tick = self.to_tick(next_ext.0);
            let remaining_ticks = next_external_tick - current_tick - 1;
            if remaining_ticks < period_ticks {
                return 0;
            }
            k_by_external = remaining_ticks / period_ticks;
        }

        let mut k_by_remaining: i64 = i64::MAX;
        for &jid in job_ids {
            let job = &jobs[jid as usize];
            let delta_iter = match profile.delta_iterations.get(&jid) {
                Some(&d) if d > 0 => d,
                _ => return 0,
            };
            let remaining_before_completion =
                job.iteration_number as i64 - 1 - job.current_iteration;
            if remaining_before_completion < 0 {
                return 0;
            }
            let k_for_job = remaining_before_completion / delta_iter;
            k_by_remaining = k_by_remaining.min(k_for_job);
        }

        if k_by_remaining <= 0 {
            return 0;
        }
        0_i64.max(k_by_external.min(k_by_remaining))
    }

    fn apply_cycle_jump(
        &mut self,
        job_ids: &[i32],
        jobs: &mut [GpuJob],
        profile: &CycleProfile,
        jump_periods: i64,
        current_time: f64,
        timer: &mut Timer,
    ) {
        let period_ticks = profile.period_ticks;
        let jump_ticks = jump_periods * period_ticks;
        let jump_seconds = self.from_tick(jump_ticks);
        let new_phase_start_time = current_time + jump_seconds;

        for &jid in job_ids {
            let job = &mut jobs[jid as usize];
            let delta_iter = profile.delta_iterations[&jid];
            job.current_iteration += delta_iter * jump_periods;
            job.completed_iterations = Some(job.current_iteration as f64);
            *job.training_time.as_mut().unwrap() += profile.delta_training[&jid] * jump_periods as f64;
            *job.consumed_compute_time.as_mut().unwrap() +=
                profile.delta_compute[&jid] * jump_periods as f64;
            *job.consumed_comms_time.as_mut().unwrap() +=
                profile.delta_comms[&jid] * jump_periods as f64;
            job.phase_start_time = Some(new_phase_start_time);
            job.last_change_time = Some(new_phase_start_time);
            timer.update_job_time(jid, new_phase_start_time + job.iteration_computing_time);
        }

        self.cycle_stats.jump_applied += 1;
        self.cycle_stats.jump_periods += jump_periods as u64;
        self.cycle_stats.jumped_ticks += jump_ticks as u64;
    }

    fn observe_or_jump_lcm_cycle(
        &mut self,
        current_time: f64,
        intra_model: bool,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) -> bool {
        if !self.interference_model.is_lcm_model() {
            return false;
        }
        if !self.enable_lcm_cycle_jump {
            return false;
        }

        let active_job_ids = self.get_unique_active_training_job_ids(jobs);
        let current_tick = self.to_tick(current_time);
        if !self.all_active_jobs_sync_compute_start(&active_job_ids, jobs, current_tick) {
            return false;
        }

        self.cycle_stats.observed_sync += 1;
        let cycle_key = self.build_cycle_key(&active_job_ids, jobs, intra_model);
        let snapshot = self.build_cycle_snapshot(&active_job_ids, jobs, current_tick);

        // Check if we have a profile
        if let Some(profile) = self.cycle_profiles.get(&cycle_key).cloned() {
            let jump_periods =
                self.compute_jump_count(&profile, &active_job_ids, jobs, current_tick, timer);
            if jump_periods > 0 {
                self.apply_cycle_jump(
                    &active_job_ids,
                    jobs,
                    &profile,
                    jump_periods,
                    current_time,
                    timer,
                );
                return true;
            }
        }

        // Check if we have a previous observation
        if let Some(previous) = self.cycle_observation_cache.get(&cycle_key).cloned() {
            let period_ticks = snapshot.tick - previous.tick;
            if period_ticks > 0 {
                let mut delta_iterations = HashMap::new();
                let mut delta_training = HashMap::new();
                let mut delta_compute = HashMap::new();
                let mut delta_comms = HashMap::new();
                let mut valid = true;

                for &jid in &active_job_ids {
                    let delta_iter =
                        snapshot.iterations[&jid] - previous.iterations[&jid];
                    if delta_iter <= 0 {
                        valid = false;
                        break;
                    }
                    delta_iterations.insert(jid, delta_iter);
                    delta_training.insert(
                        jid,
                        snapshot.training[&jid] - previous.training[&jid],
                    );
                    delta_compute.insert(
                        jid,
                        snapshot.compute[&jid] - previous.compute[&jid],
                    );
                    delta_comms
                        .insert(jid, snapshot.comms[&jid] - previous.comms[&jid]);
                }

                if valid {
                    self.cycle_profiles.insert(
                        cycle_key.clone(),
                        CycleProfile {
                            period_ticks,
                            delta_iterations,
                            delta_training,
                            delta_compute,
                            delta_comms,
                        },
                    );
                    self.cycle_stats.cycle_detected += 1;
                }
            }
        }

        self.cycle_observation_cache.insert(cycle_key, snapshot);
        false
    }

    // --- Resource allocation ---

    pub fn allocate_resources(
        &mut self,
        job_idx: i32,
        placement_method: PlacementMethod,
        current_time: f64,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) -> bool {
        match placement_method {
            PlacementMethod::K8sLoadBalancing => {
                self.k8s_load_balancing(job_idx, current_time, jobs, timer)
            }
            PlacementMethod::K8sBinPacking => {
                self.k8s_bin_packing(job_idx, current_time, jobs, timer)
            }
            PlacementMethod::Colocate => self.colocate(job_idx, current_time, jobs, timer),
            PlacementMethod::Tiresias => self.tiresias(job_idx, current_time, jobs, timer),
        }
    }

    pub fn allocate_network(
        &mut self,
        job_idx: i32,
        current_time: f64,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) {
        let job = &mut jobs[job_idx as usize];
        job.training_time = Some(job.loading_time);
        job.consumed_loading_time = Some(job.loading_time);
        job.consumed_compute_time = Some(0.0);
        job.consumed_comms_time = Some(0.0);
        job.last_change_time = Some(current_time);

        if self.interference_model.is_iter_model() {
            job.current_iteration = 0;
            job.current_phase = Phase::Compute;
            job.phase_start_time = Some(current_time);
            job.is_in_comm_phase = false;

            let comp_time = job.iteration_computing_time;
            timer.add_event(current_time + comp_time, EventType::IterCompEnd, job_idx);

            trace_log!(
                "Job {} finished loading, starting iteration 0 compute phase",
                job_idx
            );
        } else {
            // For corun-profile, refresh iteration time from the profile
            // tables now that this job is officially training (the active
            // colocation set may not have included this job before).
            if self.interference_model.is_corun_profile() {
                self.refresh_corun_iter_times(job_idx, jobs);
            }

            // Drop mutable borrow before calling methods that need &mut self
            let iter_num = jobs[job_idx as usize].iteration_number;
            let net_time = jobs[job_idx as usize].iteration_networking_time;

            self.allocate_bandwidth(jobs);

            let initial_iter_time = self.effective_iteration_time(&jobs[job_idx as usize], net_time);
            timer.add_event(
                current_time + iter_num * initial_iter_time,
                EventType::Complete,
                job_idx,
            );

            self.update_completion_times(current_time, jobs, timer);

            let job = &jobs[job_idx as usize];
            trace_log!(
                "Job {} finished loading time, network_usage: {:?}",
                job_idx, job.using_bandwidths
            );
        }
    }

    fn place_job(
        &mut self,
        job_idx: i32,
        allocated: Vec<i32>,
        ps_allocated: Vec<i32>,
        current_time: f64,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) {
        let job = &mut jobs[job_idx as usize];
        job.allocated = allocated.clone();
        job.ps_allocated = ps_allocated.clone();
        job.start_time = Some(current_time);
        job.wait_time = Some(current_time - job.arrival_time);
        job.training_time = None; // will be updated in START event
        job.consumed_compute_time = Some(0.0);
        job.consumed_comms_time = Some(0.0);
        job.completed_iterations = Some(0.0);
        job.last_change_time = Some(current_time);

        if self.interference_model.is_intra_model() {
            job.gpu_allocation = Vec::new();
        }

        let gpu_per_worker = job.gpu_per_worker;
        let cpu_per_gpu_worker = job.cpu_per_gpu_worker;
        let cpu_per_ps = job.cpu_per_ps;

        for &server in &allocated {
            self.gpus[server as usize] -= gpu_per_worker;
            self.cpu_cores[server as usize] -= cpu_per_gpu_worker;

            if self.interference_model.is_intra_model() {
                let num_gpus = gpu_per_worker;
                for gpu_idx in 0..num_gpus {
                    jobs[job_idx as usize]
                        .gpu_allocation
                        .push((server, gpu_idx));
                }
            }
        }

        for &server in &ps_allocated {
            self.cpu_cores[server as usize] -= cpu_per_ps;
        }

        // Update active_jobs
        let all_servers: HashSet<i32> = allocated.iter().chain(ps_allocated.iter()).cloned().collect();
        for server in &all_servers {
            self.active_jobs[*server as usize].push(job_idx);
        }
        self.bump_placement_epoch();

        // Snapshot loading_time at this exact placement moment when the
        // corun-profile model is active. After this point the value is
        // frozen for the rest of the job's lifetime, even if co-runners
        // come and go during the loading window.
        if self.interference_model.is_corun_profile() {
            let snapshot = self.corun_loading_time_snapshot(job_idx, jobs);
            jobs[job_idx as usize].loading_time = snapshot;
        }
        let loading_time = jobs[job_idx as usize].loading_time;

        // Add START event
        timer.add_event(current_time + loading_time, EventType::Start, job_idx);

        // Log
        trace_log!(
            "Job {} GPU allocated server: {:?} PS allocated server: {:?}",
            job_idx, allocated, ps_allocated
        );
        if let Ok(mut f) = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.allocation_log_file)
        {
            let _ = writeln!(
                f,
                "id: {} allocated server: {:?} ps allocated server: {:?} current time: {}",
                job_idx, allocated, ps_allocated, current_time
            );
        }

        // GPU occupancy just changed (acquire). Snapshot for the util log.
        self.log_gpu_util_snapshot(current_time);
    }

    // --- Placement algorithms ---

    fn k8s_load_balancing(
        &mut self,
        job_idx: i32,
        current_time: f64,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) -> bool {
        let job = &jobs[job_idx as usize];
        let gpu_workers = job.gpu_workers;
        let ps = job.ps;
        let gpu_per_worker = job.gpu_per_worker;
        let cpu_per_gpu_worker = job.cpu_per_gpu_worker;
        let cpu_per_ps = job.cpu_per_ps;

        // Sort by most available GPUs (descending)
        let mut sorted_servers: Vec<i32> = (0..self.servers).collect();
        sorted_servers.sort_by(|&a, &b| {
            self.gpus[b as usize].cmp(&self.gpus[a as usize])
        });

        let current_available_gpus: i32 = self.gpus.iter().sum();
        let current_available_cpus: i32 = self.cpu_cores.iter().sum();
        let needed_cpus = ps * cpu_per_ps + gpu_workers * cpu_per_gpu_worker;

        if current_available_gpus < gpu_workers || current_available_cpus < needed_cpus {
            trace_log!(
                "k8s-load-balancing: not enough resources CPU {}/{} GPU {}/{}",
                needed_cpus, current_available_cpus, gpu_workers, current_available_gpus
            );
            return false;
        }

        let mut allocated = Vec::new();
        let mut ps_allocated = Vec::new();
        let mut available_gpus = self.gpus.clone();

        // Place GPU workers
        for _ in 0..gpu_workers {
            for &server in &sorted_servers {
                if allocated.len() as i32 == gpu_workers {
                    break;
                }
                let server = server as usize;
                if available_gpus[server] >= gpu_per_worker
                    && self.cpu_cores[server] >= cpu_per_gpu_worker
                {
                    available_gpus[server] -= gpu_per_worker;
                    allocated.push(server as i32);
                }
            }
        }

        // Place PSs -- more cpu core servers first
        sorted_servers.sort_by(|&a, &b| {
            self.cpu_cores[b as usize].cmp(&self.cpu_cores[a as usize])
        });
        for _ in 0..ps {
            for &server in &sorted_servers {
                if ps_allocated.len() as i32 == ps {
                    break;
                }
                if self.cpu_cores[server as usize] >= cpu_per_ps {
                    ps_allocated.push(server);
                }
            }
        }

        let success =
            allocated.len() as i32 == gpu_workers && ps_allocated.len() as i32 == ps;
        if success {
            self.place_job(job_idx, allocated, ps_allocated, current_time, jobs, timer);
        }
        success
    }

    fn k8s_bin_packing(
        &mut self,
        job_idx: i32,
        current_time: f64,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) -> bool {
        let job = &jobs[job_idx as usize];
        let gpu_workers = job.gpu_workers;
        let ps = job.ps;
        let gpu_per_worker = job.gpu_per_worker;
        let cpu_per_gpu_worker = job.cpu_per_gpu_worker;
        let cpu_per_ps = job.cpu_per_ps;

        // Sort by least available GPUs (ascending)
        let mut sorted_servers: Vec<i32> = (0..self.servers).collect();
        sorted_servers.sort_by(|&a, &b| {
            self.gpus[a as usize].cmp(&self.gpus[b as usize])
        });

        let current_available_gpus: i32 = self.gpus.iter().sum();
        let current_available_cpus: i32 = self.cpu_cores.iter().sum();
        let needed_cpus = ps * cpu_per_ps + gpu_workers * cpu_per_gpu_worker;

        if current_available_gpus < gpu_workers || current_available_cpus < needed_cpus {
            trace_log!("k8s-bin-packing: not enough resources");
            return false;
        }

        let mut allocated = Vec::new();
        let mut remaining_workers = gpu_workers;

        // TRUE BIN PACKING: Fill servers sequentially
        for &server in &sorted_servers {
            if remaining_workers <= 0 {
                break;
            }
            let s = server as usize;
            let server_gpu_capacity = self.gpus[s] / gpu_per_worker;
            let server_cpu_capacity = self.cpu_cores[s] / cpu_per_gpu_worker;
            let server_capacity = server_gpu_capacity.min(server_cpu_capacity);
            let workers_in_this_server = server_capacity.min(remaining_workers);

            for _ in 0..workers_in_this_server {
                allocated.push(server);
            }
            remaining_workers -= workers_in_this_server;
        }

        // Place PSs -- least cpu core servers first
        let mut ps_allocated = Vec::new();
        sorted_servers.sort_by(|&a, &b| {
            self.cpu_cores[a as usize].cmp(&self.cpu_cores[b as usize])
        });
        for _ in 0..ps {
            for &server in &sorted_servers {
                if ps_allocated.len() as i32 == ps {
                    break;
                }
                if self.cpu_cores[server as usize] >= cpu_per_ps {
                    ps_allocated.push(server);
                }
            }
        }

        let success =
            allocated.len() as i32 == gpu_workers && ps_allocated.len() as i32 == ps;
        if success {
            self.place_job(job_idx, allocated, ps_allocated, current_time, jobs, timer);
        }
        success
    }

    fn colocate(
        &mut self,
        job_idx: i32,
        current_time: f64,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) -> bool {
        let job = &jobs[job_idx as usize];
        let gpu_workers = job.gpu_workers;
        let ps = job.ps;
        let gpu_per_worker = job.gpu_per_worker;
        let cpu_per_gpu_worker = job.cpu_per_gpu_worker;
        let cpu_per_ps = job.cpu_per_ps;

        // Sort by most available GPUs (descending)
        let mut sorted_servers: Vec<i32> = (0..self.servers).collect();
        sorted_servers.sort_by(|&a, &b| {
            self.gpus[b as usize].cmp(&self.gpus[a as usize])
        });

        let current_available_gpus: i32 = self.gpus.iter().sum();
        let current_available_cpus: i32 = self.cpu_cores.iter().sum();
        let needed_cpus = ps * cpu_per_ps + gpu_workers * cpu_per_gpu_worker;

        if current_available_gpus < gpu_workers || current_available_cpus < needed_cpus {
            trace_log!("colocate: not enough resources");
            return false;
        }

        let mut allocated = Vec::new();
        let mut ps_allocated = Vec::new();

        for &server in &sorted_servers {
            if allocated.len() as i32 == gpu_workers {
                break;
            }
            let s = server as usize;
            if self.gpus[s] >= gpu_workers
                && self.cpu_cores[s]
                    >= cpu_per_gpu_worker * gpu_workers + cpu_per_ps * ps
            {
                for _ in 0..gpu_workers {
                    allocated.push(server);
                }
                for _ in 0..ps {
                    ps_allocated.push(server);
                }
            }
        }

        let success =
            allocated.len() as i32 == gpu_workers && ps_allocated.len() as i32 == ps;
        if success {
            self.place_job(job_idx, allocated, ps_allocated, current_time, jobs, timer);
        }
        success
    }

    fn tiresias(
        &mut self,
        job_idx: i32,
        current_time: f64,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) -> bool {
        let job = &jobs[job_idx as usize];
        let gpu_workers = job.gpu_workers;
        let ps = job.ps;
        let gpu_per_worker = job.gpu_per_worker;
        let cpu_per_gpu_worker = job.cpu_per_gpu_worker;
        let cpu_per_ps = job.cpu_per_ps;
        let skewness = job.skewness;

        // Sort by least GPUs (ascending)
        let mut sorted_servers: Vec<i32> = (0..self.servers).collect();
        sorted_servers.sort_by(|&a, &b| {
            self.gpus[a as usize].cmp(&self.gpus[b as usize])
        });

        let current_available_gpus: i32 = self.gpus.iter().sum();
        let current_available_cpus: i32 = self.cpu_cores.iter().sum();
        let needed_cpus = ps * cpu_per_ps + gpu_workers * cpu_per_gpu_worker;

        if current_available_gpus < gpu_workers || current_available_cpus < needed_cpus {
            trace_log!("tiresias: not enough resources");
            return false;
        }

        if skewness > 5.0 {
            // Colocate
            let mut allocated = Vec::new();
            let mut ps_allocated = Vec::new();

            for &server in &sorted_servers {
                if allocated.len() as i32 == gpu_workers {
                    break;
                }
                let s = server as usize;
                if self.gpus[s] >= gpu_workers
                    && self.cpu_cores[s]
                        >= cpu_per_gpu_worker * gpu_workers + cpu_per_ps * ps
                {
                    for _ in 0..gpu_workers {
                        allocated.push(server);
                    }
                    for _ in 0..ps {
                        ps_allocated.push(server);
                    }
                }
            }

            let success =
                allocated.len() as i32 == gpu_workers && ps_allocated.len() as i32 == ps;
            if success {
                self.place_job(job_idx, allocated, ps_allocated, current_time, jobs, timer);
            }
            success
        } else {
            // Distribute (bin packing style)
            let mut allocated = Vec::new();
            let mut ps_allocated = Vec::new();

            for _ in 0..gpu_workers {
                for &server in &sorted_servers {
                    if allocated.len() as i32 == gpu_workers {
                        break;
                    }
                    let s = server as usize;
                    if self.gpus[s] > 0 && self.cpu_cores[s] >= cpu_per_gpu_worker {
                        allocated.push(server);
                    }
                }
            }

            // Place PSs -- more cpu core servers first
            sorted_servers.sort_by(|&a, &b| {
                self.cpu_cores[b as usize].cmp(&self.cpu_cores[a as usize])
            });
            for _ in 0..ps {
                for &server in &sorted_servers {
                    if ps_allocated.len() as i32 == ps {
                        break;
                    }
                    if self.cpu_cores[server as usize] >= cpu_per_ps {
                        ps_allocated.push(server);
                    }
                }
            }

            let success =
                allocated.len() as i32 == gpu_workers && ps_allocated.len() as i32 == ps;
            if success {
                self.place_job(job_idx, allocated, ps_allocated, current_time, jobs, timer);
            }
            success
        }
    }

    // --- Resource release ---

    pub fn release_resources(
        &mut self,
        job_idx: i32,
        release_time: f64,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) {
        let job = &jobs[job_idx as usize];
        let allocated = job.allocated.clone();
        let ps_allocated = job.ps_allocated.clone();
        let gpu_per_worker = job.gpu_per_worker;
        let cpu_per_gpu_worker = job.cpu_per_gpu_worker;
        let cpu_per_ps = job.cpu_per_ps;

        let all_servers: HashSet<i32> =
            allocated.iter().chain(ps_allocated.iter()).cloned().collect();

        for &server in &all_servers {
            let s = server as usize;
            let mut num_gpu_wk = 0;
            let mut num_ps = 0;

            if let Some(pos) = self.active_jobs[s].iter().position(|&x| x == job_idx) {
                self.active_jobs[s].remove(pos);
                num_gpu_wk = allocated.iter().filter(|&&x| x == server).count() as i32;
                num_ps = ps_allocated.iter().filter(|&&x| x == server).count() as i32;
            }

            self.gpus[s] = (self.gpus[s] + num_gpu_wk * gpu_per_worker).min(self.gpus_per_server);
            self.cpu_cores[s] = (self.cpu_cores[s]
                + num_gpu_wk * cpu_per_gpu_worker
                + num_ps * cpu_per_ps)
                .min(self.cpu_cores_per_server);
        }

        jobs[job_idx as usize].using_bandwidths.clear();
        self.bump_placement_epoch();

        self.allocate_bandwidth(jobs);
        self.update_completion_times(release_time, jobs, timer);

        // GPU occupancy just changed (release). Snapshot for the util log.
        self.log_gpu_util_snapshot(release_time);
    }

    // --- Bandwidth allocation ---

    pub fn allocate_bandwidth(&mut self, jobs: &mut [GpuJob]) {
        let mut required_bandwidths: HashMap<i32, HashMap<i32, f64>> = HashMap::new();

        for server in 0..self.servers {
            let s = server as usize;
            for &job_id in &self.active_jobs[s] {
                let job = &jobs[job_id as usize];
                let entry = required_bandwidths
                    .entry(job_id)
                    .or_insert_with(HashMap::new);

                if job.training_time.is_none() {
                    entry.insert(server, 0.0);
                } else if self.interference_model.is_iter_model() && !job.is_in_comm_phase {
                    if self.phase_overlap_weighted {
                        // Job is currently in compute phase, but with
                        // staggered iteration boundaries it spends a
                        // fraction P = T_comm / T_iter of its time in
                        // comm phase. Weight its demand by P.
                        let t_comm = job.iteration_networking_time;
                        let t_iter = job.iteration_computing_time + t_comm;
                        let p = if t_iter > 0.0 { t_comm / t_iter } else { 0.0 };
                        let r = self.required_bandwidth_capped(job, server);
                        entry.insert(server, r * p);
                    } else {
                        entry.insert(server, 0.0);
                    }
                } else {
                    entry.insert(server, self.required_bandwidth_capped(job, server));
                }
            }
        }

        let used_bandwidths = self.set_stable_bandwidth(&required_bandwidths);

        for server in 0..self.servers {
            let s = server as usize;
            let mut server_total_used = 0.0;
            let job_ids: Vec<i32> = self.active_jobs[s].clone();
            for &job_id in &job_ids {
                let job = &mut jobs[job_id as usize];
                if !job.using_bandwidths.contains_key(&server) {
                    job.using_bandwidths.insert(server, 0.0);
                }
                if let Some(server_map) = used_bandwidths.get(&job_id) {
                    if let Some(&used) = server_map.get(&server) {
                        job.using_bandwidths.insert(server, used);
                        server_total_used += used;
                    }
                }
            }
            self.bandwidths[s] = self.bandwidth_per_server - server_total_used;
        }
    }

    fn set_stable_bandwidth(
        &self,
        required_bandwidths: &HashMap<i32, HashMap<i32, f64>>,
    ) -> HashMap<i32, HashMap<i32, f64>> {
        let mut used_bandwidths: HashMap<i32, HashMap<i32, f64>> = HashMap::new();

        // HOL blocking: effective NIC capacity is `η × C` under contention
        // (Karol 1987). Set the effective capacity to `η · bandwidth_per_server`
        // when an HOL efficiency is configured.
        let effective_c = match self.hol_efficiency {
            Some(eta) if eta.is_finite() && (0.0..=1.0).contains(&eta) => {
                self.bandwidth_per_server * eta
            }
            _ => self.bandwidth_per_server,
        };

        for server in 0..self.servers {
            let s = server as usize;
            let jobs_on_server = &self.active_jobs[s];

            let total_required: f64 = jobs_on_server
                .iter()
                .map(|&jid| {
                    required_bandwidths
                        .get(&jid)
                        .and_then(|m| m.get(&server))
                        .copied()
                        .unwrap_or(0.0)
                })
                .sum();

            if total_required <= 0.0 {
                continue;
            }

            if total_required > effective_c {
                let ratio = effective_c / total_required;
                for &jid in jobs_on_server {
                    let required = required_bandwidths
                        .get(&jid)
                        .and_then(|m| m.get(&server))
                        .copied()
                        .unwrap_or(0.0);
                    used_bandwidths
                        .entry(jid)
                        .or_insert_with(HashMap::new)
                        .insert(server, required * ratio);
                }
            } else {
                for &jid in jobs_on_server {
                    let required = required_bandwidths
                        .get(&jid)
                        .and_then(|m| m.get(&server))
                        .copied()
                        .unwrap_or(0.0);
                    used_bandwidths
                        .entry(jid)
                        .or_insert_with(HashMap::new)
                        .insert(server, required);
                }
            }
        }
        used_bandwidths
    }

    // --- Completion time updates ---

    pub fn update_completion_times(
        &mut self,
        current_time: f64,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) {
        match self.interference_model {
            InterferenceModel::CommsIter => {
                self.update_completion_times_iter_model(current_time, jobs, timer);
            }
            InterferenceModel::CommsIterIntra => {
                self.update_completion_times_iter_intra_model(current_time, jobs, timer);
            }
            InterferenceModel::CommsIterLcm => {
                if !self.observe_or_jump_lcm_cycle(current_time, false, jobs, timer) {
                    self.update_completion_times_iter_model(current_time, jobs, timer);
                }
            }
            InterferenceModel::CommsIterIntraLcm => {
                if !self.observe_or_jump_lcm_cycle(current_time, true, jobs, timer) {
                    self.update_completion_times_iter_intra_model(current_time, jobs, timer);
                }
            }
            InterferenceModel::CorunProfile => {
                self.update_completion_times_corun_profile(current_time, jobs, timer);
            }
            _ => {
                self.update_completion_times_job_model(current_time, jobs, timer);
            }
        }

        // ARRIVE event re-evaluation
        if let Some((_, _, arrive_job_id)) = timer.peek_next_arrive_event() {
            let job = &mut jobs[arrive_job_id as usize];
            let new_time = current_time.max(job.temp_arrival_time);
            job.temp_arrival_time = new_time;
            timer.update_job_time(arrive_job_id, new_time);
        }
    }

    fn update_completion_times_job_model(
        &self,
        current_time: f64,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) {
        // Collect unique jobs across all servers
        let mut processed: HashSet<i32> = HashSet::new();
        for server in 0..self.servers as usize {
            for &job_id in &self.active_jobs[server] {
                if processed.contains(&job_id) {
                    continue;
                }
                processed.insert(job_id);

                let job = &mut jobs[job_id as usize];
                if job.training_time.is_none() {
                    continue;
                }

                let elapsed_time;
                if job.remaining_train_time.is_none() {
                    // New job
                    if job.last_network_factor.is_some() {
                        panic!("Job {} last network factor mismatch", job_id);
                    }
                    elapsed_time = 0.0;
                } else {
                    elapsed_time = current_time - job.last_change_time.unwrap_or(current_time);
                    let actual_networking_time =
                        job.iteration_networking_time * job.last_network_factor.unwrap_or(1.0);
                    let iteration_time =
                        self.effective_iteration_time(job, actual_networking_time);
                    let new_completed =
                        job.completed_iterations.unwrap_or(0.0) + elapsed_time / iteration_time;
                    job.completed_iterations = Some(new_completed.min(job.iteration_number));
                }

                // Calculate contention factor
                let new_contention_factor = match &self.interference_model {
                    InterferenceModel::None => 1.0,
                    InterferenceModel::Fixed => {
                        let mut shares_server = false;
                        let all_servers: HashSet<i32> = job
                            .allocated
                            .iter()
                            .chain(job.ps_allocated.iter())
                            .cloned()
                            .collect();
                        for &s in &all_servers {
                            if self.active_jobs[s as usize].len() > 1 {
                                shares_server = true;
                                break;
                            }
                        }
                        if shares_server {
                            1.0 + self.interference_ratio.unwrap_or(0.0)
                        } else {
                            1.0
                        }
                    }
                    InterferenceModel::Comms => {
                        let factors: Vec<f64> = job
                            .allocated
                            .iter()
                            .map(|&s| {
                                let required = self.required_bandwidth_capped(job, s);
                                let using = job
                                    .using_bandwidths
                                    .get(&s)
                                    .copied()
                                    .unwrap_or(0.0)
                                    .max(self.min_guaranteed_bw);
                                required / using
                            })
                            .collect();
                        factors
                            .into_iter()
                            .fold(1.0_f64, |acc, x| acc.max(x))
                    }
                    _ => 1.0,
                };

                let remaining_iterations =
                    job.iteration_number - job.completed_iterations.unwrap_or(0.0);
                if remaining_iterations > 0.0 {
                    let new_networking_time =
                        job.iteration_networking_time * new_contention_factor;
                    let new_iteration_time =
                        self.effective_iteration_time(job, new_networking_time);
                    let new_completion_time =
                        current_time + remaining_iterations * new_iteration_time;

                    *job.training_time.as_mut().unwrap() += elapsed_time;
                    if job.last_network_factor.is_none() {
                        assert!(elapsed_time == 0.0);
                        job.last_network_factor = Some(1.0);
                    }
                    let last_factor = job.last_network_factor.unwrap();
                    let last_compute_ratio = job.iteration_computing_time
                        / (job.iteration_computing_time
                            + job.iteration_networking_time * last_factor);
                    *job.consumed_compute_time.as_mut().unwrap() +=
                        elapsed_time * last_compute_ratio;
                    *job.consumed_comms_time.as_mut().unwrap() +=
                        elapsed_time * (1.0 - last_compute_ratio);
                    job.remaining_train_time =
                        Some(remaining_iterations * new_iteration_time);
                    job.last_change_time = Some(current_time);
                    job.last_network_factor = Some(new_contention_factor);

                    timer.update_job_time(job_id, new_completion_time);
                }
            }
        }
    }

    fn update_completion_times_iter_model(
        &self,
        current_time: f64,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) {
        let mut jobs_to_update: HashSet<i32> = HashSet::new();
        for server in 0..self.servers as usize {
            for &job_id in &self.active_jobs[server] {
                let job = &jobs[job_id as usize];
                if job.training_time.is_none() {
                    continue;
                }
                if job.is_in_comm_phase {
                    jobs_to_update.insert(job_id);
                }
            }
        }

        for &job_id in &jobs_to_update {
            let job = &jobs[job_id as usize];
            let factors: Vec<f64> = job
                .allocated
                .iter()
                .map(|&s| {
                    let required = self.required_bandwidth_capped(job, s);
                    let using = job
                        .using_bandwidths
                        .get(&s)
                        .copied()
                        .unwrap_or(0.0)
                        .max(self.min_guaranteed_bw);
                    required / using
                })
                .collect();
            let raw_factor = factors.into_iter().fold(1.0_f64, |acc, x| acc.max(x));
            // Apply HOL-blocking efficiency: factor /= η when in contention.
            let new_contention_factor = self.apply_hol_blocking(raw_factor);

            let job = &mut jobs[job_id as usize];
            job.update_comm_contention(current_time, new_contention_factor);

            let remaining_original_comm = job.get_comm_remaining_original(current_time);
            let new_comm_end_time = current_time + remaining_original_comm * new_contention_factor;

            timer.update_job_time(job_id, new_comm_end_time);

            trace_log!(
                "Job {} comm phase updated: contention={:.2}, remaining_original={:.2}s, new_end_time={:.2}s",
                job_id, new_contention_factor, remaining_original_comm, new_comm_end_time
            );
        }
    }

    fn update_completion_times_iter_intra_model(
        &self,
        current_time: f64,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) {
        let mut jobs_to_update: HashSet<i32> = HashSet::new();
        for server in 0..self.servers as usize {
            for &job_id in &self.active_jobs[server] {
                let job = &jobs[job_id as usize];
                if job.training_time.is_none() {
                    continue;
                }
                if job.is_in_comm_phase {
                    jobs_to_update.insert(job_id);
                }
            }
        }

        // Pre-compute per-server total demand (used by M/M/1).
        let mut server_total_demand: HashMap<i32, f64> = HashMap::new();
        if self.contention_model_mm1 {
            for s in 0..self.servers {
                let mut total = 0.0;
                for &other_jid in &self.active_jobs[s as usize] {
                    let other_job = &jobs[other_jid as usize];
                    if other_job.is_in_comm_phase {
                        total += self.required_bandwidth_capped(other_job, s);
                    }
                }
                server_total_demand.insert(s, total);
            }
        }

        for &job_id in &jobs_to_update {
            let job = &jobs[job_id as usize];
            let mut server_contention_factors = Vec::new();

            let unique_servers: HashSet<i32> = job.allocated.iter().cloned().collect();
            for &server in &unique_servers {
                let s = server as usize;
                // Inter-server
                let inter_required = self.required_bandwidth_capped(job, server);
                let inter_available = job
                    .using_bandwidths
                    .get(&server)
                    .copied()
                    .unwrap_or(0.0)
                    .max(self.min_guaranteed_bw);
                // Linear (max-min fair sharing) factor.
                let linear_factor = inter_required / inter_available;
                // Optionally REPLACE with M/M/1 PS factor (pure model test).
                let inter_contention = if self.contention_model_mm1 {
                    let total_r = server_total_demand.get(&server).copied().unwrap_or(0.0);
                    let rho = total_r / self.bandwidth_per_server;
                    let rho_clamped = rho.min(1.0 - self.mm1_epsilon);
                    1.0 / (1.0 - rho_clamped)
                } else {
                    linear_factor
                };

                // Intra-server
                let mut total_intra_demand = 0.0;
                for &other_jid in &self.active_jobs[s] {
                    let other_job = &jobs[other_jid as usize];
                    if other_job.is_in_comm_phase {
                        total_intra_demand +=
                            other_job.required_intra_bandwidth_per_server(server);
                    }
                }
                let available_intra = self.intra_bandwidths[s];
                let intra_contention = if total_intra_demand > available_intra {
                    total_intra_demand / available_intra
                } else {
                    1.0
                };

                server_contention_factors.push(inter_contention.max(intra_contention));
            }

            let raw_factor = server_contention_factors
                .into_iter()
                .fold(1.0_f64, |acc, x| acc.max(x));
            // Apply HOL-blocking efficiency: factor /= η when in contention.
            let hol_factor = self.apply_hol_blocking(raw_factor);
            // Optionally damp by `comm_contention_fraction` to model the
            // fact that only the AR-transfer portion of iter_networking
            // _time scales with bandwidth contention; sync/wait overhead
            // does not.
            let new_contention_factor = self.damp_contention_factor(hol_factor);

            let job = &mut jobs[job_id as usize];
            job.update_comm_contention(current_time, new_contention_factor);

            let remaining_original_comm = job.get_comm_remaining_original(current_time);
            let new_comm_end_time = current_time + remaining_original_comm * new_contention_factor;

            timer.update_job_time(job_id, new_comm_end_time);

            trace_log!(
                "Job {} comm phase updated (intra-model): contention={:.2}, remaining_original={:.2}s, new_end_time={:.2}s",
                job_id, new_contention_factor, remaining_original_comm, new_comm_end_time
            );
        }
    }

    /// Completion-time update for `corun-profile`. Iteration times are
    /// looked up from the profile tables based on the current colocation
    /// state, then progress is accounted for in the job-model style
    /// (single iteration_time = effective_iteration_time(compute, network)).
    fn update_completion_times_corun_profile(
        &mut self,
        current_time: f64,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) {
        // Collect unique training jobs across all servers.
        let mut processed: HashSet<i32> = HashSet::new();
        let mut active_training: Vec<i32> = Vec::new();
        for server in 0..self.servers as usize {
            for &job_id in &self.active_jobs[server] {
                if !processed.insert(job_id) {
                    continue;
                }
                let job = &jobs[job_id as usize];
                if job.training_time.is_none() {
                    continue;
                }
                if job.current_phase == Phase::Completed {
                    continue;
                }
                active_training.push(job_id);
            }
        }

        for job_id in active_training {
            // Account elapsed progress under the *previous* iteration
            // time before refreshing it.
            let job = &mut jobs[job_id as usize];
            let elapsed_time;
            if job.remaining_train_time.is_none() {
                elapsed_time = 0.0;
            } else {
                elapsed_time = current_time - job.last_change_time.unwrap_or(current_time);
                let prev_iter_time =
                    self.effective_iteration_time(job, job.iteration_networking_time);
                if prev_iter_time > 0.0 {
                    let new_completed = job.completed_iterations.unwrap_or(0.0)
                        + elapsed_time / prev_iter_time;
                    job.completed_iterations = Some(new_completed.min(job.iteration_number));
                }
            }

            // Distribute the elapsed wall-clock into compute/comms
            // buckets using the *previous* split.
            let prev_compute = job.iteration_computing_time;
            let prev_network = job.iteration_networking_time;
            let prev_iter_time = self.effective_iteration_time(job, prev_network);
            let prev_compute_share = if prev_iter_time > 0.0 {
                prev_compute / prev_iter_time
            } else {
                0.0
            };
            *job.training_time.as_mut().unwrap() += elapsed_time;
            *job.consumed_compute_time.as_mut().unwrap() +=
                elapsed_time * prev_compute_share;
            *job.consumed_comms_time.as_mut().unwrap() +=
                elapsed_time * (1.0 - prev_compute_share);

            // Refresh iteration time for the *new* colocation state.
            self.refresh_corun_iter_times(job_id, jobs);

            // Re-arm completion event under the new iteration time.
            let job = &mut jobs[job_id as usize];
            let remaining_iterations =
                job.iteration_number - job.completed_iterations.unwrap_or(0.0);
            if remaining_iterations > 0.0 {
                let new_iter_time =
                    self.effective_iteration_time(job, job.iteration_networking_time);
                let new_completion_time =
                    current_time + remaining_iterations * new_iter_time;

                job.remaining_train_time = Some(remaining_iterations * new_iter_time);
                job.last_change_time = Some(current_time);
                // last_network_factor is unused by corun-profile; pin
                // to 1.0 so any legacy reads stay sane.
                job.last_network_factor = Some(1.0);

                timer.update_job_time(job_id, new_completion_time);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cluster() -> GPUCluster {
        GPUCluster::new(
            4,
            8,
            463.0,
            16_384.0,
            256,
            String::new(),
            InterferenceModel::None,
            None,
            6,
            true,
            0.0,
            CorunProfile::default(),
            String::new(),
            None,
            None,
            None,
            None,
            false,
            0.05,
            false,
            None,
        )
    }

    #[test]
    fn k8s_load_balancing_does_not_overallocate_a_server_when_workers_exceed_server_count() {
        let mut cluster = test_cluster();
        cluster.gpus = vec![2, 3, 1, 2];
        let mut jobs = vec![GpuJob::new(
            0,
            "test-model".to_string(),
            0.0,
            1.0,
            1.0,
            0.0,
            8,
            0,
            1,
            1,
            1,
            0.0,
            0.0,
            0.0,
            0.0,
        )];
        let mut timer = Timer::new();

        let placed = cluster.allocate_resources(
            0,
            PlacementMethod::K8sLoadBalancing,
            0.0,
            &mut jobs,
            &mut timer,
        );

        assert!(placed);
        assert_eq!(jobs[0].allocated, vec![1, 0, 3, 2, 1, 0, 3, 1]);
        assert_eq!(cluster.gpus, vec![0, 0, 0, 0]);
    }
}
