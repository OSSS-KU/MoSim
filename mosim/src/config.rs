use serde::Deserialize;

/// Single source of truth for simulator configuration.
/// Produced by the Python CLI front-end (simulator-trace-timer-bw.py)
/// and consumed by the Rust simulator via stdin (JSON).
///
/// Field names match the Python `argparse` dest names so that
/// `vars(args)` can be JSON-dumped directly without renaming.
#[derive(Debug, Deserialize)]
pub struct SimConfig {
    pub num_node: i32,
    pub num_gpus_per_node: i32,
    pub num_cpus_per_node: i32,

    /// Placement algorithm. Named `schedule` because the Python CLI
    /// folded `--placement` into `--schedule`. FIFO queueing is implicit.
    pub schedule: String,
    pub jobtrace: String,
    pub allocationlog: String,
    pub log: String,

    /// Optional path for the per-event trace log (scheduling decisions and
    /// contention recomputations). Empty string disables it. Off by default
    /// because the trace is ~224 MB per 60-job run and writing it costs about
    /// half the runtime.
    #[serde(default)]
    pub trace_log: String,

    /// Optional CSV path for per-event GPU utilization log. Empty string
    /// disables logging. Written by the Rust simulator at every
    /// allocation state change (place_job success / release_resources).
    #[serde(default)]
    pub gpu_util_log: String,

    pub bandwidth: i32,
    pub intra_bandwidth: i32,

    /// Per-(model, gpu_workers) iteration timing + loading time, plus
    /// optional colocated profiling rows used by `corun-profile`.
    pub iteration_time_csv_file: String,
    /// Network summary; "Sum of Max TX+RX (MB/s)" per (Model, Number of
    /// Workers) is used as `profiled_network` (C_j).
    pub communication_volume_csv_file: String,

    pub interference_model: String,
    #[serde(default)]
    pub interference_ratio: Option<f64>,

    pub lcm_time_decimals: i32,
    pub enable_lcm_cycle_jump: bool,

    pub overlapping_ratio: f64,

    /// Multiplicative cap on `required_bandwidth(server)` expressed as a
    /// multiple of `bandwidth` (server NIC). e.g. `1.4` clamps every
    /// job's per-server bandwidth demand to `1.4 × bandwidth` MB/s.
    /// Optional; when absent, no cap is applied (preserves legacy
    /// behaviour for inputs that hand-tuned C_j to overflow L).
    #[serde(default)]
    pub required_bandwidth_cap_factor: Option<f64>,

    /// Fraction `r ∈ (0,1]` of `iteration_networking_time` that is
    /// actually transfer-bound (subject to bandwidth contention). The
    /// remainder is treated as sync/overhead and not amplified by
    /// contention. Effective factor becomes
    ///   `effective_factor = 1 + r * (raw_factor - 1)`
    /// `None` or `1.0` preserves legacy behaviour (full iter_networking
    /// _time scales with contention).
    #[serde(default)]
    pub comm_contention_fraction: Option<f64>,

    /// Exponent β for N-scaling of the `required_bandwidth` cap:
    ///   `cap = α × (N/2)^β × bandwidth_per_server`
    /// `None` or `0` = uniform cap across N. β > 0 raises the cap for
    /// jobs with more workers (more cross-server traffic).
    #[serde(default)]
    pub cap_n_exponent: Option<f64>,

    /// Minimum guaranteed per-job `using_bandwidth` floor (MB/s). When
    /// the proportional-share gives a job less than this, the contention
    /// factor uses this floor instead. Higher value → smaller possible
    /// factor (less amplification in extreme contention). Default 10.0
    /// preserves legacy behaviour.
    #[serde(default)]
    pub min_guaranteed_bw: Option<f64>,

    /// Contention factor formula. "linear" (default) uses the existing
    /// max-min-fair-sharing `R/U`. "mm1" uses the M/M/1 PS queueing
    /// model `1 / max(1 - ρ, ε)` where `ρ = total_R / L` is per-server
    /// utilization (same factor applied to all jobs on a server).
    #[serde(default)]
    pub contention_model: Option<String>,

    /// Clamping epsilon for the M/M/1 model. When `1 - ρ < ε`, the
    /// denominator is set to ε to prevent infinite factors. Default 0.05
    /// (i.e., factor capped at 20).
    #[serde(default)]
    pub mm1_epsilon: Option<f64>,

    /// Use phase-staggered overlap probability when computing per-server
    /// total demand. Currently Mosim treats jobs as binary in/out of
    /// comm phase. With this flag, jobs in compute phase still contribute
    /// `R × (T_comm / T_iter)` to the total demand (their expected
    /// probability of being in comm during a random instant). Defaults
    /// to `false` (legacy binary behaviour).
    #[serde(default)]
    pub phase_overlap_weighted: bool,

    /// Head-of-line blocking efficiency `η ∈ (0, 1]`. When `< 1` and
    /// contention is active (`R/U > 1`), the contention factor is
    /// scaled by `1/η` to model that the NIC's effective throughput
    /// under HOL blocking is only `η × L`. Karol 1987 gives
    /// `η ≈ 0.586` for input-queued switches with random Bernoulli
    /// arrivals; for fine-grained ring all-reduce we calibrate
    /// empirically. `None` or `1.0` = no HOL (legacy linear sharing).
    #[serde(default)]
    pub hol_efficiency: Option<f64>,

    // Accepted but currently unused by the Rust simulator. Kept so the
    // Python front-end can pass it through without serde rejecting
    // unknown fields.
    #[serde(default)]
    #[allow(dead_code)]
    pub lcm_algo: Option<String>,
}
