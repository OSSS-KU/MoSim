use std::collections::HashMap;

/// A GPU training job in the cluster simulator.
///
/// Uses arena pattern: jobs are stored in a Vec<GpuJob> and referenced by index (usize).
/// All numeric fields are unboxed for minimal memory footprint (~300-500 bytes per job).
#[derive(Clone)]
pub struct GpuJob {
    // Immutable fields (set at creation)
    pub job_id: i32,
    pub model: String,
    pub iteration_number: f64,
    pub iteration_computing_time: f64,
    pub iteration_networking_time: f64,
    pub gpu_workers: i32,
    pub ps: i32,
    pub gpu_per_worker: i32,
    pub cpu_per_gpu_worker: i32,
    pub cpu_per_ps: i32,
    pub tensorsizes: f64,
    pub skewness: f64,
    pub profiled_network: f64, // C_j
    pub loading_time: f64,
    pub arrival_time: f64,

    // Mutable fields
    pub temp_arrival_time: f64,
    pub allocated: Vec<i32>,      // server indices where GPU workers are placed
    pub ps_allocated: Vec<i32>,   // server indices where PSs are placed
    pub start_time: Option<f64>,
    pub end_time: Option<f64>,
    pub wait_time: Option<f64>,
    pub queueing_wait_time: Option<f64>,
    pub capacity_wait_time: Option<f64>,
    pub placement_wait_time: Option<f64>,
    pub queue_head_time: Option<f64>,
    pub capacity_met_time: Option<f64>,
    pub training_time: Option<f64>,
    pub consumed_compute_time: Option<f64>,
    pub consumed_comms_time: Option<f64>,
    pub consumed_loading_time: Option<f64>,
    pub remaining_train_time: Option<f64>,
    pub completed_iterations: Option<f64>,
    pub last_change_time: Option<f64>,
    pub last_network_factor: Option<f64>,
    pub using_bandwidths: HashMap<i32, f64>, // server -> bandwidth

    // Phase tracking for comms-iter model
    pub current_iteration: i64,
    pub current_phase: Phase,
    pub phase_start_time: Option<f64>,
    pub is_in_comm_phase: bool,

    // Accurate comm phase progress tracking
    pub comm_phase_segments: Vec<(f64, f64, f64)>, // (start_time, end_time, contention_factor)
    pub comm_phase_original_done: f64,
    pub comm_phase_current_start: Option<f64>,
    pub comm_phase_current_contention: Option<f64>,

    // For comms-iter-intra model
    pub gpu_allocation: Vec<(i32, i32)>, // (server, gpu_idx)
}

#[derive(Clone, Debug, PartialEq)]
pub enum Phase {
    Loading,
    Compute,
    Comm,
    Completed,
}

impl Phase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Phase::Loading => "loading",
            Phase::Compute => "compute",
            Phase::Comm => "comm",
            Phase::Completed => "completed",
        }
    }
}

impl GpuJob {
    pub fn new(
        job_id: i32,
        model: String,
        arrival_time: f64,
        iteration_number: f64,
        iteration_computing_time: f64,
        iteration_networking_time: f64,
        gpu_workers: i32,
        ps: i32,
        gpu_per_worker: i32,
        cpu_per_gpu_worker: i32,
        cpu_per_ps: i32,
        tensorsizes: f64,
        skewness: f64,
        profiled_network: f64,
        loading_time: f64,
    ) -> Self {
        GpuJob {
            job_id,
            model,
            iteration_number,
            iteration_computing_time,
            iteration_networking_time,
            gpu_workers,
            ps,
            gpu_per_worker,
            cpu_per_gpu_worker,
            cpu_per_ps,
            tensorsizes,
            skewness,
            profiled_network,
            loading_time,
            arrival_time,
            temp_arrival_time: arrival_time,
            allocated: Vec::new(),
            ps_allocated: Vec::new(),
            start_time: None,
            end_time: None,
            wait_time: None,
            queueing_wait_time: None,
            capacity_wait_time: None,
            placement_wait_time: None,
            queue_head_time: None,
            capacity_met_time: None,
            training_time: None,
            consumed_compute_time: None,
            consumed_comms_time: None,
            consumed_loading_time: None,
            remaining_train_time: None,
            completed_iterations: None,
            last_change_time: None,
            last_network_factor: None,
            using_bandwidths: HashMap::new(),
            current_iteration: 0,
            current_phase: Phase::Loading,
            phase_start_time: None,
            is_in_comm_phase: false,
            comm_phase_segments: Vec::new(),
            comm_phase_original_done: 0.0,
            comm_phase_current_start: None,
            comm_phase_current_contention: None,
            gpu_allocation: Vec::new(),
        }
    }

    /// Calculate required bandwidth for the job on server `server`.
    pub fn required_bandwidth(&self, server: i32) -> f64 {
        let c_j = self.profiled_network;
        if self.ps > 0 {
            // PS model
            let num_ps_inside = self.ps_allocated.iter().filter(|&&s| s == server).count() as f64;
            let num_ps_outside = self.ps as f64 - num_ps_inside;
            let num_wk_inside = self.allocated.iter().filter(|&&s| s == server).count() as f64;
            let num_wk_outside = self.gpu_workers as f64 - num_wk_inside;
            c_j * (num_ps_inside * num_wk_outside + num_ps_outside * num_wk_inside)
                / (self.ps as f64 * self.gpu_workers as f64)
        } else {
            // AllReduce or Single-GPU job
            let num_wk_i = self.allocated.iter().filter(|&&s| s == server).count() as i32;
            if self.gpu_workers == num_wk_i {
                0.0
            } else if self.gpu_workers < 2 {
                0.0
            } else if self.gpu_workers == 2 {
                c_j / self.gpu_workers as f64
            } else {
                // gpu_workers > 2
                2.0 * c_j / self.gpu_workers as f64
            }
        }
    }

    /// Calculate required intra-server bandwidth for a specific server.
    pub fn required_intra_bandwidth_per_server(&self, server: i32) -> f64 {
        if self.gpu_workers <= 0 {
            return 0.0;
        }
        let num_workers_on_server = self.allocated.iter().filter(|&&s| s == server).count() as i32;
        if num_workers_on_server == 0 {
            return 0.0;
        }
        if num_workers_on_server == self.gpu_workers {
            return self.profiled_network;
        }
        self.profiled_network * num_workers_on_server as f64 / self.gpu_workers as f64
    }

    /// Initialize communication phase with accurate progress tracking.
    pub fn start_comm_phase(&mut self, current_time: f64, initial_contention: f64) {
        self.current_phase = Phase::Comm;
        self.is_in_comm_phase = true;
        self.phase_start_time = Some(current_time);
        self.comm_phase_segments.clear();
        self.comm_phase_original_done = 0.0;
        self.comm_phase_current_start = Some(current_time);
        self.comm_phase_current_contention = Some(initial_contention);
    }

    /// Update contention factor during communication phase.
    pub fn update_comm_contention(&mut self, current_time: f64, new_contention: f64) {
        if !self.is_in_comm_phase {
            return;
        }

        let segment_start = self.comm_phase_current_start.unwrap_or(current_time);
        let segment_duration = current_time - segment_start;
        let original_progress = if let Some(contention) = self.comm_phase_current_contention {
            if contention > 0.0 {
                segment_duration / contention
            } else {
                0.0
            }
        } else {
            0.0
        };

        self.comm_phase_segments.push((
            segment_start,
            current_time,
            self.comm_phase_current_contention.unwrap_or(0.0),
        ));
        self.comm_phase_original_done += original_progress;

        self.comm_phase_current_start = Some(current_time);
        self.comm_phase_current_contention = Some(new_contention);
    }

    /// Calculate remaining original communication time.
    pub fn get_comm_remaining_original(&self, current_time: f64) -> f64 {
        if !self.is_in_comm_phase {
            return 0.0;
        }

        let segment_start = self.comm_phase_current_start.unwrap_or(current_time);
        let segment_duration = current_time - segment_start;
        let current_segment_progress = if let Some(contention) = self.comm_phase_current_contention
        {
            if contention > 0.0 {
                segment_duration / contention
            } else {
                0.0
            }
        } else {
            0.0
        };

        let total_progress = self.comm_phase_original_done + current_segment_progress;
        let remaining = self.iteration_networking_time - total_progress;
        remaining.max(0.0)
    }

    /// Finalize communication phase when it completes.
    pub fn finish_comm_phase(&mut self, current_time: f64) {
        if !self.is_in_comm_phase {
            return;
        }

        let segment_start = self.comm_phase_current_start.unwrap_or(current_time);
        let segment_duration = current_time - segment_start;
        let original_progress = if let Some(contention) = self.comm_phase_current_contention {
            if contention > 0.0 {
                segment_duration / contention
            } else {
                0.0
            }
        } else {
            0.0
        };

        self.comm_phase_segments.push((
            segment_start,
            current_time,
            self.comm_phase_current_contention.unwrap_or(0.0),
        ));
        self.comm_phase_original_done += original_progress;

        // Update consumed time (total elapsed time with slowdown)
        let phase_start = self.phase_start_time.unwrap_or(current_time);
        let total_elapsed = current_time - phase_start;
        if let Some(ref mut consumed) = self.consumed_comms_time {
            *consumed += total_elapsed;
        }

        self.is_in_comm_phase = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_job() -> GpuJob {
        GpuJob::new(
            1,
            "resnet110".to_string(),
            0.0,
            100.0,
            0.5,
            0.3,
            4,
            0,
            1,
            4,
            4,
            100.0,
            1.0,
            500.0,
            10.0,
        )
    }

    #[test]
    fn test_required_bandwidth_allreduce_all_colocated() {
        let mut job = make_test_job();
        job.allocated = vec![0, 0, 0, 0];
        assert_eq!(job.required_bandwidth(0), 0.0);
    }

    #[test]
    fn test_required_bandwidth_allreduce_distributed() {
        let mut job = make_test_job();
        job.allocated = vec![0, 0, 1, 1];
        // gpu_workers=4, >2 => 2 * C_j / gpu_workers = 2 * 500 / 4 = 250
        assert!((job.required_bandwidth(0) - 250.0).abs() < 1e-6);
        assert!((job.required_bandwidth(1) - 250.0).abs() < 1e-6);
    }

    #[test]
    fn test_required_bandwidth_ps() {
        let mut job = make_test_job();
        job.ps = 2;
        job.allocated = vec![0, 0, 1, 1];
        job.ps_allocated = vec![0, 1];
        // PS model: C_j * (ps_inside * wk_outside + ps_outside * wk_inside) / (ps * gpu_workers)
        // server 0: C_j * (1 * 2 + 1 * 2) / (2 * 4) = 500 * 4 / 8 = 250
        assert!((job.required_bandwidth(0) - 250.0).abs() < 1e-6);
    }

    #[test]
    fn test_comm_phase_tracking() {
        let mut job = make_test_job();
        job.consumed_comms_time = Some(0.0);

        // Start comm phase at time 10 with contention 1.0
        job.start_comm_phase(10.0, 1.0);
        assert!(job.is_in_comm_phase);

        // At time 10.15 (0.15s elapsed), remaining = 0.3 - 0.15 = 0.15
        let remaining = job.get_comm_remaining_original(10.15);
        assert!((remaining - 0.15).abs() < 1e-9);

        // Update contention at time 10.15 to 2.0
        job.update_comm_contention(10.15, 2.0);
        // remaining should still be 0.15
        let remaining = job.get_comm_remaining_original(10.15);
        assert!((remaining - 0.15).abs() < 1e-9);

        // Finish at time 10.45 (0.15 original remaining * 2.0 contention = 0.30 wall clock)
        job.finish_comm_phase(10.45);
        assert!(!job.is_in_comm_phase);
    }
}
