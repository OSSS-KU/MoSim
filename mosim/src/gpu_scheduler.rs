use crate::gpu_cluster::{GPUCluster, PlacementMethod};
use crate::gpu_job::GpuJob;
use crate::timer::{EventType, Timer};

pub struct JobMetrics {
    pub job_id: i32,
    pub model: String,
    pub arrival_time: f64,
    pub start_time: f64,
    pub end_time: f64,
    pub wait_time: f64,
    pub queueing_wait_time: f64,
    pub capacity_wait_time: f64,
    pub placement_wait_time: f64,
    pub training_time: f64,
    pub jct: f64,
    pub compute_time: f64,
    pub comms_time: f64,
    pub loading_time: f64,
}

pub struct SchedulerMetrics {
    pub jobs: Vec<JobMetrics>,
    pub sched_idle_time: f64,
    pub sched_capacity_wait_time: f64,
    pub sched_placement_wait_time: f64,
}

pub struct GPUScheduler {
    placement_method: PlacementMethod,
    metrics_jobs: Vec<i32>, // completed job indices

    // Scheduler-wide wait time tracking
    sched_idle_time: f64,
    sched_capacity_wait_time: f64,
    sched_placement_wait_time: f64,
    last_state_change_time: f64,
    current_state: SchedulerState,
}

#[derive(PartialEq)]
enum SchedulerState {
    Idle,
    CapacityWait,
    PlacementWait,
}

impl GPUScheduler {
    pub fn new(
        placement_method: PlacementMethod,
        jobs: &mut [GpuJob],
        timer: &mut Timer,
    ) -> Self {
        // Sort jobs by arrival time then job_id, add to arrival queue
        let mut indices: Vec<usize> = (0..jobs.len()).collect();
        indices.sort_by(|&a, &b| {
            jobs[a]
                .arrival_time
                .partial_cmp(&jobs[b].arrival_time)
                .unwrap()
                .then(jobs[a].job_id.cmp(&jobs[b].job_id))
        });

        for &idx in &indices {
            timer.add_arrival(jobs[idx].job_id);
        }
        trace_log!("Total {} jobs added to arrival queue", jobs.len());

        // Add first arrival event
        if let Some(first_job_id) = timer.get_next_arrival() {
            let arrival_time = jobs[first_job_id as usize].arrival_time;
            timer.add_event(arrival_time, EventType::Arrive, first_job_id);
        }

        GPUScheduler {
            placement_method,
            metrics_jobs: Vec::new(),
            sched_idle_time: 0.0,
            sched_capacity_wait_time: 0.0,
            sched_placement_wait_time: 0.0,
            last_state_change_time: 0.0,
            current_state: SchedulerState::Idle,
        }
    }

    fn update_scheduler_state(&mut self, new_state: SchedulerState, current_time: f64) {
        let time_in_state = current_time - self.last_state_change_time;
        match self.current_state {
            SchedulerState::Idle => self.sched_idle_time += time_in_state,
            SchedulerState::CapacityWait => self.sched_capacity_wait_time += time_in_state,
            SchedulerState::PlacementWait => self.sched_placement_wait_time += time_in_state,
        }
        self.current_state = new_state;
        self.last_state_change_time = current_time;
    }

    fn check_pending_jobs(&self, timer: &Timer) -> bool {
        if timer.has_arrivals() {
            return true;
        }
        timer.peek_next_arrive_event().is_some()
    }

    pub fn run(
        &mut self,
        jobs: &mut [GpuJob],
        cluster: &mut GPUCluster,
        timer: &mut Timer,
    ) {
        let mut past_placement_success_time: f64 = 0.0;

        loop {
            let event = timer.get_next_event();
            if event.is_none() {
                if timer.has_arrivals() {
                    trace_log!(
                        "Warning: Still have arrivals but no events - cluster resources insufficient"
                    );
                } else {
                    trace_log!("DONE");
                    let final_time = if self.metrics_jobs.is_empty() {
                        self.last_state_change_time
                    } else {
                        self.metrics_jobs
                            .iter()
                            .map(|&jid| jobs[jid as usize].end_time.unwrap_or(0.0))
                            .fold(0.0_f64, f64::max)
                    };
                    self.update_scheduler_state(SchedulerState::Idle, final_time);
                }
                break;
            }

            let (current_time, event_type, job_id) = event.unwrap();
            trace_log!(
                "current time: {}, job id: {}, event type: {:?}",
                current_time, job_id, event_type
            );
            trace_log!(
                "At {}, Event type: {:?}, Job ID: {}\n",
                current_time, event_type, job_id
            );

            match event_type {
                EventType::Arrive => {
                    self.handle_arrive(
                        job_id,
                        current_time,
                        &mut past_placement_success_time,
                        jobs,
                        cluster,
                        timer,
                    );
                }
                EventType::Start => {
                    cluster.allocate_network(job_id, current_time, jobs, timer);
                }
                EventType::IterCompEnd => {
                    self.handle_iter_comp_end(job_id, current_time, jobs, cluster, timer);
                }
                EventType::IterCommEnd => {
                    self.handle_iter_comm_end(job_id, current_time, jobs, cluster, timer);
                }
                EventType::Complete => {
                    self.handle_complete(job_id, current_time, jobs, cluster, timer);
                }
            }
        }
    }

    fn handle_arrive(
        &mut self,
        job_id: i32,
        current_time: f64,
        past_placement_success_time: &mut f64,
        jobs: &mut [GpuJob],
        cluster: &mut GPUCluster,
        timer: &mut Timer,
    ) {
        let job = &mut jobs[job_id as usize];

        // Queue head time
        if job.queue_head_time.is_none() {
            job.queue_head_time = Some((*past_placement_success_time).max(job.arrival_time));
        }

        // Capacity met time
        let total_available_gpus: i32 = cluster.gpus.iter().sum();
        let total_available_cpus: i32 = cluster.cpu_cores.iter().sum();
        let needed_gpus = job.gpu_workers * job.gpu_per_worker;
        let needed_cpus = job.ps * job.cpu_per_ps + job.gpu_workers * job.cpu_per_gpu_worker;

        if job.capacity_met_time.is_none() {
            if needed_gpus <= total_available_gpus && needed_cpus <= total_available_cpus {
                job.capacity_met_time = Some(current_time);
                self.update_scheduler_state(SchedulerState::PlacementWait, current_time);
            } else {
                self.update_scheduler_state(SchedulerState::CapacityWait, current_time);
            }
        } else {
            self.update_scheduler_state(SchedulerState::PlacementWait, current_time);
        }

        // Try to allocate
        if cluster.allocate_resources(job_id, self.placement_method, current_time, jobs, timer) {
            *past_placement_success_time = current_time;

            if let Some(next_job_id) = timer.get_next_arrival() {
                let next_arrival_time =
                    current_time.max(jobs[next_job_id as usize].arrival_time);
                jobs[next_job_id as usize].temp_arrival_time = next_arrival_time;
                timer.add_event(next_arrival_time, EventType::Arrive, next_job_id);
                trace_log!("{} Next job added to arrival queue", current_time);
            } else if !self.check_pending_jobs(timer) {
                self.update_scheduler_state(SchedulerState::Idle, current_time);
            }
        } else {
            // Allocation failed - find next retry time
            let mut next_try_allocate_time: Option<f64> = None;

            if let Some((t, _, _)) = timer.peek_next_complete_event() {
                next_try_allocate_time = Some(t);
            }
            if let Some((t, _, _)) = timer.peek_next_start_event() {
                next_try_allocate_time = Some(match next_try_allocate_time {
                    Some(prev) => prev.min(t),
                    None => t,
                });
            }

            if cluster.interference_model.is_iter_model() {
                if let Some((t, _, _)) = timer.peek_next_iter_comm_end_event() {
                    next_try_allocate_time = Some(match next_try_allocate_time {
                        Some(prev) => prev.min(t),
                        None => t,
                    });
                }
                if let Some((t, _, _)) = timer.peek_next_iter_comp_end_event() {
                    next_try_allocate_time = Some(match next_try_allocate_time {
                        Some(prev) => prev.min(t),
                        None => t,
                    });
                }
            }

            let next_try = match next_try_allocate_time {
                Some(t) => t,
                None => {
                    trace_log!(
                        "Error: Cannot allocate Job {} and no future events",
                        job_id
                    );
                    trace_log!(
                        "At {} Remaining CPU {:?} GPU {:?} Network {:?}",
                        current_time, cluster.cpu_cores, cluster.gpus, cluster.bandwidths
                    );
                    panic!(
                        "Error -- Cannot allocate Job {} and no future events",
                        job_id
                    );
                }
            };

            let job = &mut jobs[job_id as usize];
            let next_try = next_try.max(job.temp_arrival_time);
            job.temp_arrival_time = next_try;
            timer.add_event(next_try, EventType::Arrive, job_id);
        }
    }

    fn handle_iter_comp_end(
        &self,
        job_id: i32,
        current_time: f64,
        jobs: &mut [GpuJob],
        cluster: &mut GPUCluster,
        timer: &mut Timer,
    ) {
        trace_log!(
            "Job {} iteration {} compute phase ended",
            job_id, jobs[job_id as usize].current_iteration
        );

        let elapsed = current_time - jobs[job_id as usize].phase_start_time.unwrap_or(current_time);
        *jobs[job_id as usize].consumed_compute_time.as_mut().unwrap() += elapsed;
        *jobs[job_id as usize].training_time.as_mut().unwrap() += elapsed;

        // Calculate initial contention
        let job = &jobs[job_id as usize];
        let factors: Vec<f64> = job
            .allocated
            .iter()
            .map(|&s| {
                let required = job.required_bandwidth(s);
                let using = job
                    .using_bandwidths
                    .get(&s)
                    .copied()
                    .unwrap_or(0.0)
                    .max(cluster.min_guaranteed_bw);
                required / using
            })
            .collect();
        let initial_contention = factors.into_iter().fold(1.0_f64, |acc, x| acc.max(x));

        jobs[job_id as usize].start_comm_phase(current_time, initial_contention);

        cluster.allocate_bandwidth(jobs);

        let comm_duration =
            jobs[job_id as usize].iteration_networking_time * initial_contention;
        timer.add_event(current_time + comm_duration, EventType::IterCommEnd, job_id);

        cluster.update_completion_times(current_time, jobs, timer);

        trace_log!(
            "Job {} started comm phase, contention={:.2}, duration={:.2}s",
            job_id, initial_contention, comm_duration
        );
    }

    fn handle_iter_comm_end(
        &self,
        job_id: i32,
        current_time: f64,
        jobs: &mut [GpuJob],
        cluster: &mut GPUCluster,
        timer: &mut Timer,
    ) {
        trace_log!(
            "Job {} iteration {} comm phase ended",
            job_id, jobs[job_id as usize].current_iteration
        );

        let comm_phase_start = jobs[job_id as usize]
            .phase_start_time
            .unwrap_or(current_time);
        let actual_comm_duration = current_time - comm_phase_start;
        let comp_time = jobs[job_id as usize].iteration_computing_time;

        jobs[job_id as usize].finish_comm_phase(current_time);

        jobs[job_id as usize].current_iteration += 1;
        jobs[job_id as usize].completed_iterations =
            Some(jobs[job_id as usize].current_iteration as f64);

        let job = &jobs[job_id as usize];
        if job.current_iteration >= job.iteration_number as i64 {
            // Job completed
            jobs[job_id as usize].current_phase = crate::gpu_job::Phase::Completed;
            timer.add_event(current_time, EventType::Complete, job_id);
        } else {
            // Start next compute phase (overlap credit may advance the next IterCompEnd)
            let overlap_credit =
                (cluster.overlapping_ratio * comp_time).min(actual_comm_duration);
            let adjusted_compute_start = current_time - overlap_credit;

            let job = &mut jobs[job_id as usize];
            job.current_phase = crate::gpu_job::Phase::Compute;
            job.phase_start_time = Some(adjusted_compute_start);
            job.is_in_comm_phase = false;

            cluster.allocate_bandwidth(jobs);

            timer.add_event(
                adjusted_compute_start + comp_time,
                EventType::IterCompEnd,
                job_id,
            );

            cluster.update_completion_times(current_time, jobs, timer);

            trace_log!(
                "Job {} started iteration {} compute phase",
                job_id, jobs[job_id as usize].current_iteration
            );
        }
    }

    fn handle_complete(
        &mut self,
        job_id: i32,
        current_time: f64,
        jobs: &mut [GpuJob],
        cluster: &mut GPUCluster,
        timer: &mut Timer,
    ) {
        jobs[job_id as usize].end_time = Some(current_time);

        if !cluster.interference_model.is_iter_model() {
            let job = &mut jobs[job_id as usize];
            let elapsed_time =
                current_time - job.last_change_time.unwrap_or(current_time);
            *job.training_time.as_mut().unwrap() += elapsed_time;
            let last_factor = job.last_network_factor.unwrap_or(1.0);
            let last_compute_ratio = job.iteration_computing_time
                / (job.iteration_computing_time + job.iteration_networking_time * last_factor);
            *job.consumed_compute_time.as_mut().unwrap() += elapsed_time * last_compute_ratio;
            *job.consumed_comms_time.as_mut().unwrap() +=
                elapsed_time * (1.0 - last_compute_ratio);

            let actual_networking_time = job.iteration_networking_time * last_factor;
            let iteration_time = cluster.effective_iteration_time(job, actual_networking_time);
            let new_completed =
                job.completed_iterations.unwrap_or(0.0) + elapsed_time / iteration_time;
            job.completed_iterations = Some(new_completed.min(job.iteration_number));
        }

        // Calculate wait time components
        let job = &jobs[job_id as usize];
        let queue_head = job.queue_head_time.unwrap_or(job.arrival_time);
        let capacity_met = job.capacity_met_time.unwrap_or(queue_head);
        let start = job.start_time.unwrap_or(capacity_met);

        let job = &mut jobs[job_id as usize];
        job.queueing_wait_time = Some(queue_head - job.arrival_time);
        job.capacity_wait_time = Some(capacity_met - queue_head);
        job.placement_wait_time = Some(start - capacity_met);

        cluster.release_resources(job_id, current_time, jobs, timer);
        self.metrics_jobs.push(job_id);

        trace_log!("complete: {}", job_id);
        trace_log!(
            "After release, Remaining CPU {:?} GPU {:?} Network {:?}",
            cluster.cpu_cores, cluster.gpus, cluster.bandwidths
        );
    }

    pub fn metrics(&self, jobs: &[GpuJob]) -> SchedulerMetrics {
        let mut result = Vec::new();
        for &jid in &self.metrics_jobs {
            let job = &jobs[jid as usize];
            let loading = job.consumed_loading_time.unwrap_or(0.0);
            let compute = job.consumed_compute_time.unwrap_or(0.0);
            let comms = job.consumed_comms_time.unwrap_or(0.0);
            result.push(JobMetrics {
                job_id: job.job_id,
                model: job.model.clone(),
                arrival_time: job.arrival_time,
                start_time: job.start_time.unwrap_or(0.0),
                end_time: job.end_time.unwrap_or(0.0),
                wait_time: job.wait_time.unwrap_or(0.0),
                queueing_wait_time: job.queueing_wait_time.unwrap_or(0.0),
                capacity_wait_time: job.capacity_wait_time.unwrap_or(0.0),
                placement_wait_time: job.placement_wait_time.unwrap_or(0.0),
                training_time: loading + compute + comms,
                jct: job.end_time.unwrap_or(0.0) - job.arrival_time,
                compute_time: compute,
                comms_time: comms,
                loading_time: loading,
            });
        }

        SchedulerMetrics {
            jobs: result,
            sched_idle_time: self.sched_idle_time,
            sched_capacity_wait_time: self.sched_capacity_wait_time,
            sched_placement_wait_time: self.sched_placement_wait_time,
        }
    }
}
