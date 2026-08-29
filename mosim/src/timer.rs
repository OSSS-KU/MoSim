use std::collections::{HashMap, VecDeque};

/// Event types in priority order (lower value = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EventType {
    Complete = 0,
    IterCommEnd = 1,
    IterCompEnd = 2,
    Start = 3,
    Arrive = 4,
}

/// An event in the simulation event queue.
/// Stores job_id (arena index into Vec<GpuJob>) instead of a reference.
#[derive(Debug, Clone)]
pub struct Event {
    pub time: f64,
    pub event_type: EventType,
    pub job_id: i32,
}

impl Event {
    fn less_than(&self, other: &Event) -> bool {
        if self.time != other.time {
            return self.time < other.time;
        }
        (self.event_type as u8) < (other.event_type as u8)
    }
}

/// Custom indexed min-heap with O(log n) update capability.
/// Uses 1-based indexing. Maintains job_id -> heap index mapping.
pub struct Timer {
    pq: Vec<Option<Event>>, // 1-based indexing, index 0 is None
    jobid_to_idx: HashMap<i32, usize>,
    arrival_queue: VecDeque<i32>, // job_ids in arrival order
}

impl Timer {
    pub fn new() -> Self {
        Timer {
            pq: vec![None], // index 0 placeholder
            jobid_to_idx: HashMap::new(),
            arrival_queue: VecDeque::new(),
        }
    }

    /// Add job_id to arrival queue.
    pub fn add_arrival(&mut self, job_id: i32) {
        self.arrival_queue.push_back(job_id);
    }

    /// Check if arrival queue has jobs.
    pub fn has_arrivals(&self) -> bool {
        !self.arrival_queue.is_empty()
    }

    /// Get next job_id from arrival queue.
    pub fn get_next_arrival(&mut self) -> Option<i32> {
        self.arrival_queue.pop_front()
    }

    /// Number of events in the event queue.
    pub fn len(&self) -> usize {
        self.pq.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add event to event queue. Panics if job_id already exists.
    pub fn add_event(&mut self, time: f64, event_type: EventType, job_id: i32) {
        if self.jobid_to_idx.contains_key(&job_id) {
            panic!("Event for job_id {} already exists in queue", job_id);
        }

        let event = Event {
            time,
            event_type,
            job_id,
        };
        self.pq.push(Some(event));
        let idx = self.pq.len() - 1;
        self.jobid_to_idx.insert(job_id, idx);
        self.swim(idx);
    }

    /// Get and remove the highest-priority event.
    pub fn get_next_event(&mut self) -> Option<(f64, EventType, i32)> {
        if self.pq.len() <= 1 {
            return None;
        }

        let last = self.pq.len() - 1;
        self.exchange(1, last);

        let min_event = self.pq.pop().unwrap().unwrap();
        self.jobid_to_idx.remove(&min_event.job_id);

        if self.pq.len() > 1 {
            self.sink(1);
        }

        Some((min_event.time, min_event.event_type, min_event.job_id))
    }

    /// Remove a specific job's event from the queue.
    pub fn remove_job(&mut self, job_id: i32) {
        let idx = match self.jobid_to_idx.get(&job_id) {
            Some(&idx) => idx,
            None => panic!("Job ID {} not found in queue", job_id),
        };

        let last = self.pq.len() - 1;
        self.exchange(idx, last);

        let removed = self.pq.pop().unwrap().unwrap();
        self.jobid_to_idx.remove(&removed.job_id);

        if idx < self.pq.len() {
            self.swim(idx);
            self.sink(idx);
        }
    }

    /// Peek at the highest-priority event without removing it.
    pub fn peek_next_event(&self) -> Option<(f64, EventType, i32)> {
        if self.pq.len() <= 1 {
            return None;
        }
        let event = self.pq[1].as_ref().unwrap();
        Some((event.time, event.event_type, event.job_id))
    }

    /// Update the time of a specific job's event.
    pub fn update_job_time(&mut self, job_id: i32, new_time: f64) {
        let idx = match self.jobid_to_idx.get(&job_id) {
            Some(&idx) => idx,
            None => panic!("Job ID {} not found in queue", job_id),
        };

        self.pq[idx].as_mut().unwrap().time = new_time;
        self.swim(idx);
        self.sink(idx);
    }

    /// Find next event of a specific type by scanning the array (no copy needed in Rust).
    fn peek_next_event_of_type(&self, target: EventType) -> Option<(f64, EventType, i32)> {
        let mut best: Option<&Event> = None;
        for event_opt in &self.pq[1..] {
            if let Some(event) = event_opt {
                if event.event_type == target {
                    match best {
                        None => best = Some(event),
                        Some(b) => {
                            if event.less_than(b) {
                                best = Some(event);
                            }
                        }
                    }
                }
            }
        }
        best.map(|e| (e.time, e.event_type, e.job_id))
    }

    pub fn peek_next_arrive_event(&self) -> Option<(f64, EventType, i32)> {
        self.peek_next_event_of_type(EventType::Arrive)
    }

    pub fn peek_next_complete_event(&self) -> Option<(f64, EventType, i32)> {
        self.peek_next_event_of_type(EventType::Complete)
    }

    pub fn peek_next_start_event(&self) -> Option<(f64, EventType, i32)> {
        self.peek_next_event_of_type(EventType::Start)
    }

    pub fn peek_next_iter_comp_end_event(&self) -> Option<(f64, EventType, i32)> {
        self.peek_next_event_of_type(EventType::IterCompEnd)
    }

    pub fn peek_next_iter_comm_end_event(&self) -> Option<(f64, EventType, i32)> {
        self.peek_next_event_of_type(EventType::IterCommEnd)
    }

    /// Find next ARRIVE/START/COMPLETE event without removing it.
    pub fn peek_next_external_state_change_event(&self) -> Option<(f64, EventType, i32)> {
        let mut best: Option<&Event> = None;
        for event_opt in &self.pq[1..] {
            if let Some(event) = event_opt {
                match event.event_type {
                    EventType::Arrive | EventType::Start | EventType::Complete => match best {
                        None => best = Some(event),
                        Some(b) => {
                            if event.less_than(b) {
                                best = Some(event);
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
        best.map(|e| (e.time, e.event_type, e.job_id))
    }

    // --- Heap operations (1-based indexing) ---

    fn swim(&mut self, mut k: usize) {
        while k > 1 {
            let parent = k / 2;
            let child = self.pq[k].as_ref().unwrap();
            let par = self.pq[parent].as_ref().unwrap();
            if child.less_than(par) {
                self.exchange(k, parent);
                k = parent;
            } else {
                break;
            }
        }
    }

    fn sink(&mut self, mut k: usize) {
        let n = self.pq.len() - 1;
        while 2 * k <= n {
            let mut j = 2 * k;
            if j < n {
                let left = self.pq[j + 1].as_ref().unwrap();
                let right = self.pq[j].as_ref().unwrap();
                if left.less_than(right) {
                    j += 1;
                }
            }
            let child = self.pq[j].as_ref().unwrap();
            let par = self.pq[k].as_ref().unwrap();
            if child.less_than(par) {
                self.exchange(k, j);
                k = j;
            } else {
                break;
            }
        }
    }

    fn exchange(&mut self, i: usize, j: usize) {
        let id_i = self.pq[i].as_ref().unwrap().job_id;
        let id_j = self.pq[j].as_ref().unwrap().job_id;
        self.jobid_to_idx.insert(id_i, j);
        self.jobid_to_idx.insert(id_j, i);
        self.pq.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_event() {
        let mut timer = Timer::new();
        timer.add_event(10.0, EventType::Arrive, 1);
        timer.add_event(5.0, EventType::Complete, 2);

        let event = timer.get_next_event().unwrap();
        assert_eq!(event.0, 5.0);
        assert_eq!(event.1, EventType::Complete);
        assert_eq!(event.2, 2);
    }

    #[test]
    fn test_same_time_ordering() {
        let mut timer = Timer::new();
        timer.add_event(10.0, EventType::Arrive, 1);
        timer.add_event(10.0, EventType::Complete, 2);
        timer.add_event(10.0, EventType::Start, 3);

        let e1 = timer.get_next_event().unwrap();
        assert_eq!(e1.1, EventType::Complete);
        let e2 = timer.get_next_event().unwrap();
        assert_eq!(e2.1, EventType::Start);
        let e3 = timer.get_next_event().unwrap();
        assert_eq!(e3.1, EventType::Arrive);
    }

    #[test]
    fn test_update_job_time() {
        let mut timer = Timer::new();
        timer.add_event(10.0, EventType::Arrive, 1);
        timer.add_event(20.0, EventType::Arrive, 2);

        timer.update_job_time(2, 5.0);
        let event = timer.get_next_event().unwrap();
        assert_eq!(event.2, 2);
        assert_eq!(event.0, 5.0);
    }

    #[test]
    fn test_remove_job() {
        let mut timer = Timer::new();
        timer.add_event(10.0, EventType::Arrive, 1);
        timer.add_event(20.0, EventType::Complete, 2);
        timer.add_event(15.0, EventType::Start, 3);

        timer.remove_job(1);
        assert_eq!(timer.len(), 2);

        let event = timer.get_next_event().unwrap();
        assert_eq!(event.2, 3);
    }

    #[test]
    fn test_peek_event_types() {
        let mut timer = Timer::new();
        timer.add_event(10.0, EventType::Arrive, 1);
        timer.add_event(5.0, EventType::Start, 2);
        timer.add_event(15.0, EventType::Complete, 3);

        let arrive = timer.peek_next_arrive_event().unwrap();
        assert_eq!(arrive.2, 1);

        let start = timer.peek_next_start_event().unwrap();
        assert_eq!(start.2, 2);

        let complete = timer.peek_next_complete_event().unwrap();
        assert_eq!(complete.2, 3);
    }

    #[test]
    fn test_arrival_queue() {
        let mut timer = Timer::new();
        timer.add_arrival(1);
        timer.add_arrival(2);
        timer.add_arrival(3);

        assert!(timer.has_arrivals());
        assert_eq!(timer.get_next_arrival(), Some(1));
        assert_eq!(timer.get_next_arrival(), Some(2));
        assert_eq!(timer.get_next_arrival(), Some(3));
        assert!(!timer.has_arrivals());
        assert_eq!(timer.get_next_arrival(), None);
    }

    #[test]
    fn test_empty_queue() {
        let mut timer = Timer::new();
        assert!(timer.is_empty());
        assert_eq!(timer.get_next_event(), None);
        assert_eq!(timer.peek_next_event(), None);
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn test_duplicate_job_id() {
        let mut timer = Timer::new();
        timer.add_event(10.0, EventType::Arrive, 1);
        timer.add_event(20.0, EventType::Complete, 1);
    }

    #[test]
    fn test_peek_external_state_change() {
        let mut timer = Timer::new();
        timer.add_event(10.0, EventType::IterCompEnd, 1);
        timer.add_event(15.0, EventType::Arrive, 2);
        timer.add_event(20.0, EventType::IterCommEnd, 3);

        let ext = timer.peek_next_external_state_change_event().unwrap();
        assert_eq!(ext.2, 2);
        assert_eq!(ext.1, EventType::Arrive);
    }

    #[test]
    fn test_many_events_ordering() {
        let mut timer = Timer::new();
        for i in (0..100).rev() {
            timer.add_event(i as f64, EventType::Arrive, i);
        }
        let mut prev_time = -1.0;
        for _ in 0..100 {
            let (time, _, _) = timer.get_next_event().unwrap();
            assert!(time > prev_time);
            prev_time = time;
        }
    }
}
