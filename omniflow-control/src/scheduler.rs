#![forbid(unsafe_code)]
use crate::compiler::ExecutionPlan;
use crate::ir::{JobPolicy, ServiceClass};
use chrono::{DateTime, Duration, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobState {
    Created, Validating, Admitted, Queued, Leased, Running,
    Checkpointing, Verifying, Committing, Succeeded,
    WaitingInput, WaitingModel, WaitingApproval,
    RetryDelay, CancelRequested, Cancelled,
    FailedRetryable, FailedFinal, Quarantined, NeedsReview,
}
impl JobState {
    pub fn may_transition(self, to: Self) -> bool {
        use JobState::*;
        matches!((self, to),
            (Created, Validating) | (Validating, Admitted) |
            (Validating, FailedFinal) | (Validating, WaitingModel) |
            (Validating, WaitingInput) | (WaitingInput, Validating) |
            (WaitingModel, Validating) | (Admitted, Queued) |
            (Admitted, WaitingApproval) | (WaitingApproval, Queued) |
            (WaitingApproval, Cancelled) | (Queued, Leased) |
            (Queued, CancelRequested) | (Leased, Running) |
            (Leased, Queued) | (Running, Checkpointing) |
            (Checkpointing, Running) | (Running, Verifying) |
            (Running, FailedRetryable) | (Running, FailedFinal) |
            (Running, Quarantined) | (Running, NeedsReview) |
            (Running, CancelRequested) | (FailedRetryable, RetryDelay) |
            (RetryDelay, Queued) | (Verifying, Committing) |
            (Verifying, NeedsReview) | (Verifying, Quarantined) |
            (Verifying, FailedFinal) | (Committing, Succeeded) |
            (Committing, FailedRetryable) | (CancelRequested, Cancelled))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunRecord {
    pub job_id: String,
    pub graph_digest: String,
    pub idempotency_key: String,
    pub policy: JobPolicy,
    pub plan: ExecutionPlan,
    pub state: JobState,
    pub generation: u64,
    pub fencing_token: u64,
    pub worker_id: Option<String>,
    pub lease_expires_at: Option<DateTime<Utc>>,
    pub committed_artifacts: HashMap<String, String>,
}
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SchedulerError {
    #[error("job not found or tenant mismatch")]
    NotFound,
    #[error("idempotency key reused with different graph")]
    IdempotencyConflict,
    #[error("invalid state transition")]
    InvalidTransition,
    #[error("stale generation")]
    StaleGeneration,
    #[error("stale fencing token")]
    StaleFencingToken,
    #[error("invalid lease duration or GPU budget")]
    InvalidResource,
}

struct ScheduleState {
    jobs: HashMap<String, RunRecord>,
    idempotency: HashMap<(String, String), String>,
    queues: HashMap<ServiceClass, VecDeque<String>>,
    cursor: usize,
}
impl Default for ScheduleState {
    fn default() -> Self {
        let mut queues=HashMap::new();
        for cls in [ServiceClass::Interactive, ServiceClass::Editorial,
            ServiceClass::Production, ServiceClass::BatchResearch] {
            queues.insert(cls, VecDeque::new());
        }
        Self { jobs:HashMap::new(),idempotency:HashMap::new(),queues,cursor:0 }
    }
}

#[derive(Default)]
pub struct WeightedFairScheduler { state: Mutex<ScheduleState> }
impl WeightedFairScheduler {
    pub fn new() -> Self { Self::default() }
    pub fn submit(&self, idempotency_key:&str, policy:JobPolicy,
                  plan:ExecutionPlan) -> Result<RunRecord,SchedulerError> {
        if idempotency_key.trim().is_empty() ||
            plan.estimated_gpu_seconds > policy.max_gpu_seconds {
            return Err(SchedulerError::InvalidResource)
        }
        let mut state=self.state.lock();
        let key=(policy.tenant_id.clone(),idempotency_key.to_string());
        if let Some(id)=state.idempotency.get(&key) {
            let existing=&state.jobs[id];
            if existing.graph_digest != plan.graph_digest {
                return Err(SchedulerError::IdempotencyConflict)
            }
            return Ok(existing.clone())
        }
        let id=format!("job_{}",Uuid::new_v4().simple());
        let record=RunRecord {
            job_id:id.clone(),graph_digest:plan.graph_digest.clone(),
            idempotency_key:idempotency_key.into(),policy,plan,
            state:JobState::Queued,generation:1,fencing_token:0,worker_id:None,
            lease_expires_at:None,committed_artifacts:HashMap::new(),
        };
        state.queues.get_mut(&record.policy.service_class).expect("configured").push_back(id.clone());
        state.idempotency.insert(key,id.clone());
        state.jobs.insert(id,record.clone());
        Ok(record)
    }

    /// Local scheduler only: durable PostgreSQL leases are a separate unimplemented gate.
    pub fn lease_next(&self, worker_id:&str, now:DateTime<Utc>,
                      lease_seconds:i64) -> Result<Option<RunRecord>,SchedulerError> {
        if worker_id.is_empty() || lease_seconds <= 0 {
            return Err(SchedulerError::InvalidResource)
        }
        let mut state=self.state.lock();
        let expired: Vec<(String,ServiceClass)> = state.jobs.values_mut().filter_map(|job| {
            if matches!(job.state,JobState::Leased|JobState::Running)
               && job.lease_expires_at.is_some_and(|exp| exp <= now) {
                job.state=JobState::Queued;
                job.generation+=1;
                job.fencing_token+=1;
                job.worker_id=None;
                job.lease_expires_at=None;
                Some((job.job_id.clone(),job.policy.service_class))
            } else { None }
        }).collect();
        for (id,cls) in expired {
            state.queues.get_mut(&cls).expect("configured").push_back(id);
        }
        let mut wheel=Vec::with_capacity(15);
        for (class,count) in [
            (ServiceClass::Interactive,8),(ServiceClass::Editorial,4),
            (ServiceClass::Production,2),(ServiceClass::BatchResearch,1)] {
            wheel.extend(std::iter::repeat_n(class,count));
        }
        for offset in 0..wheel.len() {
            let index=(state.cursor+offset)%wheel.len();
            let class=wheel[index];
            while let Some(job_id)=state.queues.get_mut(&class).expect("configured").pop_front() {
                let Some(job)=state.jobs.get_mut(&job_id) else { continue; };
                if job.state != JobState::Queued { continue; }
                job.state=JobState::Leased;
                job.generation+=1;
                job.fencing_token+=1;
                job.worker_id=Some(worker_id.into());
                job.lease_expires_at=Some(now+Duration::seconds(lease_seconds));
                let result=job.clone();
                state.cursor=(index+1)%wheel.len();
                return Ok(Some(result))
            }
        }
        Ok(None)
    }

    pub fn transition(&self, job_id:&str,tenant_id:&str,expected_generation:u64,
                      worker_token:u64,to:JobState)
                      -> Result<RunRecord,SchedulerError> {
        let mut state=self.state.lock();
        let job=state.jobs.get_mut(job_id).ok_or(SchedulerError::NotFound)?;
        if job.policy.tenant_id != tenant_id { return Err(SchedulerError::NotFound) }
        if job.fencing_token != worker_token { return Err(SchedulerError::StaleFencingToken) }
        if job.generation != expected_generation { return Err(SchedulerError::StaleGeneration) }
        if !job.state.may_transition(to) { return Err(SchedulerError::InvalidTransition) }
        job.state=to;
        job.generation+=1;
        Ok(job.clone())
    }
    pub fn get_job(&self,job_id:&str,tenant_id:&str) -> Result<RunRecord,SchedulerError> {
        let state=self.state.lock();
        let job=state.jobs.get(job_id).ok_or(SchedulerError::NotFound)?;
        if job.policy.tenant_id != tenant_id { return Err(SchedulerError::NotFound) }
        Ok(job.clone())
    }
    pub fn cancel(&self,job_id:&str,tenant_id:&str) -> Result<RunRecord,SchedulerError> {
        let mut state=self.state.lock();
        let job=state.jobs.get_mut(job_id).ok_or(SchedulerError::NotFound)?;
        if job.policy.tenant_id != tenant_id { return Err(SchedulerError::NotFound) }
        if !matches!(job.state,JobState::Queued|JobState::Running|JobState::Leased) {
            return Err(SchedulerError::InvalidTransition)
        }
        job.state=JobState::CancelRequested;
        job.generation+=1;
        job.fencing_token+=1;
        Ok(job.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{FidelityMode,ServiceClass};
    use std::collections::BTreeSet;
    fn policy(tenant:&str,class:ServiceClass)->JobPolicy {
        JobPolicy {tenant_id:tenant.into(),principal_id:"tester".into(),
            fidelity_mode:FidelityMode::Editorial,service_class:class,
            allowed_models:BTreeSet::new(),commercial_use:false,
            allow_biometric_egress:false,require_identity_guard:false,
            max_nodes:2,max_gpu_seconds:60,available_vram_bytes:1_000_000_000,seed:0 }
    }
    fn plan(digest:&str)->ExecutionPlan {
        ExecutionPlan { graph_digest:digest.into(),topological_order:vec!["a".into()],
            predicted_peak_bytes:1234,estimated_gpu_seconds:1,node_count:1 }
    }
    #[test]
    fn idempotency_is_scoped_by_tenant_and_payload() {
        let s=WeightedFairScheduler::new();
        let a=s.submit("same",policy("a",ServiceClass::Interactive),plan("A")).unwrap();
        assert_eq!(s.submit("same",policy("a",ServiceClass::Interactive),plan("A")).unwrap().job_id,a.job_id);
        assert_eq!(s.submit("same",policy("a",ServiceClass::Interactive),plan("B")).unwrap_err(),
                   SchedulerError::IdempotencyConflict);
        let b=s.submit("same",policy("b",ServiceClass::Interactive),plan("A")).unwrap();
        assert_ne!(a.job_id,b.job_id);
        assert_eq!(s.get_job(&a.job_id,"b").unwrap_err(),SchedulerError::NotFound);
    }
    #[test]
    fn expired_worker_is_fenced_out() {
        let s=WeightedFairScheduler::new(); let now=Utc::now();
        s.submit("1",policy("a",ServiceClass::Interactive),plan("X")).unwrap();
        let old=s.lease_next("workerA",now,1).unwrap().unwrap();
        let fresh=s.lease_next("workerB",now+Duration::seconds(2),10).unwrap().unwrap();
        assert_eq!(old.job_id,fresh.job_id);
        assert!(fresh.fencing_token>old.fencing_token);
        assert_eq!(s.transition(&old.job_id,"a",old.generation,old.fencing_token,
                                 JobState::Running).unwrap_err(),SchedulerError::StaleFencingToken);
        assert_eq!(s.transition(&fresh.job_id,"a",fresh.generation,fresh.fencing_token,
                                 JobState::Running).unwrap().state,JobState::Running);
    }
    #[test]
    fn illegal_success_jump_is_rejected() {
        let s=WeightedFairScheduler::new();let now=Utc::now();
        s.submit("1",policy("a",ServiceClass::Interactive),plan("X")).unwrap();
        let job=s.lease_next("worker",now,30).unwrap().unwrap();
        assert_eq!(s.transition(&job.job_id,"a",job.generation,job.fencing_token,
                       JobState::Succeeded).unwrap_err(),SchedulerError::InvalidTransition);
    }
    #[test]
    fn cancel_bumps_fence_and_denies_old_worker() {
        let s=WeightedFairScheduler::new();let now=Utc::now();
        s.submit("1",policy("a",ServiceClass::Interactive),plan("X")).unwrap();
        let job=s.lease_next("worker",now,30).unwrap().unwrap();
        let cancelled=s.cancel(&job.job_id,"a").unwrap();
        assert!(cancelled.fencing_token>job.fencing_token);
        assert_eq!(s.transition(&job.job_id,"a",job.generation,job.fencing_token,
                JobState::Running).unwrap_err(),SchedulerError::StaleFencingToken);
    }
    #[test]
    fn wheel_has_exact_eight_four_two_one_weight() {
        assert_eq!(ServiceClass::Interactive.weight(),8);
        assert_eq!(ServiceClass::Editorial.weight(),4);
        assert_eq!(ServiceClass::Production.weight(),2);
        assert_eq!(ServiceClass::BatchResearch.weight(),1);
    }
}
