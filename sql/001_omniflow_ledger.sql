-- AETHERIS OmniFlow Omega / SYS-AETHERIS-OMNIFLOW-016
-- PostgreSQL 16: tenant-scoped durable jobs, CAS provenance and outbox.
-- WARNING: app.tenant_id MUST be set only by a trusted, authenticated service.
-- Untrusted clients must not receive SQL connections, table ownership, or BYPASSRLS.
BEGIN;
CREATE SCHEMA IF NOT EXISTS omniflow;

CREATE TABLE IF NOT EXISTS omniflow.jobs (
  job_id UUID PRIMARY KEY,
  tenant_id TEXT NOT NULL CHECK (length(tenant_id) BETWEEN 1 AND 128),
  principal_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL CHECK (length(idempotency_key) BETWEEN 1 AND 128),
  graph_digest TEXT NOT NULL CHECK (graph_digest ~ '^sha256:[a-f0-9]{64}$'),
  policy_digest TEXT NOT NULL CHECK (policy_digest ~ '^sha256:[a-f0-9]{64}$'),
  state TEXT NOT NULL CHECK (state IN (
    'CREATED','VALIDATING','ADMITTED','QUEUED','LEASED','RUNNING',
    'CHECKPOINTING','VERIFYING','COMMITTING','SUCCEEDED',
    'WAITING_INPUT','WAITING_MODEL','WAITING_APPROVAL','RETRY_DELAY',
    'CANCEL_REQUESTED','CANCELLED','FAILED_RETRYABLE','FAILED_FINAL',
    'QUARANTINED','NEEDS_REVIEW'
  )),
  generation BIGINT NOT NULL DEFAULT 1 CHECK (generation >= 1),
  fencing_token BIGINT NOT NULL DEFAULT 0 CHECK (fencing_token >= 0),
  service_class TEXT NOT NULL CHECK (service_class IN
    ('INTERACTIVE','EDITORIAL','PRODUCTION','BATCH_RESEARCH')),
  assigned_worker_id TEXT,
  lease_expires_at TIMESTAMPTZ,
  retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
  consumed_gpu_millis BIGINT NOT NULL DEFAULT 0 CHECK (consumed_gpu_millis >= 0),
  policy JSONB NOT NULL DEFAULT '{}'::jsonb,
  execution_plan JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(tenant_id,idempotency_key)
);
CREATE INDEX IF NOT EXISTS jobs_tenant_queue_idx
 ON omniflow.jobs (tenant_id,service_class,created_at)
 WHERE state='QUEUED';
CREATE INDEX IF NOT EXISTS jobs_expiring_leases_idx
 ON omniflow.jobs (lease_expires_at) WHERE state IN ('LEASED','RUNNING');

CREATE TABLE IF NOT EXISTS omniflow.artifacts (
  artifact_sha256 TEXT NOT NULL CHECK (artifact_sha256 ~ '^sha256:[a-f0-9]{64}$'),
  tenant_id TEXT NOT NULL,
  job_id UUID NOT NULL REFERENCES omniflow.jobs(job_id),
  storage_uri TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  color_profile_sha256 TEXT,
  fidelity_mode TEXT NOT NULL CHECK (fidelity_mode IN
    ('DOC','EDITORIAL','CREATIVE','IDENTITY_REVIEW')),
  verification_status TEXT NOT NULL CHECK (verification_status IN
    ('PREVIEW','UNVERIFIED','ABSTAIN','REJECTED','APPROVED')),
  published BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY(tenant_id,artifact_sha256)
);
CREATE INDEX IF NOT EXISTS artifacts_job_idx ON omniflow.artifacts(job_id);

CREATE TABLE IF NOT EXISTS omniflow.job_events (
  event_id UUID PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  job_id UUID NOT NULL REFERENCES omniflow.jobs(job_id),
  generation BIGINT NOT NULL CHECK (generation >= 1),
  fencing_token BIGINT NOT NULL CHECK (fencing_token >= 0),
  event_kind TEXT NOT NULL,
  actor_id TEXT NOT NULL,
  model_manifest_digest TEXT,
  previous_hash TEXT NOT NULL CHECK (previous_hash ~ '^sha256:[a-f0-9]{64}$'),
  event_hash TEXT NOT NULL CHECK (event_hash ~ '^sha256:[a-f0-9]{64}$'),
  payload JSONB NOT NULL,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(job_id,generation,event_kind)
);
CREATE INDEX IF NOT EXISTS events_job_idx ON omniflow.job_events(job_id,occurred_at);

CREATE TABLE IF NOT EXISTS omniflow.outbox (
  event_id UUID PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  job_id UUID NOT NULL REFERENCES omniflow.jobs(job_id),
  event_type TEXT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  processed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS outbox_pending_idx
 ON omniflow.outbox(created_at) WHERE processed_at IS NULL;

-- Row-level isolation on ALL tenant-bearing tables, INCLUDING the outbox.
ALTER TABLE omniflow.jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE omniflow.jobs FORCE ROW LEVEL SECURITY;
ALTER TABLE omniflow.artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE omniflow.artifacts FORCE ROW LEVEL SECURITY;
ALTER TABLE omniflow.job_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE omniflow.job_events FORCE ROW LEVEL SECURITY;
ALTER TABLE omniflow.outbox ENABLE ROW LEVEL SECURITY;
ALTER TABLE omniflow.outbox FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS jobs_tenant_policy ON omniflow.jobs;
CREATE POLICY jobs_tenant_policy ON omniflow.jobs
 USING (tenant_id = current_setting('omniflow.tenant_id',true))
 WITH CHECK (tenant_id = current_setting('omniflow.tenant_id',true));
DROP POLICY IF EXISTS artifacts_tenant_policy ON omniflow.artifacts;
CREATE POLICY artifacts_tenant_policy ON omniflow.artifacts
 USING (tenant_id = current_setting('omniflow.tenant_id',true))
 WITH CHECK (tenant_id = current_setting('omniflow.tenant_id',true));
DROP POLICY IF EXISTS events_tenant_policy ON omniflow.job_events;
CREATE POLICY events_tenant_policy ON omniflow.job_events
 USING (tenant_id = current_setting('omniflow.tenant_id',true))
 WITH CHECK (tenant_id = current_setting('omniflow.tenant_id',true));
DROP POLICY IF EXISTS outbox_tenant_policy ON omniflow.outbox;
CREATE POLICY outbox_tenant_policy ON omniflow.outbox
 USING (tenant_id = current_setting('omniflow.tenant_id',true))
 WITH CHECK (tenant_id = current_setting('omniflow.tenant_id',true));
COMMIT;

-- Authoritative transitions MUST use a transaction on the same PostgreSQL
-- connection: UPDATE jobs SET state=$new, generation=generation+1
-- WHERE job_id=$id AND tenant_id=$tenant AND generation=$expected_generation
--   AND fencing_token=$expected_fence RETURNING *;
-- Verify exactly 1 row, then INSERT job_events + outbox and COMMIT atomically.
-- The in-memory Rust scheduler is NOT YET CONNECTED to these durable tables.
