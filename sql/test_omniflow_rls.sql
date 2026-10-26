-- PostgreSQL 16 RLS smoke test; execute as postgres, with impersonated nonowner.
\set ON_ERROR_STOP on
BEGIN;
SET ROLE omniflow_tester;
SET LOCAL omniflow.tenant_id = 'tenant_a';
INSERT INTO omniflow.jobs(
 job_id,tenant_id,principal_id,idempotency_key,graph_digest,policy_digest,state,service_class
) VALUES (
 '00000000-0000-0000-0000-000000000001','tenant_a','operator_a','idem_a',
 'sha256:'||repeat('a',64),'sha256:'||repeat('b',64),'QUEUED','EDITORIAL'
);
DO $$
BEGIN
 IF (SELECT count(*) FROM omniflow.jobs)!=1 THEN
   RAISE EXCEPTION 'RLS failed for tenant_a';
 END IF;
END;
$$;
SET LOCAL omniflow.tenant_id = 'tenant_b';
DO $$
BEGIN
 IF (SELECT count(*) FROM omniflow.jobs)!=0 THEN
   RAISE EXCEPTION 'Cross-tenant read leak: tenant_b sees tenant_a';
 END IF;
END;
$$;
INSERT INTO omniflow.jobs(
 job_id,tenant_id,principal_id,idempotency_key,graph_digest,policy_digest,state,service_class
) VALUES (
 '00000000-0000-0000-0000-000000000002','tenant_b','operator_b','idem_b',
 'sha256:'||repeat('a',64),'sha256:'||repeat('b',64),'QUEUED','INTERACTIVE'
);
DO $$
BEGIN
 IF (SELECT count(*) FROM omniflow.jobs)!=1 THEN
   RAISE EXCEPTION 'RLS failed for tenant_b';
 END IF;
END;
$$;
SET LOCAL omniflow.tenant_id = 'tenant_a';
DO $$
BEGIN
 IF (SELECT count(*) FROM omniflow.jobs)!=1 THEN
   RAISE EXCEPTION 'Cross-tenant read leak: tenant_a sees tenant_b';
 END IF;
END;
$$;
ROLLBACK;
