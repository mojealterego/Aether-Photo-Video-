# AETHERIS OmniFlow Ω — Engine Module Contract v1.1

System: `SYS-AETHERIS-OMNIFLOW-016` / unified `ACE 2.0`.
Scope: native, independent, versioned engine components inside
`mojealterego/Aether-Photo-Video-` on **`main`**.

## Purpose

OmniFlow must absorb dozens or hundreds of additional engines without
mutating the security model or hardcoding each engine into the scheduler.
Engine discovery is **declarative** and cannot execute Python at discovery
time. A declaration is not permission to run.

Implementation:
`omniflow-worker/omega_sdk/registry.py`

## Required declarative metadata

Each `EngineDescriptor` contains:

- `op_id`, exact `op_version` (numeric `MAJOR.MINOR.PATCH`),
  `abi_version = omega.node.abi.v1.1`.
- `provider`, `family`, `trust_class`, `mode`.
- `state`: `PLANNED`, `ON_HOLD`, `EXPERIMENTAL`, `VERIFIED`, `REVOKED`.
- `provenance_sha256` (SHA-256 of the engine source/build artifact),
  `license_id` and `test_evidence` references.
- Exact-version `dependencies`, `capabilities` and named
  `input_ports`/`output_ports` with explicit semantic types.

The registry derives `sha256:...` from canonical serialized JSON and
refuses an attempt to replace an existing exact `(op_id, op_version)`
with an incompatible descriptor. Independent versions may coexist.
It resolves dependency DAGs in topological order, detecting cycles and
unknown exact dependencies. Its dependency planner has an explicit
1250-engine regression test with no Python recursion.

## Compatibility and safety gates

- **No implicit conversion:** `validate_link` requires exact semantic
  port match; scene-linear RGB, display sRGB, depth units and VAE latent
  formats must use an explicit conversion node.
- **No self-activation:** manifest parsing never imports plugin code.
  `register_verified_operator` checks a previously authorized descriptor,
  requires T0 trust, and matches ABI, family, name, version and mode
  against an already-trusted runtime instance.
- T1/T2 modules require process/VM isolation before any invocation.
  This first-stage registry does not supply such isolation.
- Any `ON_HOLD`, `PLANNED`, `EXPERIMENTAL` or `REVOKED` module
  is excluded when `require_verified=True`.
- **Critical distinction:** a `VERIFIED` label and a test-evidence
  reference inside JSON are not cryptographic attestation. A production
  control plane must validate release provenance, signed evidence and
  effective licensing against independent trusted records. The local
  T0-binding helper is for trusted callers only, not public requests.

## Future module acceptance flow

1. Receive the Argon engineering specification and inventory its exact
   node IDs, versions, data types, responsibilities and dependencies.
2. Add tests demonstrating the ABI, type compatibility, rejection
   behavior, integrity, resource budgets and missing prerequisites.
3. Implement the native engine within the same `main` codebase.
4. Register a descriptor with pinned SHA-256 and exact dependencies.
5. Compile and run unit/integration tests and full CI on the new commit.
6. Validate real weights, hardware, licensing, performance, optical
   quality and multi-tenant security separately when applicable.
7. Only then request a trusted production admission decision.

## Current paused subsystems

**Do not integrate or expand DA3, CodeFormer or ArcFace** while waiting
for Argon's proprietary replacements. These technologies are not
eligible for native production activation just because legacy reference
adapters exist in the repo.

## Open infrastructure gates

The catalog does not replace the unfinished Rust compiler, PostgreSQL
scheduler persistence, OMAs/OMAt authorisation, T1/T2 isolation, KMS/HSM,
model and plugin attestations, or real GPU/400MP E2E. See
`OMNIFLOW_OMEGA_INTEGRATION.md` for acceptance status.
