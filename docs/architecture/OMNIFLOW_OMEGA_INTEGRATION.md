# AETHERIS OmniFlow Ω — unifikacja architektury ACE 2.0

Source of architectural truth: **SYS-AETHERIS-OMNIFLOW-016 v1.1.0**, the
80-page user-supplied document *Wdrożenie Silnika Aetheris OmniFlow*.
This ledger records implementation status; it is **not** a substitute
for the source document, GPU evidence, or a production certification.

## Repository and integration policy

- One product and one code repository: `mojealterego/Aether-Photo-Video-`.
- All subsequent architectural modules belong to the existing system.
- Work is committed directly to `main`; do not create feature branches or PRs.
- Existing source material and camera originals are never overwritten.
- Add source, tests, acceptance evidence and provenance per implemented module.
- Keep unverified models and incomplete workers **blocked**, not marked done.
- The target OmniFlow runtime is native and architecturally independent
  of ComfyUI. Older Aether code referring to ComfyUI is transitional legacy
  and does not represent the intended native OmniFlow execution plane.

## Canonical subsystem structure

| Subsystem in SYS-AETHERIS-OMNIFLOW-016 | Integration target in unified repository | Status |
|---|---|---|
| OmniGraph IR, 12-stage compiler and typed ports | `omniflow-control/src/ir.rs`, `compiler.rs` | NOT IMPLEMENTED |
| 20-state job scheduler, WFQ and fencing | `omniflow-control/src/scheduler.rs` | NOT IMPLEMENTED |
| KMS/DEK-KEK, audit, SSRF and REST API | `omniflow-control/src/security.rs`, `api.rs` | NOT IMPLEMENTED |
| Native OmegaNode ABI v1.1, UDS, T0 worker | `omniflow-worker/omega_sdk/protocol.py` | INITIAL IMPLEMENTATION; T0 ONLY |
| Tenant-scoped content-addressed artifact storage | `omniflow-worker/omega_sdk/protocol.py` | BASIC FILESYSTEM CAS; NO MULTIHOST |
| Nikon NEF 16-bit ingest | `src/aether/ingest/libraw_adapter.py` | EXISTING; sRGB TRANSFER CORRECTED; scene-linear OmniFlow node pending |
| Faithful 400 MP, tile coverage and BigTIFF | `omniflow-worker/omega_sdk/photo_faithful_400mp.py` plus existing Aether core | PARTIAL LEGACY; NATIVE NODE PENDING |
| Depth Ω / Depth Anything 3 | Native `Depth` OmegaNode | NOT IMPLEMENTED; weights/E2E pending |
| ArcFace Ω / Identity Guard Ω | `src/aether/vision/identity_guard.py` + native bridge | POLICY IMPLEMENTED; real weights/E2E pending |
| CodeFormer Ω and inpainting | Native face-restoration OmegaNode | NOT IMPLEMENTED; licensing and weights pending |
| AVS / ASS / GLS | Native digital human/world operators | NOT IMPLEMENTED |
| Dual-GGUF, Wan / Temporal VAE, 30-agent studio | `omniflow-worker/omega_sdk/llm_and_video_adapters.py` | NOT IMPLEMENTED |
| Chronos Ledger PostgreSQL / RLS / outbox | `sql/001_omniflow_ledger.sql` | NOT IMPLEMENTED |
| Mobile S24 Ultra / Studio UI SDK | `omniflow-ui/src/sdk/omniflow_client.ts` | NOT IMPLEMENTED |
| CI Gate C5: no ComfyUI dependency | `ci/verify_zero_comfy_c5.sh` | NOT IMPLEMENTED |
| AetherCanvas, AWR, SLE, OmniNodes Export | Native modules inside this repository | NOT IMPLEMENTED |

## Native worker baseline — scope and limits

`OmegaWorkerRpcServer` implements the local length-prefixed UDS data plane
with the `omega.node.abi.v1.1` marker. It rejects oversize messages,
unknown operator versions, unrecognized fidelity modes and missing
fencing metadata. `ContentAddressedStorage` stores SHA-256-addressed
tenant-scoped artifacts, never overwriting a committed digest.

**Not yet guaranteed:** authoritative validation of fencing epochs,
cross-process or multi-host tenant isolation, authentication of the socket
peer, T1/T2 sandbox isolation, GPU execution, persistence in Chronos Ledger
and restart/retry semantics. Untrusted, side-effecting or human-approval
operators are refused in this baseline. Consequently, a passing unit suite
is NOT proof of E2E or production security.

## Acceptance gates for each integration

1. Check source artifact and checksum (NEF/model/video).
2. Compile or import the exact implementation in `main`.
3. Run focused tests and full existing repository suite; report failures.
4. Execute real GPU/model/hardware acceptance separately.
5. Validate color management, 16-bit export and no overwrites.
6. Verify human approval / ABSTAIN for identity-sensitive changes.
7. Reject unsupported license / missing evidence with no implicit fallback.

## Source hierarchy

The uploaded design document specifies responsibilities, contracts,
identifiers, mathematical models, node families and gates. This document
tracks the incremental execution status. No broad claim that the entire
80-page design has been implemented or production-certified is made.
