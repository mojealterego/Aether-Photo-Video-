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
| OmniGraph IR and typed DAG compiler | `omniflow-control/src/ir.rs`, `compiler.rs` | INITIAL IMPLEMENTATION + Rust unit tests; full 12-stage compiler pending |
| 20-state job scheduler, WFQ and fencing | `omniflow-control/src/scheduler.rs` | INITIAL IN-MEMORY IMPLEMENTATION + Rust tests; durable leases pending |
| KMS/DEK-KEK, audit, SSRF and REST API | `omniflow-control/src/security.rs`, `api.rs` | IN-MEMORY AES-GCM VAULT, audit chain and static URL guard; external KMS, DNS pinning and authenticated REST missing |
| Native OmegaNode ABI v1.1, UDS, T0 worker | `omniflow-worker/omega_sdk/protocol.py` | INITIAL IMPLEMENTATION; T0 ONLY |
| Tenant-scoped content-addressed artifact storage | `omniflow-worker/omega_sdk/protocol.py` | BASIC FILESYSTEM CAS; NO MULTIHOST |
| Nikon NEF 16-bit ingest | `src/aether/ingest/libraw_adapter.py`, `omniflow-worker/omega_sdk/photo_faithful_400mp.py` | sRGB transfer fix and native linear-sRGB NEF node; actual LibRaw E2E pending |
| Faithful 400 MP, tile coverage and BigTIFF | `omniflow-worker/omega_sdk/photo_faithful_400mp.py`, `optics.py` | NATIVE DISK-BACKED STITCHING, OETF, ICC16 TIFF + bounded RL/CA; full 400MP E2E pending |
| Depth Ω / Depth Anything 3 | `omniflow-worker/omega_sdk/vision_native.py` | NATIVE DA3 OUTPUT/CONF CAS ADAPTER + LOCAL MODEL LOADER; real weights/GPU E2E pending |
| ArcFace Ω / Identity Guard Ω | `src/aether/vision/identity_guard.py`, `omniflow-worker/omega_sdk/identity_guard.py` | NATIVE ABSTAIN/DRIFT/HUMAN-APPROVAL POLICY; actual ArcFace embedding inference pending |
| CodeFormer Ω and inpainting | `omniflow-worker/omega_sdk/vision_native.py` | NATIVE PYTORCH CODEFORMER FACE CANDIDATE ADAPTER; true model E2E/licensing and dedicated inpainting pending |
| AVS / ASS / GLS | Native digital human/world operators | NOT IMPLEMENTED |
| Dual-GGUF, Wan / Temporal VAE, 30-agent studio | `omniflow-worker/omega_sdk/llm_and_video_adapters.py` | NOT IMPLEMENTED |
| Chronos Ledger PostgreSQL / RLS / outbox | `sql/001_omniflow_ledger.sql` | DDL AND RLS SMOKE TEST COMMITTED; production DB integration and results pending |
| Mobile S24 Ultra / Studio UI SDK | `omniflow-ui/src/sdk/omniflow_client.ts` | NOT IMPLEMENTED |
| CI Gate C5: no ComfyUI dependency | `ci/verify_zero_comfy_c5.py` | AST STATIC CHECK + TESTS; runtime/SBOM proof pending |
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

## Latest execution evidence (2026-10-26)

- Local Python OmegaNode, NEF contract, weighted TIFF, identity, and optics tests
  were executed in the assistant's temporary workspace. 44 Python
  unit tests + 2 subtests passed in the latest full native local run. These are not hardware/model-weight acceptance tests.
- A new Rust 2024 Cargo crate and GitHub Actions Rust test/clippy job were
  committed on `main`; their remote result has not been confirmed here.
- PostgreSQL 16 test service + tenant-RLS SQL smoke are configured in CI;
  their remote result has not been confirmed here.
- ComfyUI integration in the older Aether tree is transitional; native
  OmniFlow code must not import ComfyUI modules.
- No external GPU was used, no external faces/models were uploaded, and no
  full 400 MP camera-to-output run has been certified.
- The data plane is T0-only and its supplied fencing token has not been
  validated against the authoritative control plane. Do not expose the UDS
  worker as a public multi-tenant service.
- SQL DDL alone does not make the Rust scheduler durable; the transactional
  PostgreSQL adapter is still outstanding.

The next mandatory acceptance gates are: Rust build/CI evidence,
PostgreSQL RLS proof, real NEF + ICC chart proof, native scheduler↔worker↔CAS
RPC E2E, license-approved real DA3/ArcFace/CodeFormer weights, and a secured
GPU run (only with explicit authorization to transfer photographs).

## Native executable NEF baseline

After an environment has LibRaw-compatible `rawpy` installed:

```bash
pip install -e ".[mobile,omniflow]"
aether-omniflow-nef --nef ./camera.nef --output ./camera-rgb16.tif --tile 512 --overlap 128
```

This is a cohesive first-stage invocation:
`I002_LoadNEF` -> linear-sRGB float32 -> disk-backed weighted tile
compositor -> explicit IEC 61966-2-1 OETF -> display-sRGB ICC16 BigTIFF.
It preserves the original NEF and refuses to overwrite the requested
output. The provenance reports both SHA-256 digests.

The command currently produces output at **native input dimensions**.
It does **not** perform upscale/super-resolution to 400 MP, claim
restored documentary detail, validate chart ΔE, or run neural inference.
The integrated CLI test uses an injected synthetic LibRaw decoder and
is not a substitute for actual Nikon D810/D750 acceptance.

## Neural checkpoint adapters — explicit restrictions

- `DepthOmega`: signed/scoped CAS RGB8 preview -> DA3 inference ->
  relative depth/confidence `float32` stored as immutable NPZ. Relative
  depth is never silently interpreted as metric depth.
- `CodeFormerOmega`: pre-aligned 512x512 display-sRGB face ->
  CodeFormer checkpoint output -> immutable candidate PNG explicitly
  marked `NEEDS_REVIEW` and `identity_verified=false`.
- Local checkpoint loading requires an existing weight file and
  user-supplied SHA-256; no automatic remote model/photo upload.
- Neural adapters were tested against injected fakes. They have NOT
  run actual model weights, on real faces, or through a GPU in this run.
- ComfyUI is not used by these native modules.
- CodeFormer S-Lab noncommercial restrictions and ArcFace weight rights
  must be checked before commercial deployment.

## Current recorded local test result

The native Python test suite through the integrated NEF CLI contains
**44 passing tests and 2 passing subtests** (local Python 3.13).
Remote GitHub Actions Rust/Python/PostgreSQL results and actual GPU
results remain unverified.
