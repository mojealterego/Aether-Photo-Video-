# Aether development roadmap

## R0 — bootstrap (this commit)
- [x] Initialize canonical repository and reference-device architecture.
- [x] Pure tile partition planner and 400 MP disk-budget estimates.
- [x] CLI `aether plan`, offline standard-library core tests, Linux CI definition.
- [ ] Confirm GitHub Actions run and pass on GitHub.

## R1 — migrate and audit existing Photo400 (next)
- [ ] Import previous Photo400 core, Android/Termux code, ComfyUI adapters and tests into separate packages.
- [ ] Fix known integration drift in saved work artifacts (for example ArcFace R100 import and missing helper modules).
- [ ] Make complete local test suite pass on a clean environment and pin dependency profiles.
- [ ] Add golden-image and RGB16 BigTIFF round-trip tests.

## R2 — S24 Ultra on-device, offline
- [ ] Termux installation instructions and reproducible dependency lockfile.
- [ ] Actual TIFF source decoding, linear optical corrections, focus mask, faithful frequency fusion, disk WOLA, RGB16 BigTIFF.
- [ ] Resume after Android app interruption; disk preflight/low-memory and thermal guard.
- [ ] Test 12/50/200 MP real photographs on the only available S24 Ultra.
- [ ] Document 400 MP render timings, RSS, temperature and output verification.

## R3 — optional neural worker
- [ ] Verify exact ComfyUI node contracts, model licenses and model hashes.
- [ ] Optional SDXL/ControlNet Tile/TiledDiffusion/Depth Anything V2.
- [ ] Inpaint/face restoration with explicit user review, opt-in transfer and identity similarity veto.
- [ ] Reproducible GUI/API workflow, per-tile job IDs, checksums and recovery.
- [ ] Real GPU tests only when a suitable worker exists.

## R4 — multiplatform app and video
- [ ] Choose Android UI stack based on performance profiling (Kotlin/Compose recommended candidate).
- [ ] Implement thin desktop and web clients using same job schema.
- [ ] Video: frames, timestamps, scene cuts, temporal stabilization, audio/metadata handling.
- [ ] End-to-end tests, packaging, accessibility, release engineering.

A roadmap item is complete only with reproducible evidence, not because source code has been drafted.
