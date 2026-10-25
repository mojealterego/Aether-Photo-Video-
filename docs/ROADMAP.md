# Aether development roadmap

## Current focus (2026-10-25)

- Priority: ICC16 / RAW/DNG import, offline 400 MP reconstruction, optional ComfyUI integration and Depth Anything / ArcFace / CodeFormer model contracts.
- GUI applications (Android, desktop, web) and installers are **deferred**, not removed from the long-term architecture.
- New integration details: [ICC16/RAW/ComfyUI/ONNX](INTEGRATION_ICC_RAW_COMFY_NEURAL.md).
- Real model inference, real GPU ComfyUI E2E and physical S24 Ultra acceptance remain open until evidenced.

## R0 — bootstrap (this commit)
- [x] Initialize canonical repository and reference-device architecture.
- [x] Pure tile partition planner and 400 MP disk-budget estimates.
- [x] CLI `aether plan`, offline standard-library core tests, Linux CI definition.
- [x] Confirm GitHub Actions pass on Python 3.11/3.12 (95 tests per version at commit a658be3).

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

## R4 — video engine active; applications postponed
- [ ] Choose Android UI stack based on performance profiling (Kotlin/Compose recommended candidate).
- [ ] Implement thin desktop and web clients using same job schema.
- [ ] Video: frames, timestamps, scene cuts, temporal stabilization, audio/metadata handling.
- [ ] End-to-end tests, packaging, accessibility, release engineering.

A roadmap item is complete only with reproducible evidence, not because source code has been drafted.

## Argon 400 MP PDF — requirements intake
- [x] Archive page-by-page text of all 64 pages with source hash and attribution.
- [x] Establish requirement IDs A400-01..12 and A400-X01..X10.
- [x] Implement and test disk-backed Gaussian WOLA with normalization exactly once.
- [ ] Complete remaining A400 requirements and Android/ComfyUI acceptance evidence. See docs/requirements/ARGON-400MP.md.

## Etap bieżący: Argon 400MP

- [x] Wyodrębnienie i rejestr wszystkich wymagań z 64 stron.
- [x] Migracja lokalnego S24 BigTIFF oraz zamknięcie podstawowej ścieżki CLI `aether restore`.
- [x] Dodanie modułów optycznych, Depth/Bokeh, WOLA, Frequency Fusion, ArcFace R100 i PSF.
- [x] Referencyjny JSON ComfyUI z testem statycznym.
- [ ] Integracja zaawansowanych modułów w jednej ścieżce optycznej bez dublowania algorytmów.
- [ ] Prawdziwe modele ONNX/ComfyUI oraz testy urządzenia, pełny render 400 MP.

Szczegóły: [stan wdrożenia](ARGON_IMPLEMENTATION_STATUS.md).
