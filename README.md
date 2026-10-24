# Aether Photo & Video

**Mobile-first computational photography and video restoration.**

Reference device: **Samsung Galaxy S24 Ultra (Snapdragon 8 Gen 3 for Galaxy, Android, 12 GB RAM)**. This is the only guaranteed test device. Desktop, server, and other mobile platforms are *future ports*, not required to start developing or testing.

## Design rules

1. **Android first.** The local baseline must function without CUDA, an NVIDIA GPU, or a remote service.
2. **Out-of-core images.** Process 400 MP using bounded tiles and disk-backed memory; never allocate a full-size RGB float32 ComfyUI tensor.
3. **Source fidelity.** Prefer non-generative optical correction, focus-aware frequency separation, and explicit masks. Generated details are hypotheses, not recovered evidence.
4. **Optional remote inference.** CodeFormer/GFPGAN, ArcFace, ControlNet and tiled diffusion belong to an optional GPU worker. The app must continue functioning when the worker is unavailable.
5. **Transparent tests.** Keep unit, integration, Android-device, thermal, and GPU-worker test results separate. Do not claim hardware validation without a physical run.
6. **Private by default.** No automatic upload of photos, portraits or biometric embeddings; remote processing requires an explicit user choice.

## Target outputs

- Photographs: RGB16 BigTIFF up to **24,495 × 16,330 pixels** (400,003,350 pixels), plus compact previews.
- Future: still-image pipelines reused for framewise and temporal-aware video processing; audio/video codecs are not included in the initial baseline.
- Mobile CLI in Termux as the first executable target; later an Android GUI and separate desktop/server clients.

## Architecture

```text
                  Aether Core (domain + pipeline contracts)
                                 |
       +-------------------------+-------------------------+
       |                         |                         |
 Android / S24 Ultra       Optional GPU worker       Future other platforms
 Termux CLI -> Native      ComfyUI tile-only API     Desktop / Web / iOS
       |                         |                         |
  Storage, tiled CPU       Inpaint / tiled diffusion      Shared job schema
  PSF / wavelet / WOLA     optional ArcFace veto
  RGB16 BigTIFF
```

**400 MP processing remains local for deterministic operations.** Remote neural inference is a separate opt-in capability; no local SDXL execution on the S24 Ultra is assumed.

## Current repository status

Bootstrap phase. The older Photo400 engineering code and test records exist as conversation/workspace artifacts, but are **not automatically part of this repository** until separately migrated, audited, committed, and verified. This README does not claim a working installed Android app or a completed 400 MP render.

See [Architecture](docs/ARCHITECTURE.md), [Roadmap](docs/ROADMAP.md), [Acceptance matrix](docs/ACCEPTANCE.md) and [Security](SECURITY.md).

## Licensing

Project source license and licenses for external neural checkpoints are independent. Model weights are intentionally excluded. Review rights to CodeFormer, GFPGAN, InsightFace, diffusion checkpoints and extensions before any commercial distribution.

## Imported research requirements

The full 64-page Argon-supplied 400 MP research transcription is archived in [source-material](docs/source-material/). All technical requests are tracked, without silent omission, in [ARGON-400MP requirements](docs/requirements/ARGON-400MP.md). The original PDF remains the fidelity reference for formulas and figures. New code is tested separately; archival text is not automatically runnable.
