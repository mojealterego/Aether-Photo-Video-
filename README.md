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

## Architecture — AETHERIS OmniFlow Ω

Aether is a **single ACE 2.0 system**, developed only in `main`:
the native OmniGraph IR compiler and Rust control plane, isolated OmegaNode
workers, camera RAW/NEF processing, RGB16/ICC workflows, native Depth Ω,
ArcFace/Identity Guard Ω, CodeFormer Ω, Chronos Ledger and the
AetherCanvas / AWR / SLE / OmniNodes / film-studio subsystems.

```text
      Studio UI / S24 Ultra / OCAF OMAs and OMAt
                        |
          OmniFlow Omega Control Plane (Rust)
         typed OmniGraph IR / WFQ / VRAM / policy
                        |
         OmegaNode SDK ABI v1.1 (native workers)
           |              |               |
        NEF/RGB16       Depth/Face       Video/Audio
           |              |               |
         Immutable CAS -> Chronos Ledger -> Export QC
```

**Native OmniFlow is not a ComfyUI wrapper.** Existing older ComfyUI-related
code remains a transitional, not-yet-migrated component; it cannot be
counted as proof of the target architecture's independence or GPU E2E.
No external service receives private NEF files or biometric embeddings
without explicit consent.

Implementation status and full integration registry:
[OmniFlow Ω integration ledger](docs/architecture/OMNIFLOW_OMEGA_INTEGRATION.md).
The architectural source of truth is
`SYS-AETHERIS-OMNIFLOW-016 v1.1.0` (80-page supplied PDF).

## Current repository status

Bootstrap phase. The older Photo400 engineering code and test records exist as conversation/workspace artifacts, but are **not automatically part of this repository** until separately migrated, audited, committed, and verified. This README does not claim a working installed Android app or a completed 400 MP render.

See [Architecture](docs/ARCHITECTURE.md), [Roadmap](docs/ROADMAP.md), [Acceptance matrix](docs/ACCEPTANCE.md) and [Security](SECURITY.md).

## Licensing

Project source license and licenses for external neural checkpoints are independent. Model weights are intentionally excluded. Review rights to CodeFormer, GFPGAN, InsightFace, diffusion checkpoints and extensions before any commercial distribution.

## Imported research requirements

The full 64-page Argon-supplied 400 MP research transcription is archived in [source-material](docs/source-material/). All technical requests are tracked, without silent omission, in [ARGON-400MP requirements](docs/requirements/ARGON-400MP.md). The original PDF remains the fidelity reference for formulas and figures. New code is tested separately; archival text is not automatically runnable.


## Samsung S24 Ultra — obecny offline renderer

Po instalacji `pip install -e ".[mobile,optical]"` można wykonywać lokalną rekonstrukcję RGB16 BigTIFF bez CUDA i bez uruchomionego ComfyUI. Parametry i liczbę kafelków sprawdza `aether plan`.

### Fuzja rzeczywistej hierarchii Haar DWT z kandydatem SR

```bash
aether restore \
  --source ./photo.png --output ./result-400mp.tif \
  --workdir ./work --target-mp 400 --tile 256 \
  --focus-mask ./focus-L.png \
  --sr-candidate ./sr-candidate-rgb16.tif \
  --wavelet-levels 3 --wavelet-gain 1.0 \
  --compression deflate
```

`sr-candidate` musi być przestrzennie zarejestrowany względem źródła albo rozmiaru docelowego. Dla rzeczywistego obrazu 400 MP ma to być **nieskompresowany, mapowalny RGB TIFF**; renderer pobiera tylko wycinki. Opcjonalny `--compression deflate` dotyczy **końcowego BigTIFF**; pośredni `--contiguous-output` pozostaje nieskompresowany i mapowalny. Wygenerowane szczegóły nie stanowią dowodu autentycznego odzyskania danych.

### Opcjonalna mapa ostrości z Depth Anything ONNX

Dla wcześniej przygotowanego podglądu RGB8 o rozdzielczości co najwyżej 1 MP i kompatybilnego pliku ONNX:

```bash
aether-depth \
  --preview ./preview-small-rgb.png \
  --model ./depth-anything-compatible.onnx \
  --focus-depth 0.55 --tolerance 0.15 \
  --output ./focus-L.png
```

Wymaga oddzielnej, zgodnej z Androidem instalacji `onnxruntime` i prawdziwych wag ONNX. Automatyczny wybór prawidłowej płaszczyzny ostrości nie został zweryfikowany. Bez modelu użytkownik może dostarczyć lub wyrysować maskę ręcznie.

### Zakres odbioru

Załączone regresje numeryczne obejmują: odwracalną transformację Haar również dla nieparzystych wymiarów, zachowanie bazy poza maską, test zgodności kafelków 16 vs 32 px bez różnic pikselowych, małe end-to-end BigTIFF RGB16 i bezstratny eksport DEFLATE. Nie przeprowadzono pełnego renderu 400 MP na fizycznym Samsungu Galaxy S24 Ultra ani niezweryfikowano prawdziwej inferencji ONNX/GPU. Szczegóły: [ARGON-400MP](docs/requirements/ARGON-400MP.md).

