# Aether architecture — ADR-0001

**Status:** accepted bootstrap, 2026-10-24. **Sole reference device:** Samsung Galaxy S24 Ultra, 12 GB RAM, Android. No desktop or GPU is assumed available.

## Modules and dependency direction

```
                 core/domain + typed image/tile contracts
                              ^
             application/orchestrator + checkpoints
           /                   |                      \
Android S24 local         storage / TIFF         optional ComfyUI worker
CLI -> native UI          out-of-core                over secured HTTPS
                                                  (no GPU on phone)
```

Domain and application code must not import ComfyUI, CUDA, Android SDK, UI frameworks or cloud providers. External adapters implement ports (future). The current `src/aether/core/geometry.py` is a pure computational primitive: it does not decode/render any image.

## Memory invariants

- 400 MP RGB float32: ~4.80 decimal GB / ~4.47 GiB per array; RGB16: ~2.40 decimal GB / ~2.24 GiB.
- Target example: **24,495 x 16,330 = 400,003,350 pixels** (3:2).
- Do not allocate a full 400 MP tensor; use 256 px core tiles initially, a bounded halo and disk-backed `numpy.memmap` for future working buffers.
- A three-channel RGB32 WOLA accumulator plus float32 weight map requires ~6.40 decimal GB of disk storage. Together with RGB16, minimum working+result output is ~8.80 decimal GB, *excluding* source, intermediates, reserve and filesystem overhead.
- Mobile process must measure actual peak RSS, Android memory pressure and thermal throttling; cannot guarantee OOM-free behavior before physical testing.

## Pipeline target (not all implemented)

Decode/ICC -> linear light -> calibrated CA / PSF RL -> global focus/depth -> occlusion masks -> face hypotheses (optional) -> staged SR -> faithful frequency/wavelet fusion -> gaussian WOLA -> RGB16 BigTIFF -> preview and provenance.

Neural work should always be **opt-in** and **tile-only**. The S24 Ultra must be able to complete a non-generative path entirely offline. Use HTTP/WebSocket clients only behind authenticated HTTPS or user-owned VPN. Do not store embeddings by default.

## Multiplatform direction

Android is first; Linux/Windows/macOS, web client and potentially iOS can later supply independent shell and storage adapters. Do not treat 'multiplatform' as a promise that desktop CUDA tooling executes on Android.
