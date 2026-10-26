# AETHERIS Ω-VAE — integration record (SYS-AETHERIS-VISION-VAE-017)

**Status: PARTIAL IMPLEMENTATION, UNTRAINED, NOT APPROVED FOR PRODUCTION**

Source: user-supplied 87-page "Architektura Systemowa i Kompletna Implementacja
Wdrożeniowa Modelu AETHERIS Ω-VAE", document version 1.0.0.

Repository policy: a single Aetheris OmniFlow Ω system, main only;
no additional app, branch, PR, or ComfyUI runtime dependency.
This module does not replace the existing raw/ICC16 master
and does not invoke paused DA3, CodeFormer or ArcFace.

## Five canonical model profiles

| Profile | Package ID | Source-document intent | Current state |
|---|---|---|---|
| Ω-P / P16 | AetherVAE-P16 | HDR scene-linear photo latent | Schema + untrained 2D topology; no real checkpoint or colorimetry |
| Ω-G / G16 | AetherVAE-G16 | Generative latent flow/diffusion | Schema + untrained 2D topology; generative training absent |
| Ω-T / TC16 | AetherVAE-TC16 | Causal temporal video with anchor and state | CausalConv3D primitive; full anchor/stride-4 video VAE absent |
| Ω-E / E8 | AetherVAE-E8 | Distilled/quantized mobile preview only | Schema; PTQ/QAT and mobile inference absent |
| Ω-R / R | AetherVAE-R | Deterministic bit-exact archival by residual | Tested bounded uint16 sidecar for an individual ROI; full 400 MP streaming absent |

A model must never be declared trained, calibrated, signed or operational
just because an executable architecture class exists.

## Implemented native modules

- omniflow-worker/omega_sdk/vae/contracts.py — five profiles, explicit
  LatentSchema/ColorContract, per-channel normalization, deterministic
  SHA-256 fingerprints and SafeTensor bounded/strict tensor header parser.
- omniflow-worker/omega_sdk/vae/reversible.py — Ω-R deterministic
  uint16 source + retained uint16 base + int32 residual, zlib, SHA-256,
  size-bounded recovery and prevention of false lossless claims if the
  sidecar is missing. Limit: 16,777,216 elements per ROI.
- omniflow-worker/omega_sdk/vae/governance.py — TR-0 training sample
  license, consent, color-profile and content/subject/session split gate.
  Source license information is metadata, not independent legal proof.
- omniflow-worker/omega_sdk/vae/neural.py — UNTRAINED Torch 2D
  f_s=8 encoder-decoder with Haar micro path, antialiasing, window-local
  attention, global color descriptor and variance output, Gaussian
  posterior and CausalConv3D streaming primitive. NO trained weights.
- omniflow-worker/omega_sdk/vae/ovlt.py — bounded OVLT v1
  4D/5D float32 transport with exact schema/model hashes, HMAC-SHA256
  supplied by trusted caller, per-chunk CRC32 and SHA-256, strict
  decompression limits and exclusive atomic publication; 256 MiB
  compressed payload bound.

## Test evidence (local sandbox, not GPU)

- tests/test_vae_contracts.py: 13 passing contract, model ABI,
  SafeTensor, Ω-R and TR-0 tests.
- tests/test_vae_ovlt.py: 6 passing transport integrity tests.
- tests_vae_cpu/test_vae_neural.py: 6 passing untrained PyTorch
  shape, uncertainty and causal no-future-leakage tests.
- Verified locally with Python 3.13, torch 2.10 CPU; 25 tests total.
- Separate GitHub Actions CPU VAE job added. Its results remain to
  be inspected; committing a test is not evidence that CI passed.

## Not yet implemented — the six-module acceptance map

1. Contracts/security: independent release signatures, signed bundle
   verification, efficient streaming SafeTensor validation with real weights.
2. Color/HDR: reference OCIO ACEScg, signed-log HDR, standards-based
   ΔE00 and true camera-profile calibration. Display-sRGB is not ACEScg.
3. Neural: full production P16/G16/TC16/E8, temporal f_t=4 anchor codec,
   trained weights, deterministic inference and published signatures.
4. Loss/training: full CompositeLossOmega and TR-0 through TR-6,
   real authorized dataset, multi-GPU FSDP2.
5. Tiling: bounded 400 MP neural inference and uncertainty maps;
   the existing disk-backed compositor is NOT equivalent to tiled VAE inference.
6. Runtime: authenticated model admission, model/tenant-scoped
   EncodeImage/DecodeImage/EncodeChunk/DecodeChunk, Chronos audit,
   deployed KMS/HSM, real GPU E2E and production release policy.

## Mandatory acceptance gates

- UNTRAINED or absent verified weights means no production inference.
- Mismatched schema/color/model digest or HMAC means reject before decode.
- Missing Ω-R sidecar means lossless=False; never auto-assert archival fidelity.
- Original Nikon NEF and ICC16 master remain immutable.
- DA3, CodeFormer and ArcFace remain ON_HOLD pending Argon's replacements.

This record distinguishes actual committed code from outstanding tests and
does not claim a fully trained or hardware-certified VAE.
