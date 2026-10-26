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
- omniflow-worker/omega_sdk/vae/color_hdr.py — signed-log exposure,
  reference sRGB-linear↔ACEScg matrix (NOT certified OCIO) and safe alpha.
- omniflow-worker/omega_sdk/vae/losses.py — differentiable photographic
  Charbonnier, log-luma, Laplacian, FFT, Scharr, KL and uncertainty terms;
  missing CIEDE2000, optical-flow, learned perceptual and full TR-0–TR-6.
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

- tests/test_vae_contracts.py: 14 passing contract, model ABI,
  SafeTensor, Ω-R and TR-0 tests.
- tests/test_vae_ovlt.py: 6 passing transport integrity tests.
- tests/test_vae_color.py: 5 passing HDR, signed-log and alpha tests.
- tests_vae_cpu/test_vae_neural.py: 6 passing untrained PyTorch
  shape, uncertainty and causal no-future-leakage tests.
- tests_vae_cpu/test_vae_losses.py: 4 passing differentiable loss and adversarial-veto tests.
- Verified locally with Python 3.13, torch 2.10 CPU; 35 tests total.
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

## Stage 017 extension: native tiling, CIEDE2000 and signed model releases

The following source files and tests were committed directly to main.

### 400MP tile planning

- Source: omniflow-worker/omega_sdk/vae/tiling_plan.py
- Regression: tests/test_vae_tiling_plan.py (7 tests).
- Deterministic grid with core and halo coordinates; validates approximately
  400MP image 24495 x 16329, 16-channel f_s=8 latent geometry, and
  exact full-frame core partition without a dense framebuffer.
- Peak VRAM is a conservative HEURISTIC, not a profiled GPU bound.
  Neither neural inference nor reconstruction is executed by this planner.

### CIEDE2000 metrology

- Source: omniflow-worker/omega_sdk/vae/deltae.py
- Regression: tests/test_vae_deltae.py (4 tests, 10 Sharma reference pairs).
- Operates exclusively on explicit CIE L*a*b* arrays, never raw ACEScg
  channels. Proper color-managed D60 ACEScg to D65 Lab adaptation still
  requires validation with reference color targets; no camera Delta-E
  chart result or OCIO certification is claimed.

### Trusted model release admission

- Source: omniflow-worker/omega_sdk/vae/model_admission.py
- Regression: tests_vae_security/test_vae_model_admission.py (8 tests).
- Adds independent Ed25519 signer verification using a trusted runtime
  public-key allowlist; pins VAE ABI, latent and color contract digests,
  local safetensors checkpoint SHA-256 and shape/offset validation;
  refuses path traversal, symlinks, untrained status and denied commercial
  use. Ed25519 is a deliberate ADDITIVE hardening to the PDF's HMAC
  proposal, not falsely described as identical to its wire protocol.
- Publisher-supplied TRAINED_SIGNED_VALIDATED status and signature are
  evidence of asserted provenance, NOT model training quality certification.
  Trust-store governance, immutable filesystem enforcement and multi-host
  release attestation are independent future gates.

### Verification and CI

Local: 42 non-security regular tests + 4 neural/loss tests + 8 new
model admission tests = **54 focused Ω-VAE tests**. Separate GitHub
Actions vae-security job added for cryptography dependency. CI success
has not been established by committing the workflow.

Remain open: real neural 400MP rendering, true GPU benchmark,
complete TC16 temporal anchor model and other trained profiles,
camera-chart color accuracy, trusted model promotion and OmniFlow E2E.
DA3, CodeFormer and ArcFace remain ON_HOLD.
