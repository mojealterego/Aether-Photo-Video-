# Ω-VAE-017 — execution evidence, 2026-10-26

**Scope:** continuation of native AETHERIS OmniFlow Ω on main.
Source specification: user-provided 87-page
"Wdrożenie Systemu AETHERIS Ω-VAE.pdf" (SYS-AETHERIS-VISION-VAE-017).

## Native code committed

- omega_sdk/vae/temporal_tc16.py: trainable 16-channel f_s=8,
  anchor + f_t=4 causal group codec; encode_chunk/decode_chunk have
  explicit model-scoped states. Groups are emitted after four
  source frames arrive, with up to 3-frame group latency.
  Tz = 1 + floor((T-1)/4) is honored without padding each chunk.
  Final incomplete source frames are lossy last-frame predictions.
- omega_sdk/vae/streaming_inference.py: actual PyTorch tiled FP32
  scene-linear HWC memmap inference, disjoint core, halo, finite
  result validation, atomic output and SHA-256. Untrained test mode
  is TEST_ONLY_UNAPPROVED; production is denied pending attestation.
- omega_sdk/vae/color_measure.py: reproducible RGB16 ICC TIFF versus
  JPEG CIEDE2000 comparison; ICC/sRGB input policy.
- omega_sdk/vae/training_tc16.py: bounded synthetic-only training
  smoke with governance and SafeTensors checkpoint; NOT TR-1..TR-6.

## Real Nikon source evidence

| Asset | SHA-256 |
|---|---|
| _DSC7184.NEF | 9da1a6f8ad5909cc6711dd474a05d2d9fe4e0dd32156d1de46cc9762dea53428 |
| _DSC7184_cameraWB_sRGB16.tif | 61c16b3d9d20e984d0c822e4d1c1e4fdf3c634de493cbf5ffbd89037698cc0d0 |

TIFF: 4032 × 6032 × RGB16 with embedded sRGB built-in ICC.
The NEF was not re-demosaiced in this execution and remains unchanged.

### Relative CIEDE2000 color QC

CIE Lab D65 via OpenCV; TIFF resized with INTER_AREA to
1500 × 1003 preview pixels. Untagged JPEG assumed sRGB D65.

| Candidate | Mean ΔE00 | Median | P95 | P99 |
|---|---:|---:|---:|---:|
| CameraWB JPEG | 0.914700 | 0.726326 | 2.298224 | 3.774178 |
| Final JPEG | 3.499565 | 3.375898 | 5.916171 | 6.491485 |

This measures two renders, NOT absolute camera-chart color accuracy,
certified OCIO/ACEScg, or a calibrated camera profile.

### Executed neural tiling

1. Real Nikon-derived TIFF: 24.3 MP, TIFF converted in row blocks
   from sRGB16 to FP32 scene-linear ACEScg reference matrix;
   untrained identity Conv2d(3,3,kernel=1) on CPU, 96 tiles,
   3.256 seconds, max FP32 absolute error 0,
   peak RSS 1,028,276 KiB, output SHA-256:
   54ad8d225a6017a28c206442f04022d37328341e0c506980b17eb851826ca78a
2. Synthetic 100 MP: 10,000 × 10,000 × RGB FP32,
   identity Conv2d CPU, 400 tiles, 397 exact-zero output
   tiles safely elided, 13.369 seconds, peak RSS 1,482,308 KiB,
   output SHA-256:
   23345b94b096834f4ec6cc5b2226749acd325aa41c1d1c3df79efb13d56041f5
3. Full 400 MP: planner tests exist, but 20,000 × 20,000 FP32
   stress attempts TIMED OUT. NO full 400 MP completion claim.

Both executed neural models are untrained identity convolutions;
they validate tiling/transport, not trained VAE quality.

## New CPU acceptance tests

- tests_vae_cpu/test_vae_temporal_tc16.py: 8.
- tests_vae_cpu/test_vae_streaming_inference.py: 6.
- tests_vae_cpu/test_vae_color_measure.py: 3.
- tests_vae_cpu/test_vae_training_tc16.py: 3.

Total: 20 new local CPU tests; remote GitHub Actions
success has not been verified.

## Remaining production blockers

- No governed, sufficiently large temporal training corpus with
  train/val/test splits. Only one real Nikon still image is locally
  available; synthetic smoke weights are NOT approved.
- No approved P16/G16/TC16/E8 checkpoints, training metrics,
  release signatures or independent quality validation.
- No CUDA GPU is available in the execution environment.
  GPU memory benchmarks, real 400 MP trained-VAE render and GPU E2E
  have NOT run.
- Rust OmniFlow Control Plane is currently a library, not a
  deployed authenticated HTTP worker-control service. No live
  Control Plane-to-GPU worker E2E has run.
- Absolute camera ΔE, chart reference, target-host tenant isolation
  and production model promotion remain open.

DA3, CodeFormer and ArcFace remain ON_HOLD pending Argon's native versions.
