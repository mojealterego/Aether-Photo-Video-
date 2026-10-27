# AETHERIS Ω-TC16 — CPU Research Candidate (2026-10-27)

**RESEARCH_CANDIDATE_NOT_PRODUCTION_APPROVED**

This is a published **technical training/validation record**, not a
production model release. Do not present it as a high-fidelity film encoder.

## Model and source evidence

- Native model: `omniflow-worker/omega_sdk/vae/temporal_tc16.py`
- Model configuration: 24 base channels, 16 latent channels,
  60,067 trainable parameters; anchor frame + 4-frame groups;
  spatial downsample ×8.
- Reproducible train corpus: 22 user-uploaded AI-generated video clips,
  split 16 train / 3 validation / 3 held-out test by file SHA-256.
  Grouping by underlying scene/subject remains unverified.
- Total optimizer updates **executed**: 7,600 CPU only, no cloud or paid
  accelerator. Intermediate steps selected by validation:
  2,400 initial + best 3,500 of 4,000 resolution-continuation updates
  + best 1,000 of 1,200 EOS-tail fine-tuning updates.
- Sampled material: 4 FPS, at most **128 × 128 RGB**, 9–12-frame windows.
  Not a verified 24-FPS, 720p, 1080p or 4K model.
- EOS fix is opt-in with `encode_eos_tail=True`: a padded last group
  now creates a genuine final latent for source tails of 1–3 frames.
  The decoder crops synthetic padded positions rather than freezing
  the final image. Real frames affect the latent. It remains lossy.

## Experimental checkpoint

Filename: `tc16_cpu_eos_experimental.safetensors`

SHA-256:
`3141e00e2238543000f9e131c81fbe3695c05fb1e70d7977e5c25145920522d7`

Size: 242,876 bytes.

The experimental checkpoint was saved and reloaded strictly. Original
media was unchanged during training and is not in this repository.
**The weights have been delivered privately to the uploader as a
downloadable package; they are NOT a public GitHub model binary or an
official production release.**

## Held-out measurements

On the same held-out 128×128, 9-frame samples:

| Measurement | Prior checkpoint | EOS research candidate |
|---|---:|---:|
| RGB MSE | 0.012298 | **0.009327** |
| PSNR | 19.10 dB | **20.30 dB** |
| Mean SSIM | 0.568 | **0.637** |

Mean output/input Laplacian sharpness ratio was **0.162**.

For a separate held-out 9–12-frame evaluation (not directly comparable
to the 9-frame values): RGB MSE **0.009264**, PSNR **20.33 dB**,
tail-only MSE **0.008890**.

EOS 11-frame streaming versus one-shot inference: maximum absolute
latent and decoded-frame error **0** in the tested scenario.
One 11-frame example: tail MSE **0.012604** with encoded EOS vs
**0.018579** with legacy last-frame hold. This does not prove broader
temporal-quality superiority.

Local test evidence: 14/14 focused tests of training authorization,
original immutability, duplicate rejection, optional temporal tail,
chunk equivalence, checkpoint loading and bounded CPU demo.

## Release gate results

- Held-out PSNR ≥26 dB: **FAIL**.
- Mean SSIM ≥0.90: **FAIL**.
- Laplacian detail retention ≥0.60: **FAIL**.
- Verified 24-FPS 720p+ temporal reconstruction: **NOT DONE**.
- Signed model admission, Control Plane / GPU E2E: **NOT DONE**.
- Rights/provenance review of all AI-generated clips: **NOT COMPLETE**.
- Camera RAW color calibration: **NOT DONE**.

The main repository code and this record are public. Private video
filenames, checksums, user content and binary weights remain outside the
GitHub source tree. No commercial license is granted by this report.

**Status:** training experiment executed; model release **BLOCKED**.
