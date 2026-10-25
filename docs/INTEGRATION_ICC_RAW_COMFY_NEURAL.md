# ICC16 / RAW-DNG / ComfyUI / ONNX — integration contracts

Status: source and synthetic contract tests added; hardware and real model execution
require independent evidence. A GUI is explicitly **out of scope** for this phase.

## 1. Preserve RGB16 and camera provenance

Use a new output filename on the same accessible filesystem. Aether refuses
to overwrite the destination.

### ICC-tagged RGB TIFF (bounded LittleCMS2 strips)

```bash
aether-ingest --source photo-rgb16.tif --normalized master-srgb16.tif
```

An untagged RGB TIFF needs the explicit `--assume-srgb` decision. Camera RAW
must **never** be treated as an untagged standard RGB image.

### Uncompressed Bayer DNG (strict tiled decoder)

```bash
aether-ingest --source photo.dng --normalized master-srgb16.tif \
  --raw-backend strict --raw-tile 256 --max-dng-pixels 225000000
```

Only the documented, validated unpacked 16-bit Bayer subset is supported.
Camera opcodes, compressed mosaics, active crops and unsupported DNG tags fail
closed. The strict path applies camera black/white levels, white balance,
ColorMatrix1 and sRGB conversion, but does **not** certify color accuracy
without chart measurements and per-camera calibration.

### Complex camera RAW/DNG (explicit opt-in LibRaw)

```bash
aether-ingest --source camera.nef --normalized master-srgb16.tif \
  --raw-backend libraw --max-libraw-pixels 40000000 --max-decoded-mb 900
```

The LibRaw/rawpy backend is optional and *not out-of-core*: it creates a
full-frame working array. Conservative caps avoid silently assuming that
12 GB of RAM makes full-resolution RAW safe. This path requires a compatible
installation of rawpy/LibRaw. It never auto-falls back from strict DNG.

Append `--render result-400mp.tif --workdir ./work` to perform optional offline
CPU reconstruction after a successful RGB16 normalization. A 400 MP RGB16
output alone is approximately 2.4 GB; working buffers and reserves add to it.
No 400 MP run on physical Galaxy S24 Ultra is claimed.

## 2. ComfyUI: optional external worker

```bash
aether-gpu-check --server https://your-consented-worker.example \
  --tile ./small-rgb8-tile.png --graphs ./api-stages.json \
  --workdir ./gpu-evidence
```

The supplied API graphs may contain installed node classes for Depth Anything,
ArcFace, CodeFormer, ControlNet and other extensions. The preflight uses live
`/object_info/<class>` metadata: it checks node presence, required fields,
graph connections, output ports and every declared widget choice (including
extension-specific model names). Arbitrary class names or example filenames
do not imply that a model is actually installed.

The existing transaction layer journals input/output SHA-256 and rejects
ambiguous submissions, unsafe URLs and unexpected output image geometry.
Only bounded PNG tiles may be transferred; bulk 400 MP source images are not
uploaded automatically. Server TLS should be configured with trusted
certificates. Actual worker compatibility, selected checkpoints, VRAM capacity
and model licenses must be verified at deployment time.

## 3. Local/optional ONNX model interfaces

- `aether-depth` produces a relative depth / focus-mask hypothesis from a
  bounded preview using a user-supplied compatible Depth Anything ONNX.
- `ArcFaceR100` in `aether.vision.arcface_r100` accepts a licensed local ONNX,
  plus five landmarks, to calculate a similarity *veto*, not prove identity.
- `aether-codeformer` accepts a pre-aligned RGB8 512x512 face crop and a
  compatible single-input, single-output CodeFormer ONNX wrapper:

```bash
aether-codeformer --model ./codeformer-compatible.onnx \
  --input-crop ./aligned-face-rgb512.png \
  --output-hypothesis ./face-hypothesis.png
```

The CodeFormer wrapper contract is NCHW float32 RGB512 in [-1,1] on both
sides. Exports with an extra fidelity-weight input or other tensor contracts
are rejected; they require an explicit compatible wrapper. The output is
**generated** RGB8 and must be fused into an RGB16 original only under an
explicit user-approved mask. It cannot reconstruct identity lost to strong
occlusion from absent evidence. Face detection/alignment and ArcFace reference
matching must be completed by an independent, licensed path.

For forward-pass evidence, supply actual ONNX model files:

```bash
aether-neural-check --depth ./depth.onnx --arcface ./arcface.onnx \
  --codeformer ./codeformer-compatible.onnx --report ./onnx-evidence.json
```

The SHA hashes and provider are recorded by the acceptance command. A
successful synthetic test or a successful forward pass alone does **not**
establish perceptual accuracy, compatibility with a real camera, faithful
identity preservation, or Android NPU acceleration.

## 4. Acceptance still required

1. Golden camera RAW/DNG images with independently verified ICC and color-chart output.
2. Physically measured RAM, thermals, storage, interruption/recovery and RGB16
   400 MP output on Galaxy S24 Ultra.
3. ONNX execution with real licensed weights and reviewed mask/alignment quality.
4. End-to-end ComfyUI GPU inference with exact node versions, model hashes,
   proper server authentication and output fidelity validation.


## 5. Current acceptance focus: Nikon NEF (no GUI, no merge)

The current user-provided camera input for acceptance is **Nikon NEF**. Other
formats remain in the codebase but are not part of this acceptance round.
Do not infer a camera model from a filename: read the camera model and
CFA dimensions from the file itself. Never upload a private NEF to a public
GPU worker without explicit consent.

LibRaw camera-to-sRGB output is tagged with an sRGB ICC profile. The
`rawpy.postprocess` gamma parameters must be `(2.4, 12.92)` to match the
sRGB transfer function, not the default `(2.222, 4.5)` curve. The test
`test_nef_color_transfer.py` guards this contract. This does **not** replace
a color-chart measurement or per-camera colorimetric validation.

`aether.vision.identity_guard.evaluate_identity_guard` now provides a
fail-closed policy gate for a CodeFormer candidate and the original face:
both images need valid five-point landmarks, the same supplied ArcFace
encoder produces finite 512-dimensional embeddings, and the cosine similarity
must meet an explicitly supplied calibrated threshold. Passing the veto
without human approval returns `needs_review`, never `approved`. A failed
veto returns `rejected` even with human approval. This policy does not
automatically write to the RGB16 master, prove historical identity, or
substitute for real-weight ArcFace E2E testing.

The acceptance gate remains **OPEN** until actual licensed Depth Anything,
ArcFace and CodeFormer weights, a consented authenticated ComfyUI GPU worker,
and real end-to-end evidence are available. Do not mark a mock or a CI
unit test as a real neural inference result.
