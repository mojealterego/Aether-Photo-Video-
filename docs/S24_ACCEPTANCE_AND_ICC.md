# S24 Ultra: RGB16 ICC and measured hardware acceptance

**Reference target:** Samsung Galaxy S24 Ultra (Android/Termux), with no external GPU.

## 16-bit ICC ingest

An explicit normalization path has been added before the offline renderer:

```bash
# From the checked-out repository, with numpy/tifffile/Pillow installed:
python -m aether.ingest.cli \
  --source original_rgb16.tif \
  --normalized sRGB16_master.tif \
  --render final_400mp.tif \
  --workdir ./work \
  --target-mp 400 \
  --tile 256
```

Installed entrypoint: `aether-ingest`.

This uses LittleCMS2 at RGB16 precision for input TIFF carrying an ICC profile. The library must be installed and discoverable (or set `AETHER_LCMS2_PATH`). It processes TIFF strips in bounded memory, creates a contiguous 16-bit sRGB BigTIFF, and then optionally invokes the local renderer.

A TIFF without an embedded profile is rejected unless `--assume-srgb` is set explicitly. A TIFF with unsupported compression, pixel format, orientation or non-RGB photometric interpretation is rejected. This command does **not** demosaic DNG/RAW; incorrectly labeling camera RAW as standard sRGB is prohibited. Source RGB16 pixel values may shift slightly even for sRGB→sRGB color-profile processing due to LittleCMS lookup-table interpolation. This is not an 8-bit intermediate.

## Android/Termux measured acceptance

```bash
aether-device-test \
  --source /path/to/photograph.tif \
  --output /path/to/final_400mp.tif \
  --workdir /path/to/aether-working \
  --report /path/to/acceptance.json \
  --target-mp 400 --tile 256 \
  --reserve-free-gb 1
```

The tool attempts the real **offline**, non-generative, mobile BigTIFF render. On successful completion it validates the final TIFF and writes JSON evidence:

- Pixel dimensions and RGB16 BigTIFF metadata
- Wall-clock time, sampled peak process RSS and operating-system reported high-water RSS
- Available filesystem space before/after processing
- Maximum readable thermal-zone temperature, or explicit `null` if Android denies sensor access

A normal Linux synthetic run tests the measurement **mechanism**, not S24 Ultra thermal behavior. Android's low-memory killer or a hard reboot may kill the process before a report is written. Absence of a report is not success. High-water RSS, thermal zones and filesystem cache do not alone prove immunity to OOM.

## Remaining blockers

- A **real S24 Ultra run** using 12/50/200 MP source photos and final 400 MP TIFF.
- Verification of DNG/RAW demosaicing (not yet integrated), color matrices, white balance and full ICC/HDR provenance.
- Native ONNX inference with actual Depth Anything / ArcFace / CodeFormer weights and compatible Android runtime; no weights are included.
- Live ComfyUI import/inference and automated GPU-worker E2E; it is optional and cannot be asserted functional without a server.
- Measured peak memory pressure, disk reserve, heat and interrupted-resume recovery on hardware.

All such items remain listed as acceptance requirements rather than silently declared complete.
