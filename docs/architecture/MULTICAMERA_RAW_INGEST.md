# AETHERIS — native multi-camera RAW intake

This is a **capability and acceptance ledger**, not authorization to publish
camera images, metadata, proprietary source files or training examples.

## Scope

AETHERIS photo ingest must preserve source RAW files and camera metadata.
The Nikon NEF pipeline remains intact. Camera-specific import is extended by
validation rather than by assuming all RAW files share one TIFF-like layout.

## Host-side Canon decoder compatibility — October 2026

A bounded, local, read-only decoder acceptance was conducted on **10
user-supplied Canon RAW originals** with system LibRaw 0.21.4:

| Type | Files | Open | Unpack | 16-bit 3-channel RGB demosaic |
|---|---:|---:|---:|---:|
| DNG (including CHDK variants) | 2 | 2/2 | 2/2 | 2/2 |
| CR2 | 3 | 3/3 | 3/3 | 3/3 |
| Legacy CRW / CIFF | 5 | 5/5 | 5/5 | 5/5 |
| **Total** | **10** | **10/10** | **10/10** | **10/10** |

Every original retained its byte-identical SHA-256 after decoding.
The camera file payloads, private filenames and SHA hashes are intentionally
**not included in the public repository**. The private local intake manifest
remains separate from this source tree.

**Limits:** The local test invoked the system LibRaw C ABI, not the
`rawpy` Python package used by
`src/aether/ingest/libraw_adapter.py`. Therefore package portability,
mobile Android support and exact production processing results remain
**UNVERIFIED**. Decoded RGB16 has no certified camera ICC/DCP or
scene-linear ACEScg processing in this test. No photographic ΔE00 target
measurements were performed on these sources.

## Format-specific gates

- **CRW/CIFF:** use LibRaw's CIFF decoder and validate metadata extraction,
  orientation, white/black calibration and endianness. Never reinterpret
  CRW as CR2 or DNG.
- **CR2:** validate camera-specific dimensions, Bayer mapping, masking,
  white balance and image orientation for each supported body.
- **DNG/CHDK:** do not infer all DNG files share original camera metadata;
  verify CFA, black/white levels, linearization, active area, camera matrices
  and provenance before ICC16 master export.
- **NEF:** retain the already implemented Nikon path and preserve exact
  byte-for-byte source integrity.

## Production acceptance still required

1. Confirm the exact `rawpy`/LibRaw build in deploy targets supports each
   camera variant, not only the CI host.
2. Verify full-fledged camera-specific color transforms and DCP/ICC under
   calibrated reference conditions, without silently changing the color
   contract to display-sRGB.
3. Run a bounded-memory Android/mobile test and verify BigTIFF/ICC16 output,
   EXIF provenance, image rotation and source file immutability.
4. Commit tests and non-sensitive fixtures only; never add copyrighted
   source originals, location metadata, private thumbnails or corpus
   manifest into a public repository without explicit authorization.
5. Default all uploaded materials to **validation only**; no permission to
   train weights is inferred from upload.

DA3, CodeFormer and ArcFace remain ON_HOLD pending native replacements.
