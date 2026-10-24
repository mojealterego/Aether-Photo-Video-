# Verification and acceptance matrix

| Layer | Minimum acceptance evidence | Current status |
| --- | --- | --- |
| Pure core | Unit tests for geometry, tile boundaries, memory estimates | 7 tests passed locally on Linux before commit |
| Repository CI | GitHub Actions checkout/install/test on Python 3.11/3.12 | Configured, not yet checked remotely |
| Photo400 migration | Baseline source imported, clean install, full regression | Pending |
| Android | Real Galaxy S24 Ultra runtime, RSS/thermal measurements | Pending |
| 400 MP output | Real camera photo, exact dimensions, RGB16 BigTIFF integrity | Pending |
| ComfyUI neural | Installed checkpoint hashes, node discovery, safe tile transactions | Pending |
| Identity/occlusion | Manual review, fallback, opt-in, accurate 'generated' label | Pending |
| Video | Deterministic frame processing and temporal consistency | Pending |

The earlier Photo400 '132/132 local tests' referred to a different working snapshot; the present bootstrap does **not** claim that those tests have been committed or reproduced here.
