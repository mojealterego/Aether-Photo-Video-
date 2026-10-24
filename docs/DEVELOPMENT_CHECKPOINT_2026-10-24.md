# Punkt kontrolny rozwoju Aether — 2026-10-24

Repozytorium: mojealterego/Aether-Photo-Video- (z końcowym myślnikiem).
Platforma referencyjna i jedyna zakładana fizycznie: **Samsung Galaxy S24 Ultra, 12 GB RAM**.

## Co wykonuje się bez CUDA i serwera

- CLI Aether plan, restore; docelowo RGB16 BigTIFF.
- Mobilny renderer: linearyzacja, resampling Lanczos w świetle liniowym, maskowane hipotezy inpaintingu, optyka RL-TV i radialna LCA, opcjonalne focus/protected mask.
- Dyskowe bufory / checkpointy / kontrola wolnej przestrzeni i suma SHA-256 dla źródła.
- Silnik Gaussian WOLA i separacja częstotliwościowa ze zintegrowanymi testami numerycznymi.
- Kalibracja empirycznego PSF i rozszerzonej CA jako niezależne narzędzia.

## Kod przygotowany pod backend opcjonalny

- ArcFace R100 ONNX (bez wag i autentycznego testu), preflight ComfyUI i oddzielne mosty wymiany kafelków.
- Statyczny graf 51 węzłów / 73 połączenia / 24 grupy ComfyUI 0.4 jako referencja dla **zewnętrznego** GPU.
- Realny CodeFormer/ControlNet, pełny ONNX Depth Anything i weryfikacja tożsamości na urządzeniu lub GPU — NIEPRZETESTOWANE.

## Dowody

- 33/33 testy lokalne, w tym CLI→GPU-independent BigTIFF, przebieg optyczny RL/LCA, WOLA, focus/frequency, ArcFace z atrapą ONNX, PSF kalibracja i poprawność schematu grafu.
- Linuxowy test małego TIFF; brak fizycznego renderu 400 MP na S24 Ultra.
- Przy 24495 x 16330 pikseli docelowy RGB16 to 2.40 GB, bufor WOLA RGB float32 i wagi około 6.40 GB na dysku, łącznie 8.80 GB bez pozostałych etapów.

## Otwarte

Pełne źródłowe RAW/ICC, Depth Anything na rzeczywistym backendzie, migracja kompletnego backendu ComfyUI, niezawodne 400 MP po przerwaniu zasilania, fizyczne pomiary RSS/termiki/UFS i ocena rzeczywistej jakości detalu. **Nie oznaczamy 12 etapów jako ukończonych produkcyjnie** bez takich dowodów.

Kolejne dostarczone dokumenty Argon dodajemy do rejestru wymagań bez usuwania niedokończonych elementów. Brak paczki ZIP.
