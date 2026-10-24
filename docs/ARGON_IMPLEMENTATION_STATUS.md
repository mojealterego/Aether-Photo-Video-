# Aether — stan wdrożenia Argon 400 MP

Stan: 2026-10-24. Referencyjna platforma: Samsung Galaxy S24 Ultra, Android, 12 GB RAM.

Wszystkie 12 etapów i 10 wymagań przekrojowych pozostają obowiązującą specyfikacją: [ARGON-400MP](requirements/ARGON-400MP.md). Rejestr nie pozwala skreślić wymagań tylko dlatego, że potrzebują dalszej pracy lub sprzętu.

## Kod przeniesiony i przetestowany lokalnie

| Moduł | Konkretny zakres | Granice walidacji |
|---|---|---|
| S24 offline renderer | kafelkowa interpolacja linear-light, ochrona źródła, przygotowanie i zapis 16-bit BigTIFF | test małego TIFF, nie 400 MP na urządzeniu |
| core/optical.py | EOTF sRGB, LCA radial/tangential do stopnia 8, anizotropowy PSF, RL-TV z czynnikiem A-transpose-ones | testy syntetyczne; nie zintegrowano wszystkich ustawień ze starym rendererem |
| core/psf_calibration.py | empiryczne PSF i radialna CA | bez danych kalibracyjnych aparatu |
| core/focus_frequency.py | guided relative-depth focus mask, bokeh protection, adaptive master base, Gaussian frequency fusion | nie jest równoważne pełnemu DWT Wavelet; brak sieci Depth Anything na telefonie |
| core/wola.py | dyskowy Gaussian WOLA z jednorazową normalizacją | testy małych kafelków |
| vision/arcface_r100.py | 5-landmark alignment i 512D ONNX R100 adapter | testy z symulowaną sesją; brak prawdziwych wag i detektora |
| remote/preflight.py | kontrola schematów żywego ComfyUI | nie ma podłączonego i uruchomionego serwera |
| workflows/photo400_reference_400mp.gui.json | schemat 0.4: 51 węzłów, 73 linki, 24 grupy | tylko jeden aktywny tor kafelkowy; nie full GPU E2E |
| aether restore | bezpośrednio z CLI uruchamia lokalny S24 BigTIFF | wymaga zależności mobilnych i wolnej przestrzeni |

## Otwarte wymagania produkcyjne

1. Połączenie wszystkich zaawansowanych modułów optycznych i frequency z jedną ścieżką renderowania bez powielania implementacji.
2. Strumieniowe RAW/DNG oraz obsługa ICC poza sRGB z kontrolą RAM dla fotografii wejściowych 200 MP.
3. Depth Anything / ONNX NPU na telefonie (jeśli możliwe); rzeczywisty model i jakość masek.
4. Prawdziwa inferencja CodeFormer, ControlNet Inpaint i ArcFace na zweryfikowanym backendzie. Nie wolno uznawać generowanych cech za historyczną prawdę.
5. Integracja automatycznego zewnętrznego ComfyUI E2E z checkpointami i pełnym wznowieniem, jeśli kiedykolwiek dostępny będzie serwer.
6. Próby na **fizycznym S24 Ultra**: wykonanie 400 MP, szczytowy RSS, presja LMK, temperatura, przerwania, miejsce w UFS i weryfikacja BigTIFF.

**Nie jest potwierdzone**, że GPU/NPU S24 wykonuje SDXL/ComfyUI; podstawowy renderer pozostaje lokalny i bez modeli.

## Integracja optyki z rendererem (2026-10-24)

Renderer `mobile_s24.s24_render` używa bezpośrednio funkcji `aether.core.optical.correct_lca` oraz `aether.core.optical.richardson_lucy_tv` podczas przetwarzania kafelków, gdy parametry optyczne są włączone. Przed RL stosuje odbity margines odpowiadający promieniowi PSF i liczbie iteracji, co ogranicza artefakty od sztucznego zerowego brzegu. Test integracyjny sprawdza wykonanie obu funkcji i wynikowy BigTIFF.

Potwierdzenie lokalne: **33/33 testy** (Python, Linux, syntetyczne małe obrazy), `compileall` oraz `aether plan --width 6000 --height 4000 --target-mp 400`. Nie jest to jeszcze test fizycznego S24 Ultra, pełnego obrazu 400 MP ani rzeczywistych wag ONNX/GPU. Model PSF w CLI jest nadal gaussowski; API skalibrowanych/wyższych wielomianów pozostaje do połączenia z konfiguracją renderera.
