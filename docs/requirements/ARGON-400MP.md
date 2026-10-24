# ARGON-400MP — obowiązujący rejestr wymagań źródłowych

**Źródło:** użytkownik, „Potok Rekonstrukcji Obrazu 400 MP.PDF”, 64 strony.
**Deklarowany autor dokumentacji:** Argon (wg informacji użytkownika).
**SHA-256 oryginalnego PDF:** `d5f955b871a66196229d75c911fe4dd02def4211e7e77d908ec955c820b48afc`.
**Pełna ekstrakcja:** cztery pliki w [source-material](../source-material/), obejmujące strony 1–64.
**Reguła:** wszystkie funkcje zapisujemy jako wymagania Aether; nie wolno pomijać trudnych funkcji. Implementacja, weryfikacja i ograniczenia są prowadzone oddzielnie. Kod i JSON wyekstrahowane z PDF NIE są traktowane jako poprawne składniowo artefakty do wykonania bez przeglądu.

## Kanoniczny profil docelowy

- **Urządzenie referencyjne:** Galaxy S24 Ultra, 12 GB RAM, Android i UFS 4.0; tylko ten sprzęt jest aktualnie dostępny.
- **Docelowy obraz:** 400 MP z zachowaniem proporcji oryginalnego wejścia (1:1, 4:3, 3:2 lub inne; odpowiednio 20000 × 20000 albo 24495 × 16330).
- **Priorytet:** lokalny offline renderer, bez CUDA, ComfyUI ani zewnętrznego GPU jako twardych zależności.
- **Rozwój:** multiplatforma z oddzielnymi adapterami klienta i opcjonalnym workerem neuronowym.
- **Niepotwierdzone:** deklaracje „OOM-free”, „brak halucynacji”, „matematyczna gwarancja tożsamości” i sztywne limity RAM wymagają rzeczywistych pomiarów; nie mogą zastępować kryteriów odbioru.

## 12 etapów ze stron 7–9 — rejestr obowiązkowy

| ID | Etap dokumentacji / zakres | Wymagane kryterium odbioru | Status repozytorium |
|---|---|---|---|
| A400-01 | Ingestia RAW/TIFF i dokładna sRGB EOTF → linear RGB32 | Strumieniowe wejście; wzorce kolorymetryczne/ICC; pamięć szczytowa mierzona | Częściowo: RGB8/16 TIFF/PNG i sRGB EOTF; strumieniowe RAW/DNG i ICC do wdrożenia |
| A400-02 | LCA R/B → G: radialny wielomian do 8. rzędu, decentracja | Testy subpikselowej kalibracji, niezmiennego kanału G, krawędzi | Częściowo: radial/tangential LCA i kanał G; kalibracja urządzenia niezweryfikowana |
| A400-03 | Depth + energia Laplace'a + guided-filter M_DoF | Maski globalne, rejestracja współrzędnych, ochrona nieostrości | Częściowo: guided filter + Depth ONNX adapter, maska CLI; rzeczywiste wagi niezweryfikowane |
| A400-04 | Richardson–Lucy ML-EM/TV, anizotropowy PSF i normalizacja brzegowa Aᵀ1 | Tłumienie halo/ringingu, wierność dla syntetycznej PSF | Częściowo: RL-TV z Aᵀ1 i anizotropowym PSF; pełna walidacja optyczna otwarta |
| A400-05 | Detekcja twarzy, 5 landmarks, ArcFace R100 512D | Prawdziwy model z pliku ONNX, fail-closed bez referencji | Częściowo: adapter ArcFace R100 ONNX; bez prawdziwego checkpointu/detektora |
| A400-06 | Dyskowe bufory docelowe, interpolacja Lanczos4 do 400 MP | Dokładna geometria bez przesunięć na granicach kafelków | Częściowo: streaming Lanczos, memmap i test geometrii; pełne 400 MP niewykonane |
| A400-07 | Separacja częstotliwościowa f_L/f_H, pasma na dysku | Numeryczny test rekonstrukcji i ciągłości, budżet buforów | Częściowo: multilevel Haar DWT + Gaussian fusion, testy odd-size i szwów |
| A400-08 | Kaskadowy SR, Gaussian WOLA, normalizacja ΣW | WOLA lokalnie działa (12/12 testów całego bootstrapu); modele SR nadal TODO | Częściowo: Gaussian WOLA + mobile Haar candidate; rzeczywiste modele SR niewykonane |
| A400-09 | CodeFormer, ControlNet Inpaint, Identity Guard | Rzeczywista inferencja albo wyraźny tryb offline bez wniosków o tożsamości | Częściowo: protokoły zewnętrznego workera i maski; realna inferencja niezweryfikowana |
| A400-10 | Faithful Fusion (low original + masked high SR + bokeh) | Kolory/światłocienie i geometria poza maską bez naruszenia | Częściowo: maskowana baza i Haar fusion przed RGB16 BigTIFF; walidacja percepcyjna otwarta |
| A400-11 | Tone mapping / ICC / transfer / kwantyzacja RGB16 | Bez niejawnego przejścia przez RGB8 i bez obcięcia tonów | Częściowo: sRGB linear→RGB16; pełny profil ICC/HDR i 16-bit źródła niezamknięte |
| A400-12 | Strumieniowy BigTIFF RGB16, kafelki i DEFLATE | Odczyt kontrolny TIFF, rozmiar/piksele/checkpoint/resume na S24 | Częściowo: RGB16 BigTIFF tiled/DEFLATE + checkpoint; mały roundtrip przetestowany, brak fizycznego 400 MP |

**Uwaga do etapu 09:** wymaganie musi być wykonane semantycznie przed zablokowaniem bazy niskoczęstotliwościowej; etapowanie fizyczne dopasowujemy tak, by nie odtworzyć mozaiki / cenzury z oryginalnego f_L.

## Wymagania przekrojowe z pozostałych stron

| ID | Zakres | Strony | Status |
|---|---|---|---|
| A400-X01 | Clean Architecture: Domain → Infrastructure → Application → CLI | 11–44 | TODO / migracja Photo400 |
| A400-X02 | Samodzielny batch CLI, checkpointy, monitor pamięci, swap | 11–44 | TODO |
| A400-X03 | GPU/SR silnik podmienialny: ONNX, PyTorch, CPU, opcjonalny worker | 23–35 | TODO |
| A400-X04 | Gaussian WOLA bez wielokrotnej normalizacji i zero-weight holes | 10, 24–25, 37–39 | **Wdrożony moduł** |
| A400-X05 | Face Prior i weryfikacja embeddingiem bez podstawiania DCT za ArcFace | 10–11, 27–31, 38–40 | TODO |
| A400-X06 | Kompletny workflow GUI ComfyUI schema 0.4 oraz osobny API DAG | 44–62 | TODO, backend opcjonalny |
| A400-X07 | Depth Anything, FaceDetailer, CodeFormer, ControlNet Tile/Inpaint, UltimateSDUpscale | 45–49 | TODO; sprawdzić zgodność SD1.5/SDXL wag |
| A400-X08 | Ograniczenia RAM, UFS, cache, pomiary peak RSS / LMK | 1–3, 7–9, 63 | TODO; profilowanie fizycznego S24 |
| A400-X09 | Korekty naukowe: source-vs-synthesis, istotna niepewność i brak gwarancji tożsamości | 10–11, 63 | Wymóg jakościowy |
| A400-X10 | Dalsze moduły wideo i multiplatformowe w architekturze Aether | nadrzędny projekt | TODO |

**Żaden wpis TODO nie jest odrzucony.** Wymogi sprzeczne z ograniczeniami lokalnego sprzętu realizujemy przez rozdzielenie backends/fallbacków i publikujemy zmierzony poziom jakości. Parametry przypisane przez Argon to wartości wejściowe do strojenia, nie gwarancje pracy na telefonie.

## Integracja z repo

Pierwszy zaakceptowany fragment wdrożeniowy: `src/aether/core/wola.py` oraz `tests/test_wola.py`. Następne moduły wymagają portów I/O oraz testów obrazów rzeczywistych. Cały oryginalny materiał jest archiwizowany tekstowo na wypadek rozbudowy wymagań. Dokładny plik PDF pozostaje przesłanym oryginałem i powinien zostać dołączony jako osobny binarny artefakt, jeśli ma być przechowywany bez strat formatowania.

## Aktualizacja wdrożeniowa — 24 października 2026

Statusy powyżej oznaczają stan **programowania i testów lokalnych**, nie produkcyjne ukończenie wszystkich wymagań. W tym przyroście dodano: `src/aether/core/wavelet.py` (Haar DWT, wiele skal), `src/aether/vision/depth_onnx.py` (bounded ONNX depth), `src/aether/vision/depth_cli.py` (`aether-depth`), obsługę `--sr-candidate` w `aether restore`, globalne wyrównanie fazy kafelków i stratny-informacyjnie bezpieczny eksport RGB16 BigTIFF z opcjonalną **bezstratną** kompresją DEFLATE. Odpowiednie testy wykonano na małych danych syntetycznych.

Pozostałe **blokery odbioru produkcyjnego**: fizyczne uruchomienie na S24 Ultra, pełnowymiarowy render 400 MP i monitoring Android LMK/termiki; strumieniowe RAW/DNG oraz pełna polityka ICC; prawdziwe wagi Depth Anything, ArcFace, CodeFormer i ControlNet wraz z licencjami; środowisko GPU/ComfyUI; ocena rzetelności generowanych szczegółów i szwów na rzeczywistych fotografiach.

Weryfikacja nowego przyrostu: lokalnie 18 testów zakresu wavelet/depth/mobile/BigTIFF, w tym identyczne piksele przy kafelkach 16 i 32 px. Wynik wcześniej istniejącego zestawu 33 testów nie został niezależnie powtórzony w tej samej sesji; nie sumować liczb bez ponownego uruchomienia całego repozytorium. GitHub Actions również wymagają osobnego potwierdzenia.
