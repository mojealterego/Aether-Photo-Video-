# Samsung Galaxy S24 Ultra — automatyczna maska ostrości

**Status:** wdrożony kod, testy syntetyczne Linux (24.10.2026); brak pomiarów na rzeczywistym urządzeniu Samsung Galaxy S24 Ultra.

## Komenda Termux

```bash
pkg install python
python -m pip install -e '.[mobile]'
termux-setup-storage
bash scripts/termux_restore_auto.sh "$HOME/storage/shared/DCIM/Camera/source.jpg" "$HOME/storage/shared/Download/result_400mp.tif"
```

Albo `aether restore-auto --source input.tif --output out.tif --workdir "$HOME/aether-work" --target-mp 400 --tile 256`.

## Przebieg

1. `Source` czyta źródłowy TIFF przez `memmap` albo wejściowy JPEG/PNG przy ograniczeniu zdekodowanego wejścia. Wstępne przygotowanie RAW/DNG nie jest jeszcze strumieniowe.
2. Generowany jest globalny podgląd z ROI wejścia: max 768 px na dłuższym boku, jeden crop wejścia max. 8 MP. Nie ma alokacji pełnego RGB float32.
3. Maska ostrości bazuje na wysokoprzepustowej energii oraz lokalnym operatorze Laplace'a. Dodatkowa zewnętrzna mapa głębi `--depth-map` (8-bit L) może modulować maskę. **Nie ma lokalnej inferencji Depth Anything V2.**
4. System atomowo zapisuje `focus_mask.png` i `focus_manifest.json` zawierający SHA-256 źródła, mapy oraz parametrów; zmienione wejście blokuje przypadkowe wznowienie starej maski.
5. Ta sama maska działa przed RL-TV i podczas opcjonalnego podbicia wysokich częstotliwości w rendererze. Finalny wynik jest strumieniowo zapisywany do 16-bit BigTIFF.

**Ograniczenia:** maska jest propozycją ostrości na podstawie faktury, a nie pomiarem głębi. Faktura tła może zostać sklasyfikowana jako ostra. Nie przywraca autentycznie utraconych pikseli. Model PSF bez pomiarów kalibracyjnych pozostaje aproksymacją. Nie gwarantujemy braku OOM/termicznego throttlingu bez fizycznego testu S24.

## Weryfikacja

`tests/test_auto_focus_unittest.py`: 6 przypadków obejmujących teksturę, cache SHA-256, naruszone wejście, zewnętrzną mapę L, brak OpenCV i mały, rzeczywisty BigTIFF RGB16.

## Mapa do specyfikacji Argon

- A400-03: maska ostrości i ochrona bokeh — **częściowo**, bez inferencji relatywnej głębi ONNX na S24.
- A400-04: używany dotychczasowy RL-TV — funkcja zrealizowana programowo; brak fizycznej kalibracji PSF.
- A400-06: kafelkowe skalowanie i zapis pośredni — istnieje w rendererze.
- A400-10/11/12: separacja częstotliwościowa i RGB16 BigTIFF — dotychczasowa ścieżka działa na małych obrazach, bez pełnego testu 400 MP na S24.
- A400-05/09: ArcFace/CodeFormer/ControlNet to funkcje odrębnych opcjonalnych backendów; testy na rzeczywistych wagach nadal wymagane.
