# Uruchomienie Aether na Galaxy S24 Ultra (Termux)

To jest **lokalny, niegeneratywny** renderer RGB16 BigTIFF. Nie wymaga CUDA ani ComfyUI. Jego przetwarzanie jest ograniczone do kafelków, a duży bufor jest przechowywany na dysku.

## Instalacja (aktualizacje pakietów Termux mogą wymagać innych nazw)

```bash
pkg update
pkg install python python-numpy python-pillow
python -m pip install --no-deps tifffile
termux-setup-storage
python -m pip install -e ".[mobile]" --no-deps
```

W środowiskach z niedostępnymi pakietami binarnymi należy zweryfikować alternatywną instalację; nie wymuszać kompilacji dużych kół na telefonie.

## Pierwszy render

```bash
PYTHONPATH=src python -m mobile_s24.s24_render \
  --input "$HOME/storage/shared/DCIM/Camera/photo.jpg" \
  --output "$HOME/aether-output/test.tif" \
  --workdir "$HOME/aether-stage" \
  --width 960 --height 640 --tile 256 --detail-gain 0 --reserve-free-gb 1
```

Zacznij od rozdzielczości testowej, dopiero po sprawdzeniu miejsca na dysku i zachowania systemu uruchom `--target-mp 400`. Pełen plik 400 MP wymaga gigabajtów przestrzeni w pamięci masowej oraz długiego przetwarzania.

## Ograniczenia

- Fotografie JPEG/PNG RGB8 muszą zmieścić się w limitach dekodowania; obecny tryb pełnego 200 MP JPEG może przekroczyć budżet RAM. Nieskompresowany RGB TIFF może być mapowany na dysk.
- `CodeFormer/ControlNet/ArcFace` wymagają oddzielnego trybu z modelem; offline renderer nie udaje neuronowej rekonstrukcji.
- Moduł jest przeniesiony z Photo400. Walidacja Android/Termux i pełny render 400 MP **nie zostały wykonane**.
- S24 Ultra jest referencyjnym urządzeniem testowym; inne platformy są opcjonalne.

