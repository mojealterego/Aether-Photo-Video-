#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Użycie: bash scripts/termux_render.sh SOURCE.JPG [OUTPUT.TIF]' >&2; exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$1"; OUTPUT="${2:-$HOME/aether-output/final-400mp.tif}"
mkdir -p "$(dirname "$OUTPUT")" "$HOME/aether-work"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock || true
  trap 'termux-wake-unlock || true' EXIT
fi
python -m mobile_s24.s24_render --input "$INPUT" --output "$OUTPUT" \
  --workdir "$HOME/aether-work" --target-mp 400 --tile 256 --detail-gain 0
