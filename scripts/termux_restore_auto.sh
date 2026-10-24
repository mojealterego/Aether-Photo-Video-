#!/data/data/com.termux/files/usr/bin/bash
# S24 Ultra: offline output in RGB16 BigTIFF; no GPU required.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/termux_restore_auto.sh INPUT.jpg OUTPUT.tif [WORKDIR]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$1"
OUTPUT="$2"
WORK="${3:-$HOME/aether-s24-auto-work}"
mkdir -p "$(dirname "$OUTPUT")" "$WORK"
if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock || true
  trap 'termux-wake-unlock || true' EXIT
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m aether.mobile.cli restore-auto \
  --source "$SOURCE" --output "$OUTPUT" --workdir "$WORK" \
  --target-mp 400 --tile 256 --detail-gain 0
