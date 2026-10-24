#!/data/data/com.termux/files/usr/bin/bash
# Physical Galaxy S24 Ultra only. No emulated GPU or synthetic hardware results.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: bash scripts/s24_acceptance_400mp.sh SOURCE.jpg OUT.tif WORKDIR REPORT.json" >&2
  exit 2
fi
SRC="$1"; OUT="$2"; WORK="$3"; REPORT="$4"
if ! command -v getprop >/dev/null; then
  echo "Android getprop unavailable; not running on physical Android" >&2
  exit 3
fi
MODEL="$(getprop ro.product.model)"
ANDROID="$(getprop ro.build.version.release)"
if [[ "$MODEL" != *S24* && "$MODEL" != *SM-S928* ]]; then
  echo "Unknown S24 Ultra device model: $MODEL" >&2
  exit 4
fi
if command -v termux-wake-lock >/dev/null; then
  termux-wake-lock || true
  trap 'termux-wake-unlock || true' EXIT
fi
mkdir -p "$WORK" "$(dirname "$OUT")" "$(dirname "$REPORT")"
if [[ -e "$OUT" || -e "$REPORT" ]]; then
  echo "Existing report or output: refusing to overwrite" >&2
  exit 5
fi
echo "Aether S24 Ultra acceptance; model=$MODEL Android=$ANDROID"
if command -v termux-battery-status >/dev/null; then
  termux-battery-status || true
fi
df -h "$WORK" || true
aether-device-test --source "$SRC" --output "$OUT" --workdir "$WORK" \
  --report "$REPORT" --target-mp 400 --tile 256 --reserve-free-gb 2
python - "$REPORT" "$MODEL" "$ANDROID" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]);report=json.loads(path.read_text())
h,w,c=report['output_shape_hwc']
assert h*w>=399_000_000 and c==3,'Output is not approximately 400 MP RGB16'
report['device_model_verified_by_getprop']=sys.argv[2]
report['android_release_reported']=sys.argv[3]
report['full_400mp_verified']=True
path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
print('Full S24 render verified:',w,'x',h)
PY
echo "Acceptance report: $REPORT"
