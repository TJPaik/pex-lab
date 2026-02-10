#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_pex_from_gds.sh <input.gds> [output.csv] [pdk_root]
#
# Example:
#   ./run_pex_from_gds.sh OTA_FF_992_0.gds OTA_FF_992_0_bem_pred.csv /home/paiktj/skywater-pdk

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input.gds> [output.csv] [pdk_root]"
  exit 1
fi

GDS_INPUT="$1"
OUTPUT_CSV="${2:-$(basename "${GDS_INPUT%.gds}")_bem_pred.csv}"
PDK_ROOT="${3:-/home/paiktj/skywater-pdk}"
STACK_JSON="sky130a_stack_from_pdk.json"
POLYGONS_TMP="/tmp/$(basename "${GDS_INPUT%.gds}")_from_gds_polygons.txt"

if [[ ! -f "$GDS_INPUT" ]]; then
  echo "Error: GDS file not found: $GDS_INPUT"
  exit 1
fi

if [[ ! -d "$PDK_ROOT" ]]; then
  echo "Error: PDK root not found: $PDK_ROOT"
  exit 1
fi

echo "[1/3] Generate stack JSON from official PDK docs..."
PYENV_VERSION=torch PYTHONDONTWRITEBYTECODE=1 pyenv exec python generate_stack_from_pdk.py \
  --pdk-root "$PDK_ROOT" \
  -o "$STACK_JSON"

echo "[2/3] Run Python BEM prediction from GDS..."
PYENV_VERSION=torch PYTHONDONTWRITEBYTECODE=1 pyenv exec python cap_extract.py "$GDS_INPUT" \
  --from-gds \
  --polygons-out "$POLYGONS_TMP" \
  --stack "$STACK_JSON" \
  -o "$OUTPUT_CSV"

echo "[3/3] Done."
echo "Stack JSON: $STACK_JSON"
echo "PEX output : $OUTPUT_CSV"
echo "Temp poly  : $POLYGONS_TMP"
