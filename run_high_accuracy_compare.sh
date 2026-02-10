#!/usr/bin/env bash
set -euo pipefail

# High-accuracy reference extraction with FasterCap, then compare vs Python BEM.
#
# Usage:
#   ./run_high_accuracy_compare.sh <input.gds> [pdk_root] [fc_accuracy] [fc_timeout_s] [bem_panel_size]
#
# Example:
#   ./run_high_accuracy_compare.sh OTA_FF_992_0.gds /home/paiktj/skywater-pdk 0.01 21600 1.0

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input.gds> [pdk_root] [fc_accuracy] [fc_timeout_s] [bem_panel_size]"
  exit 1
fi

GDS_INPUT="$1"
PDK_ROOT="${2:-/home/paiktj/skywater-pdk}"
FC_ACCURACY="${3:-0.01}"
FC_TIMEOUT="${4:-21600}"
BEM_PANEL="${5:-1.0}"

if [[ ! -f "$GDS_INPUT" ]]; then
  echo "Error: GDS file not found: $GDS_INPUT"
  exit 1
fi
if [[ ! -d "$PDK_ROOT" ]]; then
  echo "Error: PDK root not found: $PDK_ROOT"
  exit 1
fi

BASE="$(basename "${GDS_INPUT%.gds}")"
ACC_TAG="${FC_ACCURACY/./p}"
PANEL_TAG="${BEM_PANEL/./p}"

STACK_JSON="sky130a_stack_from_pdk.json"
FC_OUT_DIR="/tmp/${BASE}_fastercap_a${ACC_TAG}"
FC_POLY="/tmp/${BASE}_fc_from_gds_polygons.txt"
BEM_POLY="/tmp/${BASE}_bem_from_gds_polygons.txt"
FC_CSV="${BASE}_fastercap_ref_a${ACC_TAG}.csv"
FC_LOG="${BASE}_fastercap_ref_a${ACC_TAG}.log"
BEM_CSV="${BASE}_bem_pred_panel${PANEL_TAG}.csv"
CMP_CSV="${BASE}_fc_vs_bem_a${ACC_TAG}_panel${PANEL_TAG}.csv"
CMP_SUMMARY="${BASE}_fc_vs_bem_a${ACC_TAG}_panel${PANEL_TAG}.txt"
CMP_SCATTER="${BASE}_fc_vs_bem_a${ACC_TAG}_panel${PANEL_TAG}.png"

echo "[1/5] Generate stack JSON from official PDK..."
PYENV_VERSION=torch PYTHONDONTWRITEBYTECODE=1 pyenv exec python generate_stack_from_pdk.py \
  --pdk-root "$PDK_ROOT" \
  -o "$STACK_JSON"

echo "[2/5] Run FasterCap high-accuracy reference..."
PYENV_VERSION=torch PYTHONDONTWRITEBYTECODE=1 pyenv exec python fastercap_export.py "$GDS_INPUT" \
  --from-gds \
  --polygons-out "$FC_POLY" \
  --stack "$STACK_JSON" \
  -o "$FC_OUT_DIR" \
  --run \
  --accuracy "$FC_ACCURACY" \
  --timeout "$FC_TIMEOUT" \
  --galerkin \
  --csv-out "$FC_CSV" \
  > "$FC_LOG" 2>&1

echo "[3/5] Run Python BEM prediction..."
PYENV_VERSION=torch PYTHONDONTWRITEBYTECODE=1 pyenv exec python cap_extract.py "$GDS_INPUT" \
  --from-gds \
  --polygons-out "$BEM_POLY" \
  --stack "$STACK_JSON" \
  --panel-size "$BEM_PANEL" \
  --ground-net GND \
  -o "$BEM_CSV"

echo "[4/5] Compare FasterCap vs Python BEM..."
PYENV_VERSION=torch PYTHONDONTWRITEBYTECODE=1 pyenv exec python compare_fc_vs_bem.py \
  --fastercap "$FC_CSV" \
  --bem "$BEM_CSV" \
  --out-csv "$CMP_CSV" \
  --summary "$CMP_SUMMARY" \
  --scatter "$CMP_SCATTER"

echo "[5/5] Done."
echo "Reference CSV : $FC_CSV"
echo "Reference log : $FC_LOG"
echo "BEM CSV       : $BEM_CSV"
echo "Compare CSV   : $CMP_CSV"
echo "Summary       : $CMP_SUMMARY"
echo "Scatter       : $CMP_SCATTER"
