#!/usr/bin/env bash
#
# package_data.sh — build the publishable data archives from a populated data/.
#
# Produces two zips under dist/ :
#   dist/a2a-features.zip    data/processed*/   (.npz feature matrices + labels.json)
#                            — small; enough for scripts/reproduce.sh
#   dist/a2a-raw-pcaps.zip   data/raw*/         (header-only pcaps + label sidecars)
#                            — large; only needed for defense-overhead numbers
#                              and re-extracting features from scratch
#
# The canonical results (data/results/) are committed in the repo and are NOT
# packaged here.  Upload the two zips to Zenodo/figshare/OSF, then paste the
# resulting DOI/URL into DATA.md and scripts/reproduce.sh.
#
# Usage:
#   bash scripts/package_data.sh
#
set -euo pipefail

OUT="dist"
mkdir -p "$OUT"

# Exclude caches and any stray result dirs that may live under data/.
EXCLUDES=(-x "*/__pycache__/*" -x "*.DS_Store")

echo "== packaging feature matrices =="
proc_dirs=(data/processed*/)
if [ ! -e "${proc_dirs[0]}" ]; then
  echo "  no data/processed*/ found — nothing to package (did you unpack/collect data?)" >&2
else
  rm -f "$OUT/a2a-features.zip"
  zip -r -q "$OUT/a2a-features.zip" data/processed*/ "${EXCLUDES[@]}"
  echo "  wrote $OUT/a2a-features.zip ($(du -h "$OUT/a2a-features.zip" | cut -f1))"
fi

echo "== packaging raw captures =="
raw_dirs=(data/raw*/)
if [ ! -e "${raw_dirs[0]}" ]; then
  echo "  no data/raw*/ found — skipping raw archive" >&2
else
  rm -f "$OUT/a2a-raw-pcaps.zip"
  zip -r -q "$OUT/a2a-raw-pcaps.zip" data/raw*/ "${EXCLUDES[@]}"
  echo "  wrote $OUT/a2a-raw-pcaps.zip ($(du -h "$OUT/a2a-raw-pcaps.zip" | cut -f1))"
fi

echo ""
echo "Done. Next steps:"
echo "  1. Upload dist/*.zip to Zenodo/figshare/OSF (human step)."
echo "  2. Paste the DOI/URL into DATA.md (Archive location) and scripts/reproduce.sh header."
