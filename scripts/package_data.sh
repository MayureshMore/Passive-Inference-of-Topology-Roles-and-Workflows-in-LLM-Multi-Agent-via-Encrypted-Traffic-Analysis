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
# resulting DOI/URL AND the SHA-256 from dist/SHA256SUMS.txt into DATA.md (and
# the DOI/URL into scripts/reproduce.sh).  The checksum lets an artifact
# evaluator verify the download matches the release before reproducing.
#
# Usage:
#   bash scripts/package_data.sh
#
set -euo pipefail

OUT="dist"
mkdir -p "$OUT"

# Exclude caches and any stray result dirs that may live under data/.
EXCLUDES=(-x "*/__pycache__/*" -x "*.DS_Store")

# Portable SHA-256 (macOS: shasum -a 256; Linux: sha256sum).
sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

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

# ── Checksums (so a downloader can verify integrity before reproducing) ───────
echo "== writing checksums =="
( cd "$OUT" && rm -f SHA256SUMS.txt && for z in a2a-features.zip a2a-raw-pcaps.zip; do
    [ -e "$z" ] && sha256 "$z" >> SHA256SUMS.txt; done )
if [ -s "$OUT/SHA256SUMS.txt" ]; then
  echo "  wrote $OUT/SHA256SUMS.txt:"; sed 's/^/    /' "$OUT/SHA256SUMS.txt"
else
  echo "  (no archives were built, so no checksums written)" >&2
fi

echo ""
echo "Done. Next steps:"
echo "  1. Upload dist/*.zip + dist/SHA256SUMS.txt to Zenodo/figshare/OSF (human step)."
echo "  2. Paste the DOI/URL into DATA.md (Archive location) + scripts/reproduce.sh header,"
echo "     and the a2a-features.zip SHA-256 into DATA.md so downloaders can verify."
