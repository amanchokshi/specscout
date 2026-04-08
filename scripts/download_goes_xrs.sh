YEAR=2025

BASE_URL="https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/xrsf-l2-flx1s_science/${YEAR}/"
OUTDIR="/Volumes/specscout/goes18_xrs_1s/${YEAR}"

mkdir -p "$OUTDIR"

wget \
  --recursive \
  --no-parent \
  --no-host-directories \
  --cut-dirs=9 \
  --reject "index.html*" \
  --accept "*.nc" \
  --directory-prefix="$OUTDIR" \
  "$BASE_URL"
