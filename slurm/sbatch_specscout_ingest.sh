#!/usr/bin/env bash
#SBATCH --job-name=specscout-ingest
#SBATCH --output=logs/ingest-%j.out
#SBATCH --error=logs/ingest-%j.err
#SBATCH --ntasks-per-node=192
#SBATCH --time=3:00:00
#SBATCH --nodes=1

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 <STATION> <SEASON>"
    exit 1
fi

STATION="$1"
SEASON="$2"

DATA_DIR="/project/rrg-sievers/achokshi/data/specscout/data_auto_cross/${STATION}/${SEASON}"
OUT_ZARR="/scratch/achokshi/specscout/data/zarr/${SEASON}/${STATION}_${SEASON}.zarr"

module purge
module use /project/rrg-sievers/achokshi/software/modulefiles
module load specscout

mkdir -p "$(dirname "$OUT_ZARR")"

echo "STATION  : $STATION"
echo "SEASON   : $SEASON"
echo "DATA_DIR : $DATA_DIR"
echo "OUT_ZARR : $OUT_ZARR"

specscout ingest "$DATA_DIR" \
    --station "$STATION" \
    --batch-size 512 \
    --out-zarr "$OUT_ZARR"
