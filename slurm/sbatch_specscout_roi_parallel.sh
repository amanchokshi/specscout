#!/usr/bin/env bash
#SBATCH --job-name=specscout-roi
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=logs/roi-%j.out
#SBATCH --error=logs/roi-%j.err

set -euo pipefail

# ----------------------- user settings -----------------------

SEASON="202407"
STARTUTC="20240720_000000"
STOPUTC="20250501_000000"

ZARR_ROOT="/scratch/achokshi/specscout/data/zarr/${SEASON}"
OUT_ROOT="/scratch/achokshi/specscout/data/roi/${SEASON}"

STATIONS=(MARS1 MARS2 MARS3 MARS4 MARS5 MARS6)
N_JOBS=6

module purge
module use /project/rrg-sievers/achokshi/software/modulefiles
module load specscout

mkdir -p "${OUT_ROOT}" logs


echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node list   : ${SLURM_NODELIST}"
echo "Season      : ${SEASON}"
echo "Start UTC   : ${STARTUTC}"
echo "Stop UTC    : ${STOPUTC}"
echo "Zarr root   : ${ZARR_ROOT}"
echo "Output root : ${OUT_ROOT}"
echo "Stations    : ${STATIONS[*]}"
echo "Parallel jobs: ${N_JOBS}"
echo

run_station() {
    local station="$1"
    local zarr_path="${ZARR_ROOT}/${station}_${SEASON}.zarr"
    local out_dir="${OUT_ROOT}/${station}/p99"

    mkdir -p "${out_dir}"

    {
        echo "============================================================"
        echo "Station   : ${station}"
        echo "Host      : $(hostname)"
        echo "Start UTC : $(date -u '+%Y-%m-%d %H:%M:%S')"
        echo "Zarr path : ${zarr_path}"
        echo "Out dir   : ${out_dir}"
        echo "============================================================"

        specscout roi-search "${zarr_path}" \
            --station "${station}" \
            --startutc "${STARTUTC}" \
            --stoputc "${STOPUTC}" \
            --nsig 2 \
            --out-dir "${out_dir}"

        echo
        echo "Finished ${station} at $(date -u '+%Y-%m-%d %H:%M:%S')"
    } > "${out_dir}/run.log" 2>&1
}

export SEASON STARTUTC STOPUTC ZARR_ROOT OUT_ROOT
export -f run_station

printf '%s\n' "${STATIONS[@]}" | parallel -j "${N_JOBS}" --joblog "${OUT_ROOT}/parallel_joblog.txt" run_station

echo "All station runs completed successfully."
