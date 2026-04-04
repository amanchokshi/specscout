#!/usr/bin/env bash
#SBATCH --job-name=specscout-roi
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=32
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

# ----------------------- environment -------------------------

module purge
module use /project/rrg-sievers/achokshi/software/modulefiles
module load specscout

mkdir -p "${OUT_ROOT}"

# Keep each station confined to its allocated 32 cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node list   : ${SLURM_NODELIST}"
echo "Season      : ${SEASON}"
echo "Start UTC   : ${STARTUTC}"
echo "Stop UTC    : ${STOPUTC}"
echo "Zarr root   : ${ZARR_ROOT}"
echo "Output root : ${OUT_ROOT}"
echo "CPUs/task   : ${SLURM_CPUS_PER_TASK}"
echo

# ----------------------- launch all stations -----------------

pids=()

for STATION in "${STATIONS[@]}"; do
    ZARR_PATH="${ZARR_ROOT}/${STATION}_${SEASON}.zarr"
    OUT_DIR="${OUT_ROOT}/${STATION}/p99"

    mkdir -p "${OUT_DIR}"

    echo "Launching ${STATION}"
    echo "  zarr : ${ZARR_PATH}"
    echo "  out  : ${OUT_DIR}"

    srun --exclusive -N1 -n1 -c "${SLURM_CPUS_PER_TASK}" \
        specscout roi-search "${ZARR_PATH}" \
        --station "${STATION}" \
        --startutc "${STARTUTC}" \
        --stoputc "${STOPUTC}" \
        --nsig 2 \
        --out-dir "${OUT_DIR}" \
        > "${OUT_DIR}/run.log" 2>&1 &

    pids+=($!)
done

# ----------------------- wait for all ------------------------

fail=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        fail=1
    fi
done

if [[ "${fail}" -ne 0 ]]; then
    echo "One or more station runs failed."
    exit 1
fi

echo "All station runs completed successfully."
