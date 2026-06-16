#!/bin/bash
# ============================================================================
# run_slurm.sh — SLURM batch script for a CppNCorr DIC run on an HPC cluster.
# ----------------------------------------------------------------------------
# Target environment: a SLURM-managed cluster where CppNCorr has been built
# (the `proxyncorr` executable is on PATH or referenced via PROXYNCORR below).
#
# DIC is OpenMP-parallel within a single node. This script requests one node and
# binds the OpenMP thread count to the allocated CPUs. Adjust the resource
# requests (--cpus-per-task, --mem, --time) to your dataset and cluster limits.
#
# Submit:
#   sbatch deploy/run_slurm.sh /path/to/images /path/to/output
# ============================================================================
#SBATCH --job-name=cppncorr-dic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=cppncorr-%j.out
#SBATCH --error=cppncorr-%j.err

set -euo pipefail

IMAGES="${1:?usage: sbatch run_slurm.sh <images_dir> <output_dir>}"
OUTPUT="${2:?usage: sbatch run_slurm.sh <images_dir> <output_dir>}"

# Path to the DIC executable. Override by exporting PROXYNCORR before submitting.
PROXYNCORR="${PROXYNCORR:-proxyncorr}"

# Bind OpenMP to the CPUs SLURM allocated to this task.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Match the engine's recommended affinity (see recent engine tuning):
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "Host: $(hostname)"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "Images: ${IMAGES}"
echo "Output: ${OUTPUT}"

srun "${PROXYNCORR}" \
    --folder "${IMAGES}" \
    --output "${OUTPUT}" \
    --threads "${OMP_NUM_THREADS}" \
    --no-videos

echo "Done."
