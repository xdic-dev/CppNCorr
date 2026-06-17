# CppNCorr deployment assets

This directory holds everything needed to build and run the CppNCorr DIC tool
(`proxyncorr`) in containerized and cluster environments. CppNCorr's parallelism
is **OpenMP within a single node**; there is no MPI/multi-node distribution, so
the cluster recipe scales by CPU cores per node, not across nodes.

## Files

| File | Target environment | Purpose |
|------|--------------------|---------|
| `Dockerfile` | Any Docker host / CI | Single-stage Ubuntu 24.04 image that installs all dependencies from the distro and builds the `proxyncorr` executable. `ENTRYPOINT` is `proxyncorr`. |
| `run_slurm.sh` | SLURM-managed HPC cluster | `sbatch` script that runs one DIC job on a single node, binding `OMP_NUM_THREADS` to the allocated CPUs. |
| `../.dockerignore` | (repo root) | Keeps local build trees / caches out of the Docker build context. Lives at the repo root because that is the Docker build context. |

## Docker

The build context is the **repository root** (so the image can copy the whole
source tree), and the Dockerfile is referenced with `-f`:

```bash
# from the repository root
docker build -f deploy/Dockerfile -t cppncorr .
```

Run it against a local data folder that contains the numbered frames plus a
`roi.png` mask:

```bash
docker run --rm -v "$PWD/data:/data" cppncorr \
    --folder /data/images --output /data/out --no-videos
```

Arguments after the image name are passed straight to `proxyncorr` (see the
[user guide](../docs/user_guide.md) for all flags). `docker run --rm cppncorr`
with no extra args prints `--help`.

The image builds `proxyncorr` with `BUILD_TESTING=OFF`, so the Catch2 test suite
is not part of the image.

## SLURM / HPC

Build CppNCorr on the cluster first (or use the container via Singularity/Apptainer
— see below), then submit a job:

```bash
# images_dir must contain the frames + roi.png; output_dir is created as needed
sbatch deploy/run_slurm.sh /scratch/$USER/run1/images /scratch/$USER/run1/out
```

Tunables (edit the `#SBATCH` directives or export before submitting):

- `--cpus-per-task` — number of OpenMP threads (the script sets
  `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`).
- `--mem`, `--time` — scale to your dataset.
- `PROXYNCORR` — path to the executable if it is not on `PATH`
  (`export PROXYNCORR=/path/to/proxyncorr`).

The script sets `OMP_PROC_BIND=spread` and `OMP_PLACES=cores`, matching the
engine's recommended thread affinity.

### Running the Docker image under Singularity/Apptainer

Many clusters disallow the Docker daemon but support Apptainer/Singularity:

```bash
# convert the Docker image to a SIF (after `docker build` + push, or from a local daemon)
apptainer build cppncorr.sif docker-daemon://cppncorr:latest

# then call it from run_slurm.sh by setting:
#   export PROXYNCORR="apptainer run cppncorr.sif"
```

## Notes

- All dependencies (OpenCV, FFTW, SuiteSparse, OpenBLAS/LAPACK, nlohmann/json,
  OpenMP) are installed from Ubuntu packages in the Dockerfile. On clusters
  without these as modules, configure with `-DFORCE_FETCH_DEPENDENCIES=ON` to
  build them from source via FetchContent (slower first build).
- Video rendering (`.avi`) is disabled in the sample commands (`--no-videos`) to
  keep batch runs fast and headless; drop the flag to produce field videos.
