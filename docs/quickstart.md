# CppNCorr Quick-start

CppNCorr is a C++17 implementation of the **ncorr** 2D Digital Image Correlation
(DIC) engine. It takes a reference image plus a series of deformed images and
computes 2D displacement and strain fields. It builds as a static library
(`libncorr.a`) and ships a command-line driver, **`proxyncorr`**, that runs a
full DIC analysis on a folder of images.

> **Note on target names.** `ncorr` is the static *library* target. The actual
> DIC *executable* is `proxyncorr` (defined in `test/CMakeLists.txt`). Throughout
> the docs, "the DIC tool" means `proxyncorr`.

## Prerequisites

- CMake >= 3.14
- A C++17 compiler (GCC 10+ / Clang 12+ / MSVC 2019+)
- OpenCV (image/video I/O), FFTW3, SuiteSparse, BLAS/LAPACK, nlohmann/json,
  OpenMP

Every dependency is searched on the system first and only fetched from source if
missing (or when `-DFORCE_FETCH_DEPENDENCIES=ON`). See
[`../CMakeOptions.md`](../CMakeOptions.md) for the full dependency table.

On macOS with Homebrew:

```bash
brew install cmake opencv fftw suite-sparse libomp nlohmann-json
```

On Debian/Ubuntu:

```bash
sudo apt-get install cmake g++ libopencv-dev libfftw3-dev \
     libsuitesparse-dev libopenblas-dev nlohmann-json3-dev libomp-dev
```

## Build

### Library

```bash
cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --target ncorr -j
```

This produces `lib/libncorr.a`. The convenience script `./build.sh` does the
same (clean + configure + make).

### The DIC tool (`proxyncorr`) and the example executables

```bash
cmake -S test -B test/build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build test/build -j
```

Executables land in `test/bin/` (e.g. `test/bin/proxyncorr`).

## Minimal run

`proxyncorr` works on a **folder of frames** plus a ROI mask. The simplest
invocation points it at an image folder and an output directory:

```bash
./test/bin/proxyncorr --folder path/to/images --output out/
```

By convention the folder contains numbered frames (`frame_00.png`,
`frame_01.png`, ...) and a `roi.png` mask; the first frame (or `ref.png` if
present) is used as the reference. See the [user guide](user_guide.md) for the
full naming convention and every CLI flag.

## Try it on the bundled `ohtcfrp` dataset

The repository ships a small fixture under `test/examples/ohtcfrp/images`
(12 PNG frames + `roi.png`):

```bash
./test/bin/proxyncorr \
    --folder test/examples/ohtcfrp/images \
    --output test/examples/ohtcfrp \
    --scalefactor 1 --radius 30 --threads 4
```

Outputs are written under the chosen output directory:

- `save/` — binary serialized DIC/strain inputs and outputs
- `save_json/` — the same data as JSON
- `video/` — rendered displacement/strain field videos (`.avi`)

To skip the (slower) video rendering, add `--no-videos`.

## Configuration

Parameters resolve through a three-tier override chain (highest priority first):

1. Direct CLI arguments
2. A config file (`--config path.cfg`, INI format; defaults to
   `config/default.cfg` when present)
3. Compiled defaults (`include/ncorr/config.h`)

See [`config/default.cfg`](../config/default.cfg) for every tuneable key.

## Next steps

- [User guide](user_guide.md) — image formats, all CLI flags, output layout,
  troubleshooting.
- [Developer guide](developer_guide.md) — repository layout, extending the CLI,
  the in-memory `NcorrSession` API, config schema, running tests.

## Reference papers

CppNCorr is a port of the ncorr 2D DIC algorithm. If you use it, please cite the
original ncorr work:

- [Ncorr: open-source 2D digital image correlation MATLAB software](DOI_PLACEHOLDER)
- [Investigation of Reliability-Guided Digital Image Correlation](DOI_PLACEHOLDER)

(Replace `DOI_PLACEHOLDER` with the appropriate DOIs.)
