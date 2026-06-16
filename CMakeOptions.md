# CppNCorr CMake Options & Build Reference

This document describes every CMake option, language requirement, and dependency
handling rule used by the CppNCorr build. It complements `CMakeLists.txt` (the
library) and `test/CMakeLists.txt` (the example/tool executables).

## Language standard

CppNCorr requires **C++17** (it relies on `std::optional` and other modern
features). The standard is enforced project-wide in the top of `CMakeLists.txt`:

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
```

and is additionally pinned on the `ncorr` target via `set_property(... CXX_STANDARD 17)`.
`test/CMakeLists.txt` sets the same global variables before defining its targets.

- **Minimum CMake**: 3.14 (declared via `CMAKE_MINIMUM_REQUIRED`).
- **Compilers**: GCC 10+ / Clang 12+ / MSVC 2019+ (mirrors CPPxDIC).

## User-configurable options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `FORCE_FETCH_DEPENDENCIES` | BOOL | `OFF` | When `ON`, ignore system-installed packages and build **all** third-party dependencies from source via CMake `FetchContent`. Useful for reproducible builds and on machines without root / package-manager access. When `OFF` (default), each dependency is first searched with `find_package` / `find_library` and only fetched if missing. |

Pass options on the command line, e.g.:

```bash
cmake -S . -B build -DFORCE_FETCH_DEPENDENCIES=ON
```

## Compile flags applied by the build

| Flag / definition | Where | Purpose |
|-------------------|-------|---------|
| `-O3` | both | Added when the compiler supports it (probed with `CheckCXXCompilerFlag`). |
| `-Wall -Wextra` | library (`target_compile_options(ncorr ...)`) | Warning hygiene for the core library. |
| `-DNDEBUG` | both (`ADD_DEFINITIONS`) | Disables `assert()` in release builds. |
| `-isystem /opt/homebrew/include` | APPLE only | Silences warnings from Homebrew system headers. |

## Targets

| Target | Defined in | Type | Notes |
|--------|------------|------|-------|
| `ncorr` | `CMakeLists.txt` | STATIC library | Core DIC engine → `lib/libncorr.a`. |
| `proxyncorr` | `test/CMakeLists.txt` | executable | The de-facto **main DIC tool**: folder discovery, full CLI (getopt), config file, JSON/binary/video output. Used by the parent CPPxDIC project. |
| `ncorr_test` | `test/CMakeLists.txt` | executable | Minimal end-to-end example on the `ohtcfrp` dataset. |
| `ncorr_test_*`, `*_test` | `test/CMakeLists.txt` | executables | Assorted regression/unit-style harnesses (interpolator, cubic, parity, RGDIC, sequential, parallel, chain seam, ROI reduce, etc.). |

## Dependencies (resolved by `cmake/FetchDependencies.cmake`)

Each dependency is found on the system first; if missing (or when
`FORCE_FETCH_DEPENDENCIES=ON`) it is fetched and built from source. The pinned
fetch versions are:

| Dependency | Role | System lookup | Fetch tag/version |
|------------|------|---------------|-------------------|
| **OpenCV** | image / video I/O, image processing | `find_package(OpenCV)` (+ `opencv4` include hints on Debian/Ubuntu) | 4.9.0 (minimal `core,imgproc,imgcodecs,highgui,videoio` build) |
| **FFTW3** | FFT routines | `find_library(fftw3)` + `fftw3.h` | 3.3.10 |
| **SuiteSparse** | sparse solvers (SPQR, CHOLMOD, AMD, COLAMD, config) | `find_library(spqr/cholmod/...)` | v7.6.0 |
| **BLAS / LAPACK** | dense linear algebra (SuiteSparse backend) | `find_package(BLAS/LAPACK)` | OpenBLAS v0.3.26 (fallback) |
| **nlohmann/json** | JSON output (header-only) | `find_package(nlohmann_json CONFIG)` | v3.11.3 |
| **OpenMP** | parallelism | `find_package(OpenMP)` (Homebrew `libomp` paths on macOS) | system only (not fetched) |

> The `FORCE`-flagged `set(... CACHE ...)` lines inside `FetchDependencies.cmake`
> (e.g. `WITH_CUDA OFF`, `BUILD_TESTS OFF`) are **internal** build-tuning for the
> fetched dependency sub-builds. They are not meant to be set by users.

## Quick build

```bash
# Library only
cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --target ncorr -j

# Executables (proxyncorr, ncorr_test, ...) — configure the test/ tree
cmake -S test -B test/build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build test/build -j
```

The convenience script `build.sh` performs the library build (clean + configure +
make) and refreshes `lib/libncorr.a`.

## Known pre-existing warnings

The header `include/Array2D.h` emits a handful of `-Wall -Wextra` warnings that
predate this work and are intrinsic to its template iterator design:

- `-Winjected-class-name` on qualified iterator definitions.
- `-Wunused-parameter` for the `type` tag parameter of `eye(...)`.
- `-Wdeprecated-declarations` for `allocator::rebind<bool>`.

These are non-fatal and are left untouched to avoid risky edits to the large
template header; they are tracked with `// FIXME(newversion):` markers where it
is safe to annotate. New code added on this branch compiles cleanly under
`-Wall -Wextra`.
