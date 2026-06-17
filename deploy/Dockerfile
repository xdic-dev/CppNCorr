# syntax=docker/dockerfile:1
#
# Dockerfile for CppNCorr — builds the main DIC executable `proxyncorr`.
#
# Builds on Ubuntu using system-installed dependencies (OpenCV, FFTW,
# SuiteSparse, OpenBLAS/LAPACK, nlohmann/json, OpenMP). The CMake build will
# find these via find_package/find_library; FetchContent is only used as a
# fallback when a dependency is missing.
#
# Build (from the repository root):
#   docker build -f deploy/Dockerfile -t cppncorr .
#
# Run (mount a data folder containing frames + roi.png):
#   docker run --rm -v "$PWD/data:/data" cppncorr \
#       --folder /data/images --output /data/out --no-videos
#
# This is a single-stage build kept deliberately simple for reproducibility.

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# Toolchain + all CppNCorr dependencies from the distro.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        libopencv-dev \
        libfftw3-dev \
        libsuitesparse-dev \
        libopenblas-dev \
        liblapack-dev \
        nlohmann-json3-dev \
        libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cppncorr

# Copy the whole repository (build context = repo root).
COPY . .

# Build the static library and the proxyncorr executable.
#   - The library target `ncorr` is built from the root CMakeLists.
#   - The executables (incl. proxyncorr) are built from the test/ tree, which
#     links libncorr.a.
# BUILD_TESTING=OFF keeps the Catch2 test suite out of the image build.
RUN cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_TESTING=OFF \
    && cmake --build build --target ncorr -j "$(nproc)" \
    && cmake -S test -B test/build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && cmake --build test/build --target proxyncorr -j "$(nproc)"

# Put the DIC tool on PATH.
RUN install -m 0755 test/bin/proxyncorr /usr/local/bin/proxyncorr

# Default to running the DIC tool; arguments are appended after `docker run img`.
ENTRYPOINT ["proxyncorr"]
CMD ["--help"]
