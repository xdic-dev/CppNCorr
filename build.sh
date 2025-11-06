#!/bin/bash

# Build script for CppNCorr library and provide libncorr.a in the lib/ directory

set -e

# Create build directory
echo "Creating build directory..."
mkdir -p build
pushd build

# Clean previous build
echo "Cleaning previous build..."
rm -rf CMakeCache.txt CMakeFiles Makefile cmake_install.cmake ../lib/libncorr.a

# Run CMake
echo "Running CMake..."
cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5

# Build library
echo "Building library..."
make

# Return to previous directory
echo "Returning to previous directory..."
popd
