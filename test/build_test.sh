#!/bin/bash

# Build script for CppNCorr test in test

set -e

# Create build directory
echo "Creating build directory..."
cd test
mkdir -p build
pushd build

# Clean previous build
echo "Cleaning previous build..."
rm -rf CMakeCache.txt CMakeFiles Makefile cmake_install.cmake

# Run CMake
echo "Running CMake..."
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5

# Build library
echo "Building test..."
make clean
make

# Return to previous directory
echo "Returning to previous directory..."
popd
cd ..
