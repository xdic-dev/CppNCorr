#!/bin/bash

# Build script for CppNCorr library and provide libncorr.a in the lib/ directory

# Clean previous build
echo "Cleaning previous build..."
rm -rf build
rm lib/libncorr.a

# Create build directory
echo "Creating build directory..."
mkdir -p build
pushd build

# Run CMake
echo "Running CMake..."
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5

# Build library
echo "Building library..."
make

# Return to previous directory
echo "Returning to previous directory..."
popd
