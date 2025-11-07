# CppNCorr

This is the fork of the official repo for the complete C++ port of [Ncorr_2D_cpp](https://github.com/justinblaber/ncorr_2D_cpp).

In this fork, we have added the following features:

- Update some modules to make it runnable
- Parallelization using OpenMP
- Loading of images from a memory-mapped file
- Manual seeding point picking
- Smart seeding update on failure

## Build

To build the library, run the following command:

```bash
./build.sh
```

This will create a `libncorr.a` file in the `lib` directory. You can then link this library to your own projects.

## Usage

To use the library, you can include the header file `ncorr.h` and link against the library `libncorr.a`. Check the `test` directory for examples. For example, `ncorr_test.cpp` is a simple example that shows how to use the library. You can compile it using the following command:

```bash
cd test
./../build.sh
```

the build result will be in the `bin` directory. Then run the executable, for example, for the ohtcfrp example:

```bash
cd examples/ohtcfrp
./../../bin/ncorr_test
```

Check the video directory in the `examples/ohtcfrp` directory for the result video.




