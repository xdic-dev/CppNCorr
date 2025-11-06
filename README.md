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

