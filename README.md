[![CI](https://github.com/xdic-dev/CppNCorr/actions/workflows/ci.yml/badge.svg)](https://github.com/xdic-dev/CppNCorr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/xdic-dev/CppNCorr/branch/main/graph/badge.svg)](https://codecov.io/gh/xdic-dev/CppNCorr)
[![Docs](https://img.shields.io/badge/docs-Doxygen-blue)](https://xdic-dev.github.io/CppNCorr/)
[![Release](https://img.shields.io/github/v/release/xdic-dev/CppNCorr?include_prereleases&sort=semver)](https://github.com/xdic-dev/CppNCorr/releases)
[![License](https://img.shields.io/github/license/xdic-dev/CppNCorr)](https://github.com/xdic-dev/CppNCorr/blob/main/LICENSE)


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

## Logging

CppNCorr uses a small, dependency-free leveled logger (`include/ncorr/log.h`)
instead of ad-hoc `std::cout`/`std::cerr`. Messages carry a severity — **TRACE,
DEBUG, INFO, WARN, ERROR** — and go to the console (INFO/DEBUG/TRACE to `stdout`,
WARN/ERROR to `stderr`) and, optionally, a full-detail log file. The stream-style
macros (`NLOG_INFO << ...`) short-circuit message construction when the level is
disabled, so they are cheap inside the per-frame / per-iteration DIC loops.

Because the library has no command line of its own, configuration is driven by
environment variables (applied lazily on first use) and by the engine's existing
`debug` flag:

```bash
export NCORR_LOG_LEVEL=info     # trace|debug|info|warn|error|off (console threshold)
export NCORR_LOG_FILE=ncorr.log # also write a full debug-level log here (empty = none)
export NCORR_LOG_CONSOLE=0      # disable console output entirely
```

Setting `debug = true` in the DIC analysis input lowers the console threshold to
DEBUG so the engine's debug diagnostics appear. A parent application can also
configure the logger programmatically via `ncorr::log::set_level()`,
`ncorr::log::set_file()`, and `ncorr::log::Logger::instance()`. When consumed by
CPPxDIC, that project propagates its own logging settings here via the
`NCORR_LOG_*` environment variables so engine logs share the same verbosity and
destination.


