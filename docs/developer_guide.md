# CppNCorr Developer Guide

This guide is for contributors working on CppNCorr itself: the repository layout,
how to extend the CLI, how to finish the in-memory `NcorrSession` API, the
config-file schema, and how to run the tests.

## Repository layout

```
CppNCorr/
├── CMakeLists.txt          # Library target `ncorr`; guarded test include (5b)
├── CMakeOptions.md         # Every CMake option / dependency / flag
├── build.sh                # Convenience library build
├── cmake/
│   └── FetchDependencies.cmake   # find-or-fetch for OpenCV/FFTW/SuiteSparse/BLAS/json
├── config/
│   └── default.cfg         # Shipped INI config (mirrors compiled defaults)
├── include/
│   ├── ncorr.h             # Public engine API (DIC_analysis, strain_analysis, ...)
│   ├── Array2D.h Data2D.h Disp2D.h Strain2D.h ROI2D.h Image2D.h
│   └── ncorr/
│       ├── config.h        # struct Config + load_config_file (override chain)
│       ├── ini.h           # Header-only INI parser
│       ├── session.h       # In-memory NcorrSession API (stub)
│       └── frame_reader.h  # discover_frames / has_image_extension / natural_less
├── src/                    # Engine + config.cpp + session.cpp implementations
├── test/
│   ├── CMakeLists.txt      # Legacy example/tool executables (incl. proxyncorr)
│   ├── tests.cmake         # Catch2 test suite (included only when top-level)
│   ├── src/                # Legacy harnesses + proxyncorr.cpp (the DIC tool)
│   ├── unit/               # Catch2 unit tests (ini/config/frame_reader/session)
│   ├── integration/        # Catch2 pipeline-stage tests
│   ├── e2e/                # Catch2 end-to-end test (ohtcfrp)
│   └── examples/ohtcfrp/   # Lightweight fixture dataset
├── deploy/                 # Docker / cluster deployment assets
└── docs/                   # This guide + quickstart + user guide
```

### Targets

| Target | Type | Notes |
|--------|------|-------|
| `ncorr` | static library | Core DIC engine → `lib/libncorr.a`. |
| `proxyncorr` | executable | The de-facto **main DIC tool** (folder discovery, full CLI, config file, JSON/binary/video output). |
| `ncorr_unit_tests` | executable | Catch2 unit tests; no heavy deps. |
| `ncorr_engine_tests` | executable | Catch2 integration + e2e tests; links `ncorr`. |
| `ncorr_test*`, `*_test` | executables | Legacy regression/example harnesses. |

See [`../CMakeOptions.md`](../CMakeOptions.md) for the complete options and
dependency reference.

## How to extend the CLI with a new post-DIC operation

The CLI lives in `test/src/proxyncorr.cpp` and uses `getopt_long`. To add a new
post-DIC operation (e.g. a new export format):

1. **Add a field** to `struct ProxyConfig` for the option's state.
2. **Register the long option** in the `long_options[]` array. Use a numeric
   value `>= 1000` for options without a short flag.
3. **Parse it** in the second `getopt_long` loop's `switch`.
4. **Document it** in `print_usage()` (and in [`user_guide.md`](user_guide.md)).
5. **Wire the behaviour** after argument parsing. If the backing implementation
   does not exist yet, follow the existing stub convention: print
   `"[--flag] not yet implemented"` and `return 0` (exit cleanly), and mark the
   gap with a `// FIXME(newversion):` comment. See the handling of
   `--change-perspective` for an example of a flag with both an implemented mode
   and a not-yet-implemented mode.

If the new parameter is also a tuneable DIC parameter (rather than pure I/O),
add it to `ncorr::Config` (`include/ncorr/config.h`), to `config/default.cfg`
(same key name), to the overlay in `src/config.cpp`, and to `apply_core_config()`
in `proxyncorr.cpp` so it participates in the three-tier override chain.

## How to implement the in-memory API (`NcorrSession`)

`include/ncorr/session.h` declares a small, dependency-light API that lets a
caller (e.g. CPPxDIC's `proxyncorr`) push image buffers directly to the engine
without writing frames to disk. The interface is **final**; the bodies in
`src/session.cpp` are currently **stubs** (validate inputs, return a
"not yet implemented" `DICResult`).

To implement it, follow the file-based pipeline in `test/src/proxyncorr.cpp`:

1. Wrap each `ImageBuffer` in a `cv::Mat` (row-major, interleaved, 8-bit) and
   build an `ncorr::Image2D` via `Image2D::from_mat(...)`.
2. Build a `ROI2D` from the optional mask (or a full-frame default).
3. Construct a `DIC_analysis_input { {ref, def}, roi, scalefactor, interp,
   subregion, radius, num_threads, DIC_analysis_config, debug }`.
4. Run `DIC_analysis(...)` and copy the resulting `Disp2D` `u`/`v` arrays
   (`disp.get_u().get_array()`, `disp.get_v().get_array()`) into the flat
   row-major `DICResult::u` / `DICResult::v` vectors, writing `NaN` outside the
   ROI. Set `result.valid = true` on success.

The `Impl` PIMPL struct in `session.cpp` should cache the converted reference
`Image2D` and the `ROI2D` so repeated `process_frame` calls reuse them.

When done, **update the unit test** `session_stub_contract` in
`test/unit/test_session.cpp` (which currently pins the stub message) and add the
deferred integration test `pipeline_session_matches_folder` from `TESTS_PLAN.md`
(assert the session matches the folder pipeline on the same frames).

## Config-file schema reference

Format: minimal **INI** (`key = value`), parsed by the header-only
`include/ncorr/ini.h`. Full-line and inline `#`/`;` comments are supported;
values may be quoted to preserve spaces or comment characters; CRLF line endings
are handled.

Keys MUST exactly match the field names of `ncorr::Config` (case-sensitive) so
there are no silent mismatches — this is enforced by the
`config_key_field_parity` unit test.

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `scalefactor` | int | `3` | Pyramid downsampling level for seed search. |
| `subregion_type` | string | `CIRCLE` | `CIRCLE` or `SQUARE`. |
| `subregion_radius` | int | `20` | Correlation window radius (px). |
| `interp_type` | string | `QUINTIC_BSPLINE_PRECOMPUTE` | `NEAREST`, `LINEAR`, `CUBIC_KEYS[_PRECOMPUTE]`, `QUINTIC_BSPLINE[_PRECOMPUTE]`. |
| `strain_subregion_type` | string | `CIRCLE` | `CIRCLE` or `SQUARE`. |
| `strain_radius` | int | `5` | Strain window radius (px). |
| `dic_config` | string | `NO_UPDATE` | `NO_UPDATE`, `KEEP_MOST_POINTS`, `REMOVE_BAD_POINTS`. |
| `num_threads` | int | `4` | Worker threads for parallel analysis. |
| `debug` | bool | `false` | Verbose engine output. |
| `perspective_interp` | string | `CUBIC_KEYS` | Interpolation used by the perspective change. |
| `units` | string | `mm` | Displacement units label. |
| `units_per_pixel` | double | `0.2` | Physical scaling. |
| `alpha` | double | `0.5` | Video overlay alpha (0..1). |
| `fps` | double | `15.0` | Output video FPS. |

The shipped [`config/default.cfg`](../config/default.cfg) mirrors this table.

## How to run tests locally

The Catch2 test suite is configured **only when CppNCorr is the top-level
project** (gated by `PROJECT_IS_TOP_LEVEL AND BUILD_TESTING`), so it never
affects the parent CPPxDIC build.

```bash
# Configure + build the whole top-level project (incl. tests)
cmake -S . -B build_tests -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build_tests -j

# Run everything via CTest
cd build_tests && ctest --output-on-failure
```

### Test tiers and filtering

- **`ncorr_unit_tests`** — fast, dependency-light (config/ini/frame_reader/
  session). Run directly: `./build_tests/ncorr_unit_tests`.
- **`ncorr_engine_tests`** — links the full engine; the integration/e2e cases
  run real DIC and are slow (tens of seconds each). Filter by Catch2 tag:

  ```bash
  ./build_tests/ncorr_engine_tests "[integration]"
  ./build_tests/ncorr_engine_tests "[e2e]"
  ```

To re-capture the e2e golden baseline (after an intentional behavioural change):

```bash
NCORR_E2E_CAPTURE=1 ./build_tests/ncorr_engine_tests "[e2e]"
```

then update `kGoldenMeanDispMag` in `test/e2e/test_e2e.cpp`.

### Adding a test

Drop a new `.cpp` under `test/unit/`, `test/integration/`, or `test/e2e/`, add it
to the corresponding `add_executable(...)` list in `test/tests.cmake`, and use
the Catch2 `TEST_CASE("name", "[tier][topic]")` macro. `catch_discover_tests`
registers each `TEST_CASE` with CTest automatically.

## Coding conventions

- New public headers: `#pragma once` and Doxygen doc-comments on every public
  symbol.
- Conventional-commit messages, one logical unit per commit.
- New code must compile cleanly under `-Wall -Wextra`. Pre-existing template
  warnings in `Array2D.h` are documented in `CMakeOptions.md` and left untouched.
