# CppNCorr Test Plan (`newversion`)

> STATUS: **proposal — awaiting confirmation.** This document lists candidate
> tests only. No test framework has been added yet and no test code has been
> written. Per the task spec, implementation (section 5b) must not start until
> this plan is explicitly confirmed.

## Scope & context

The source tree contains:

- **Core library** (`src/`, `include/`): `Array2D`, `Data2D`, `ROI2D`, `Disp2D`,
  `Strain2D`, `Image2D`, the DIC engine `ncorr.cpp` (RGDIC, `DIC_analysis*`,
  `change_perspective`, `set_units`, `strain_analysis`, interpolators,
  nloptimizers), plus the new `session.cpp` (in-memory API) and `config.cpp`
  (+ header-only `ini.h`).
- **Existing harnesses** (`test/src/`): ad-hoc executables that already exercise
  parts of the engine — `interpolator_test`, `ncorr_interpolator_test`,
  `ncorr_cubic_interpolator_test`, `ncorr_nloptimizer_test`, `roi2d_reduce_test`,
  `ncorr_test_parity`, `ncorr_test_verif`, `ncorr_test_rgdic_verification`,
  `ncorr_test_sequential`, `ncorr_test_parallel`, `ncorr_test_chain_seam`,
  `ncorr_test`, `proxyncorr`. These are standalone `main()`s, not a unit-test
  framework, and mostly dump JSON for external (MATLAB) comparison.
- **Fixture dataset**: `test/examples/ohtcfrp/` (12 PNG frames + `roi.png`) — the
  lightweight end-to-end fixture referenced by the spec.

### Open question for the reviewer (needed before 5b)

The spec offers two options: reuse the existing ad-hoc harness style, or add
**GoogleTest / Catch2** (which requires explicit approval as a new dependency).
**Recommendation:** add **Catch2 v3 via the existing FetchContent pattern**
(header/single-lib, no system install, matches how nlohmann_json/OpenCV are
already fetched). It gives real assertions, test discovery, and CTest
integration. If you prefer zero new dependencies, the alternative is a minimal
in-tree assertion macro + a CTest registration of pass/fail executables.
**Please confirm the framework choice before I implement.**

---

## 1. Unit tests (individual functions)

### 1a. Config parser (new code — highest-value, dependency-free)
- `ini_parse_basic` — `key = value`, whitespace trimming, returns expected map.
- `ini_parse_comments` — full-line `#` and `;` comments are skipped; inline
  comments after unquoted `#`/`;` are stripped.
- `ini_parse_quoted_values` — single/double quotes preserve spaces and embedded
  `#`/`;`.
- `ini_parse_sections` — `[section]` headers produce `section.key` addressing.
- `ini_parse_crlf` — CRLF line endings parse identically to LF.
- `ini_typed_getters` — `get_int`/`get_double`/`get_bool` parse valid values;
  bool accepts true/false/1/0/yes/no/on/off.
- `ini_typed_getters_invalid` — non-numeric int/double and bad bool throw with a
  descriptive message.
- `ini_missing_file` — `load()` of a non-existent path returns false (not throw).
- `config_defaults` — a fresh `Config` has the documented compiled defaults.
- `config_overlay` — `load_config_file` overrides only keys present in the file,
  leaving others at their incoming values.
- `config_key_field_parity` — every key in `config/default.cfg` maps to a
  `Config` field (guards against silent mismatches).

### 1b. In-memory session API (new code)
- `imagebuffer_valid` — `valid()` / `size_bytes()` for good and degenerate inputs.
- `session_requires_reference` — `process_frame` before `set_reference` throws
  `std::logic_error`.
- `session_rejects_invalid_reference` — `set_reference` with a null/empty buffer
  throws `std::invalid_argument`.
- `session_geometry_mismatch` — deformed frame of a different size returns an
  invalid `DICResult` with a message (no crash).
- `session_stub_contract` — current stub returns `valid == false` and the
  "not yet implemented" message (update this test when 3b is implemented).

### 1c. Image folder reader (`discover_frames` + helpers in proxyncorr)
- `has_image_extension` — accepts png/tif/tiff/bmp/jpg/jpeg (any case), rejects
  others and length-edge names.
- `natural_less` — `frame_2 < frame_10`; zero-padded names also order correctly.
- `discover_frames_sorted` — returns frames in natural order from a temp dir.
- `discover_frames_excludes_reserved` — skips `roi.png`, `ref.png`, hidden files,
  sub-directories, and explicitly named ref/roi basenames.
- `discover_frames_empty` — empty folder returns an empty vector.
- `discover_frames_missing_folder` — throws `std::runtime_error`.
  > Note: these helpers currently live in `proxyncorr.cpp`. Testing them cleanly
  > may require extracting them into a small reader unit (e.g.
  > `include/ncorr/frame_reader.h`). Flag if you want that refactor in 5b.

### 1d. Image reader (`Image2D`)
- `image2d_read_grayscale` — `get_gs()` on a known PNG returns values in [0,1]
  with expected dimensions.
- `image2d_read_missing` — constructing from / reading a missing file throws
  `std::invalid_argument`.
- `image2d_from_mat_roundtrip` — `from_mat` then `get_gs()` matches the source.
- `imageprocessor_mat_array_roundtrip` — `array2d_to_mat` / `mat_to_array2d`.

### 1e. Array2D core
- `array2d_construct_index` — element access, `height`/`width`, bounds.
- `array2d_arithmetic` — `+ - * /`, scalar ops, equality.
- `array2d_eye` — identity construction.
- (Lower priority: this is heavily templated and large; smoke-level coverage.)

### 1f. Interpolation (engine)
- `interp_nearest_linear` — exact recovery of node values; linear midpoint.
- `interp_cubic_keys` — reproduces a known cubic polynomial within tolerance.
- `interp_quintic_bspline` — round-trip / partition-of-unity property.
  > Existing `interpolator_test` / `ncorr_cubic_interpolator_test` provide
  > reference behaviour to port into assertions.

### 1g. Subregion / RGDIC building blocks
- `subregion_circle_square` — generated subregion point sets for CIRCLE/SQUARE at
  a few radii.
- `rgdic_synthetic_shift` — `RGDIC` on a synthetically translated image recovers
  the known uniform displacement within tolerance (small fixture).

### 1h. Strain computation
- `strain_zero_for_rigid_translation` — pure translation yields ~0 strain.
- `strain_uniform_field` — a synthetic linear displacement field yields the
  expected constant `exx`/`eyy`/`exy`.

### 1i. ROI2D
- `roi_from_mask` — threshold mask -> region count and mask round-trip.
- `roi_reduce` — port assertions from existing `roi2d_reduce_test`.
- `roi_update` — `update(ROI, Disp, INTERP)` SKIP_ALL vs SKIP_INVALID behaviour.

### 1j. Units / perspective
- `set_units_scaling` — `set_units` scales displacement by units/pixel.
- `change_perspective_identity` — converting and (where applicable) inverting
  returns a field close to the original on a synthetic case.

---

## 2. Integration tests (pipeline stages)

- `pipeline_load_to_dic` — load `ohtcfrp` ref + one deformed frame, build
  `DIC_analysis_input`, run `DIC_analysis`, assert output dimensions and a
  finite displacement inside the ROI.
- `pipeline_dic_to_strain` — feed `DIC_analysis_output` into `strain_analysis`,
  assert strain field dimensions and finiteness in-ROI.
- `pipeline_perspective_units` — `DIC_analysis -> change_perspective ->
  set_units` chain runs and preserves dimensions.
- `pipeline_sequential_vs_parallel` — `DIC_analysis_sequential` and
  `DIC_analysis(parallel)` agree within tolerance on the same small input.
- `pipeline_config_drives_run` — load params from a temp INI, confirm the run
  uses them (e.g. radius/scalefactor reflected in `DIC_analysis_input`).
- `pipeline_export_outputs` — JSON and binary save/round-trip (`save`/`load`) of
  DIC/strain inputs and outputs to a temp dir.
- `pipeline_session_matches_folder` *(deferred until 3b implemented)* — once
  `NcorrSession::process_frame` is real, assert it matches the folder pipeline on
  the same frames.

## 3. End-to-end test

- `e2e_ohtcfrp_smoke` — run the full pipeline (or the `proxyncorr` executable) on
  `test/examples/ohtcfrp` with a reduced frame count, writing to a temp output
  dir; assert it exits 0 and produces the expected output files.
- `e2e_ohtcfrp_known_value` — at a chosen in-ROI control point on a specific
  frame, assert the displacement magnitude matches a recorded golden value
  within tolerance. **Action needed:** capture the golden value from a trusted
  run during 5b (or supply one) — please confirm the control point/frame and the
  acceptable tolerance.
- Keep CI-friendly: use 2-3 frames, small scalefactor, no video rendering, no
  large downloads.

---

## Proposed mechanics (pending confirmation)

- **Framework**: Catch2 v3 via FetchContent (recommended) **or** in-tree minimal
  asserts + CTest — your call.
- **Layout**: `test/unit/`, `test/integration/`, `test/e2e/`, registered with
  `enable_testing()` / `add_test` so `ctest` runs everything.
- **Fixtures**: reuse `test/examples/ohtcfrp`; generate synthetic shifted images
  in-memory for deterministic unit tests (no new data files).
- **Tolerances**: define a shared `kDispTol` / `kStrainTol`; sub-pixel DIC tests
  use loose tolerances (e.g. 0.05 px) to stay robust across platforms.
- **Possible small refactor**: extract `discover_frames` / `has_image_extension`
  / `natural_less` into a reusable header so 1c can test them without compiling
  all of `proxyncorr.cpp` — flag if acceptable.

---

### Please confirm before I proceed to 5b
1. Test framework: **Catch2 v3 (FetchContent)** or in-tree asserts?
2. OK to extract the folder-reader helpers into a small unit for testability?
3. The e2e golden control point + tolerance (or approve capturing one during 5b).
