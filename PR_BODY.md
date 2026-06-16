# feat: CppNCorr newversion — in-memory API, params, tests, docs, CI

> **Status:** prepared locally on the `newversion` branch. Not yet pushed and no
> PR has been opened (per request). This file is the proposed PR description for
> when the branch is published.

## Summary

This branch hardens and extends CppNCorr (the standalone C++ 2D DIC engine that
is also vendored as a submodule of CPPxDIC) without changing the behaviour of the
existing image-folder DIC pipeline. Work spans repository hygiene, CLI auditing,
input handling, a parameter override chain, a real test suite, documentation,
deployment assets, and CI.

### Section 1 — Repository hygiene
- Enforced **C++17** project-wide and per-target; documented every CMake option,
  flag, dependency, and target in `CMakeOptions.md`.
- Commit: `chore: enforce C++17, document CMake options`.

### Section 2 — Outputs / CLI (the DIC executable is `proxyncorr`)
- Audited the `proxyncorr` CLI and added post-DIC output flags:
  `--export-video`, `--export-strains`, `--change-perspective <mode>`. Flags whose
  backing code exists are wired to existing behaviour; modes without dedicated
  code print a clear "not yet implemented" notice and exit cleanly. (`ncorr` is
  the static **library** target; `proxyncorr` is the actual DIC tool.)
- Commit: `feat(ncorr): audit CLI; stub missing post-DIC output flags`.

### Section 3 — Inputs
- **3a (folder reader):** hardened edge cases (sorted/natural filename order,
  supported formats, reserved `roi.png`/`ref.png`, hidden files, sub-directories,
  empty/missing folders) and documented the naming convention.
  - Commit: `fix(input/images): harden image-folder reader edge cases`.
  - Follow-up refactor (for testability): extracted `discover_frames`,
    `has_image_extension`, `natural_less` into the reusable header-only
    `include/ncorr/frame_reader.h`; `proxyncorr.cpp` now includes it. No
    behavioural change.
    - Commit: `refactor(input/images): extract frame-reader helpers into reusable header`.
- **3b (in-memory input):** added the `NcorrSession` API
  (`include/ncorr/session.h` + `src/session.cpp`) with `ImageBuffer`, `DICResult`,
  `SessionConfig`, and a PIMPL session class. **Interface is final; bodies are
  stubs** (validate inputs, return a graceful "not yet implemented" result).
  - Commit: `feat(input/memory): add in-memory NcorrSession API (stub)`.

### Section 4 — Parameters
- Implemented the three-tier override chain **CLI > config file > compiled
  defaults**: `include/ncorr/config.h` (`struct Config` + `load_config_file`),
  header-only INI parser `include/ncorr/ini.h`, and `config/default.cfg` whose
  keys match `Config` field names exactly.
- Commit: `feat(params): implement CLI > config-file > compiled-defaults override chain`.

### Section 5 — Tests
- **5a:** `TESTS_PLAN.md` (committed earlier, plan approved).
  - Commit: `docs(tests): add TESTS_PLAN.md — awaiting confirmation before implementation`.
- **5b:** implemented the confirmed plan with **Catch2 v3 via FetchContent**
  (approved new dependency, matches the repo's existing FetchContent pattern).
  - **Unit tests** (`test/unit/`, target `ncorr_unit_tests`, dependency-light):
    INI parser, Config + key/field parity, frame-reader helpers, and the
    `NcorrSession` stub contract — **24 test cases / 130 assertions**.
  - **Integration tests** (`test/integration/`, target `ncorr_engine_tests`):
    load→DIC, change_perspective+set_units, DIC→strain, and config-driven input
    on the `ohtcfrp` fixture — **4 test cases**.
  - **End-to-end tests** (`test/e2e/`): full pipeline on `ohtcfrp` (reference + 2
    frames) with a captured golden baseline — **2 test cases**.
  - Registered with CTest (`catch_discover_tests`, labelled `unit` / `integration`
    / `e2e`). **All 30 CTest tests pass locally.**
  - Commits: `test(unit): ...`, `test(integration): ...`, `test(e2e): ...`.

### Section 6 — Documentation
- `docs/quickstart.md`, `docs/developer_guide.md`, `docs/user_guide.md`.
- Commits: `docs: add quickstart guide`, `docs: add developer guide`,
  `docs: add user guide`.

### Section 7 — Parallel / cluster / Docker
- New `deploy/` directory (no such assets existed before): `Dockerfile`
  (builds `proxyncorr` on Ubuntu 24.04 with apt deps), `run_slurm.sh`
  (single-node OpenMP sbatch script), `deploy/README.md`, and a repo-root
  `.dockerignore`. The Dockerfile's native build sequence was verified to build
  `proxyncorr` cleanly (the Docker daemon was unavailable to run the container
  build itself in the prep environment).
- Commit: `chore(deploy): organise parallel/cluster/Docker scripts under deploy/`.

### Section 8 — CI/CD
- `.github/workflows/ci.yml` (push + PR to `main`/`newversion`): `build`,
  `test` (unit + integration on `ohtcfrp` via ctest), and `lint`
  (`clang-format --dry-run` on added files). Added a `.clang-format`. Fast: GCC
  on ubuntu-latest, apt deps, no GPU/large downloads; slow e2e excluded from CI.
- Commit: `ci: add GitHub Actions workflow for build, test, lint`.

## Stubs and TODO markers

- **In-memory DIC** (`src/session.cpp`): `process_frame` returns
  `"NcorrSession::process_frame not yet implemented"`.
  - `src/session.cpp:10` `TODO(newversion):` implement bodies following the
    file-based pipeline.
  - `src/session.cpp:58` `TODO(newversion):` convert reference to `Image2D`.
  - `src/session.cpp:70` `TODO(newversion):` convert mask to `ROI2D`.
  - `src/session.cpp:94` `FIXME(newversion):` in-memory DIC not implemented.
  - When implemented, update the `session_stub_contract` unit test and add the
    deferred `pipeline_session_matches_folder` integration test from
    `TESTS_PLAN.md`.
- **CLI post-DIC stubs** (`test/src/proxyncorr.cpp`):
  - `--change-perspective` non-`eulerian` modes print "not yet implemented"
    (`FIXME(newversion):` near line 619).
  - `--export-strains` relies on the JSON/binary bundle; `TODO(newversion):`
    (near line 608) to add a dedicated standalone strain export format.
  - Two `FIXME(newversion):` notes document the `DIC_analysis_sequential`
    overload-disambiguation already handled explicitly.
- **e2e golden value**: baked in as a regression guard
  (`kGoldenMeanDispMag = 0.180423 mm`, tolerance `±0.02 mm`). Re-capture with
  `NCORR_E2E_CAPTURE=1`.

## Items that were confirmed / need awareness

- **Test framework:** Catch2 v3 via FetchContent — confirmed.
- **Folder-reader extraction** into `frame_reader.h` — confirmed.
- **e2e golden control point + tolerance:** captured during 5b (mean in-ROI
  |displacement| on the final frame). Reviewers may wish to confirm the chosen
  metric and ±0.02 mm band.

## Critical: test infra is guarded; parent build is unaffected

All test infrastructure (Catch2 FetchContent, `enable_testing()`, the test
targets, and CTest registration) is gated behind
`if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)` in the root `CMakeLists.txt`
(with a manual `PROJECT_IS_TOP_LEVEL` fallback for CMake < 3.21, and
`BUILD_TESTING` defaulting ON only when top-level). Verified by configuring a
fake parent project that does `add_subdirectory(...)`: CppNCorr printed
`tests skipped (PROJECT_IS_TOP_LEVEL=OFF, BUILD_TESTING=OFF)`, no Catch2 was
fetched, no test targets were generated, and the `ncorr` library still built.

## How to review

1. **Read the plan & docs:** `TESTS_PLAN.md`, then `docs/quickstart.md`,
   `docs/developer_guide.md`, `docs/user_guide.md`.
2. **Confirm the parent is unaffected:** configure a throwaway parent project
   that `add_subdirectory()`s this repo and confirm the "tests skipped" message
   and that no Catch2/test targets appear.
3. **Build + test top-level:**
   ```bash
   cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
   cmake --build build -j
   cd build && ctest --output-on-failure -L unit          # fast
   ctest --output-on-failure -L integration --timeout 1200 # slower DIC
   ctest --output-on-failure -L e2e --timeout 1200         # slowest
   ```
4. **Spot-check the override chain:** `include/ncorr/config.h`,
   `include/ncorr/ini.h`, `config/default.cfg`, and the overlay wiring in
   `test/src/proxyncorr.cpp`.
5. **Review the API surface:** `include/ncorr/session.h` (final interface) and
   `src/session.cpp` (stub bodies + TODO markers).
6. **Deploy/CI:** `deploy/Dockerfile`, `deploy/run_slurm.sh`, `deploy/README.md`,
   `.github/workflows/ci.yml`, `.clang-format`.

## Commit list (`git log --oneline main..newversion`)

```
8273d66 ci: add GitHub Actions workflow for build, test, lint
cfc0640 chore(deploy): organise parallel/cluster/Docker scripts under deploy/
87585d4 docs: add user guide
4af29db docs: add developer guide
fb7eb2a docs: add quickstart guide
417b89c test(e2e): add ohtcfrp end-to-end regression guard with golden baseline
7e5f89b test(integration): add DIC pipeline-stage tests on ohtcfrp fixture
6634af9 test(unit): add Catch2 v3 suite for ini/config/frame_reader/session
ea1d73e refactor(input/images): extract frame-reader helpers into reusable header
b2bf99c docs(tests): add TESTS_PLAN.md — awaiting confirmation before implementation
9c31435 feat(params): implement CLI > config-file > compiled-defaults override chain
f241076 feat(input/memory): add in-memory NcorrSession API (stub)
6aafea2 fix(input/images): harden image-folder reader edge cases
4bc4ee1 feat(ncorr): audit CLI; stub missing post-DIC output flags
251b538 chore: enforce C++17, document CMake options
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)
