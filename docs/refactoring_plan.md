# CppNCorr Refactoring Plan — Round 2 (`ja/refactoring/2`)

Status: **proposal** — this document is the working plan for the second refactoring
round. It supersedes the stale `ja/refactoring/1` branch (PR #10), which forked
before the newversion restructure (#12) and no longer applies to `main`.

## 1. Where round 1 stands

`ja/refactoring/1` (9 commits, forked at `74a6deb`) pursued two themes:

1. **Extraction by algorithm** — seed analysis + serialization + visualization
   split out of `ncorr.cpp` (5815 → 3386 lines), `ImageProcessor` /
   `VideoImporter` modules, opt-in diagnostics (`NCORR_ENABLE_DIAGNOSTICS`).
2. **Extraction by dependency** — `Array2D.h` split into OpenCV / FFTW / BLAS /
   linear-solver headers, plus an umbrella `include/ncorr/` forwarding-header tree.

Since then `main` absorbed the newversion (#12: in-memory API, params, tests,
docs, CI), the exact quintic interpolation / fake-jump fix, the recursive
B-spline filter, multi-reference chaining, and leveled logging (#18). Verdict
per theme:

| Round-1 theme | Verdict |
|---|---|
| ImageProcessor / VideoImporter / FilterConfig | **Already integrated** — now static members / factories in `Image2D.h` (in-memory modes, `from_mat()`, `from_file_filtered()`) |
| Diagnostics env-var system | **Superseded** — `ncorr/log.h` leveled logging (#18) is strictly better |
| Seed analysis module | **Still open, high value** — must be redone against main (both `matlab_*` and `exact_matlab_*` variants now exist) |
| Serialization module | **Still open, moderate value** |
| Array2D dependency split | **Defer** — header-only templates; split buys no compile-time or link benefit today; revisit only if cuNCorr needs a subset |
| Umbrella `ncorr/core/*.hpp` forwarding headers | **Drop** — main's pragmatic layout (flat legacy headers + `include/ncorr/` for new modules) is fine |

The branch diff no longer applies (`ncorr.cpp` was rewritten underneath it);
nothing is worth cherry-picking as code. The ideas below are re-implemented
against current `main`.

## 2. Current pain points on `main`

- `src/ncorr.cpp` — **5815 lines**: DIC engine (~1258 ln), I/O + visualization
  (~650 ln), seed-based parallel DIC (~340 ln), chain composition
  (~790 ln: `add_with_rois` + `exact_*` + legacy), helpers/optimizers (~2780 ln).
- `include/Array2D.h` — **4241 lines**, monolithic but well-sectioned.
- **Duplication**: `matlab_DIC_analysis_impl()` (ncorr.cpp:5085) vs
  `exact_matlab_DIC_analysis_impl()` (ncorr.cpp:5684) share ~65% of their
  bodies; they differ only in the chain-composition call
  (`add_with_rois` vs `exact_add_with_rois`).
- **Missing test seams**: seed scheduling and chain composition are private
  helpers, only covered end-to-end (`test/src/ncorr_test_chain_seam.cpp`).
- **Unclear API surface**: `DIC_analysis()`, `matlab_DIC_analysis_*()` and
  `exact_matlab_DIC_analysis_*()` are all public; optimizers
  (`subregion_nloptimizer`, `disp_nloptimizer`) live in `ncorr.h` without a
  documented contract. This matters now that CPPxDIC treats CppNCorr as one of
  two interchangeable DIC engines (alongside cuNCorr).

## 3. Plan

### Phase 1 — extract by algorithm (high value)

1. **Seed analysis module** (`include/ncorr_seed.hpp` + `src/ncorr_seed.cpp`)
   - Move `matlab_compute_seed_segment()` (ncorr.cpp:4877), `matlab_run_segment_dic()`
     (ncorr.cpp:4997), the `matlab_seed_*` helpers (ncorr.cpp:4804–4875) and the
     `matlab_seed_segment` struct behind a small API
     (`compute_seed_segment()`, `run_segment_dic()`).
   - Must serve both the canonical and `exact_*` analysis paths.
   - Add unit tests for seed scheduling (currently untestable in isolation).
   - Est. ~300 lines moved; ncorr.cpp shrinks accordingly.

2. **Exact chain composition module** (`include/ncorr_exact_chain.hpp` + `.cpp`)
   - Move the `exact_detail` namespace (ncorr.cpp:5409–5551),
     `exact_add_with_rois()` (ncorr.cpp:5556) and
     `exact_matlab_DIC_analysis_*` (ncorr.cpp:5684–5815) into a dedicated
     `ncorr::exact` module. Makes explicit that the MATLAB byte-for-byte path
     is an *alternative strategy*, not part of the core engine.
   - Existing regression test (`ncorr_test_chain_seam.cpp`) guards the move.
   - Est. ~400 lines moved.

### Phase 2 — de-duplicate (clarity)

3. **Unify `matlab_DIC_analysis` variants** via a chain-composition strategy
   parameter (template or callable): one `matlab_DIC_analysis_impl(input,
   parallel, chain_strategy)` with canonical and exact strategies. Removes the
   ~65% duplicated body; the substitution point becomes explicit.

4. **Document/tighten the public API** — clarify in `ncorr.h` + developer guide
   which entry points are the supported engine surface (what CPPxDIC's adapter
   calls) vs internal; move internal optimizer declarations out of the public
   header if feasible.

### Phase 3 — optional / deferred

5. **I/O module extraction** (`DIC_analysis_step_data` save/load,
   Disp2D/ROI2D/Strain2D stream operators, ~600 lines) — only if format
   versioning or isolated I/O tests become necessary.
6. **Array2D split** — deferred; revisit if the cuNCorr port needs a
   CPU-reference subset of Array2D without FFTW/OpenCV.

### Guardrails

- One extraction per PR-sized commit; each keeps the full test suite green.
- No behavior changes: extractions are pure moves + renames; the strategy
  unification (item 3) must be provably equivalent (existing regression tests
  + chain-seam test).
- Public header layout stays flat for legacy types; new modules go under
  `include/ncorr/` per the newversion convention.
