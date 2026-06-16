# ============================================================================
# tests.cmake — Catch2-based test suite for CppNCorr (section 5b)
# ----------------------------------------------------------------------------
# This file is included ONLY when CppNCorr is the top-level project and testing
# is enabled (see the guard in the root CMakeLists.txt). It must never run when
# the parent CPPxDIC project does add_subdirectory(Tools/CppNCorr), so that the
# Catch2 FetchContent and CTest targets do not leak into the parent build.
#
# Test tiers:
#   - ncorr_unit_tests   : dependency-light (config/ini/frame_reader/session).
#   - ncorr_engine_tests : links the full ncorr library (integration + e2e).
# ============================================================================

include(FetchContent)

# Keep test-tree build artifacts (Catch2 static libs, test executables) inside
# the build directory instead of the source-tree lib/ that the root CMakeLists
# uses for libncorr.a. This avoids polluting the source tree with Catch2.a etc.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---- Catch2 v3 via FetchContent (matches the repo's FetchContent pattern) ---
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# Path constants injected into the tests as compile definitions.
set(NCORR_DEFAULT_CFG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/config/default.cfg")
set(NCORR_FIXTURE_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/test/examples/ohtcfrp/images")

# ----------------------------------------------------------------------------
# 1. Lightweight unit tests — no OpenCV/FFTW/SuiteSparse/BLAS required.
#    Compiles only the small, dependency-free sources directly.
# ----------------------------------------------------------------------------
add_executable(ncorr_unit_tests
    test/unit/test_ini.cpp
    test/unit/test_config.cpp
    test/unit/test_frame_reader.cpp
    test/unit/test_session.cpp
    src/config.cpp
    src/session.cpp
)
target_include_directories(ncorr_unit_tests PRIVATE include)
target_compile_definitions(ncorr_unit_tests PRIVATE
    NCORR_DEFAULT_CFG="${NCORR_DEFAULT_CFG_PATH}")
target_link_libraries(ncorr_unit_tests PRIVATE Catch2::Catch2WithMain)
# Label every discovered case "unit" so CI can select it with `ctest -L unit`.
catch_discover_tests(ncorr_unit_tests PROPERTIES LABELS "unit")

# ----------------------------------------------------------------------------
# 2. Engine tests (integration + e2e) — link the full ncorr library, which the
#    root CMakeLists has already defined as target `ncorr`. The ncorr static
#    library does not carry its transitive system deps, so the executable must
#    link OpenCV/FFTW/SuiteSparse/BLAS itself (resolved here the same way the
#    legacy test/CMakeLists.txt does, honouring the FetchContent fallbacks).
# ----------------------------------------------------------------------------
find_or_fetch_suitesparse()
find_or_fetch_openblas()

add_executable(ncorr_engine_tests
    test/integration/test_pipeline.cpp
    test/e2e/test_e2e.cpp
)
target_include_directories(ncorr_engine_tests PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(ncorr_engine_tests PRIVATE
    NCORR_FIXTURE_DIR="${NCORR_FIXTURE_DIR_PATH}")
target_link_libraries(ncorr_engine_tests PRIVATE ncorr Catch2::Catch2WithMain ${OpenCV_LIBS})
if(FFTW_LIBRARY)
    target_link_libraries(ncorr_engine_tests PRIVATE ${FFTW_LIBRARY})
endif()
if(SUITESPARSE_LIBRARIES)
    target_link_libraries(ncorr_engine_tests PRIVATE ${SUITESPARSE_LIBRARIES})
endif()
if(LAPACK_LIBRARIES OR BLAS_LIBRARIES)
    target_link_libraries(ncorr_engine_tests PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif()
find_package(Threads QUIET)
if(Threads_FOUND)
    target_link_libraries(ncorr_engine_tests PRIVATE Threads::Threads)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(ncorr_engine_tests PRIVATE OpenMP::OpenMP_CXX)
endif()
# Register the integration and e2e cases with distinct CTest labels so CI can
# run them selectively (e.g. `ctest -L integration`, `ctest -L e2e`). Each
# invocation filters by Catch2 tag via TEST_SPEC.
catch_discover_tests(ncorr_engine_tests
    TEST_SPEC "[integration]"
    TEST_PREFIX "integration:"
    PROPERTIES LABELS "integration")
catch_discover_tests(ncorr_engine_tests
    TEST_SPEC "[e2e]"
    TEST_PREFIX "e2e:"
    PROPERTIES LABELS "e2e")
