/**
 * @file test_e2e.cpp
 * @brief End-to-end regression test on the ohtcfrp fixture.
 *
 * Runs the full file-based pipeline in-process (load -> DIC -> change_perspective
 * -> set_units -> strain_analysis) on the reference frame plus a small number of
 * deformed frames, then asserts:
 *   - the run completes and produces one displacement/strain field per frame;
 *   - a recorded GOLDEN baseline (mean in-ROI |displacement| on the last frame)
 *     is reproduced within a tolerance band.
 *
 * The golden value was captured from a known-good run during section 5b on this
 * fixture with the parameters below. It is a regression guard, not a physical
 * ground truth; the tolerance band is intentionally generous so the test stays
 * robust across platforms/compilers.
 *
 * Set the environment variable NCORR_E2E_CAPTURE=1 to print the freshly computed
 * statistics (used to (re)capture the baseline) instead of failing.
 *
 * NCORR_FIXTURE_DIR is injected as a compile definition.
 */

#include <catch2/catch_test_macros.hpp>

#include "ncorr.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace ncorr;

namespace {

const std::string kFixtureDir = NCORR_FIXTURE_DIR;

// E2E run parameters (kept small for CI). Reference + two deformed frames.
constexpr int kScaleFactor = 1;
constexpr int kRadius = 30;
constexpr int kStrainRadius = 5;
constexpr double kUnitsPerPixel = 0.2;

// ---- GOLDEN BASELINE (captured during section 5b) -------------------------//
// Mean in-ROI displacement magnitude (sqrt(u^2 + v^2), in the configured units)
// over the final analysed frame. See NCORR_E2E_CAPTURE above to re-capture.
// Captured 2026-06-16 on macOS/AppleClang, OpenCV 4.13, single-threaded; value
// was identical across 3 consecutive runs. Tolerance is a generous +/-0.02 mm
// (~11%) so the regression guard stays robust across platforms/compilers/thread
// counts while still catching a real behavioural regression.
constexpr double kGoldenMeanDispMag = 0.180423;  // mm (units_per_pixel = 0.2)
constexpr double kGoldenTolerance = 0.02;        // mm
constexpr bool kGoldenRecorded = true;           // baseline baked in

// Mean magnitude of finite (u,v) inside the ROI for the given displacement field.
double mean_disp_magnitude(const Disp2D& disp) {
    const Array2D<double>& u = disp.get_u().get_array();
    const Array2D<double>& v = disp.get_v().get_array();
    double sum = 0.0;
    long n = 0;
    for (int i = 0; i < u.height(); ++i) {
        for (int j = 0; j < u.width(); ++j) {
            double uu = u(i, j), vv = v(i, j);
            if (std::isfinite(uu) && std::isfinite(vv)) {
                sum += std::sqrt(uu * uu + vv * vv);
                ++n;
            }
        }
    }
    return n > 0 ? sum / static_cast<double>(n) : 0.0;
}

}  // namespace

TEST_CASE("e2e_ohtcfrp_smoke", "[e2e]") {
    std::vector<Image2D> imgs;
    imgs.push_back(Image2D(kFixtureDir + "/ohtcfrp_00.png"));  // reference
    imgs.push_back(Image2D(kFixtureDir + "/ohtcfrp_01.png"));
    imgs.push_back(Image2D(kFixtureDir + "/ohtcfrp_02.png"));

    ROI2D roi(Image2D(kFixtureDir + "/roi.png").get_gs() > 0.5);

    DIC_analysis_input input(imgs, roi, kScaleFactor, INTERP::QUINTIC_BSPLINE_PRECOMPUTE,
                             SUBREGION::CIRCLE, kRadius, /*num_threads=*/1,
                             DIC_analysis_config::NO_UPDATE, /*debug=*/false);

    DIC_analysis_output output = DIC_analysis(input);
    output = change_perspective(output, INTERP::CUBIC_KEYS);
    output = set_units(output, "mm", kUnitsPerPixel);

    strain_analysis_input strain_in(input, output, SUBREGION::CIRCLE, kStrainRadius);
    strain_analysis_output strain_out = strain_analysis(strain_in);

    // Two deformed frames -> two displacement and two strain fields.
    REQUIRE(output.disps.size() == 2);
    REQUIRE(strain_out.strains.size() == 2);

    double mag = mean_disp_magnitude(output.disps.back());
    INFO("mean in-ROI |displacement| (last frame) = " << mag);
    CHECK(std::isfinite(mag));
    CHECK(mag >= 0.0);

    if (std::getenv("NCORR_E2E_CAPTURE")) {
        std::cout << "[E2E CAPTURE] mean_disp_magnitude(last) = " << mag << std::endl;
    }
}

TEST_CASE("e2e_ohtcfrp_known_value", "[e2e]") {
    if (!kGoldenRecorded) {
        SUCCEED(
            "golden baseline not yet recorded; run with NCORR_E2E_CAPTURE=1 "
            "to capture it (see e2e_ohtcfrp_smoke).");
        return;
    }

    std::vector<Image2D> imgs;
    imgs.push_back(Image2D(kFixtureDir + "/ohtcfrp_00.png"));
    imgs.push_back(Image2D(kFixtureDir + "/ohtcfrp_01.png"));
    imgs.push_back(Image2D(kFixtureDir + "/ohtcfrp_02.png"));

    ROI2D roi(Image2D(kFixtureDir + "/roi.png").get_gs() > 0.5);

    DIC_analysis_input input(imgs, roi, kScaleFactor, INTERP::QUINTIC_BSPLINE_PRECOMPUTE,
                             SUBREGION::CIRCLE, kRadius, /*num_threads=*/1,
                             DIC_analysis_config::NO_UPDATE, /*debug=*/false);

    DIC_analysis_output output = DIC_analysis(input);
    output = change_perspective(output, INTERP::CUBIC_KEYS);
    output = set_units(output, "mm", kUnitsPerPixel);

    double mag = mean_disp_magnitude(output.disps.back());
    INFO("computed = " << mag << ", golden = " << kGoldenMeanDispMag
                       << ", tol = " << kGoldenTolerance);
    CHECK(std::abs(mag - kGoldenMeanDispMag) <= kGoldenTolerance);
}
