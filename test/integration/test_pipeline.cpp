/**
 * @file test_pipeline.cpp
 * @brief Integration tests for the DIC pipeline stages on the ohtcfrp fixture.
 *
 * These tests link the full ncorr engine. To stay CI-friendly they use the
 * reference frame plus a single deformed frame at a modest scale factor and
 * larger subregion radius, asserting structural properties (dimensions,
 * finiteness inside the ROI) rather than exact values. Exact-value regression
 * is covered by the e2e test.
 *
 * NCORR_FIXTURE_DIR is injected as a compile definition pointing at
 * test/examples/ohtcfrp/images.
 */

#include <catch2/catch_test_macros.hpp>

#include "ncorr.h"

#include <cmath>
#include <string>
#include <vector>

using namespace ncorr;

namespace {

const std::string kFixtureDir = NCORR_FIXTURE_DIR;

// Build a DIC input from the reference + first deformed frame of the fixture.
DIC_analysis_input make_small_input() {
    std::vector<Image2D> imgs;
    imgs.push_back(Image2D(kFixtureDir + "/ohtcfrp_00.png"));  // reference
    imgs.push_back(Image2D(kFixtureDir + "/ohtcfrp_01.png"));  // one deformed frame

    ROI2D roi(Image2D(kFixtureDir + "/roi.png").get_gs() > 0.5);

    // scalefactor 1, large radius -> few seeds, fast, deterministic enough.
    return DIC_analysis_input(imgs, roi, /*scalefactor=*/1,
                              INTERP::QUINTIC_BSPLINE_PRECOMPUTE,
                              SUBREGION::CIRCLE, /*radius=*/30,
                              /*num_threads=*/1, DIC_analysis_config::NO_UPDATE,
                              /*debug=*/false);
}

// Count finite (non-NaN) values inside the displacement array.
int count_finite(const Array2D<double>& a) {
    int n = 0;
    for (int i = 0; i < a.height(); ++i)
        for (int j = 0; j < a.width(); ++j)
            if (std::isfinite(a(i, j))) ++n;
    return n;
}

}  // namespace

TEST_CASE("pipeline_load_to_dic", "[integration][pipeline]") {
    DIC_analysis_input input = make_small_input();
    DIC_analysis_output output = DIC_analysis(input);

    REQUIRE(output.disps.size() == 1);  // one deformed frame analysed
    const Disp2D& disp = output.disps.front();
    CHECK(disp.get_u().get_array().height() > 0);
    CHECK(disp.get_u().get_array().width() > 0);
    // At least some points inside the ROI must have a finite displacement.
    CHECK(count_finite(disp.get_u().get_array()) > 0);
    CHECK(count_finite(disp.get_v().get_array()) > 0);
}

TEST_CASE("pipeline_perspective_units", "[integration][pipeline]") {
    DIC_analysis_input input = make_small_input();
    DIC_analysis_output output = DIC_analysis(input);

    int w0 = output.disps.front().get_u().get_array().width();
    int h0 = output.disps.front().get_u().get_array().height();

    output = change_perspective(output, INTERP::CUBIC_KEYS);
    output = set_units(output, "mm", 0.2);

    REQUIRE(output.disps.size() == 1);
    CHECK(output.units == "mm");
    CHECK(output.units_per_pixel == 0.2);
    // Perspective change preserves field dimensions.
    CHECK(output.disps.front().get_u().get_array().width() == w0);
    CHECK(output.disps.front().get_u().get_array().height() == h0);
}

TEST_CASE("pipeline_dic_to_strain", "[integration][pipeline]") {
    DIC_analysis_input input = make_small_input();
    DIC_analysis_output output = DIC_analysis(input);
    output = change_perspective(output, INTERP::CUBIC_KEYS);
    output = set_units(output, "mm", 0.2);

    strain_analysis_input strain_in(input, output, SUBREGION::CIRCLE, /*r=*/5);
    strain_analysis_output strain_out = strain_analysis(strain_in);

    REQUIRE(strain_out.strains.size() == 1);
    const Strain2D& strain = strain_out.strains.front();
    CHECK(strain.get_exx().get_array().height() > 0);
    CHECK(strain.get_eyy().get_array().width() > 0);
    CHECK(count_finite(strain.get_exx().get_array()) > 0);
}

TEST_CASE("pipeline_config_drives_run", "[integration][pipeline]") {
    // Confirm the tuneable parameters set on DIC_analysis_input are reflected in
    // the constructed input (the config tier feeds these fields in the driver).
    DIC_analysis_input input = make_small_input();
    CHECK(input.scalefactor == 1);
    CHECK(input.subregion_type == SUBREGION::CIRCLE);
    CHECK(input.r == 30);
    CHECK(input.interp_type == INTERP::QUINTIC_BSPLINE_PRECOMPUTE);
}
