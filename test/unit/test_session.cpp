/**
 * @file test_session.cpp
 * @brief Tests for the in-memory NcorrSession API.
 *
 * The contract cases (tagged "[session]") pin input validation, the
 * reference-required precondition, and geometry checks; they do not run DIC and
 * are labelled "unit" by the engine target's discovery.
 *
 * The parity case (tagged "[integration]") runs a real in-memory DIC through
 * NcorrSession::process_frame and asserts it matches the file/Image2D +
 * DIC_analysis path on the same reference/deformed pair, bit-for-bit within a
 * tight tolerance (it is the same computation).
 *
 * This translation unit links the full ncorr engine (see tests.cmake) so it can
 * include ncorr.h and OpenCV.
 */

#include <catch2/catch_test_macros.hpp>

#include "ncorr/session.h"
#include "ncorr.h"

#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

// Build a small valid grayscale buffer kept alive by the returned vector.
ncorr::ImageBuffer make_buffer(std::vector<std::uint8_t>& storage, int w, int h, int ch = 1) {
    storage.assign(static_cast<size_t>(w) * h * ch, 0);
    return ncorr::ImageBuffer(storage.data(), w, h, ch);
}

}  // namespace

TEST_CASE("imagebuffer_valid", "[session]") {
    std::vector<std::uint8_t> storage;
    auto good = make_buffer(storage, 4, 3);
    CHECK(good.valid());
    CHECK(good.size_bytes() == 12u);

    ncorr::ImageBuffer empty;
    CHECK_FALSE(empty.valid());

    std::uint8_t one = 0;
    ncorr::ImageBuffer zero_dim(&one, 0, 5, 1);
    CHECK_FALSE(zero_dim.valid());

    ncorr::ImageBuffer null_data(nullptr, 4, 4, 1);
    CHECK_FALSE(null_data.valid());
}

TEST_CASE("session_requires_reference", "[session]") {
    ncorr::NcorrSession session;
    CHECK_FALSE(session.has_reference());
    std::vector<std::uint8_t> storage;
    auto def = make_buffer(storage, 8, 8);
    CHECK_THROWS_AS(session.process_frame(def), std::logic_error);
}

TEST_CASE("session_rejects_invalid_reference", "[session]") {
    ncorr::NcorrSession session;
    ncorr::ImageBuffer bad;  // null / zero dims
    CHECK_THROWS_AS(session.set_reference(bad), std::invalid_argument);
    CHECK_FALSE(session.has_reference());
}

TEST_CASE("session_geometry_mismatch", "[session]") {
    ncorr::NcorrSession session;
    std::vector<std::uint8_t> ref_storage;
    auto ref = make_buffer(ref_storage, 8, 8);
    session.set_reference(ref);
    CHECK(session.has_reference());

    std::vector<std::uint8_t> def_storage;
    auto def = make_buffer(def_storage, 8, 9);  // different height
    auto result = session.process_frame(def);
    CHECK_FALSE(result.valid);
    CHECK_FALSE(result.message.empty());
}

// ---------------------------------------------------------------------------
// Real in-memory DIC parity test: NcorrSession::process_frame must reproduce
// the file/Image2D + DIC_analysis path exactly on the ohtcfrp fixture.
// NCORR_FIXTURE_DIR is injected as a compile definition (see tests.cmake).
// ---------------------------------------------------------------------------
TEST_CASE("session_dic_parity", "[integration]") {
    using namespace ncorr;

    const std::string fixture_dir = NCORR_FIXTURE_DIR;
    const std::string ref_path = fixture_dir + "/ohtcfrp_00.png";
    const std::string def_path = fixture_dir + "/ohtcfrp_01.png";
    const std::string roi_path = fixture_dir + "/roi.png";

    // Load both frames and the ROI mask through OpenCV (BGR, as the session
    // expects an interleaved 8-bit buffer matching OpenCV's default layout).
    cv::Mat ref_bgr = cv::imread(ref_path, cv::IMREAD_COLOR);
    cv::Mat def_bgr = cv::imread(def_path, cv::IMREAD_COLOR);
    cv::Mat roi_bgr = cv::imread(roi_path, cv::IMREAD_COLOR);
    REQUIRE_FALSE(ref_bgr.empty());
    REQUIRE_FALSE(def_bgr.empty());
    REQUIRE_FALSE(roi_bgr.empty());
    REQUIRE(ref_bgr.isContinuous());
    REQUIRE(def_bgr.isContinuous());
    REQUIRE(roi_bgr.isContinuous());

    // Common config / ROI for both paths.
    SessionConfig cfg;  // defaults
    ROI2D roi(Image2D(roi_path).get_gs() > 0.5);

    // (a) Direct file/Image2D + DIC_analysis path.
    std::vector<Image2D> imgs{Image2D(ref_path), Image2D(def_path)};
    DIC_analysis_input in(imgs, roi, cfg.scalefactor, INTERP::QUINTIC_BSPLINE_PRECOMPUTE,
                          SUBREGION::CIRCLE, cfg.subregion_radius, cfg.num_threads,
                          DIC_analysis_config::NO_UPDATE, cfg.debug);
    DIC_analysis_output out = DIC_analysis(in);
    REQUIRE(out.disps.size() == 1);
    const Disp2D& disp = out.disps.front();
    const Array2D<double>& u_arr = disp.get_u().get_array();
    const Array2D<double>& v_arr = disp.get_v().get_array();
    const Array2D<bool>& mask = disp.get_roi().get_mask();

    // (b) In-memory NcorrSession path on the same data.
    NcorrSession session(cfg);
    session.set_reference(
        ImageBuffer(ref_bgr.data, ref_bgr.cols, ref_bgr.rows, ref_bgr.channels()));
    session.set_roi(ImageBuffer(roi_bgr.data, roi_bgr.cols, roi_bgr.rows, roi_bgr.channels()));
    DICResult result = session.process_frame(
        ImageBuffer(def_bgr.data, def_bgr.cols, def_bgr.rows, def_bgr.channels()));

    REQUIRE(result.valid);
    REQUIRE(result.message.empty());
    REQUIRE(result.width == u_arr.width());
    REQUIRE(result.height == u_arr.height());
    REQUIRE(result.u.size() == static_cast<size_t>(result.width) * result.height);

    // In-ROI u/v must match the direct DIC_analysis arrays within tolerance.
    const double tol = 1e-6;
    int checked = 0;
    for (int i = 0; i < result.height; ++i) {
        for (int j = 0; j < result.width; ++j) {
            const size_t idx = static_cast<size_t>(i) * result.width + j;
            if (i < mask.height() && j < mask.width() && mask(i, j)) {
                REQUIRE(std::isfinite(result.u[idx]));
                REQUIRE(std::isfinite(result.v[idx]));
                CHECK(std::abs(result.u[idx] - u_arr(i, j)) <= tol);
                CHECK(std::abs(result.v[idx] - v_arr(i, j)) <= tol);
                ++checked;
            }
        }
    }
    CHECK(checked > 0);  // the ROI must actually contain analysed points
}
