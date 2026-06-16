/**
 * @file session.cpp
 * @brief In-memory NcorrSession implementation backed by the DIC engine.
 *
 * Mirrors the file-based pipeline in test/src/proxyncorr.cpp but drives the DIC
 * engine directly from caller-provided in-memory image buffers:
 *   1. Wrap each ImageBuffer in an owning cv::Mat and build an ncorr::Image2D.
 *   2. Build a ROI2D from the optional mask (or a full-frame default).
 *   3. Construct a DIC_analysis_input { ref + def imgs, roi, scalefactor,
 *      interp, subregion, radius, num_threads, DIC_analysis_config, debug }.
 *   4. Run DIC_analysis(...) and copy the resulting Disp2D u/v/cc arrays into
 *      the flat DICResult vectors (NaN outside the ROI).
 *
 * The returned fields are the native Lagrangian displacements in pixels on the
 * reduced analysis grid (their own dims, not necessarily the input image size).
 */

#include "ncorr/session.h"

#include "ncorr.h"

#include <opencv2/opencv.hpp>

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ncorr {

namespace {

// Wrap a (row-major, interleaved 8-bit) ImageBuffer into an owning cv::Mat. The
// clone() decouples the cv::Mat from the caller's storage so the session can
// outlive the original buffer.
cv::Mat to_owning_mat(const ImageBuffer& b) {
    return cv::Mat(b.height, b.width, CV_8UC(b.channels),
                   const_cast<std::uint8_t*>(b.data))
        .clone();
}

}  // namespace

// ---------------------------------------------------------------------------
// Private implementation (PIMPL). Holds the cached reference Image2D, its
// geometry, and the optional ROI2D.
// ---------------------------------------------------------------------------
struct NcorrSession::Impl {
    SessionConfig config;

    bool has_reference = false;
    Image2D ref_img;
    int ref_width = 0;
    int ref_height = 0;

    bool has_roi = false;
    ROI2D roi;

    explicit Impl(const SessionConfig& cfg) : config(cfg) {}
};

NcorrSession::NcorrSession(const SessionConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

NcorrSession::~NcorrSession() = default;

NcorrSession::NcorrSession(NcorrSession&&) noexcept = default;
NcorrSession& NcorrSession::operator=(NcorrSession&&) noexcept = default;

void NcorrSession::set_reference(const ImageBuffer& ref) {
    if (!ref.valid()) {
        throw std::invalid_argument(
            "NcorrSession::set_reference: invalid ImageBuffer (null data or "
            "non-positive dimensions).");
    }
    impl_->ref_img = Image2D(to_owning_mat(ref));
    impl_->ref_width = ref.width;
    impl_->ref_height = ref.height;
    impl_->has_reference = true;
    // A new reference invalidates any previously set ROI.
    impl_->has_roi = false;
}

void NcorrSession::set_roi(const ImageBuffer& roi_mask) {
    if (!roi_mask.valid()) {
        throw std::invalid_argument(
            "NcorrSession::set_roi: invalid ROI mask ImageBuffer.");
    }
    if (impl_->has_reference &&
        (roi_mask.width != impl_->ref_width ||
         roi_mask.height != impl_->ref_height)) {
        throw std::invalid_argument(
            "NcorrSession::set_roi: ROI mask geometry does not match the "
            "reference frame.");
    }
    Array2D<double> gs = Image2D(to_owning_mat(roi_mask)).get_gs();
    impl_->roi = ROI2D(gs > 0.5);
    impl_->has_roi = true;
}

DICResult NcorrSession::process_frame(const ImageBuffer& def) {
    if (!impl_->has_reference) {
        throw std::logic_error(
            "NcorrSession::process_frame: no reference frame set; call "
            "set_reference() first.");
    }

    DICResult result;
    result.valid = false;

    if (!def.valid()) {
        result.message = "process_frame: invalid deformed ImageBuffer.";
        return result;
    }
    if (def.width != impl_->ref_width || def.height != impl_->ref_height) {
        result.message =
            "process_frame: deformed frame geometry does not match reference.";
        return result;
    }

    try {
        Image2D def_img(to_owning_mat(def));

        ROI2D roi = impl_->has_roi
                        ? impl_->roi
                        : ROI2D(Array2D<bool>(impl_->ref_height,
                                              impl_->ref_width, true));

        std::vector<Image2D> imgs{impl_->ref_img, def_img};

        DIC_analysis_input in(
            imgs, roi, impl_->config.scalefactor,
            INTERP::QUINTIC_BSPLINE_PRECOMPUTE, SUBREGION::CIRCLE,
            impl_->config.subregion_radius, impl_->config.num_threads,
            DIC_analysis_config::NO_UPDATE, impl_->config.debug);

        DIC_analysis_output out = DIC_analysis(in);
        if (out.disps.empty()) {
            result.message =
                "process_frame: DIC produced no displacement fields.";
            return result;
        }

        const Disp2D& disp = out.disps.front();
        const Array2D<double>& u_arr = disp.get_u().get_array();
        const Array2D<double>& v_arr = disp.get_v().get_array();
        const Array2D<double>& cc_arr = disp.get_cc().get_array();
        const Array2D<bool>& mask = disp.get_roi().get_mask();

        const int h = u_arr.height();
        const int w = u_arr.width();
        result.width = w;
        result.height = h;

        const double nan = std::numeric_limits<double>::quiet_NaN();
        const std::size_t n = static_cast<std::size_t>(w) * h;
        result.u.assign(n, nan);
        result.v.assign(n, nan);
        result.corrcoef.assign(n, nan);

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                if (i < mask.height() && j < mask.width() && mask(i, j)) {
                    const std::size_t idx =
                        static_cast<std::size_t>(i) * w + j;
                    if (i < u_arr.height() && j < u_arr.width())
                        result.u[idx] = u_arr(i, j);
                    if (i < v_arr.height() && j < v_arr.width())
                        result.v[idx] = v_arr(i, j);
                    if (i < cc_arr.height() && j < cc_arr.width())
                        result.corrcoef[idx] = cc_arr(i, j);
                }
            }
        }

        result.valid = true;
        result.message.clear();
        return result;
    } catch (const std::exception& e) {
        result.valid = false;
        result.message =
            std::string("process_frame: DIC failed: ") + e.what();
        return result;
    }
}

bool NcorrSession::has_reference() const {
    return impl_->has_reference;
}

}  // namespace ncorr
