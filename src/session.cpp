/**
 * @file session.cpp
 * @brief Stub implementation of the in-memory NcorrSession API.
 *
 * STUB (newversion branch): the interface is final and behaves gracefully, but
 * the DIC computation is NOT yet wired up. Each entry point validates its inputs
 * and stores state; process_frame() returns an explicit "not implemented"
 * DICResult instead of fabricating numbers.
 *
 * TODO(newversion): implement the bodies by following the file-based pipeline in
 * test/src/proxyncorr.cpp:
 *   1. Wrap each ImageBuffer in a cv::Mat (no copy of caller memory needed for
 *      the wrap) and build an ncorr::Image2D via Image2D::from_mat(...).
 *   2. Build a ROI2D from the optional mask (or full-frame default).
 *   3. Construct a DIC_analysis_input { ref + def imgs, roi, scalefactor,
 *      interp, subregion, radius, num_threads, DIC_analysis_config, debug }.
 *   4. Run DIC_analysis(...) and copy the resulting Disp2D u/v arrays into the
 *      flat DICResult::u / DICResult::v vectors (NaN outside the ROI).
 */

#include "ncorr/session.h"

#include <iostream>
#include <stdexcept>

namespace ncorr {

// ---------------------------------------------------------------------------
// Private implementation (PIMPL). Kept minimal for the stub; will later hold
// ncorr::Image2D / ROI2D / cached engine state.
// ---------------------------------------------------------------------------
struct NcorrSession::Impl {
    SessionConfig config;

    bool has_reference = false;
    int ref_width = 0;
    int ref_height = 0;

    bool has_roi = false;

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
    // TODO(newversion): copy/convert ref into an ncorr::Image2D and cache its
    // grayscale representation. For now we only record geometry.
    impl_->ref_width = ref.width;
    impl_->ref_height = ref.height;
    impl_->has_reference = true;
}

void NcorrSession::set_roi(const ImageBuffer& roi_mask) {
    if (!roi_mask.valid()) {
        throw std::invalid_argument(
            "NcorrSession::set_roi: invalid ROI mask ImageBuffer.");
    }
    // TODO(newversion): convert the mask into an ncorr::ROI2D.
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

    // FIXME(newversion): in-memory DIC is not yet implemented. Behave gracefully
    // rather than returning fabricated displacement fields.
    result.width = impl_->ref_width;
    result.height = impl_->ref_height;
    result.message = "NcorrSession::process_frame not yet implemented";
    std::cerr << "[NcorrSession] process_frame not yet implemented" << std::endl;
    return result;
}

bool NcorrSession::has_reference() const {
    return impl_->has_reference;
}

}  // namespace ncorr
