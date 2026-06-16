#pragma once
/**
 * @file session.h
 * @brief In-memory DIC session API for CppNCorr.
 *
 * This header defines a small, dependency-light interface that lets a caller
 * (for example CPPxDIC's @c proxyncorr target) push raw image buffers directly
 * to the ncorr DIC engine without first writing frames to disk.
 *
 * Design goals:
 *  - The public surface depends only on standard-library types and the thin
 *    @ref ncorr::ImageBuffer struct, so callers need not include the heavy
 *    internal ncorr headers (Array2D, Disp2D, ...) just to drive a session.
 *  - The usage pattern mirrors the file-based pipeline: set a reference frame
 *    once, then process one or more deformed frames.
 *
 * @note This is a STUB in the @c newversion branch: the interface is final but
 *       the bodies are not yet wired to the DIC engine. See session.cpp and the
 *       @c TODO markers. The image-folder reader path in @c proxyncorr.cpp
 *       (discover_frames + DIC_analysis) is the model implementation to follow.
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ncorr {

/**
 * @brief Thin, non-owning view over a raw image in memory.
 *
 * Wraps a contiguous pixel buffer plus its geometry. The buffer is interpreted
 * as row-major, with @c channels interleaved per pixel (e.g. BGR for 3-channel
 * data, matching OpenCV's default layout). 8-bit unsigned samples are assumed.
 *
 * The struct does NOT own @c data; the caller must keep the underlying storage
 * alive for the duration of any call that receives the ImageBuffer.
 */
struct ImageBuffer {
    /// Pointer to the first byte of the (row-major, interleaved) pixel data.
    const std::uint8_t* data = nullptr;
    /// Image width in pixels.
    int width = 0;
    /// Image height in pixels.
    int height = 0;
    /// Number of interleaved channels per pixel (1 = grayscale, 3 = BGR, ...).
    int channels = 1;

    /// Default-construct an empty (invalid) buffer.
    ImageBuffer() = default;

    /**
     * @brief Construct an image buffer view.
     * @param data_     Pointer to row-major interleaved 8-bit pixel data.
     * @param width_    Width in pixels.
     * @param height_   Height in pixels.
     * @param channels_ Channels per pixel (default 1 = grayscale).
     */
    ImageBuffer(const std::uint8_t* data_, int width_, int height_, int channels_ = 1)
        : data(data_), width(width_), height(height_), channels(channels_) {}

    /// @return true if the buffer is non-null and has positive dimensions.
    bool valid() const {
        return data != nullptr && width > 0 && height > 0 && channels > 0;
    }

    /// @return total number of bytes the buffer is expected to span.
    std::size_t size_bytes() const {
        return static_cast<std::size_t>(width) * height * channels;
    }
};

/**
 * @brief Result of running DIC on a single deformed frame.
 *
 * Displacement fields are returned as flat row-major arrays of length
 * @c width * @c height. Points outside the analysed region of interest are set
 * to NaN. The fields are deliberately plain @c std::vector<double> so callers do
 * not need any internal ncorr types to consume the result.
 */
struct DICResult {
    /// Width of the displacement fields (pixels).
    int width = 0;
    /// Height of the displacement fields (pixels).
    int height = 0;
    /// Horizontal displacement field (u), row-major, size width*height. NaN outside ROI.
    std::vector<double> u;
    /// Vertical displacement field (v), row-major, size width*height. NaN outside ROI.
    std::vector<double> v;
    /// Per-point correlation coefficient, row-major, size width*height (optional; may be empty).
    std::vector<double> corrcoef;
    /// True if the frame was processed successfully.
    bool valid = false;
    /// Human-readable status / error message (empty on success).
    std::string message;
};

/**
 * @brief Configuration for an in-memory DIC session.
 *
 * Mirrors the tuneable parameters of the file-based pipeline. Field names match
 * the compiled defaults in @ref Config (see config.h) so the two stay in sync.
 */
struct SessionConfig {
    /// Pyramid scale factor (downsampling level for the seed search).
    int scalefactor = 3;
    /// Subregion (correlation window) radius in pixels.
    int subregion_radius = 20;
    /// Strain subregion radius in pixels.
    int strain_radius = 5;
    /// Number of worker threads for parallel analysis.
    int num_threads = 4;
    /// Enable verbose debug output from the engine.
    bool debug = false;
};

/**
 * @brief Drives an in-memory Digital Image Correlation session.
 *
 * Typical lifecycle:
 * @code
 *   ncorr::NcorrSession session(cfg);
 *   session.set_reference(ref_buffer);          // once
 *   auto r1 = session.process_frame(def1);      // per deformed frame
 *   auto r2 = session.process_frame(def2);
 * @endcode
 *
 * The session owns a private implementation (PIMPL) so that this header stays
 * free of heavy internal ncorr includes.
 *
 * @warning STUB: methods are declared and behave gracefully (see session.cpp)
 *          but DIC is not yet computed. Implement following the file-based path
 *          in proxyncorr.cpp (Image2D::from_mat -> DIC_analysis_input -> DIC_analysis).
 */
class NcorrSession {
public:
    /**
     * @brief Construct a session with the given configuration.
     * @param config Tuneable DIC parameters (defaults match Config).
     */
    explicit NcorrSession(const SessionConfig& config = SessionConfig());

    /// Destructor (defined in session.cpp because of the PIMPL).
    ~NcorrSession();

    // Movable, non-copyable (owns engine state).
    NcorrSession(NcorrSession&&) noexcept;
    NcorrSession& operator=(NcorrSession&&) noexcept;
    NcorrSession(const NcorrSession&) = delete;
    NcorrSession& operator=(const NcorrSession&) = delete;

    /**
     * @brief Set the reference (undeformed) frame. Call once before processing.
     *
     * The pixel data is copied into the session, so @p ref need not outlive the
     * call. Calling again replaces the reference and invalidates prior state.
     *
     * @param ref Reference image buffer.
     * @throws std::invalid_argument if @p ref is not valid().
     */
    void set_reference(const ImageBuffer& ref);

    /**
     * @brief Optionally supply a region-of-interest mask.
     *
     * The mask is a single-channel buffer the same size as the reference; any
     * non-zero pixel marks a point to analyse. If never called, the whole frame
     * is analysed.
     *
     * @param roi_mask Single-channel ROI mask buffer.
     */
    void set_roi(const ImageBuffer& roi_mask);

    /**
     * @brief Push a deformed frame and run DIC against the reference.
     *
     * @param def Deformed image buffer (same geometry as the reference).
     * @return A DICResult; on failure @c valid is false and @c message explains why.
     * @throws std::logic_error if no reference has been set.
     */
    DICResult process_frame(const ImageBuffer& def);

    /// @return true once a valid reference frame has been set.
    bool has_reference() const;

private:
    // PIMPL: hides internal ncorr engine types from this public header.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ncorr
