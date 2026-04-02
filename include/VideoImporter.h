/*
 * File:   VideoImporter.h
 * Author: justin
 *
 * Extracted from Image2D.h during refactor to separate video import concerns
 * from image storage.
 */

#ifndef VIDEOIMPORTER_H
#define VIDEOIMPORTER_H

#include "Image2D.h"

#include <optional>
#include <string>
#include <vector>

namespace ncorr {

struct VideoImportParams {
    int frame_start = 1;
    int frame_end = -1;
    int frame_jump = 1;
    bool use_grayscale = true;
    bool use_red_channel = true;

    VideoImportParams() = default;
    VideoImportParams(int start, int end, int jump)
        : frame_start(start), frame_end(end), frame_jump(jump) {}
};

class VideoImporter {
    public:
        typedef std::ptrdiff_t difference_type;

        static std::vector<Image2D> import_video(
            const std::string& video_path,
            const VideoImportParams& params = VideoImportParams(),
            const FilterConfig& filter_config = FilterConfig(),
            const std::string& name_prefix = "frame");

        static std::vector<Image2D> import_video_to_files(
            const std::string& video_path,
            const std::string& output_dir,
            const VideoImportParams& params = VideoImportParams(),
            const FilterConfig& filter_config = FilterConfig(),
            const std::string& name_prefix = "frame");

        struct VideoInfo {
            int total_frames;
            int width;
            int height;
            double fps;
            std::string codec;
        };

        static std::optional<VideoInfo> get_video_info(const std::string& video_path);

    private:
        static cv::Mat extract_grayscale_frame(const cv::Mat& frame, bool use_red_channel);
};

} // namespace ncorr

#endif /* VIDEOIMPORTER_H */
