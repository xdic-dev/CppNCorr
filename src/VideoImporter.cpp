#include "VideoImporter.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace ncorr {

std::optional<VideoImporter::VideoInfo> VideoImporter::get_video_info(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        return std::nullopt;
    }

    VideoInfo info;
    info.total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    info.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    info.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    info.fps = cap.get(cv::CAP_PROP_FPS);

    const int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    info.codec = std::string(1, fourcc & 255) +
                 std::string(1, (fourcc >> 8) & 255) +
                 std::string(1, (fourcc >> 16) & 255) +
                 std::string(1, (fourcc >> 24) & 255);

    cap.release();
    return info;
}

cv::Mat VideoImporter::extract_grayscale_frame(const cv::Mat& frame, bool use_red_channel) {
    cv::Mat gray;

    if (frame.channels() == 1) {
        gray = frame.clone();
    } else if (use_red_channel) {
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);
        gray = channels[2];
    } else {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    }

    return gray;
}

std::vector<Image2D> VideoImporter::import_video(
    const std::string& video_path,
    const VideoImportParams& params,
    const FilterConfig& filter_config,
    const std::string& name_prefix) {

    std::vector<Image2D> images;

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video file: " << video_path << std::endl;
        return images;
    }

    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const int frame_start = std::max(1, params.frame_start);
    const int frame_end = (params.frame_end < 0) ? total_frames : std::min(params.frame_end, total_frames);
    const int frame_jump = std::max(1, params.frame_jump);

    std::cout << "Importing video: " << video_path << std::endl;
    std::cout << "  Total frames: " << total_frames << std::endl;
    std::cout << "  Import range: " << frame_start << " to " << frame_end << " (step " << frame_jump << ")" << std::endl;

    int imported_count = 0;
    for (int f = frame_start; f <= frame_end; f += frame_jump) {
        cap.set(cv::CAP_PROP_POS_FRAMES, f - 1);

        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "  Warning: Failed to read frame " << f << std::endl;
            break;
        }

        cv::Mat gray = extract_grayscale_frame(frame, params.use_red_channel);
        if (!filter_config.empty()) {
            gray = ImageProcessor::apply_filters(gray, filter_config);
        }

        std::ostringstream name;
        name << name_prefix << "_" << std::setw(6) << std::setfill('0') << f;

        images.emplace_back(std::move(gray), name.str());
        imported_count++;
    }

    cap.release();
    std::cout << "  Imported " << imported_count << " frames" << std::endl;

    return images;
}

std::vector<Image2D> VideoImporter::import_video_to_files(
    const std::string& video_path,
    const std::string& output_dir,
    const VideoImportParams& params,
    const FilterConfig& filter_config,
    const std::string& name_prefix) {

    std::vector<Image2D> images;
    std::filesystem::create_directories(output_dir);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video file: " << video_path << std::endl;
        return images;
    }

    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const int frame_start = std::max(1, params.frame_start);
    const int frame_end = (params.frame_end < 0) ? total_frames : std::min(params.frame_end, total_frames);
    const int frame_jump = std::max(1, params.frame_jump);

    std::cout << "Importing video to files: " << video_path << std::endl;
    std::cout << "  Output directory: " << output_dir << std::endl;
    std::cout << "  Total frames: " << total_frames << std::endl;
    std::cout << "  Import range: " << frame_start << " to " << frame_end << " (step " << frame_jump << ")" << std::endl;

    int imported_count = 0;
    for (int f = frame_start; f <= frame_end; f += frame_jump) {
        cap.set(cv::CAP_PROP_POS_FRAMES, f - 1);

        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "  Warning: Failed to read frame " << f << std::endl;
            break;
        }

        cv::Mat gray = extract_grayscale_frame(frame, params.use_red_channel);
        if (!filter_config.empty()) {
            gray = ImageProcessor::apply_filters(gray, filter_config);
        }

        std::ostringstream filepath;
        filepath << output_dir << "/" << name_prefix << "_" << std::setw(6) << std::setfill('0') << f << ".png";

        cv::imwrite(filepath.str(), gray);
        images.emplace_back(filepath.str());
        imported_count++;
    }

    cap.release();
    std::cout << "  Imported and saved " << imported_count << " frames" << std::endl;

    return images;
}

} // namespace ncorr
