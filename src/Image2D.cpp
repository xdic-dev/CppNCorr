/* 
 * File:   Image2D.cpp
 * Author: justin
 *
 * Created on January 28, 2015, 3:10 PM
 * Extended to support in-memory images, video import, and image filtering
 */

#include "Image2D.h"
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace ncorr {

// ============================================================================
// Image2D Implementation
// ============================================================================

// Static factory methods ----------------------------------------------------//
Image2D Image2D::load(std::ifstream &is) {
    // Form empty Image2D then fill in values in accordance to how they are saved
    Image2D img;
    
    // Load length
    difference_type length = 0;
    is.read(reinterpret_cast<char*>(&length), std::streamsize(sizeof(difference_type)));
    
    // Allocate new string
    img.filename_ptr = std::make_shared<std::string>(length,' ');
        
    // Read data
    is.read(const_cast<char*>(img.filename_ptr->c_str()), std::streamsize(length));
    
    // Loaded images are always file-based
    img.storage_mode = StorageMode::FILE_PATH;
    
    return img;
}

Image2D Image2D::from_mat(const cv::Mat& mat, const std::string& name, const FilterConfig& filter_config) {
    cv::Mat processed;
    if (!filter_config.empty()) {
        processed = ImageProcessor::apply_filters(mat, filter_config);
    } else {
        processed = mat.clone();
    }
    
    // Ensure grayscale
    cv::Mat gray;
    if (processed.channels() == 3) {
        cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = processed;
    }
    
    return Image2D(std::move(gray), name);
}

Image2D Image2D::from_file_filtered(const std::string& filename, const FilterConfig& filter_config) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::invalid_argument("Image file: " + filename + " cannot be found or read.");
    }
    
    if (!filter_config.empty()) {
        img = ImageProcessor::apply_filters(img, filter_config);
    }
    
    // Extract just the filename for naming
    std::filesystem::path p(filename);
    return Image2D(std::move(img), p.filename().string());
}

// Operations interface ------------------------------------------------------//
std::ostream& operator<<(std::ostream &os, const Image2D &img) {
    os << *img.filename_ptr;
    if (img.storage_mode == Image2D::StorageMode::IN_MEMORY_MAT) {
        os << " [in-memory cv::Mat]";
    } else if (img.storage_mode == Image2D::StorageMode::IN_MEMORY_GS) {
        os << " [in-memory Array2D]";
    }
    return os;
}

void imshow(const Image2D &img, Image2D::difference_type delay) { 
    imshow(img.get_gs(), delay);
}  

bool isequal(const Image2D &img1, const Image2D &img2) {
    // For in-memory images, compare data; for file-based, compare filenames
    if (img1.storage_mode != img2.storage_mode) {
        return false;
    }
    
    if (img1.storage_mode == Image2D::StorageMode::FILE_PATH) {
        return *img1.filename_ptr == *img2.filename_ptr;
    }
    
    // For in-memory, do a data comparison
    auto gs1 = img1.get_gs();
    auto gs2 = img2.get_gs();
    
    if (gs1.height() != gs2.height() || gs1.width() != gs2.width()) {
        return false;
    }
    
    for (Image2D::difference_type r = 0; r < gs1.height(); ++r) {
        for (Image2D::difference_type c = 0; c < gs1.width(); ++c) {
            if (std::abs(gs1(r, c) - gs2(r, c)) > 1e-10) {
                return false;
            }
        }
    }
    
    return true;
}

void save(const Image2D &img, std::ofstream &os) {    
    typedef Image2D::difference_type difference_type;
    
    // For in-memory images, we save the name but note that load() will fail
    // unless the data has been saved to disk
    difference_type length = img.filename_ptr->size();
    os.write(reinterpret_cast<const char*>(&length), std::streamsize(sizeof(difference_type)));
    os.write(img.filename_ptr->c_str(), std::streamsize(img.filename_ptr->size()));
}

// Access --------------------------------------------------------------------//
Array2D<double> Image2D::get_gs() const {
    switch (storage_mode) {
        case StorageMode::IN_MEMORY_GS: {
            // Already have grayscale data, return a copy
            return *gs_ptr;
        }
        
        case StorageMode::IN_MEMORY_MAT: {
            // Convert cv::Mat to Array2D
            return ImageProcessor::mat_to_array2d(*mat_ptr);
        }
        
        case StorageMode::FILE_PATH:
        default: {
            // Original behavior: load from file
            cv::Mat cv_img = cv::imread((*filename_ptr), cv::IMREAD_GRAYSCALE);
            if (!cv_img.data) {
                throw std::invalid_argument("Image file : " + *filename_ptr + " cannot be found or read.");
            }           

            // Images will be read as 8-bit grayscale values; convert these to double 
            // precision with values ranging from 0 - 1.
            Array2D<double> A(cv_img.rows, cv_img.cols);    
            double conversion = 1.0 / 255.0;
            for (difference_type p2 = 0; p2 < cv_img.rows; ++p2) {
                for (difference_type p1 = 0; p1 < cv_img.cols; ++p1) {
                    A(p2, p1) = cv_img.data[p1 + p2 * cv_img.cols] * conversion;
                }
            }           
            return A;
        }
    }
}

cv::Mat Image2D::get_mat() const {
    switch (storage_mode) {
        case StorageMode::IN_MEMORY_MAT: {
            return mat_ptr->clone();
        }
        
        case StorageMode::IN_MEMORY_GS: {
            return ImageProcessor::array2d_to_mat(*gs_ptr);
        }
        
        case StorageMode::FILE_PATH:
        default: {
            cv::Mat img = cv::imread(*filename_ptr, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                throw std::invalid_argument("Image file: " + *filename_ptr + " cannot be found or read.");
            }
            return img;
        }
    }
}

bool Image2D::save_to_file(const std::string& filename) const {
    cv::Mat mat = get_mat();
    return cv::imwrite(filename, mat);
}

// ============================================================================
// ImageProcessor Implementation
// ============================================================================

cv::Mat ImageProcessor::saturate(const cv::Mat& input, int level) {
    // Convert to grayscale if needed
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    
    // Ensure uint8 format
    if (gray.type() != CV_8UC1) {
        gray.convertTo(gray, CV_8UC1);
    }
    
    cv::Mat output = gray.clone();
    
    // Saturate: clip values above level to level
    // MATLAB satur.m: var(var > level) = level
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            uint8_t val = output.at<uint8_t>(y, x);
            if (val > level) {
                output.at<uint8_t>(y, x) = static_cast<uint8_t>(level);
            }
        }
    }
    
    return output;
}

std::vector<cv::Mat> ImageProcessor::saturate(const std::vector<cv::Mat>& input, int level) {
    std::vector<cv::Mat> output;
    output.reserve(input.size());
    
    for (const auto& img : input) {
        output.push_back(saturate(img, level));
    }
    
    return output;
}

cv::Mat ImageProcessor::apply_bandpass_filter(const cv::Mat& input, const std::vector<int>& params) {
    if (params.size() < 2) {
        throw std::invalid_argument("Bandpass filter requires two parameters: {r1, r2}");
    }
    
    // Convert to float for processing
    cv::Mat float_img;
    input.convertTo(float_img, CV_64F);
    
    double r1 = static_cast<double>(params[0]);
    double r2 = static_cast<double>(params[1]);
    
    int rows = float_img.rows;
    int cols = float_img.cols;
    
    // Create Gaussian bandpass filter (matching MATLAB bandpassfft.m)
    cv::Mat filter_mask(rows, cols, CV_64F);
    int cx = cols / 2;
    int cy = rows / 2;
    
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            double dx = x - cx;
            double dy = y - cy;
            double r_sq = dx * dx + dy * dy;
            
            // Gaussian bandpass: exp(-r^2/(2*r2^2)) * (1 - exp(-r^2/(2*r1^2)))
            double low_pass = std::exp(-r_sq / (2.0 * r2 * r2));
            double high_pass = (r1 != 0) ? (1.0 - std::exp(-r_sq / (2.0 * r1 * r1))) : 1.0;
            filter_mask.at<double>(y, x) = low_pass * high_pass;
        }
    }
    
    // Perform FFT
    cv::Mat planes[] = {float_img, cv::Mat::zeros(float_img.size(), CV_64F)};
    cv::Mat complex_img;
    cv::merge(planes, 2, complex_img);
    cv::dft(complex_img, complex_img);
    
    // fftshift the filter
    int cx2 = filter_mask.cols / 2;
    int cy2 = filter_mask.rows / 2;
    cv::Mat q0(filter_mask, cv::Rect(0, 0, cx2, cy2));
    cv::Mat q1(filter_mask, cv::Rect(cx2, 0, cols - cx2, cy2));
    cv::Mat q2(filter_mask, cv::Rect(0, cy2, cx2, rows - cy2));
    cv::Mat q3(filter_mask, cv::Rect(cx2, cy2, cols - cx2, rows - cy2));
    
    cv::Mat filter_shifted(rows, cols, CV_64F);
    q3.copyTo(filter_shifted(cv::Rect(0, 0, cols - cx2, rows - cy2)));
    q0.copyTo(filter_shifted(cv::Rect(cols - cx2, rows - cy2, cx2, cy2)));
    q1.copyTo(filter_shifted(cv::Rect(0, rows - cy2, cols - cx2, cy2)));
    q2.copyTo(filter_shifted(cv::Rect(cols - cx2, 0, cx2, rows - cy2)));
    
    // Apply filter in frequency domain
    cv::split(complex_img, planes);
    planes[0] = planes[0].mul(filter_shifted);
    planes[1] = planes[1].mul(filter_shifted);
    cv::merge(planes, 2, complex_img);
    
    // Inverse DFT
    cv::Mat filtered_img;
    cv::idft(complex_img, filtered_img, cv::DFT_SCALE);
    cv::split(filtered_img, planes);
    
    return planes[0];  // Return real part
}

std::vector<cv::Mat> ImageProcessor::apply_bandpass_filter(const std::vector<cv::Mat>& input, 
                                                           const std::vector<int>& params) {
    std::vector<cv::Mat> output;
    output.reserve(input.size());
    
    for (const auto& img : input) {
        output.push_back(apply_bandpass_filter(img, params));
    }
    
    return output;
}

cv::Mat ImageProcessor::apply_filters(const cv::Mat& input, const FilterConfig& config) {
    cv::Mat result = input.clone();
    
    // Apply saturation first (if enabled)
    if (config.has_saturation()) {
        result = saturate(result, config.saturation_level);
    }
    
    // Apply bandpass filter (if enabled)
    if (config.has_bandpass()) {
        result = apply_bandpass_filter(result, config.bandpass_params);
        // Convert back to 8-bit after bandpass (which produces float output)
        if (result.type() != CV_8UC1) {
            cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
            result.convertTo(result, CV_8UC1);
        }
    }
    
    return result;
}

std::vector<cv::Mat> ImageProcessor::apply_filters(const std::vector<cv::Mat>& input, const FilterConfig& config) {
    std::vector<cv::Mat> output;
    output.reserve(input.size());
    
    for (const auto& img : input) {
        output.push_back(apply_filters(img, config));
    }
    
    return output;
}

std::pair<std::vector<cv::Mat>, std::pair<double, double>> 
ImageProcessor::filter_like_ben(const std::vector<cv::Mat>& input,
                               const cv::Mat& mask,
                               const std::vector<int>& params,
                               const std::pair<double, double>* gs_boundaries) {
    
    std::vector<cv::Mat> filtered_images;
    std::pair<double, double> boundaries;
    
    // Apply bandpass filter to all images first
    for (const auto& img : input) {
        cv::Mat filtered = apply_bandpass_filter(img, params);
        filtered_images.push_back(filtered);
    }
    
    // Compute boundaries from FIRST FILTERED image using percentiles (5th, 95th) if not provided
    if (gs_boundaries == nullptr) {
        boundaries = compute_percentile_boundaries(filtered_images[0], mask, 5.0, 95.0);
        std::cout << "  Computed filter boundaries: [" << boundaries.first << ", " << boundaries.second << "]" << std::endl;
    } else {
        boundaries = *gs_boundaries;
    }
    
    // Normalize each filtered image using the boundaries
    for (size_t i = 0; i < filtered_images.size(); ++i) {
        filtered_images[i] = normalize_and_clamp(filtered_images[i], boundaries);
    }
    
    return {filtered_images, boundaries};
}

cv::Mat ImageProcessor::filter_like_ben_single(const cv::Mat& input,
                                               const cv::Mat& mask,
                                               const std::vector<int>& params,
                                               const std::pair<double, double>* gs_boundaries) {
    std::vector<cv::Mat> input_vec = {input};
    auto result = filter_like_ben(input_vec, mask, params, gs_boundaries);
    return result.first[0];
}

std::pair<double, double> ImageProcessor::compute_grayscale_boundaries(const cv::Mat& image,
                                                                       const cv::Mat& mask) {
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F);
    
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (mask.empty() || mask.at<uint8_t>(y, x) > 0) {
                double val = float_img.at<float>(y, x);
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }
    }
    
    return {min_val, max_val};
}

std::pair<double, double> ImageProcessor::compute_percentile_boundaries(const cv::Mat& image,
                                                                        const cv::Mat& mask,
                                                                        double lower_percentile,
                                                                        double upper_percentile) {
    // Collect all pixel values within masked region
    std::vector<double> values;
    
    // Convert to double if needed
    cv::Mat double_img;
    if (image.type() == CV_64F) {
        double_img = image;
    } else {
        image.convertTo(double_img, CV_64F);
    }
    
    for (int y = 0; y < double_img.rows; ++y) {
        for (int x = 0; x < double_img.cols; ++x) {
            if (mask.empty() || mask.at<uint8_t>(y, x) > 0) {
                values.push_back(double_img.at<double>(y, x));
            }
        }
    }
    
    if (values.empty()) {
        std::cerr << "Warning: No pixels in mask for percentile computation" << std::endl;
        return {0.0, 255.0};
    }
    
    // Sort values
    std::sort(values.begin(), values.end());
    
    // Compute percentile indices
    size_t lower_idx = static_cast<size_t>(values.size() * lower_percentile / 100.0);
    size_t upper_idx = static_cast<size_t>(values.size() * upper_percentile / 100.0);
    
    lower_idx = std::min(lower_idx, values.size() - 1);
    upper_idx = std::min(upper_idx, values.size() - 1);
    
    return {values[lower_idx], values[upper_idx]};
}

cv::Mat ImageProcessor::normalize_and_clamp(const cv::Mat& image,
                                            const std::pair<double, double>& boundaries) {
    cv::Mat double_img;
    if (image.type() == CV_64F) {
        double_img = image;
    } else {
        image.convertTo(double_img, CV_64F);
    }
    
    cv::Mat normalized;
    double range = boundaries.second - boundaries.first;
    if (range > 0) {
        normalized = (double_img - boundaries.first) / range;
        // Saturate (clamp) to [0, 1]
        cv::threshold(normalized, normalized, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(normalized, normalized, 0.0, 0.0, cv::THRESH_TOZERO);
    } else {
        normalized = cv::Mat::zeros(image.size(), CV_64F);
    }
    
    // Convert to uint8 [0, 255]
    cv::Mat output;
    normalized.convertTo(output, CV_8UC1, 255.0);
    
    return output;
}

cv::Mat ImageProcessor::array2d_to_mat(const Array2D<double>& arr) {
    cv::Mat mat(static_cast<int>(arr.height()), static_cast<int>(arr.width()), CV_8UC1);
    
    for (int r = 0; r < mat.rows; ++r) {
        for (int c = 0; c < mat.cols; ++c) {
            // Array2D stores values in 0-1 range, convert to 0-255
            double val = arr(r, c) * 255.0;
            val = std::max(0.0, std::min(255.0, val));
            mat.at<uint8_t>(r, c) = static_cast<uint8_t>(val);
        }
    }
    
    return mat;
}

Array2D<double> ImageProcessor::mat_to_array2d(const cv::Mat& mat) {
    cv::Mat gray;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = mat;
    }
    
    // Ensure 8-bit format
    if (gray.type() != CV_8UC1) {
        cv::Mat temp;
        gray.convertTo(temp, CV_8UC1);
        gray = temp;
    }
    
    Array2D<double> arr(gray.rows, gray.cols);
    double conversion = 1.0 / 255.0;
    
    for (int r = 0; r < gray.rows; ++r) {
        for (int c = 0; c < gray.cols; ++c) {
            arr(r, c) = gray.at<uint8_t>(r, c) * conversion;
        }
    }
    
    return arr;
}

// ============================================================================
// VideoImporter Implementation
// ============================================================================

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
    
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
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
        // OpenCV reads as BGR, so R is at index 2
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);
        gray = channels[2];  // R channel (BGR -> index 2)
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
    
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int frame_start = std::max(1, params.frame_start);
    int frame_end = (params.frame_end < 0) ? total_frames : std::min(params.frame_end, total_frames);
    int frame_jump = std::max(1, params.frame_jump);
    
    std::cout << "Importing video: " << video_path << std::endl;
    std::cout << "  Total frames: " << total_frames << std::endl;
    std::cout << "  Import range: " << frame_start << " to " << frame_end << " (step " << frame_jump << ")" << std::endl;
    
    int imported_count = 0;
    for (int f = frame_start; f <= frame_end; f += frame_jump) {
        cap.set(cv::CAP_PROP_POS_FRAMES, f - 1);  // 0-indexed
        
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "  Warning: Failed to read frame " << f << std::endl;
            break;
        }
        
        // Extract grayscale
        cv::Mat gray = extract_grayscale_frame(frame, params.use_red_channel);
        
        // Apply filters if configured
        if (!filter_config.empty()) {
            gray = ImageProcessor::apply_filters(gray, filter_config);
        }
        
        // Create frame name
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
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video file: " << video_path << std::endl;
        return images;
    }
    
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int frame_start = std::max(1, params.frame_start);
    int frame_end = (params.frame_end < 0) ? total_frames : std::min(params.frame_end, total_frames);
    int frame_jump = std::max(1, params.frame_jump);
    
    std::cout << "Importing video to files: " << video_path << std::endl;
    std::cout << "  Output directory: " << output_dir << std::endl;
    std::cout << "  Total frames: " << total_frames << std::endl;
    std::cout << "  Import range: " << frame_start << " to " << frame_end << " (step " << frame_jump << ")" << std::endl;
    
    int imported_count = 0;
    for (int f = frame_start; f <= frame_end; f += frame_jump) {
        cap.set(cv::CAP_PROP_POS_FRAMES, f - 1);  // 0-indexed
        
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "  Warning: Failed to read frame " << f << std::endl;
            break;
        }
        
        // Extract grayscale
        cv::Mat gray = extract_grayscale_frame(frame, params.use_red_channel);
        
        // Apply filters if configured
        if (!filter_config.empty()) {
            gray = ImageProcessor::apply_filters(gray, filter_config);
        }
        
        // Create file path
        std::ostringstream filepath;
        filepath << output_dir << "/" << name_prefix << "_" << std::setw(6) << std::setfill('0') << f << ".png";
        
        // Save to file
        cv::imwrite(filepath.str(), gray);
        
        // Create Image2D pointing to the saved file
        images.emplace_back(filepath.str());
        imported_count++;
    }
    
    cap.release();
    std::cout << "  Imported and saved " << imported_count << " frames" << std::endl;
    
    return images;
}

}