#include "ImageProcessor.h"
#include "ncorr/internal/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ncorr {

cv::Mat ImageProcessor::saturate(const cv::Mat& input, int level) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }

    if (gray.type() != CV_8UC1) {
        gray.convertTo(gray, CV_8UC1);
    }

    cv::Mat output = gray.clone();
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            const uint8_t val = output.at<uint8_t>(y, x);
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

    cv::Mat float_img;
    input.convertTo(float_img, CV_64F);

    const double r1 = static_cast<double>(params[0]);
    const double r2 = static_cast<double>(params[1]);

    const int rows = float_img.rows;
    const int cols = float_img.cols;

    cv::Mat filter_mask(rows, cols, CV_64F);
    const int cx = cols / 2;
    const int cy = rows / 2;

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const double dx = x - cx;
            const double dy = y - cy;
            const double r_sq = dx * dx + dy * dy;
            const double low_pass = std::exp(-r_sq / (2.0 * r2 * r2));
            const double high_pass = (r1 != 0) ? (1.0 - std::exp(-r_sq / (2.0 * r1 * r1))) : 1.0;
            filter_mask.at<double>(y, x) = low_pass * high_pass;
        }
    }

    cv::Mat planes[] = {float_img, cv::Mat::zeros(float_img.size(), CV_64F)};
    cv::Mat complex_img;
    cv::merge(planes, 2, complex_img);
    cv::dft(complex_img, complex_img);

    const int cx2 = filter_mask.cols / 2;
    const int cy2 = filter_mask.rows / 2;
    cv::Mat q0(filter_mask, cv::Rect(0, 0, cx2, cy2));
    cv::Mat q1(filter_mask, cv::Rect(cx2, 0, cols - cx2, cy2));
    cv::Mat q2(filter_mask, cv::Rect(0, cy2, cx2, rows - cy2));
    cv::Mat q3(filter_mask, cv::Rect(cx2, cy2, cols - cx2, rows - cy2));

    cv::Mat filter_shifted(rows, cols, CV_64F);
    q3.copyTo(filter_shifted(cv::Rect(0, 0, cols - cx2, rows - cy2)));
    q0.copyTo(filter_shifted(cv::Rect(cols - cx2, rows - cy2, cx2, cy2)));
    q1.copyTo(filter_shifted(cv::Rect(0, rows - cy2, cols - cx2, cy2)));
    q2.copyTo(filter_shifted(cv::Rect(cols - cx2, 0, cx2, rows - cy2)));

    cv::split(complex_img, planes);
    planes[0] = planes[0].mul(filter_shifted);
    planes[1] = planes[1].mul(filter_shifted);
    cv::merge(planes, 2, complex_img);

    cv::Mat filtered_img;
    cv::idft(complex_img, filtered_img, cv::DFT_SCALE);
    cv::split(filtered_img, planes);

    return planes[0];
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

    if (config.has_saturation()) {
        result = saturate(result, config.saturation_level);
    }

    if (config.has_bandpass()) {
        result = apply_bandpass_filter(result, config.bandpass_params);
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

    for (const auto& img : input) {
        filtered_images.push_back(apply_bandpass_filter(img, params));
    }

    if (gs_boundaries == nullptr) {
        boundaries = compute_percentile_boundaries(filtered_images[0], mask, 5.0, 95.0);
        details::diagnostic_log(std::cout,
                                "  Computed filter boundaries: [",
                                boundaries.first,
                                ", ",
                                boundaries.second,
                                "]");
    } else {
        boundaries = *gs_boundaries;
    }

    for (auto& filtered_image : filtered_images) {
        filtered_image = normalize_and_clamp(filtered_image, boundaries);
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
                const double val = float_img.at<float>(y, x);
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
    std::vector<double> values;

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
        details::diagnostic_log(std::cerr, "Warning: No pixels in mask for percentile computation");
        return {0.0, 255.0};
    }

    std::sort(values.begin(), values.end());

    std::size_t lower_idx = static_cast<std::size_t>(values.size() * lower_percentile / 100.0);
    std::size_t upper_idx = static_cast<std::size_t>(values.size() * upper_percentile / 100.0);

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
    const double range = boundaries.second - boundaries.first;
    if (range > 0) {
        normalized = (double_img - boundaries.first) / range;
        cv::threshold(normalized, normalized, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(normalized, normalized, 0.0, 0.0, cv::THRESH_TOZERO);
    } else {
        normalized = cv::Mat::zeros(image.size(), CV_64F);
    }

    cv::Mat output;
    normalized.convertTo(output, CV_8UC1, 255.0);
    return output;
}

cv::Mat ImageProcessor::array2d_to_mat(const Array2D<double>& arr) {
    cv::Mat mat(static_cast<int>(arr.height()), static_cast<int>(arr.width()), CV_8UC1);

    for (int r = 0; r < mat.rows; ++r) {
        for (int c = 0; c < mat.cols; ++c) {
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

    if (gray.type() != CV_8UC1) {
        cv::Mat temp;
        gray.convertTo(temp, CV_8UC1);
        gray = temp;
    }

    Array2D<double> arr(gray.rows, gray.cols);
    const double conversion = 1.0 / 255.0;

    for (int r = 0; r < gray.rows; ++r) {
        for (int c = 0; c < gray.cols; ++c) {
            arr(r, c) = gray.at<uint8_t>(r, c) * conversion;
        }
    }

    return arr;
}

} // namespace ncorr
