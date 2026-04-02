/*
 * File:   ImageProcessor.h
 * Author: justin
 *
 * Extracted from Image2D.h during refactor to separate image processing
 * utilities from image storage.
 */

#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include "Array2D.h"
#include <opencv2/opencv.hpp>

#include <set>
#include <utility>
#include <vector>

namespace ncorr {

enum class FilterType {
    SATURATION,
    BANDPASS
};

struct FilterConfig {
    std::set<FilterType> filters;
    int saturation_level = 200;
    std::vector<int> bandpass_params = {5, 50};

    FilterConfig() = default;
    FilterConfig(std::set<FilterType> f) : filters(std::move(f)) {}
    FilterConfig(std::set<FilterType> f, int sat_level) : filters(std::move(f)), saturation_level(sat_level) {}
    FilterConfig(std::set<FilterType> f, int sat_level, std::vector<int> bp)
        : filters(std::move(f)), saturation_level(sat_level), bandpass_params(std::move(bp)) {}

    bool has_saturation() const { return filters.count(FilterType::SATURATION) > 0; }
    bool has_bandpass() const { return filters.count(FilterType::BANDPASS) > 0; }
    bool empty() const { return filters.empty(); }
};

class ImageProcessor {
    public:
        typedef std::ptrdiff_t difference_type;

        static cv::Mat saturate(const cv::Mat& input, int level = 200);
        static cv::Mat apply_bandpass_filter(const cv::Mat& input, const std::vector<int>& params = {5, 50});
        static cv::Mat apply_filters(const cv::Mat& input, const FilterConfig& config);

        static std::vector<cv::Mat> saturate(const std::vector<cv::Mat>& input, int level = 200);
        static std::vector<cv::Mat> apply_bandpass_filter(const std::vector<cv::Mat>& input,
                                                          const std::vector<int>& params = {5, 50});
        static std::vector<cv::Mat> apply_filters(const std::vector<cv::Mat>& input, const FilterConfig& config);

        static std::pair<std::vector<cv::Mat>, std::pair<double, double>>
        filter_like_ben(const std::vector<cv::Mat>& input,
                        const cv::Mat& mask,
                        const std::vector<int>& params = {5, 50},
                        const std::pair<double, double>* gs_boundaries = nullptr);

        static cv::Mat filter_like_ben_single(const cv::Mat& input,
                                              const cv::Mat& mask,
                                              const std::vector<int>& params = {5, 50},
                                              const std::pair<double, double>* gs_boundaries = nullptr);

        static std::pair<double, double> compute_grayscale_boundaries(const cv::Mat& image, const cv::Mat& mask);
        static std::pair<double, double> compute_percentile_boundaries(const cv::Mat& image,
                                                                       const cv::Mat& mask,
                                                                       double lower_pct = 5.0,
                                                                       double upper_pct = 95.0);
        static cv::Mat normalize_and_clamp(const cv::Mat& image, const std::pair<double, double>& boundaries);

        static cv::Mat array2d_to_mat(const Array2D<double>& arr);
        static Array2D<double> mat_to_array2d(const cv::Mat& mat);
};

} // namespace ncorr

#endif /* IMAGEPROCESSOR_H */
