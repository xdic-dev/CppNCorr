/* 
 * File:   Image2D.h
 * Author: justin
 *
 * Created on January 28, 2015, 3:10 PM
 * Extended to support in-memory images and video import
 */

#ifndef IMAGE2D_H
#define	IMAGE2D_H

#include "Array2D.h"
#include <opencv2/opencv.hpp>
#include <set>
#include <optional>

namespace ncorr {     

// Forward declarations
class ImageProcessor;
class VideoImporter;

// Filter types for image preprocessing
enum class FilterType {
    SATURATION,    // Clip pixel values above a threshold
    BANDPASS       // Gaussian bandpass filter (Ben's method)
};

// Configuration for image filters
struct FilterConfig {
    std::set<FilterType> filters;           // Which filters to apply
    int saturation_level = 200;             // Level for saturation filter (0-255)
    std::vector<int> bandpass_params = {5, 50};  // {r1, r2} for bandpass filter
    
    FilterConfig() = default;
    FilterConfig(std::set<FilterType> f) : filters(std::move(f)) {}
    FilterConfig(std::set<FilterType> f, int sat_level) : filters(std::move(f)), saturation_level(sat_level) {}
    FilterConfig(std::set<FilterType> f, int sat_level, std::vector<int> bp) 
        : filters(std::move(f)), saturation_level(sat_level), bandpass_params(std::move(bp)) {}
    
    bool has_saturation() const { return filters.count(FilterType::SATURATION) > 0; }
    bool has_bandpass() const { return filters.count(FilterType::BANDPASS) > 0; }
    bool empty() const { return filters.empty(); }
};

// Video import parameters
struct VideoImportParams {
    int frame_start = 1;       // 1-indexed start frame
    int frame_end = -1;        // -1 means all frames
    int frame_jump = 1;        // Frame step/skip
    bool use_grayscale = true; // Convert to grayscale
    bool use_red_channel = true; // If color, extract R channel (for DIC speckle patterns)
    
    VideoImportParams() = default;
    VideoImportParams(int start, int end, int jump) 
        : frame_start(start), frame_end(end), frame_jump(jump) {}
};

class Image2D final { 
    public:                 
        typedef std::ptrdiff_t                                  difference_type; 
        
        // Storage mode enum
        enum class StorageMode {
            FILE_PATH,      // Traditional: load from file on get_gs()
            IN_MEMORY_MAT,  // cv::Mat stored in memory
            IN_MEMORY_GS    // Array2D<double> grayscale already computed
        };
        
        // Rule of 5 and destructor ------------------------------------------//        
        Image2D() = default;
        Image2D(const Image2D&) = default;
        Image2D(Image2D&&) noexcept = default;
        Image2D& operator=(const Image2D&) = default;
        Image2D& operator=(Image2D&&) = default; 
        ~Image2D() noexcept = default;
        
        // Additional Constructors -------------------------------------------//
        // File path constructor (original behavior)
        Image2D(std::string filename) 
            : filename_ptr(std::make_shared<std::string>(std::move(filename))), 
              storage_mode(StorageMode::FILE_PATH) { }
        
        // In-memory cv::Mat constructor
        Image2D(const cv::Mat& mat, const std::string& name = "in_memory") 
            : filename_ptr(std::make_shared<std::string>(name)),
              mat_ptr(std::make_shared<cv::Mat>(mat.clone())),
              storage_mode(StorageMode::IN_MEMORY_MAT) { }
        
        // In-memory cv::Mat constructor (move semantics)
        Image2D(cv::Mat&& mat, const std::string& name = "in_memory") 
            : filename_ptr(std::make_shared<std::string>(name)),
              mat_ptr(std::make_shared<cv::Mat>(std::move(mat))),
              storage_mode(StorageMode::IN_MEMORY_MAT) { }
        
        // In-memory Array2D<double> constructor (already grayscale)
        Image2D(const Array2D<double>& gs_array, const std::string& name = "in_memory_gs") 
            : filename_ptr(std::make_shared<std::string>(name)),
              gs_ptr(std::make_shared<Array2D<double>>(gs_array)),
              storage_mode(StorageMode::IN_MEMORY_GS) { }
        
        // In-memory Array2D<double> constructor (move semantics)
        Image2D(Array2D<double>&& gs_array, const std::string& name = "in_memory_gs") 
            : filename_ptr(std::make_shared<std::string>(name)),
              gs_ptr(std::make_shared<Array2D<double>>(std::move(gs_array))),
              storage_mode(StorageMode::IN_MEMORY_GS) { }
                
        // Static factory methods --------------------------------------------//
        static Image2D load(std::ifstream&);
        
        // Create from cv::Mat with optional filtering
        static Image2D from_mat(const cv::Mat& mat, 
                               const std::string& name = "in_memory",
                               const FilterConfig& filter_config = FilterConfig());
        
        // Create from file with optional filtering (loads and stores in memory)
        static Image2D from_file_filtered(const std::string& filename,
                                          const FilterConfig& filter_config);
            
        // Interface functions -----------------------------------------------//
        friend std::ostream& operator<<(std::ostream&, const Image2D&); 
        friend void imshow(const Image2D&, difference_type delay);  
        friend bool isequal(const Image2D&, const Image2D&);
        friend void save(const Image2D&, std::ofstream&);
        
        // Access ------------------------------------------------------------//
        std::string get_filename() const { return *filename_ptr; }
        Array2D<double> get_gs() const; // Returns image as double precision grayscale array with values from 0 - 1.
        cv::Mat get_mat() const;        // Returns image as cv::Mat (grayscale, 8-bit)
        StorageMode get_storage_mode() const { return storage_mode; }
        bool is_in_memory() const { return storage_mode != StorageMode::FILE_PATH; }
        
        // Save in-memory image to file
        bool save_to_file(const std::string& filename) const;
        
    private:
        std::shared_ptr<std::string> filename_ptr;      // Name/path identifier
        std::shared_ptr<cv::Mat> mat_ptr;               // In-memory cv::Mat storage
        std::shared_ptr<Array2D<double>> gs_ptr;        // In-memory grayscale storage
        StorageMode storage_mode = StorageMode::FILE_PATH;
};

// ImageProcessor class for applying filters to images
class ImageProcessor {
    public:
        typedef std::ptrdiff_t difference_type;
        
        // Single image operations
        static cv::Mat saturate(const cv::Mat& input, int level = 200);
        static cv::Mat apply_bandpass_filter(const cv::Mat& input, const std::vector<int>& params = {5, 50});
        static cv::Mat apply_filters(const cv::Mat& input, const FilterConfig& config);
        
        // Batch operations
        static std::vector<cv::Mat> saturate(const std::vector<cv::Mat>& input, int level = 200);
        static std::vector<cv::Mat> apply_bandpass_filter(const std::vector<cv::Mat>& input, 
                                                          const std::vector<int>& params = {5, 50});
        static std::vector<cv::Mat> apply_filters(const std::vector<cv::Mat>& input, const FilterConfig& config);
        
        // Ben's filtering method (bandpass + normalization)
        static std::pair<std::vector<cv::Mat>, std::pair<double, double>> 
        filter_like_ben(const std::vector<cv::Mat>& input,
                       const cv::Mat& mask,
                       const std::vector<int>& params = {5, 50},
                       const std::pair<double, double>* gs_boundaries = nullptr);
        
        static cv::Mat filter_like_ben_single(const cv::Mat& input,
                                              const cv::Mat& mask,
                                              const std::vector<int>& params = {5, 50},
                                              const std::pair<double, double>* gs_boundaries = nullptr);
        
        // Utility functions
        static std::pair<double, double> compute_grayscale_boundaries(const cv::Mat& image, const cv::Mat& mask);
        static std::pair<double, double> compute_percentile_boundaries(const cv::Mat& image, const cv::Mat& mask,
                                                                       double lower_pct = 5.0, double upper_pct = 95.0);
        static cv::Mat normalize_and_clamp(const cv::Mat& image, const std::pair<double, double>& boundaries);
        
        // Conversion utilities
        static cv::Mat array2d_to_mat(const Array2D<double>& arr);
        static Array2D<double> mat_to_array2d(const cv::Mat& mat);
};

// VideoImporter class for extracting frames from video files
class VideoImporter {
    public:
        typedef std::ptrdiff_t difference_type;
        
        // Import video to in-memory Image2D vector
        static std::vector<Image2D> import_video(
            const std::string& video_path,
            const VideoImportParams& params = VideoImportParams(),
            const FilterConfig& filter_config = FilterConfig(),
            const std::string& name_prefix = "frame");
        
        // Import video and save frames to disk
        static std::vector<Image2D> import_video_to_files(
            const std::string& video_path,
            const std::string& output_dir,
            const VideoImportParams& params = VideoImportParams(),
            const FilterConfig& filter_config = FilterConfig(),
            const std::string& name_prefix = "frame");
        
        // Get video information
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

}

#endif	/* IMAGE2D_H */