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
#include "ImageProcessor.h"
#include <opencv2/opencv.hpp>

namespace ncorr {     

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

}

#include "VideoImporter.h"

#endif	/* IMAGE2D_H */
