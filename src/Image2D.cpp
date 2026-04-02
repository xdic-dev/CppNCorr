/* 
 * File:   Image2D.cpp
 * Author: justin
 *
 * Created on January 28, 2015, 3:10 PM
 * Extended to support in-memory images, video import, and image filtering
 */

#include "Image2D.h"
#include "Array2DOpenCV.h"
#include "ncorr/io/binary_io.hpp"
#include <cmath>
#include <filesystem>

namespace ncorr {

// ============================================================================
// Image2D Implementation
// ============================================================================

// Static factory methods ----------------------------------------------------//
Image2D Image2D::load(std::ifstream &is) {
    Image2D img;
    img.filename_ptr = std::make_shared<std::string>(io::read_string<difference_type>(is));
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

    io::write_string<difference_type>(os, *img.filename_ptr);
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
} // namespace ncorr
