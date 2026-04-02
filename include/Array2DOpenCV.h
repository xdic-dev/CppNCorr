#ifndef ARRAY2D_OPENCV_H
#define ARRAY2D_OPENCV_H

#include "Array2D.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace ncorr {

template <typename T, typename T_alloc, typename T_min, typename T_max>
typename std::enable_if<std::is_arithmetic<T>::value &&
                            std::is_arithmetic<T_min>::value &&
                            std::is_arithmetic<T_max>::value,
                        cv::Mat>::type
get_cv_img(const Array2D<T, T_alloc> &A, T_min min_val, T_max max_val) {
    if (A.height() < 1 || A.width() < 1) {
        throw std::invalid_argument("Attempted to use get_cv_img operator on array of size " +
                                    A.size_2D_string() +
                                    ". Array must be of size (1,1) or greater.");
    }

    cv::Mat cv_img(A.height(), A.width(), CV_8UC1);
    const double min_double = static_cast<double>(min_val);
    const double max_double = static_cast<double>(max_val);
    const double range = std::abs(max_double - min_double);
    const double conversion =
        (range > std::numeric_limits<double>::epsilon()) ? 255.0 / range : 0.0;

    for (typename Array2D<T, T_alloc>::difference_type p2 = 0; p2 < A.width(); ++p2) {
        for (typename Array2D<T, T_alloc>::difference_type p1 = 0; p1 < A.height(); ++p1) {
            const double val = (static_cast<double>(A(p1, p2)) - min_double) * conversion;
            if (val < 0.0) {
                cv_img.at<unsigned char>(p1, p2) = 0;
            } else if (val > 255.0) {
                cv_img.at<unsigned char>(p1, p2) = 255;
            } else {
                cv_img.at<unsigned char>(p1, p2) = static_cast<unsigned char>(val);
            }
        }
    }

    return cv_img;
}

template <typename T, typename T_alloc>
typename std::enable_if<std::is_arithmetic<T>::value, void>::type
imshow(const Array2D<T, T_alloc> &A, typename Array2D<T, T_alloc>::difference_type delay = -1) {
    if (A.height() < 1 || A.width() < 1) {
        throw std::invalid_argument("Attempted to use imshow operator on array of size " +
                                    A.size_2D_string() +
                                    ". Array must be of size (1,1) or greater.");
    }

    cv::Mat cv_img = get_cv_img(A, min(A), max(A));
    cv::imshow("Array2D", cv_img);
    delay == -1 ? cv::waitKey() : cv::waitKey(delay);
}

} // namespace ncorr

#endif
