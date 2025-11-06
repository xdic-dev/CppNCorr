#include "ncorr.h"
#include "Array2D.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sys/stat.h> // for mkdir
#include <algorithm> // for std::min

using namespace ncorr;
using json = nlohmann::json;

Array2D<double> hrrr(const Array2D<double> &A, const Array2D<double> &kernel, int fftw_type) {
    using details::fftw_allocator;

    int h = A.height();
    int w = A.width();
    int s = A.size();
    int kh = kernel.height();
    int kw = kernel.width();
    int ks = kernel.size();

    if (kh % 2 == 0 || kw % 2 == 0 || kh > h || kw > w ) {
        throw std::runtime_error("Kernel size must be odd and smaller than or equal to array");
    }
    
    // According to FFTW documentation, the only thread safe function is 
    // fftw_execute
    std::unique_lock<std::mutex> fftw_lock(details::fftw_mutex);

    // Allocate arrays based on size of A with padding for FFTW. Look at 
    // FFTW documentation to see explanation for the need of padding.
    int padded_height = 2 * (h/2 + 1);
    int padded_width = w;
    Array2D<double> A_fftw_padded(padded_height,padded_width);
    Array2D<double> kernel_fftw_padded(padded_height,padded_width);
    int kfph = kernel_fftw_padded.height();
    int kfpw = kernel_fftw_padded.width();

    std::cout << ">> 1: " << std::endl;
    // Copy A - must do conversions to fftw_allocator explicitly
    A_fftw_padded({0,h-1},{0,w-1}) = A;

    std::cout << ">> 2: " << std::endl;
    // Form kernel - must pad and place kernel's center in the top-left
    Array2D<double> kernel_fftw(kernel); 
    int kfh = kernel_fftw.height();
    int kfw = kernel_fftw.width();

    kernel_fftw_padded({0,(kfh-1)/2},{0,(kfw-1)/2}) = kernel_fftw({(kfh-1)/2,kfw-1},{(kfw-1)/2,kfw-1});
    kernel_fftw_padded({0,(kfh-1)/2},{kfph-(kfw-1)/2,kfpw-1}) = kernel_fftw({(kfh-1)/2,kfw-1},{0,(kfw-1)/2-1});
    kernel_fftw_padded({h-(kfh-1)/2,h-1},{0,(kfw-1)/2}) = kernel_fftw({0,(kfh-1)/2-1},{(kfw-1)/2,kfw-1});
    kernel_fftw_padded({h-(kfh-1)/2,h-1},{kfph-(kfw-1)/2,kfpw-1}) = kernel_fftw({0,(kfh-1)/2-1},{0,(kfw-1)/2-1});
    
    std::cout << ">> 3: " << std::endl;
    // Create plans; output is stored in-place in A_fftw.
    fftw_plan plan_A = fftw_plan_dft_r2c_2d(w, h, A_fftw_padded.get_pointer(), reinterpret_cast<fftw_complex*>(A_fftw_padded.get_pointer()), FFTW_ESTIMATE);
    fftw_plan plan_kernel = fftw_plan_dft_r2c_2d(w, h, kernel_fftw_padded.get_pointer(), reinterpret_cast<fftw_complex*>(kernel_fftw_padded.get_pointer()), FFTW_ESTIMATE);
    fftw_plan plan_output = fftw_plan_dft_c2r_2d(w, h, reinterpret_cast<fftw_complex*>(A_fftw_padded.get_pointer()), A_fftw_padded.get_pointer(), FFTW_ESTIMATE);

    // Create a lambda for deleter and wrap in unique ptr - these will
    // cause the plans to get deleted once plan_*_delete goes out of scope,
    // or if an exception is thrown.
    auto plan_deleter = [](fftw_plan *p) { return fftw_destroy_plan(*p); };
    std::unique_ptr<fftw_plan,decltype(plan_deleter)> plan_A_delete(&plan_A, plan_deleter);
    std::unique_ptr<fftw_plan,decltype(plan_deleter)> plan_kernel_delete(&plan_kernel, plan_deleter);
    std::unique_ptr<fftw_plan,decltype(plan_deleter)> plan_output_delete(&plan_output, plan_deleter);

    // Unlock here
    fftw_lock.unlock();
    
    // Perform forward FFT of A and B
    fftw_execute(plan_A);
    fftw_execute(plan_kernel);        

    // Dispatch based on type
    switch (fftw_type) {
        case 1: 
            // Multiply element-wise
            for (int p = 0; p < A_fftw_padded.size(); p += 2) {
                double real_A = A_fftw_padded(p);
                double imag_A = A_fftw_padded(p+1);
                double real_kernel = kernel_fftw_padded(p);
                double imag_kernel = kernel_fftw_padded(p+1);
                A_fftw_padded(p) = real_A*real_kernel - imag_A*imag_kernel;
                A_fftw_padded(p+1) = real_A*imag_kernel + imag_A*real_kernel;
            }
            break;
        case 2:
            // Divide element-wise
            for (int p = 0; p < A_fftw_padded.size(); p += 2) {
                double real_A = A_fftw_padded(p);
                double imag_A = A_fftw_padded(p+1);
                double real_kernel = kernel_fftw_padded(p);
                double imag_kernel = kernel_fftw_padded(p+1);          
                double norm_kernel = std::pow(real_kernel,2) + std::pow(imag_kernel,2);                
                A_fftw_padded(p) = (real_A*real_kernel + imag_A*imag_kernel)/norm_kernel;
                A_fftw_padded(p+1) = (imag_A*real_kernel - real_A*imag_kernel)/norm_kernel;
            }
            break;
        case 3:
            // Multiple element-wise by complex conjugate of kernel
            for (int p = 0; p < A_fftw_padded.size(); p += 2) {
                double real_A = A_fftw_padded(p);
                double imag_A = A_fftw_padded(p+1);
                double real_kernel = kernel_fftw_padded(p);
                double imag_kernel = kernel_fftw_padded(p+1);
                A_fftw_padded(p) = real_A*real_kernel - imag_A*(-imag_kernel);
                A_fftw_padded(p+1) = real_A*(-imag_kernel) + imag_A*real_kernel;
            }
            break;
    }
    
    std::cout << ">> 4: " << std::endl;
    // Perform inverse FFT to form output
    fftw_execute(plan_output);  
        
    // Lock here
    fftw_lock.lock();
       
    // Store results in output array
    Array2D<double> C(A_fftw_padded({0,h-1},{0,w-1}));
    // Scale results since fftw isn't normalized
    C = std::move(C) * (1.0 / s);

    return C;
}  

// Helper functions to convert ncorr data structures to JSON
json array2d_to_json(const Array2D<double>& array) {
    json j;
    j["rows"] = array.height();
    j["cols"] = array.width();
    
    // Convert data to vector
    std::vector<double> data;
    for (int i = 0; i < array.height(); ++i) {
        for (int j = 0; j < array.width(); ++j) {
            data.push_back(array(i, j));
        }
    }
    j["data"] = data;
    
    return j;
}

json array2d_to_json(const Array2D<ROI2D::difference_type>& array) {
    json j;
    j["rows"] = array.height();
    j["cols"] = array.width();
    
    // Convert data to vector
    std::vector<ROI2D::difference_type> data;
    for (int i = 0; i < array.height(); ++i) {
        for (int j = 0; j < array.width(); ++j) {
            data.push_back(array(i, j));
        }
    }
    j["data"] = data;
    
    return j;
}

json array2d_to_json(const Array2D<bool>& array) {
    json j;
    j["rows"] = array.height();
    j["cols"] = array.width();
    
    // Convert data to vector
    std::vector<bool> data;
    for (int i = 0; i < array.height(); ++i) {
        for (int j = 0; j < array.width(); ++j) {
            data.push_back(array(i, j));
        }
    }
    j["data"] = data;
    
    return j;
}

json region_nlinfo_to_json(const ROI2D::region_nlinfo& nlinfo) {
    json j;
    j["top"] = nlinfo.top;
    j["bottom"] = nlinfo.bottom;
    j["left"] = nlinfo.left;
    j["right"] = nlinfo.right;
    j["points"] = nlinfo.points;
    j["nodelist"] = array2d_to_json(nlinfo.nodelist);
    j["left_nl"] = nlinfo.left_nl;
    j["noderange"] = array2d_to_json(nlinfo.noderange);
    return j;
}

json roi_to_json(const ROI2D& roi) {
    json j;
    j["width"] = roi.width();
    j["height"] = roi.height();
    j["mask"] = array2d_to_json(roi.get_mask());
    j["regions"] = json::array();
    for (int i = 0; i < roi.size_regions(); ++i) {
        j["regions"][i] = region_nlinfo_to_json(roi.get_nlinfo(i));
    }
    return j;
}

json data2d_to_json(const Data2D& data) {
    json j;
    j["scalefactor"] = data.get_scalefactor();
    j["width"] = data.data_width();
    j["height"] = data.data_height();
    j["array"] = array2d_to_json(data.get_array());
    j["roi"] = roi_to_json(data.get_roi());
    
    return j;
}

json disp2d_to_json(const Disp2D& disp) {
    json j;
    j["scalefactor"] = disp.get_scalefactor();
    j["width"] = disp.data_width();
    j["height"] = disp.data_height();
    j["v"] = data2d_to_json(disp.get_v());
    j["u"] = data2d_to_json(disp.get_u());
    j["roi"] = roi_to_json(disp.get_roi());
    
    return j;
}

// Function to save data as JSON
void save_as_json(const std::string& filename, const json& data, const std::string& directory) {
    // Create directory if it doesn't exist
    system(("mkdir -p " + directory).c_str());
    
    // Save to file
    std::ofstream file(directory + "/" + filename);
    file << std::setw(4) << data << std::endl;
}

// Function to load Disp2D from JSON file
Disp2D load_disp_from_json(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    json j;
    file >> j;
    
    // The JSON is an array, so we need to get the first element
    // We'll use the first element that has the necessary data
    if (j.empty() || !j.is_array()) {
        throw std::runtime_error("JSON file is empty or not an array");
    }
    
    // Use the first element in the array
    json disp_data = j[0];
    
    // Verify that the element has the required structure
    if (!disp_data.contains("roi") || !disp_data.contains("v") || !disp_data.contains("u")) {
        throw std::runtime_error("JSON element does not have required fields (roi, v, u)");
    }
    
    // Extract scalefactor (default to 1 if not present)
    int scalefactor = disp_data.contains("scalefactor") ? disp_data["scalefactor"].get<int>() : 1;
    
    // Extract ROI
    json roi_data = disp_data["roi"];
    json mask_data_json = roi_data["mask"];
    
    int mask_rows = mask_data_json.contains("rows") ? mask_data_json["rows"].get<int>() : roi_data["height"].get<int>();
    int mask_cols = mask_data_json["cols"].get<int>();
    
    Array2D<bool> mask(mask_rows, mask_cols);
    std::vector<bool> mask_data = mask_data_json["data"].get<std::vector<bool>>();
    
    for (int i = 0; i < mask.height(); ++i) {
        for (int j = 0; j < mask.width(); ++j) {
            if (i * mask.width() + j < mask_data.size()) {
                mask(i, j) = mask_data[i * mask.width() + j];
            }
        }
    }
    
    ROI2D roi(mask);
    
    // Extract v array
    json v_data_json = disp_data["v"]["array"];
    int v_rows = v_data_json.contains("rows") ? v_data_json["rows"].get<int>() : mask_rows;
    int v_cols = v_data_json["cols"].get<int>();
    
    Array2D<double> v_array(v_rows, v_cols);
    std::vector<double> v_data = v_data_json["data"].get<std::vector<double>>();
    
    for (int i = 0; i < v_array.height(); ++i) {
        for (int j = 0; j < v_array.width(); ++j) {
            if (i * v_array.width() + j < v_data.size()) {
                v_array(i, j) = v_data[i * v_array.width() + j];
            }
        }
    }
    
    // Extract u array
    json u_data_json = disp_data["u"]["array"];
    int u_rows = u_data_json.contains("rows") ? u_data_json["rows"].get<int>() : mask_rows;
    int u_cols = u_data_json["cols"].get<int>();
    
    Array2D<double> u_array(u_rows, u_cols);
    std::vector<double> u_data = u_data_json["data"].get<std::vector<double>>();
    
    for (int i = 0; i < u_array.height(); ++i) {
        for (int j = 0; j < u_array.width(); ++j) {
            if (i * u_array.width() + j < u_data.size()) {
                u_array(i, j) = u_data[i * u_array.width() + j];
            }
        }
    }
    
    // Create and return Disp2D
    return Disp2D(v_array, u_array, roi, scalefactor);
}

// Function to test disp_nloptimizer with different interpolation types
json test_disp_nloptimizer(const Disp2D& disp, ROI2D::difference_type region_idx, INTERP interp_type, const std::string& interp_name) {
    json result;
    result["test_type"] = "disp_nloptimizer";
    result["interpolator_type"] = interp_name;
    result["region_idx"] = region_idx;
    
    // Create a disp_nloptimizer
    using namespace ncorr::details;
    disp_nloptimizer optimizer(disp, region_idx, interp_type);
    
    // Test points to sample (use points within the region)
    auto nlinfo = disp.get_roi().get_nlinfo(region_idx);
    double center_p1 = (nlinfo.top + nlinfo.bottom) / 2.0;
    double center_p2 = (nlinfo.left + nlinfo.right) / 2.0;
    
    // Scale by scalefactor
    int sf = disp.get_scalefactor();
    std::vector<std::pair<double, double>> test_points = {
        {center_p1 * sf, center_p2 * sf},                   // Center of region
        {(nlinfo.top + 1) * sf, (nlinfo.left + 1) * sf},    // Near top-left
        {(nlinfo.bottom - 1) * sf, (nlinfo.right - 1) * sf} // Near bottom-right
    };
    
    // Test global method
    json global_tests = json::array();
    for (const auto& point : test_points) {
        json test;
        test["p1_new"] = point.first;
        test["p2_new"] = point.second;
        
        // Create params for global method
        Array2D<double> params_init(12, 1);
        params_init(0) = point.first;  // p1_new
        params_init(1) = point.second; // p2_new
        
        try {
            // Call global method
            auto result_pair = optimizer.global(params_init);
            
            // Extract results
            test["success"] = result_pair.second;
            if (result_pair.second) {
                const Array2D<double>& params = result_pair.first;
                test["p1_old"] = params(2);
                test["p2_old"] = params(3);
                test["v_old"] = params(4);
                test["u_old"] = params(5);
                test["dv_dp1_old"] = params(6);
                test["dv_dp2_old"] = params(7);
                test["du_dp1_old"] = params(8);
                test["du_dp2_old"] = params(9);
                test["dist"] = params(10);
                test["grad_norm"] = params(11);
            }
        } catch (const std::exception& e) {
            test["success"] = false;
            test["error"] = e.what();
        }
        
        global_tests.push_back(test);
    }
    result["global_tests"] = global_tests;
    
    // Test operator() method
    json operator_tests = json::array();
    for (const auto& point : test_points) {
        json test;
        test["p1_new"] = point.first;
        test["p2_new"] = point.second;
        
        // Create params for operator() method
        Array2D<double> params_guess(12, 1);
        params_guess(0) = point.first;   // p1_new
        params_guess(1) = point.second;  // p2_new
        params_guess(2) = point.first;   // p1_old (initial guess)
        params_guess(3) = point.second;  // p2_old (initial guess)
        
        try {
            // Call operator() method
            auto result_pair = optimizer(params_guess);
            
            // Extract results
            test["success"] = result_pair.second;
            if (result_pair.second) {
                const Array2D<double>& params = result_pair.first;
                test["p1_old"] = params(2);
                test["p2_old"] = params(3);
                test["v_old"] = params(4);
                test["u_old"] = params(5);
                test["dv_dp1_old"] = params(6);
                test["dv_dp2_old"] = params(7);
                test["du_dp1_old"] = params(8);
                test["du_dp2_old"] = params(9);
                test["dist"] = params(10);
                test["grad_norm"] = params(11);
            }
        } catch (const std::exception& e) {
            test["success"] = false;
            test["error"] = e.what();
        }
        
        operator_tests.push_back(test);
    }
    result["operator_tests"] = operator_tests;
    
    return result;
}

// Function to test disp_nloptimizer with different linsolver types
json test_disp_nloptimizer_linsolver(const Disp2D& disp, ROI2D::difference_type region_idx) {
    json result;
    result["test_type"] = "disp_nloptimizer_linsolver";
    result["region_idx"] = region_idx;
    
    // Test with different linsolver types
    std::vector<std::pair<LINSOLVER, std::string>> linsolver_types = {
        {LINSOLVER::CHOL, "CHOL"},
        {LINSOLVER::QR, "QR"},
        {LINSOLVER::LU, "LU"}
    };
    // Create a test matrix and vector
    Array2D<double> A(2, 2);
    A(0, 0) = 4.0; A(0, 1) = 1.0;
    A(1, 0) = 1.0; A(1, 1) = 3.0;
    
    Array2D<double> b(2, 1);
    b(0) = 1.0;
    b(1) = 2.0;

    json data;
    data["A"] = array2d_to_json(A);
    data["b"] = array2d_to_json(b);
    
    result["data"] = data;
    json linsolver_tests = json::array();
    for (const auto& linsolver_pair : linsolver_types) {
        json test;
        test["linsolver_type"] = linsolver_pair.second;
        
        try {
            // Get linsolver
            auto linsolver = A.get_linsolver(linsolver_pair.first);
            
            // Solve system
            const auto& x = linsolver.solve(b);
            
            // Store results
            test["success"] = true;
            test["x"] = {x(0), x(1)};
            
            // Verify solution
            Array2D<double> Ax = A * x;
            test["Ax"] = {Ax(0), Ax(1)};
            test["b"] = {b(0), b(1)};
            test["residual_norm"] = std::sqrt(std::pow(Ax(0) - b(0), 2) + std::pow(Ax(1) - b(1), 2));
        } catch (const std::exception& e) {
            test["success"] = false;
            test["error"] = e.what();
        }
        
        linsolver_tests.push_back(test);
    }
    result["linsolver_tests"] = linsolver_tests;
    
    return result;
}

// Function to test FFT-related operations
json test_fft_operations() {
    json result;
    result["test_type"] = "fft_operations";
    
    // Load test data from step1_images.json
    std::ifstream file("/Users/jaoga/devlab/OWN/ncorr_2D_py/ncorr_2D_cpp-master/test/bin/verification_json/step1_images.json");
    if (!file.is_open()) {
        json error_result;
        error_result["success"] = false;
        error_result["error"] = "Failed to open step1_images.json";
        result["fft_tests"] = json::array({error_result});
        return result;
    }
    
    json images_data;
    file >> images_data;
    file.close();
    
    // Extract the first two images as reference and current
    if (images_data.size() < 2) {
        json error_result;
        error_result["success"] = false;
        error_result["error"] = "Not enough images in step1_images.json";
        result["fft_tests"] = json::array({error_result});
        return result;
    }
    
    // Create arrays from the first two images
    json& ref_data = images_data[0]["gs"];
    json& curr_data = images_data[1]["gs"];
    
    int ref_rows = ref_data["rows"].get<int>();
    int ref_cols = ref_data["cols"].get<int>();
    int curr_rows = curr_data["rows"].get<int>();
    int curr_cols = curr_data["cols"].get<int>();
    
    // Create smaller test regions to make computation faster
    // Extract a 32x32 region from the center of each image
    int region_size = 32;
    int ref_start_row = (ref_rows - region_size) / 2;
    int ref_start_col = (ref_cols - region_size) / 2;
    int curr_start_row = (curr_rows - region_size) / 2;
    int curr_start_col = (curr_cols - region_size) / 2;
    
    Array2D<double> ref_img(region_size, region_size);
    Array2D<double> curr_img(region_size, region_size);
    
    // Fill the arrays with data
    for (int i = 0; i < region_size; ++i) {
        for (int j = 0; j < region_size; ++j) {
            int ref_idx = (ref_start_row + i) * ref_cols + (ref_start_col + j);
            int curr_idx = (curr_start_row + i) * curr_cols + (curr_start_col + j);
            
            ref_img(i, j) = ref_data["data"][ref_idx].get<double>();
            curr_img(i, j) = curr_data["data"][curr_idx].get<double>();
        }
    }
    
    json fft_tests;
    
    // Test cross-correlation
    json xcorr_test;
    xcorr_test["fft_type"] = "Cross-correlation";
    
    try {
        // Create a smaller template (kernel) with odd dimensions for cross-correlation
        // Extract a 31x31 region from the center of ref_img (odd dimensions)
        int kernel_size = 31; // Odd number
        int start_row = (ref_img.height() - kernel_size) / 2;
        int start_col = (ref_img.width() - kernel_size) / 2;
        
        Array2D<double> kernel(kernel_size, kernel_size);
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                kernel(i, j) = ref_img(start_row + i, start_col + j);
            }
        }
        
        // Perform cross-correlation
        Array2D<double> xcorr_result = xcorr(curr_img, kernel);
        // Array2D<double> test_result = hrrr(curr_img, kernel, 3);
        // std::cout << ">>>>> Test" << std::endl;
        
        // std::cout << "xcorr_result: " << array2d_to_json(xcorr_result) << std::endl;
        // std::cout << "test_result: " << array2d_to_json(test_result) << std::endl;
        
        // Store results
        xcorr_test["success"] = true;
        xcorr_test["result_height"] = xcorr_result.height();
        xcorr_test["result_width"] = xcorr_result.width();
        
        // Find the maximum correlation value and its position
        double max_val = -std::numeric_limits<double>::max();
        int max_row = -1, max_col = -1;
        
        for (int i = 0; i < xcorr_result.height(); ++i) {
            for (int j = 0; j < xcorr_result.width(); ++j) {
                if (xcorr_result(i, j) > max_val) {
                    max_val = xcorr_result(i, j);
                    max_row = i;
                    max_col = j;
                }
            }
        }
        
        xcorr_test["max_value"] = max_val;
        xcorr_test["max_position_row"] = max_row;
        xcorr_test["max_position_col"] = max_col;
        
        // Store a sample of the result
        std::vector<double> sample_data;
        // for (int i = 0; i < std::min<int>(5, xcorr_result.height()); ++i) {
        //     for (int j = 0; j < std::min<int>(5, xcorr_result.width()); ++j) {
        //         sample_data.push_back(xcorr_result(i, j));
        //     }
        // }
        for (int i = 0; i < xcorr_result.height(); ++i) {
            for (int j = 0; j < xcorr_result.width(); ++j) {
                sample_data.push_back(xcorr_result(i, j));
            }
        }
        xcorr_test["sample_data"] = sample_data;
    } catch (const std::exception& e) {
        xcorr_test["success"] = false;
        xcorr_test["error"] = e.what();
    }
    
    fft_tests["xcorr"] = xcorr_test;
    
    // Test convolution
    json conv_test;
    conv_test["fft_type"] = "Convolution";
    
    try {
        // Create a small kernel for convolution
        Array2D<double> kernel(5, 5);
        for (int i = 0; i < kernel.height(); ++i) {
            for (int j = 0; j < kernel.width(); ++j) {
                // Simple Gaussian-like kernel
                double x = i - kernel.height() / 2;
                double y = j - kernel.width() / 2;
                kernel(i, j) = std::exp(-(x*x + y*y) / 4.0);
            }
        }
        
        // Normalize the kernel
        double sum = 0.0;
        for (int i = 0; i < kernel.height(); ++i) {
            for (int j = 0; j < kernel.width(); ++j) {
                sum += kernel(i, j);
            }
        }
        kernel = kernel * (1.0 / sum);
        
        // Perform convolution
        Array2D<double> conv_result = conv(ref_img, kernel);
        
        // Store results
        conv_test["success"] = true;
        conv_test["result_height"] = conv_result.height();
        conv_test["result_width"] = conv_result.width();
        
        // Store a sample of the result
        std::vector<double> sample_data;
        for (int i = 0; i < std::min<int>(5, conv_result.height()); ++i) {
            for (int j = 0; j < std::min<int>(5, conv_result.width()); ++j) {
                sample_data.push_back(conv_result(i, j));
            }
        }
        conv_test["sample_data"] = sample_data;
        fft_tests["conv"] = conv_test;
        
        // Also test deconvolution
        json deconv_test;
        deconv_test["fft_type"] = "Deconvolution";
        
        try {
            // Perform deconvolution
            Array2D<double> deconv_result = deconv(conv_result, kernel);
            
            // Store results
            deconv_test["success"] = true;
            deconv_test["result_height"] = deconv_result.height();
            deconv_test["result_width"] = deconv_result.width();
            
            // Calculate error between original and deconvolved image
            double mse = 0.0;
            for (int i = 0; i < ref_img.height(); ++i) {
                for (int j = 0; j < ref_img.width(); ++j) {
                    mse += std::pow(ref_img(i, j) - deconv_result(i, j), 2);
                }
            }
            mse /= (ref_img.height() * ref_img.width());
            deconv_test["mse"] = mse;
            
            // Store a sample of the result
            std::vector<double> sample_data;
            for (int i = 0; i < std::min<int>(5, deconv_result.height()); ++i) {
                for (int j = 0; j < std::min<int>(5, deconv_result.width()); ++j) {
                    sample_data.push_back(deconv_result(i, j));
                }
            }
            deconv_test["sample_data"] = sample_data;
        } catch (const std::exception& e) {
            deconv_test["success"] = false;
            deconv_test["error"] = e.what();
        }
        
        fft_tests["deconv"] = deconv_test;
    } catch (const std::exception& e) {
        conv_test["success"] = false;
        conv_test["error"] = e.what();
    }
    
    result["fft_tests"] = fft_tests;
    
    return result;
}

// Function to test FFT-related operations
json test_simple_fft_operations() {
    json result;
    result["test_type"] = "simple_fft_operations";
    int h(8), w(8), kh(5), kw(5);

    Array2D<double> arr(h, w);
    Array2D<double> kernel(kh, kw);


    
    // Fill the arrays with data
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            arr(i, j) = 0;
        }
    }
    arr(0, 0) = 1;

    // create raw kernel vector: [[-0.61110434, -0.01396059, -1.29399164, 1.09071183, -0.06743282], [, 0.51110501, 0.26535201, -0.33943594, 2.00539449, -1.05166092], [, 0.35865613, 0.2461254, -1.5352936, 0.86629866, -0.55628964], [, 1.0939041, 0.32810662, 0.61213902, 0.11558149, 0.2319589, ], [-1.2898878, 1.41775119, -0.07773476, -1.1176096, -0.48717525]] 
    double* raw_kern = new double[kh * kw]{ -0.61110434, -0.01396059, -1.29399164, 1.09071183, -0.06743282, 0.51110501, 0.26535201, -0.33943594, 2.00539449, -1.05166092, 0.35865613, 0.2461254, -1.5352936, 0.86629866, -0.55628964, 1.0939041, 0.32810662, 0.61213902, 0.11558149, 0.2319589, -1.2898878, 1.41775119, -0.07773476, -1.1176096, -0.48717525 };
    std::cout << "raw_kern: "<< std::endl;
    for (int i = 0; i < kh; ++i) {
        for (int j = 0; j < kw; ++j) {
            kernel(i, j) = raw_kern[i * kw + j];
        }
    }
    
    json fft_tests = json::array();
    std::cout << "start test: " << std::endl;
    
    // Test cross-correlation
    json xcorr_test;
    xcorr_test["fft_type"] = "Cross-correlation";
    Array2D<double> test_result_xcorr = hrrr(arr, kernel, 3);
    xcorr_test["success"] = true;
    xcorr_test["result_height"] = test_result_xcorr.height();
    xcorr_test["result_width"] = test_result_xcorr.width();
    xcorr_test["data"] = array2d_to_json(test_result_xcorr);
    
    fft_tests.push_back(xcorr_test);
    
    // Test convolution
    json conv_test;
    conv_test["fft_type"] = "Convolution";
    Array2D<double> test_result_conv = hrrr(arr, kernel, 1);
    conv_test["success"] = true;
    conv_test["result_height"] = test_result_conv.height();
    conv_test["result_width"] = test_result_conv.width();
    conv_test["data"] = array2d_to_json(test_result_conv);
    fft_tests.push_back(conv_test);

    // Test deconvolution
    json deconv_test;
    deconv_test["fft_type"] = "Deconvolution";
    Array2D<double> test_result_deconv = hrrr(arr, kernel, 2);
    deconv_test["success"] = true;
    deconv_test["result_height"] = test_result_deconv.height();
    deconv_test["result_width"] = test_result_deconv.width();
    deconv_test["data"] = array2d_to_json(test_result_deconv);
    fft_tests.push_back(deconv_test);
    
    result["fft_tests"] = fft_tests;
    
    return result;
}

// Function to test subregion_nloptimizer
json test_subregion_nloptimizer(const Array2D<double>& ref_img, const Array2D<double>& cur_img, 
    const ROI2D& roi, ROI2D::difference_type region_idx, 
    SUBREGION subregion_type, const std::string& subregion_name, INTERP interp_type, const std::string& interp_name, int scalefactor=1, int radius = 7) {
    
    json result;
    result["test_type"] = "subregion_nloptimizer";
    result["subregion_type"] = subregion_name;
    result["region_idx"] = region_idx;
    result["interp_type"] = interp_name;
    result["scalefactor"] = scalefactor;
    result["radius"] = radius;

    // Create a subregion_nloptimizer with required parameters
    // Note: subregion_nloptimizer requires 7 parameters according to the error message
    using namespace ncorr::details;
    subregion_nloptimizer optimizer(ref_img, cur_img, roi, scalefactor, interp_type, subregion_type, radius);

    ROI2D::contig_subregion_generator gen = roi.get_contig_subregion_generator(subregion_type, radius);

    // Test points to sample (use points within the region)
    auto nlinfo = roi.get_nlinfo(region_idx);
    double center_p1 = (nlinfo.top + nlinfo.bottom) / 2.0;
    double center_p2 = (nlinfo.left + nlinfo.right) / 2.0;

    // Scale by scalefactor
    int sf = scalefactor;
    std::vector<std::pair<double, double>> test_points = {
        {center_p1 * sf, center_p2 * sf},                   // Center of region
        {(nlinfo.top + 1) * sf, (nlinfo.left + 1) * sf},    // Near top-left
        {(nlinfo.bottom - 1) * sf, (nlinfo.right - 1) * sf} // Near bottom-right
    };

    // Test global method
    json global_tests = json::array();
    for (const auto& point : test_points) {
        json test;
        test["p1"] = point.first;
        test["p2"] = point.second;
        
        // Create params for global method
        Array2D<double> params_init(10, 1);
        params_init(0) = point.first;  // p1
        params_init(1) = point.second; // p2

        auto region_idx_pair = roi.get_region_idx(point.first, point.second);
        test["region_idx_pair_region_idx"] = region_idx_pair.first;
        test["region_idx_pair_nl_idx"] = region_idx_pair.second;
        if (region_idx_pair.first >= 0) {
            auto &nlinfo_roi = roi.get_nlinfo(region_idx_pair.first);
            test["nlinfo_roi"] = region_nlinfo_to_json(nlinfo_roi);
        }

        auto p1p2_nlinfo = gen(point.first, point.second);
        test["nlinfo_subregion"] = region_nlinfo_to_json(p1p2_nlinfo);
        
        try {
            // Call global method
            auto result_pair = optimizer.global(params_init);
            
            // Extract results
            test["success"] = result_pair.second;
            if (result_pair.second) {
                const Array2D<double>& params = result_pair.first;
                test["u"] = params(2);
                test["v"] = params(3);
                test["dv_dp1"] = params(4);
                test["dv_dp2"] = params(5);
                test["du_dp1"] = params(6);
                test["du_dp2"] = params(7);
                test["corr_coef"] = params(8);
                test["diff_norm"] = params(9);
            }
        } catch (const std::exception& e) {
            test["success"] = false;
            test["error"] = e.what();
        }
        
        global_tests.push_back(test);
    }
    result["global_tests"] = global_tests;

    // Test operator() method
    json operator_tests = json::array();
    for (const auto& point : test_points) {
        json test;
        test["p1"] = point.first;
        test["p2"] = point.second;
        
        // Create params for operator() method
        Array2D<double> params_guess(10, 1);
        params_guess(0) = point.first;   // p1
        params_guess(1) = point.second;  // p2

        auto p1p2_nlinfo = gen(point.first, point.second);
        test["nlinfo_subregion"] = region_nlinfo_to_json(p1p2_nlinfo);

        try {
            // Call operator() method
            auto result_pair = optimizer(params_guess);
            
            // Extract results
            test["success"] = result_pair.second;
            if (result_pair.second) {
                const Array2D<double>& params = result_pair.first;
                test["v"] = params(2);
                test["u"] = params(3);
                test["dv_dp1"] = params(4);
                test["dv_dp2"] = params(5);
                test["du_dp1"] = params(6);
                test["du_dp2"] = params(7);
                test["corr_coef"] = params(8);
                test["diff_norm"] = params(9);
            }
        } catch (const std::exception& e) {
            test["success"] = false;
            test["error"] = e.what();
        }
        
        operator_tests.push_back(test);
    }
    result["operator_tests"] = operator_tests;

    return result;
}

// Main function
int main(int argc, char *argv[]) {
    //try {
        // Create output directory
        std::string output_dir = "../bin/nloptimizer_test_output";
        system(("mkdir -p " + output_dir).c_str());
        
        // Load displacement data from JSON file
        std::string disp_json_path = "interpolator_test_output/disp_interpolators.json";
        std::cout << "Loading displacement data from: " << disp_json_path << std::endl;
        Disp2D disp = load_disp_from_json(disp_json_path);
        
        // Create a JSON object to store all test results
        json all_results;
        all_results["test_name"] = "ncorr_nloptimizer_test";
        all_results["timestamp"] = std::time(nullptr);
        all_results["d_nloptimizer"] = json::array();
        all_results["linsolver"] = json::array();
        all_results["sr_nloptimizer"] = json::array();
        all_results["fftw"] = json::array();
        //all_results["simple_fft"] = json::array();
        
        // Test disp_nloptimizer with different interpolation types
        std::cout << "Testing disp_nloptimizer with different interpolation types..." << std::endl;
        std::vector<std::pair<INTERP, std::string>> interp_types = {
            {INTERP::NEAREST, "NEAREST"},
            {INTERP::CUBIC_KEYS, "BICUBIC"},
            {INTERP::CUBIC_KEYS_PRECOMPUTE, "BICUBIC_PRECOMPUTE"},
            {INTERP::LINEAR, "BILINEAR"},
            {INTERP::QUINTIC_BSPLINE, "BIQUINTIC"},
            {INTERP::QUINTIC_BSPLINE_PRECOMPUTE, "BIQUINTIC_PRECOMPUTE"}
        };
        
        for (ROI2D::difference_type region_idx = 0; region_idx < disp.get_roi().size_regions(); ++region_idx) {
            std::cout << "Testing region " << region_idx << std::endl;
            
            for (const auto& interp_pair : interp_types) {
                std::cout << "  Testing interpolation type: " << interp_pair.second << std::endl;
                json result = test_disp_nloptimizer(disp, region_idx, interp_pair.first, interp_pair.second);
                all_results["d_nloptimizer"].push_back(result);
            }
            
            // Test linsolver types
            std::cout << "  Testing linsolver types" << std::endl;
            json linsolver_result = test_disp_nloptimizer_linsolver(disp, region_idx);
            all_results["linsolver"].push_back(linsolver_result);
            
            // Only test one region to keep the test manageable
            if (region_idx == 0) {
                break;
            }
        }
        
        // Test FFT operations
        std::cout << "Testing FFT operations..." << std::endl;
        json fft_result = test_fft_operations();
        all_results["fftw"].push_back(fft_result);
        
        // // Test simple FFT operations
        // std::cout << "Testing simple FFT operations..." << std::endl;
        // json simple_fft_result = test_simple_fft_operations();
        // all_results["simple_fft"].push_back(simple_fft_result);
        
        // Test subregion_nloptimizer
        std::cout << "Testing subregion_nloptimizer..." << std::endl;
        
        // Get test images
       // Load test data from step1_images.json
        std::ifstream file("/Users/jaoga/devlab/OWN/ncorr_2D_py/ncorr_2D_cpp-master/test/bin/verification_json/step1_images.json");
        if (!file.is_open()) {
            std::cerr << "Failed to open step1_images.json" << std::endl;
            return 1;
        }

        json images_data;
        file >> images_data;
        file.close();

        // Extract the first two images as reference and current
        if (images_data.size() < 2) {
            std::cerr << "Not enough images in step1_images.json" << std::endl;
            return 1;
        }

        // Create arrays from the first two images
        json& ref_data = images_data[0]["gs"];
        json& curr_data = images_data[1]["gs"];

        int ref_rows = ref_data["rows"].get<int>();
        int ref_cols = ref_data["cols"].get<int>();
        int curr_rows = curr_data["rows"].get<int>();
        int curr_cols = curr_data["cols"].get<int>();

        Array2D<double> ref_img(ref_rows, ref_cols);
        Array2D<double> curr_img(curr_rows, curr_cols);

        // Fill the arrays with data
        for (int i = 0; i < ref_rows; ++i) {
            for (int j = 0; j < ref_cols; ++j) {
                int ref_idx = i * ref_cols + j;
                int curr_idx = i * curr_cols + j;
                
                ref_img(i, j) = ref_data["data"][ref_idx].get<double>();
                curr_img(i, j) = curr_data["data"][curr_idx].get<double>();
            }
        }


        
        // Create ROI
        std::ifstream file2("/Users/jaoga/devlab/OWN/ncorr_2D_py/ncorr_2D_cpp-master/test/bin/verification_json/step2_roi.json");
        if (!file2.is_open()) {
            std::cerr << "Failed to open step2_roi.json" << std::endl;
            return 1;
        }
        
        json roi_data;
        file2 >> roi_data;
        file2.close();
        
        // Create ROI
        Array2D<bool> mask(roi_data["mask"]["rows"].get<int>(), roi_data["mask"]["cols"].get<int>());
        for (int i = 0; i < mask.height(); ++i) {
            for (int j = 0; j < mask.width(); ++j) {
                mask(i, j) = roi_data["mask"]["data"][i * mask.width() + j].get<bool>();
            }
        }
        ROI2D roi(mask);
        
        // Test with different subregion types
        std::vector<std::pair<SUBREGION, std::string>> subregion_types = {
            {SUBREGION::CIRCLE, "FORWARDHESSIAN"},
            {SUBREGION::SQUARE, "INVERSEHESSIAN"}
        };
        
        for (const auto& interp_pair : interp_types) {
            std::cout << "Testing interpolation type: " << interp_pair.second << std::endl;
            for (ROI2D::difference_type region_idx = 0; region_idx < roi.size_regions(); ++region_idx) {
                std::cout << "Testing region " << region_idx << std::endl;
                
                for (const auto& subregion_pair : subregion_types) {
                    std::cout << "  Testing subregion type: " << subregion_pair.second << std::endl;
                    json result = test_subregion_nloptimizer(ref_img, curr_img, roi, region_idx, 
                                                           subregion_pair.first, subregion_pair.second, interp_pair.first, interp_pair.second, 3, 7);
                    all_results["sr_nloptimizer"].push_back(result);
                }
            }
        }
        
        // Save all results to JSON file
        std::string output_file = "nloptimizer_test_results.json";
        std::cout << "Saving results to: " << output_dir << "/" << output_file << std::endl;
        save_as_json(output_file, all_results, output_dir);
        
        std::cout << "All tests completed successfully!" << std::endl;
        return 0;
    // } catch (const std::exception& e) {
    //     std::cerr << "Error: " << e.what() << std::endl;
    //     return 1;
    // }
}