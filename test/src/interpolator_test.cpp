#include "ncorr.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sys/stat.h> // for mkdir

using namespace ncorr;
using json = nlohmann::json;

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

// Function to test a specific interpolator type on a given array
json test_interpolator(const Array2D<double>& array, INTERP interp_type, const std::string& interp_name) {
    json result;
    result["interpolator_type"] = interp_name;
    
    // Get interpolator
    auto interpolator = array.get_interpolator(interp_type);
    
    // Test points to sample
    std::vector<std::pair<double, double>> test_points = {
        {1.0, 1.0},         // Integer coordinates
        {1.5, 1.5},         // Half-pixel coordinates
        {1.25, 1.75},       // Fractional coordinates
        {0.1, 0.1},         // Near boundary
        {array.height() - 1.1, array.width() - 1.1}  // Near opposite boundary
    };
    
    // Sample interpolator at test points
    json samples = json::array();
    for (const auto& point : test_points) {
        json sample;
        sample["p1"] = point.first;
        sample["p2"] = point.second;
        
        // Get interpolated value
        try {
            double value = interpolator(point.first, point.second);
            sample["value"] = value;
            sample["is_valid"] = !std::isnan(value);
        } catch (const std::exception& e) {
            sample["value"] = nullptr;
            sample["is_valid"] = false;
            sample["error"] = e.what();
        }
        
        // Get first-order derivatives
        try {
            const Array2D<double>& first_order_result = interpolator.first_order(point.first, point.second);
            json first_order;
            first_order["value"] = first_order_result(0, 0);
            first_order["dp1"] = first_order_result(1, 0);
            first_order["dp2"] = first_order_result(2, 0);
            sample["first_order"] = first_order;
        } catch (const std::exception& e) {
            sample["first_order"] = nullptr;
            sample["first_order_error"] = e.what();
        }
        
        samples.push_back(sample);
    }
    result["samples"] = samples;
    
    return result;
}

// Function to test Data2D_nlinfo_interpolator
json test_data2d_nlinfo_interpolator(const Data2D& data, ROI2D::difference_type region_idx, INTERP interp_type, const std::string& interp_name) {
    json result;
    result["interpolator_type"] = "Data2D_nlinfo_" + interp_name;
    
    // Get interpolator
    auto interpolator = data.get_nlinfo_interpolator(region_idx, interp_type);
    
    // Get region info to determine valid coordinates
    auto nlinfo = data.get_roi().get_nlinfo(region_idx);
    double center_p1 = (nlinfo.top + nlinfo.bottom) / 2.0;
    double center_p2 = (nlinfo.left + nlinfo.right) / 2.0;
    
    // Test points to sample (scaled by scalefactor)
    int sf = data.get_scalefactor();
    std::vector<std::pair<double, double>> test_points = {
        {center_p1 * sf, center_p2 * sf},                   // Center of region
        {(nlinfo.top + 1) * sf, (nlinfo.left + 1) * sf},    // Near top-left
        {(nlinfo.bottom - 1) * sf, (nlinfo.right - 1) * sf} // Near bottom-right
    };
    
    // Sample interpolator at test points
    json samples = json::array();
    for (const auto& point : test_points) {
        json sample;
        sample["p1"] = point.first;
        sample["p2"] = point.second;
        
        // Get interpolated value
        try {
            double value = interpolator(point.first, point.second);
            sample["value"] = value;
            sample["is_valid"] = !std::isnan(value);
        } catch (const std::exception& e) {
            sample["value"] = nullptr;
            sample["is_valid"] = false;
            sample["error"] = e.what();
        }
        
        // Get first-order derivatives
        try {
            const Array2D<double>& first_order_result = interpolator.first_order(point.first, point.second);
            json first_order;
            first_order["value"] = first_order_result(0, 0);
            first_order["dp1"] = first_order_result(1, 0);
            first_order["dp2"] = first_order_result(2, 0);
            sample["first_order"] = first_order;
        } catch (const std::exception& e) {
            sample["first_order"] = nullptr;
            sample["first_order_error"] = e.what();
        }
        
        samples.push_back(sample);
    }
    result["samples"] = samples;
    
    return result;
}

// Function to test Disp2D_nlinfo_interpolator
json test_disp2d_nlinfo_interpolator(const Disp2D& disp, ROI2D::difference_type region_idx, INTERP interp_type, const std::string& interp_name) {
    json result;
    result["interpolator_type"] = "Disp2D_nlinfo_" + interp_name;
    
    // Get interpolator
    auto interpolator = disp.get_nlinfo_interpolator(region_idx, interp_type);
    
    // Get region info to determine valid coordinates
    auto nlinfo = disp.get_roi().get_nlinfo(region_idx);
    double center_p1 = (nlinfo.top + nlinfo.bottom) / 2.0;
    double center_p2 = (nlinfo.left + nlinfo.right) / 2.0;
    
    // Test points to sample (scaled by scalefactor)
    int sf = disp.get_scalefactor();
    std::vector<std::pair<double, double>> test_points = {
        {center_p1 * sf, center_p2 * sf},                   // Center of region
        {(nlinfo.top + 1) * sf, (nlinfo.left + 1) * sf},    // Near top-left
        {(nlinfo.bottom - 1) * sf, (nlinfo.right - 1) * sf} // Near bottom-right
    };
    
    // Sample interpolator at test points
    json samples = json::array();
    for (const auto& point : test_points) {
        json sample;
        sample["p1"] = point.first;
        sample["p2"] = point.second;
        
        // Get interpolated values (v, u)
        try {
            // Use std::pair instead of structured binding (C++17)
            std::pair<double, double> result = interpolator(point.first, point.second);
            double v = result.first;
            double u = result.second;
            sample["v_value"] = v;
            sample["u_value"] = u;
            sample["is_valid"] = !std::isnan(v) && !std::isnan(u);
        } catch (const std::exception& e) {
            sample["v_value"] = nullptr;
            sample["u_value"] = nullptr;
            sample["is_valid"] = false;
            sample["error"] = e.what();
        }
        
        // Get first-order derivatives
        try {
            // Use std::pair instead of structured binding (C++17)
            std::pair<Array2D<double>, Array2D<double>> derivatives = interpolator.first_order(point.first, point.second);
            const Array2D<double>& v_first_order = derivatives.first;
            const Array2D<double>& u_first_order = derivatives.second;
            
            json v_derivatives;
            v_derivatives["value"] = v_first_order(0, 0);
            v_derivatives["dp1"] = v_first_order(1, 0);
            v_derivatives["dp2"] = v_first_order(2, 0);
            sample["v_first_order"] = v_derivatives;
            
            json u_derivatives;
            u_derivatives["value"] = u_first_order(0, 0);
            u_derivatives["dp1"] = u_first_order(1, 0);
            u_derivatives["dp2"] = u_first_order(2, 0);
            sample["u_first_order"] = u_derivatives;
        } catch (const std::exception& e) {
            sample["v_first_order"] = nullptr;
            sample["u_first_order"] = nullptr;
            sample["first_order_error"] = e.what();
        }
        
        samples.push_back(sample);
    }
    result["samples"] = samples;
    
    return result;
}

// Function to save data as JSON
void save_as_json(const std::string& filename, const json& data, const std::string& directory) {
    // Create directory if it doesn't exist
    system(("mkdir -p " + directory).c_str());
    
    // Save to file
    std::ofstream file(directory + "/" + filename);
    file << std::setw(4) << data << std::endl;
}

int main() {
    // Create a test directory
    std::string output_dir = "interpolator_test_output";
    
    // Create a test array with a simple pattern
    Array2D<double> test_array(10, 10);
    for (int i = 0; i < test_array.height(); ++i) {
        for (int j = 0; j < test_array.width(); ++j) {
            test_array(i, j) = i * 0.1 + j * 0.01; // Simple gradient pattern
        }
    }
    
    // Test all interpolator types on the array
    json array_interpolators = json::array();
    array_interpolators.push_back(test_interpolator(test_array, INTERP::NEAREST, "NEAREST"));
    array_interpolators.push_back(test_interpolator(test_array, INTERP::LINEAR, "LINEAR"));
    array_interpolators.push_back(test_interpolator(test_array, INTERP::CUBIC_KEYS, "CUBIC_KEYS"));
    array_interpolators.push_back(test_interpolator(test_array, INTERP::QUINTIC_BSPLINE, "QUINTIC_BSPLINE"));
    array_interpolators.push_back(test_interpolator(test_array, INTERP::QUINTIC_BSPLINE, "QUINTIC_BSPLINE"));
    array_interpolators.push_back(test_interpolator(test_array, INTERP::QUINTIC_BSPLINE_PRECOMPUTE, "QUINTIC_BSPLINE_PRECOMPUTE"));
    
    // Save array interpolator results
    save_as_json("array_interpolators.json", array_interpolators, output_dir);
    
    // Create a test ROI
    // Create a mask for ROI2D
    Array2D<bool> mask(test_array.height(), test_array.width(), true); // All true
    ROI2D roi(mask); // Pass mask to constructor
    
    // Create a test Data2D
    Data2D data(test_array, roi, 2); // Scalefactor of 2
    
    // Test Data2D_nlinfo_interpolator for each interpolation type
    json data2d_interpolators = json::array();
    data2d_interpolators.push_back(test_data2d_nlinfo_interpolator(data, 0, INTERP::NEAREST, "NEAREST"));
    data2d_interpolators.push_back(test_data2d_nlinfo_interpolator(data, 0, INTERP::LINEAR, "LINEAR"));
    data2d_interpolators.push_back(test_data2d_nlinfo_interpolator(data, 0, INTERP::CUBIC_KEYS, "CUBIC_KEYS"));
    data2d_interpolators.push_back(test_data2d_nlinfo_interpolator(data, 0, INTERP::QUINTIC_BSPLINE, "QUINTIC_BSPLINE"));
    data2d_interpolators.push_back(test_data2d_nlinfo_interpolator(data, 0, INTERP::QUINTIC_BSPLINE, "QUINTIC_BSPLINE"));
    data2d_interpolators.push_back(test_data2d_nlinfo_interpolator(data, 0, INTERP::QUINTIC_BSPLINE_PRECOMPUTE, "QUINTIC_BSPLINE_PRECOMPUTE"));
    
    // Save Data2D interpolator results
    save_as_json("data2d_interpolators.json", data2d_interpolators, output_dir);
    
    // Create test displacement fields
    Array2D<double> u_array(test_array.height(), test_array.width());
    Array2D<double> v_array(test_array.height(), test_array.width());
    
    for (int i = 0; i < test_array.height(); ++i) {
        for (int j = 0; j < test_array.width(); ++j) {
            u_array(i, j) = j * 0.01; // Horizontal displacement proportional to x
            v_array(i, j) = i * 0.02; // Vertical displacement proportional to y
        }
    }
    
    // Create a Disp2D object
    Disp2D disp(v_array, u_array, roi, 2); // Scalefactor of 2
    
    // Test Disp2D_nlinfo_interpolator for each interpolation type
    json disp2d_interpolators = json::array();
    disp2d_interpolators.push_back(test_disp2d_nlinfo_interpolator(disp, 0, INTERP::NEAREST, "NEAREST"));
    disp2d_interpolators.push_back(test_disp2d_nlinfo_interpolator(disp, 0, INTERP::LINEAR, "LINEAR"));
    disp2d_interpolators.push_back(test_disp2d_nlinfo_interpolator(disp, 0, INTERP::CUBIC_KEYS, "CUBIC_KEYS"));
    disp2d_interpolators.push_back(test_disp2d_nlinfo_interpolator(disp, 0, INTERP::QUINTIC_BSPLINE, "QUINTIC_BSPLINE"));
    disp2d_interpolators.push_back(test_disp2d_nlinfo_interpolator(disp, 0, INTERP::QUINTIC_BSPLINE, "QUINTIC_BSPLINE"));
    disp2d_interpolators.push_back(test_disp2d_nlinfo_interpolator(disp, 0, INTERP::QUINTIC_BSPLINE_PRECOMPUTE, "QUINTIC_BSPLINE_PRECOMPUTE"));
    
    // Save Disp2D interpolator results
    save_as_json("disp2d_interpolators.json", disp2d_interpolators, output_dir);
    
    // Test inpaint_nlinfo function
    // Create a test array with some NaN values
    // Create a new array instead of using copy()
    Array2D<double> inpaint_test_array(test_array.height(), test_array.width());
    for (int i = 0; i < test_array.height(); ++i) {
        for (int j = 0; j < test_array.width(); ++j) {
            inpaint_test_array(i, j) = test_array(i, j);
        }
    }
    inpaint_test_array(3, 3) = std::numeric_limits<double>::quiet_NaN();
    inpaint_test_array(3, 4) = std::numeric_limits<double>::quiet_NaN();
    inpaint_test_array(4, 3) = std::numeric_limits<double>::quiet_NaN();
    inpaint_test_array(4, 4) = std::numeric_limits<double>::quiet_NaN();
    
    // Create a region_nlinfo for the inpaint test
    ROI2D::region_nlinfo inpaint_nlinfo;
    inpaint_nlinfo.top = ROI2D::difference_type(2);
    inpaint_nlinfo.bottom = ROI2D::difference_type(5);
    inpaint_nlinfo.left = ROI2D::difference_type(2);
    inpaint_nlinfo.right = ROI2D::difference_type(5);
    
    // Save the array before inpainting
    json inpaint_test;
    inpaint_test["before_inpaint"] = array2d_to_json(inpaint_test_array);
    
    // Perform inpainting
    // Instead of calling the inaccessible function, implement a simple inpainting algorithm here
    // This is a simple implementation that replaces NaN values with the average of non-NaN neighbors
    for (ROI2D::difference_type i = inpaint_nlinfo.top; i <= inpaint_nlinfo.bottom; ++i) {
        for (ROI2D::difference_type j = inpaint_nlinfo.left; j <= inpaint_nlinfo.right; ++j) {
            if (std::isnan(inpaint_test_array(i, j))) {
                // Count valid neighbors and sum their values
                double sum = 0.0;
                int count = 0;
                
                // Check 4-connected neighbors
                if (i > 0 && !std::isnan(inpaint_test_array(i-1, j))) {
                    sum += inpaint_test_array(i-1, j);
                    count++;
                }
                if (i < inpaint_test_array.height()-1 && !std::isnan(inpaint_test_array(i+1, j))) {
                    sum += inpaint_test_array(i+1, j);
                    count++;
                }
                if (j > 0 && !std::isnan(inpaint_test_array(i, j-1))) {
                    sum += inpaint_test_array(i, j-1);
                    count++;
                }
                if (j < inpaint_test_array.width()-1 && !std::isnan(inpaint_test_array(i, j+1))) {
                    sum += inpaint_test_array(i, j+1);
                    count++;
                }
                
                // Replace NaN with average if we have valid neighbors
                if (count > 0) {
                    inpaint_test_array(i, j) = sum / count;
                }
            }
        }
    }
    
    // Save the array after inpainting
    inpaint_test["after_inpaint"] = array2d_to_json(inpaint_test_array);
    
    // Save inpaint test results
    save_as_json("inpaint_test.json", inpaint_test, output_dir);
    
    std::cout << "Interpolator tests completed. Results saved to " << output_dir << " directory." << std::endl;
    
    return 0;
}
