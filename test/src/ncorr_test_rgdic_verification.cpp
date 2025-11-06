#include "ncorr.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sys/stat.h>

using namespace ncorr;
using json = nlohmann::json;

Array2D<double> get_seed_params(const Array2D<ROI2D::difference_type> &seeds_pos,
                                const ncorr::details::subregion_nloptimizer &sr_nloptimizer,
                                Array2D<double> &params_buf) {
    typedef ROI2D::difference_type                          difference_type; 
            
    // Note: params = {p1, p2, v, u, dv_dp1, dv_dp2, du_dp1, du_dp2, corr_coef, diff_norm}
            
    // Cycle over seeds and return the seed which has the lowest correlation 
    // coefficient
    
    Array2D<double> seed_params;
    for (difference_type seed_idx = 0; seed_idx < seeds_pos.height(); ++seed_idx) {
        params_buf(0) = seeds_pos(seed_idx,0);
        params_buf(1) = seeds_pos(seed_idx,1);
        std::cout << "\n seed_idx: " << seed_idx << " params_buf(0): " << params_buf(0) << " params_buf(1): " << params_buf(1) << std::endl;
        auto seed_params_pair = sr_nloptimizer.global(params_buf);
        std::cout << "\n new_seed_params: " << seed_params_pair.first << " success: " << seed_params_pair.second << std::endl;

        if (seed_params_pair.second && (seed_params.empty() || seed_params_pair.first(8) < seed_params(8))) {
            // This is either the first seed, or a seed which has a lower
            // correlation coefficient - so store it.
            seed_params = seed_params_pair.first;
        }
    }       

    return seed_params;
}

// Helper functions to convert ncorr data structures to JSON
json array2d_to_json(const Array2D<double>& array) {
    json j;
    j["rows"] = array.height();
    j["cols"] = array.width();
    
    std::vector<double> data;
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
    
    std::vector<bool> data;
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
    
    std::vector<int> data;
    for (int i = 0; i < array.height(); ++i) {
        for (int j = 0; j < array.width(); ++j) {
            data.push_back(array(i, j));
        }
    }
    j["data"] = data;
    
    return j;
}

json seeds_to_json(const std::vector<Array2D<ROI2D::difference_type>>& seeds) {
    json j = json::array();
    for (const auto& seed_array : seeds) {
        j.push_back(array2d_to_json(seed_array));
    }
    return j;
}

json roi2d_to_json(const ROI2D& roi) {
    json j;
    j["num_regions"] = roi.size_regions();
    j["height"] = roi.height();
    j["width"] = roi.width();
    
    Array2D<bool> mask = roi.get_mask();
    j["mask"] = array2d_to_json(mask);
    
    return j;
}

// Save verification data for RGDIC analysis
void save_rgdic_verification(const std::string& directory) {
    std::cout << "Starting RGDIC verification data generation..." << std::endl;
    
    // Create directory if it doesn't exist
    system(("mkdir -p " + directory).c_str());
    
    // Load images
    Image2D ref_img("images/ohtcfrp_01.png");
    Image2D cur_img("images/ohtcfrp_02.png");
    Array2D<double> A_ref = ref_img.get_gs();
    Array2D<double> A_cur = cur_img.get_gs();
    
    // Load ROI
    ROI2D roi(Image2D("images/roi.png").get_gs() > 0.5);
    
    // Parameters
    int scalefactor = 3;
    INTERP interp_type = INTERP::CUBIC_KEYS_PRECOMPUTE;
    SUBREGION subregion_type = SUBREGION::CIRCLE;
    int radius = 20;
    int num_threads = 4;
    double cutoff_corrcoef = 0.5;
    
    std::cout << "Images and ROI loaded" << std::endl;
    std::cout << "Reference image size: " << A_ref.height() << "x" << A_ref.width() << std::endl;
    std::cout << "ROI regions: " << roi.size_regions() << std::endl;
    
    // Reduce ROI
    auto roi_reduced = roi.reduce(scalefactor);
    std::cout << "Reduced ROI size: " << roi_reduced.height() << "x" << roi_reduced.width() << std::endl;
    std::cout << "Reduced ROI regions: " << roi_reduced.size_regions() << std::endl;
    
        
    // Get partition seeds
    std::cout << "Getting partition seeds..." << std::endl;
    std::vector<Array2D<ROI2D::difference_type>> partition_seeds = {{
        Array2D<ROI2D::difference_type>({{876, 153},{663, 249},{159, 156},{390, 153}})
    }};
    
    // Get subregion nonlinear optimizer
    std::cout << "Creating subregion optimizer..." << std::endl;
    using namespace ncorr::details;
    subregion_nloptimizer sr_nloptimizer(A_ref, A_cur, roi, scalefactor, interp_type, subregion_type, radius);
    
    // Get seed params (for first region, first seed)
    std::cout << "Getting seed params..." << std::endl;
    Array2D<double> seed_params(10, 1);
    bool seed_success = false;
    
    if (roi_reduced.size_regions() > 0 && !partition_seeds[0].empty()) {
        Array2D<double> params_buf(10, 1);
        seed_params = get_seed_params(partition_seeds[0], sr_nloptimizer, params_buf);
        seed_success = !seed_params.empty();
    }
    
    // Perform one iteration of analyze_point to get result for all four neighbors
    std::cout << "Analyzing sample point (all four directions)..." << std::endl;
    std::vector<std::pair<int, int>> neighbors = {
        {-scalefactor, 0},    // left
        {scalefactor, 0},     // right
        {0, -scalefactor},    // up
        {0, scalefactor}      // down
    };
    std::vector<Array2D<double>> analyze_results;
    std::vector<bool> analyze_success_flags;
    
    if (seed_success && roi_reduced.size_regions() > 0) {
        // Get first region's nlinfo
        auto nlinfo = roi_reduced.get_nlinfo(0);
        
        // Set up parameters for analyze_point
        Array2D<double> queue_params = seed_params;
        double cutoff_delta_disp = static_cast<double>(scalefactor);
        
        // Analyze all four neighbors
        for (const auto& neighbor : neighbors) {
            int p1_delta = neighbor.first;
            int p2_delta = neighbor.second;
            
            Array2D<double> params_buf(10, 1);
            
            // Call analyze_point equivalent (simplified version for testing)
            int p1 = static_cast<int>(queue_params(0, 0)) + p1_delta;
            int p2 = static_cast<int>(queue_params(1, 0)) + p2_delta;
            
            bool success = false;
            Array2D<double> result;
            
            if (nlinfo.in_nlinfo(p1 / scalefactor, p2 / scalefactor)) {
                // Fill in initial guess
                params_buf(0, 0) = p1;
                params_buf(1, 0) = p2;
                params_buf(2, 0) = queue_params(2, 0) + p1_delta * queue_params(4, 0) + p2_delta * queue_params(5, 0);
                params_buf(3, 0) = queue_params(3, 0) + p1_delta * queue_params(6, 0) + p2_delta * queue_params(7, 0);
                params_buf(4, 0) = queue_params(4, 0);
                params_buf(5, 0) = queue_params(5, 0);
                params_buf(6, 0) = queue_params(6, 0);
                params_buf(7, 0) = queue_params(7, 0);

                std::cout << "params_buf: " << params_buf << std::endl;
                
                // Optimize
                std::pair<Array2D<double>, bool> opt_result = sr_nloptimizer(params_buf);
                if (opt_result.second) {
                    result = opt_result.first;
                    success = true;
                }
            }
            
            analyze_results.push_back(result);
            analyze_success_flags.push_back(success);
        }
    }
    
    // Now perform full RGDIC analysis to get final results
    std::cout << "Performing full RGDIC analysis..." << std::endl;
    
    // Initialize arrays
    Array2D<double> A_v(roi_reduced.height(), roi_reduced.width());
    Array2D<double> A_u(roi_reduced.height(), roi_reduced.width());
    Array2D<double> A_cc(roi_reduced.height(), roi_reduced.width());
    Array2D<bool> A_vp(roi_reduced.height(), roi_reduced.width());
    
    // Perform RGDIC (this would normally be done internally, but we'll simulate)
    // For this test, we'll just use the DIC_analysis function
    DIC_analysis_input DIC_input;
    std::vector<Image2D> imgs;
    imgs.push_back(ref_img);
    imgs.push_back(cur_img);
    
    DIC_input = DIC_analysis_input(imgs, roi, scalefactor, interp_type, subregion_type, 
                                    radius, num_threads, DIC_analysis_config::NO_UPDATE, false);
    
    DIC_analysis_output DIC_output = DIC_analysis(DIC_input);
    
    // Extract results from first displacement field
    if (!DIC_output.disps.empty()) {
        Data2D v_data = DIC_output.disps[0].get_v();
        Data2D u_data = DIC_output.disps[0].get_u();
        ROI2D result_roi = DIC_output.disps[0].get_roi();
        
        A_v = v_data.get_array();
        A_u = u_data.get_array();
        A_vp = result_roi.get_mask();
        
        // Get correlation coefficients (if available)
        // Note: In actual implementation, we'd need to extract this from the analysis
    }
    
    // Save before displacement jump filtering
    Array2D<double> A_v_before = A_v;
    Array2D<double> A_u_before = A_u;
    Array2D<bool> A_vp_before = A_vp;
    Array2D<double> A_cc_before = A_cc;
    
    // Apply displacement jump filtering (simplified)
    Array2D<bool> A_vp_after = A_vp;
    double cutoff_delta_disp = static_cast<double>(scalefactor);
    
    // Filter displacement jumps
    for (int p1 = 0; p1 < A_vp.height(); ++p1) {
        for (int p2 = 0; p2 < A_vp.width(); ++p2) {
            if (!A_vp(p1, p2)) continue;
            
            // Check 4-connected neighbors
            std::vector<std::pair<int, int>> neighbors = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
            for (const auto& delta : neighbors) {
                int np1 = p1 + delta.first;
                int np2 = p2 + delta.second;
                
                if (np1 >= 0 && np1 < A_vp.height() && np2 >= 0 && np2 < A_vp.width() && A_vp(np1, np2)) {
                    double delta_v = A_v(np1, np2) - A_v(p1, p2);
                    double delta_u = A_u(np1, np2) - A_u(p1, p2);
                    double delta_disp = std::sqrt(delta_v * delta_v + delta_u * delta_u);
                    
                    if (delta_disp > cutoff_delta_disp) {
                        A_vp_after(p1, p2) = false;
                        break;
                    }
                }
            }
        }
    }
    
    // Create ROI from valid points
    ROI2D roi_valid(A_vp_after);
    
    // Create JSON output
    json output;
    
    // Input parameters
    output["input"] = {
        {"scalefactor", scalefactor},
        {"radius", radius},
        {"num_threads", num_threads},
        {"cutoff_corrcoef", cutoff_corrcoef},
        {"interp_type", static_cast<int>(interp_type)},
        {"subregion_type", static_cast<int>(subregion_type)}
    };
    
    // ROI information
    output["roi"] = roi2d_to_json(roi);
    output["roi_reduced"] = roi2d_to_json(roi_reduced);
    
    // Partition diagram and seeds
    output["partition_seeds"] = seeds_to_json(partition_seeds);
    
    // Seed params
    if (seed_success) {
        output["seed_params"] = array2d_to_json(seed_params);
    }
    
    // Analyze point results for all four neighbors
    json analyze_point_results = json::array();
    for (size_t i = 0; i < analyze_results.size(); ++i) {
        json neighbor_result;
        neighbor_result["success"] = analyze_success_flags[i];
        if (analyze_success_flags[i]) {
            neighbor_result["params"] = array2d_to_json(analyze_results[i]);
        } else {
            neighbor_result["params"] = nullptr;
        }
        // Store direction for reference
        neighbor_result["direction"] = {
            {"p1_delta", neighbors[i].first},
            {"p2_delta", neighbors[i].second}
        };
        analyze_point_results.push_back(neighbor_result);
    }
    output["analyze_point_results"] = analyze_point_results;
    
    // Results before displacement jump filtering
    output["before_filtering"] = {
        {"A_v", array2d_to_json(A_v_before)},
        {"A_u", array2d_to_json(A_u_before)},
        {"A_vp", array2d_to_json(A_vp_before)},
        {"A_cc", array2d_to_json(A_cc_before)}
    };
    
    // Results after displacement jump filtering
    output["after_filtering"] = {
        {"A_v", array2d_to_json(A_v)},
        {"A_u", array2d_to_json(A_u)},
        {"A_vp", array2d_to_json(A_vp_after)},
        {"A_cc", array2d_to_json(A_cc)}
    };
    
    // Final ROI
    output["roi_valid"] = roi2d_to_json(roi_valid);
    
    // Save JSON
    std::ofstream output_file(directory + "/rgdic_verification.json");
    output_file << std::setw(4) << output << std::endl;
    
    std::cout << "Verification data saved to " << directory << "/rgdic_verification.json" << std::endl;
}

int main(int argc, char *argv[]) {
    try {
        save_rgdic_verification("verification_json");
        std::cout << "RGDIC verification data generation complete!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
