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

// Specialized version for boolean arrays
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

json image2d_to_json(const Image2D& image) {
    json j;
    j["filename"] = image.get_filename();
    j["gs"] = array2d_to_json(image.get_gs());
    
    return j;
}

json roi2d_to_json(const ROI2D& roi) {
    json j;
    j["num_regions"] = roi.size_regions();
    
    // Get mask as array
    Array2D<bool> mask = roi.get_mask();
    j["mask"] = array2d_to_json(mask);
    
    return j;
}

json disp2d_to_json(const Disp2D& disp) {
    json j;
    
    // Get v and u components
    j["v"] = array2d_to_json(disp.get_v().get_array());
    j["u"] = array2d_to_json(disp.get_u().get_array());
    j["roi"] = roi2d_to_json(disp.get_roi());
    j["scalefactor"] = disp.get_scalefactor();
    
    return j;
}

json strain2d_to_json(const Strain2D& strain) {
    json j;
    
    // Get strain components
    j["eyy"] = array2d_to_json(strain.get_eyy().get_array());
    j["exy"] = array2d_to_json(strain.get_exy().get_array());
    j["exx"] = array2d_to_json(strain.get_exx().get_array());
    j["roi"] = roi2d_to_json(strain.get_roi());
    j["scalefactor"] = strain.get_scalefactor();
    
    return j;
}

json dic_input_to_json(const DIC_analysis_input& dic_input) {
    json j;
    
    // Convert ROI
    j["roi"] = roi2d_to_json(dic_input.roi);
    
    // Convert other parameters
    j["scalefactor"] = dic_input.scalefactor;
    j["interp_type"] = static_cast<int>(dic_input.interp_type);
    j["subregion_type"] = static_cast<int>(dic_input.subregion_type);
    j["radius"] = dic_input.r;
    j["num_threads"] = dic_input.num_threads;
    j["cutoff_corrcoef"] = dic_input.cutoff_corrcoef;
    j["update_corrcoef"] = dic_input.update_corrcoef;
    j["prctile_corrcoef"] = dic_input.prctile_corrcoef;
    j["debug"] = dic_input.debug;
    
    // Image paths
    std::vector<std::string> img_paths;
    for (const auto& img : dic_input.imgs) {
        img_paths.push_back(img.get_filename());
    }
    j["img_paths"] = img_paths;
    
    return j;
}

json dic_output_to_json(const DIC_analysis_output& dic_output) {
    json j;
    
    // Convert displacement fields
    json disps_json = json::array();
    for (const auto& disp : dic_output.disps) {
        disps_json.push_back(disp2d_to_json(disp));
    }
    j["disps"] = disps_json;
    
    // Convert other parameters
    j["perspective_type"] = static_cast<int>(dic_output.perspective_type);
    j["units"] = dic_output.units;
    j["units_per_pixel"] = dic_output.units_per_pixel;
    
    return j;
}

json strain_input_to_json(const strain_analysis_input& strain_input) {
    json j;
    
    // Convert DIC input and output
    j["dic_input"] = dic_input_to_json(strain_input.DIC_input);
    j["dic_output"] = dic_output_to_json(strain_input.DIC_output);
    
    // Convert other parameters
    j["subregion_type"] = static_cast<int>(strain_input.subregion_type);
    j["radius"] = strain_input.r;
    
    return j;
}

json strain_output_to_json(const strain_analysis_output& strain_output) {
    json j;
    
    // Convert strain fields
    json strains_json = json::array();
    for (const auto& strain : strain_output.strains) {
        strains_json.push_back(strain2d_to_json(strain));
    }
    j["strains"] = strains_json;
    
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

int main(int argc, char *argv[]) {
    if (argc != 2) {
        throw std::invalid_argument("Must have 1 command line input of either 'calculate' or 'load'");    
    }

    // Create verification directory
    std::string verif_dir = "verification_json";
    system(("mkdir -p " + verif_dir).c_str());

    // Initialize DIC and strain information
    DIC_analysis_input DIC_input;
    DIC_analysis_output DIC_output;
    strain_analysis_input strain_input;
    strain_analysis_output strain_output;

    // Determine whether or not to perform calculations or load data
    std::string input(argv[1]);
    if (input == "load") {
        // Load inputs
        DIC_input = DIC_analysis_input::load("save/DIC_input.bin");
        DIC_output = DIC_analysis_output::load("save/DIC_output.bin");
        strain_input = strain_analysis_input::load("save/strain_input.bin");
        strain_output = strain_analysis_output::load("save/strain_output.bin");
        
        // Save all data as JSON
        save_as_json("DIC_input.json", dic_input_to_json(DIC_input), verif_dir);
        save_as_json("DIC_output.json", dic_output_to_json(DIC_output), verif_dir);
        save_as_json("strain_input.json", strain_input_to_json(strain_input), verif_dir);
        save_as_json("strain_output.json", strain_output_to_json(strain_output), verif_dir);
    } else if (input == "calculate") {
        // STEP 1: Load Images
        std::vector<Image2D> imgs;
        for (int i = 0; i <= 2; ++i) {
            std::ostringstream ostr;
            ostr << "images/ohtcfrp_" << std::setfill('0') << std::setw(2) << i << ".png";
            imgs.push_back(ostr.str());
        }
        
        // Save loaded images
        json images_json = json::array();
        for (size_t i = 0; i < imgs.size(); ++i) {
            images_json.push_back(image2d_to_json(imgs[i]));
        }
        save_as_json("step1_images.json", images_json, verif_dir);
        
        // STEP 2: Create ROI
        ROI2D roi = ROI2D(Image2D("images/roi.png").get_gs() > 0.5);
        save_as_json("step2_roi.json", roi2d_to_json(roi), verif_dir);
        
        // STEP 3: Set DIC_input
        DIC_input = DIC_analysis_input(imgs,                            // Images
                           ROI2D(Image2D("images/roi.png").get_gs() > 0.5),        // ROI
                           3,                                             // scalefactor
                           INTERP::CUBIC_KEYS,            // Interpolation
                           SUBREGION::CIRCLE,                    // Subregion shape
                           20,                                            // Subregion radius
                           4,                                             // # of threads
                           DIC_analysis_config::NO_UPDATE,                // DIC configuration for reference image updates
                           true);                            // Debugging enabled/disabled
        
        save_as_json("step3_dic_input.json", dic_input_to_json(DIC_input), verif_dir);
        
        // STEP 4: Perform DIC_analysis    
        DIC_output = DIC_analysis(DIC_input);
        save_as_json("step4_dic_output_raw.json", dic_output_to_json(DIC_output), verif_dir);
        
        // // STEP 5: Convert DIC_output to Eulerian perspective
        // DIC_output = change_perspective(DIC_output, INTERP::CUBIC_KEYS);
        // save_as_json("step5_dic_output_eulerian.json", dic_output_to_json(DIC_output), verif_dir);
        
        // // STEP 6: Set units of DIC_output
        // DIC_output = set_units(DIC_output, "mm", 0.2);
        // save_as_json("step6_dic_output_with_units.json", dic_output_to_json(DIC_output), verif_dir);
        
        // // STEP 7: Set strain input
        // strain_input = strain_analysis_input(DIC_input,
        //                               DIC_output,
        //                               SUBREGION::CIRCLE,                    // Strain subregion shape
        //                               5);                        // Strain subregion radius
        
        // save_as_json("step7_strain_input.json", strain_input_to_json(strain_input), verif_dir);
        
        // // STEP 8: Perform strain_analysis
        // strain_output = strain_analysis(strain_input); 
        // save_as_json("step8_strain_output.json", strain_output_to_json(strain_output), verif_dir);
        
        // // Save outputs as binary
        // save(DIC_input, "save/DIC_input.bin");
        // save(DIC_output, "save/DIC_output.bin");
        // save(strain_input, "save/strain_input.bin");
        // save(strain_output, "save/strain_output.bin");
    } else {
        throw std::invalid_argument("Input of " + input + " is not recognized. Must be either 'calculate' or 'load'");    
    }

    std::cout << "Verification data saved to " << verif_dir << " directory." << std::endl;
    return 0;
}
