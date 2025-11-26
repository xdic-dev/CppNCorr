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
void save_as_json(const DIC_analysis_input& dic_input, 
                 const DIC_analysis_output& dic_output,
                 const strain_analysis_input& strain_input,
                 const strain_analysis_output& strain_output,
                 const std::string& directory) {
    // Create JSON objects
    json dic_input_json = dic_input_to_json(dic_input);
    json dic_output_json = dic_output_to_json(dic_output);
    json strain_input_json = strain_input_to_json(strain_input);
    json strain_output_json = strain_output_to_json(strain_output);
    
    // Create directory if it doesn't exist
    system(("mkdir -p " + directory).c_str());
    
    // Save to files
    std::ofstream dic_input_file(directory + "/DIC_input.json");
    dic_input_file << std::setw(4) << dic_input_json << std::endl;
    
    std::ofstream dic_output_file(directory + "/DIC_output.json");
    dic_output_file << std::setw(4) << dic_output_json << std::endl;
    
    std::ofstream strain_input_file(directory + "/strain_input.json");
    strain_input_file << std::setw(4) << strain_input_json << std::endl;
    
    std::ofstream strain_output_file(directory + "/strain_output.json");
    strain_output_file << std::setw(4) << strain_output_json << std::endl;
}

int main(int argc, char *argv[]) {
	// Initialize DIC and strain information ---------------//
	DIC_analysis_input DIC_input;
	DIC_analysis_output DIC_output;
	strain_analysis_input strain_input;
	strain_analysis_output strain_output;

	
	// Set images
	std::vector<Image2D> imgs;
	for (int i = 0; i <= 11; ++i) {
		std::ostringstream ostr;
		ostr << "images/ohtcfrp_" << std::setfill('0') << std::setw(2) << i << ".png";
		imgs.push_back(ostr.str());
	}
	
	// Set DIC_input
	DIC_input = DIC_analysis_input(imgs, 							// Images
									ROI2D(Image2D("images/roi.png").get_gs() > 0.5),		// ROI
									3,                                         		// scalefactor
									INTERP::QUINTIC_BSPLINE_PRECOMPUTE,			// Interpolation
									SUBREGION::CIRCLE,					// Subregion shape
									20,                                        		// Subregion radius
									4,                                         		// # of threads
									DIC_analysis_config::NO_UPDATE,				// DIC configuration for reference image updates
									false);							// Debugging enabled/disabled

	// Perform DIC_analysis    
	DIC_output = DIC_analysis(DIC_input);

	// Convert DIC_output to Eulerian perspective
	DIC_output = change_perspective(DIC_output, INTERP::CUBIC_KEYS);

	// Set units of DIC_output (provide units/pixel)
	DIC_output = set_units(DIC_output, "mm", 0.2);

	// Set strain input
	strain_input = strain_analysis_input(DIC_input,
										DIC_output,
										SUBREGION::CIRCLE,					// Strain subregion shape
										5);						// Strain subregion radius
	
	// Perform strain_analysis
	strain_output = strain_analysis(strain_input); 
	
	// Save outputs as binary
	save(DIC_input, "save/DIC_input.bin");
	save(DIC_output, "save/DIC_output.bin");
	save(strain_input, "save/strain_input.bin");
	save(strain_output, "save/strain_output.bin");	
	
	// Save as JSON
	save_as_json(DIC_input, DIC_output, strain_input, strain_output, "save_json");
        
    // Create Videos ---------------------------------------//
	// Note that more inputs can be used to modify plots. 
	// If video is not saving correctly, try changing the 
	// input codec using cv::VideoWriter::fourcc(...)). Check 
	// the opencv documentation on video codecs. By default, 
	// ncorr uses cv::VideoWriter::fourcc('M','J','P','G')).
	save_DIC_video("video/test_v_eulerian.avi", 
					DIC_input, 
					DIC_output, 
					DISP::V,
					0.5,		// Alpha		
					15);		// FPS

	save_DIC_video("video/test_u_eulerian.avi", 
					DIC_input, 
					DIC_output, 
					DISP::U, 
					0.5,		// Alpha
					15);		// FPS

	save_strain_video("video/test_eyy_eulerian.avi", 
						strain_input, 
						strain_output, 
						STRAIN::EYY, 
						0.5,		// Alpha
						15);		// FPS

	save_strain_video("video/test_exy_eulerian.avi", 
						strain_input, 
						strain_output, 
						STRAIN::EXY, 
						0.5,		// Alpha
						15);		// FPS
	
	save_strain_video("video/test_exx_eulerian.avi", 
						strain_input, 
						strain_output, 
						STRAIN::EXX, 
						0.5,		// Alpha
						15); 		// FPS

  	return 0;
}
