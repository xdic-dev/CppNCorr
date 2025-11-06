#include "ncorr.h"
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace ncorr;

// Helper function to save Array2D<bool> as JSON
void save_mask_json(const Array2D<bool>& mask, std::ofstream& file, const std::string& name) {
    file << "\"" << name << "\": {\n";
    file << "  \"rows\": " << mask.height() << ",\n";
    file << "  \"cols\": " << mask.width() << ",\n";
    file << "  \"data\": [";
    
    for (std::ptrdiff_t i = 0; i < mask.height(); ++i) {
        for (std::ptrdiff_t j = 0; j < mask.width(); ++j) {
            if (i > 0 || j > 0) file << ", ";
            file << (mask(i, j) ? "true" : "false");
        }
    }
    file << "]\n}";
}

// Helper function to save region_nlinfo as JSON
void save_nlinfo_json(const ROI2D::region_nlinfo& nlinfo, std::ofstream& file, const std::string& name) {
    file << "\"" << name << "\": {\n";
    file << "  \"empty\": " << (nlinfo.empty() ? "true" : "false") << ",\n";
    file << "  \"points\": " << nlinfo.points << ",\n";
    file << "  \"top\": " << nlinfo.top << ",\n";
    file << "  \"bottom\": " << nlinfo.bottom << ",\n";
    file << "  \"left\": " << nlinfo.left << ",\n";
    file << "  \"right\": " << nlinfo.right << ",\n";
    file << "  \"left_nl\": " << nlinfo.left_nl << ",\n";
    file << "  \"right_nl\": " << nlinfo.right_nl << ",\n";
    
    // Save nodelist
    file << "  \"nodelist\": {\n";
    file << "    \"rows\": " << nlinfo.nodelist.height() << ",\n";
    file << "    \"cols\": " << nlinfo.nodelist.width() << ",\n";
    file << "    \"data\": [";
    for (std::ptrdiff_t i = 0; i < nlinfo.nodelist.height(); ++i) {
        for (std::ptrdiff_t j = 0; j < nlinfo.nodelist.width(); ++j) {
            if (i > 0 || j > 0) file << ", ";
            file << nlinfo.nodelist(i, j);
        }
    }
    file << "]\n  },\n";
    
    // Save noderange
    file << "  \"noderange\": {\n";
    file << "    \"rows\": " << nlinfo.noderange.height() << ",\n";
    file << "    \"cols\": " << nlinfo.noderange.width() << ",\n";
    file << "    \"data\": [";
    for (std::ptrdiff_t i = 0; i < nlinfo.noderange.height(); ++i) {
        for (std::ptrdiff_t j = 0; j < nlinfo.noderange.width(); ++j) {
            if (i > 0 || j > 0) file << ", ";
            file << nlinfo.noderange(i, j);
        }
    }
    file << "]\n  }\n";
    file << "}";
}

// Helper function to save ROI2D as JSON
void save_roi_json(const ROI2D& roi, std::ofstream& file, const std::string& name) {
    file << "\"" << name << "\": {\n";
    file << "  \"empty\": " << (roi.empty() ? "true" : "false") << ",\n";
    file << "  \"points\": " << roi.get_points() << ",\n";
    file << "  \"height\": " << roi.height() << ",\n";
    file << "  \"width\": " << roi.width() << ",\n";
    file << "  \"size_regions\": " << roi.size_regions() << ",\n";
    
    // Save mask
    save_mask_json(roi.get_mask(), file, "mask");
    file << ",\n";
    
    // Save regions
    file << "  \"regions\": [\n";
    for (std::ptrdiff_t i = 0; i < roi.size_regions(); ++i) {
        if (i > 0) file << ",\n";
        file << "    {\n      ";
        save_nlinfo_json(roi.get_nlinfo(i), file, "nlinfo");
        file << "\n    }";
    }
    file << "\n  ]\n";
    file << "}";
}

void save_array2d_json(const Array2D<std::vector<ROI2D::difference_type>> &vector, std::ofstream& file, const std::string& name) {
    file << "\"" << name << "\": {\n";
    file << "  \"rows\": " << vector.height() << ",\n";
    file << "  \"cols\": " << vector.width() << ",\n";
    file << "  \"data\": [";
    
    for (std::ptrdiff_t i = 0; i < vector.height(); ++i) {
        for (std::ptrdiff_t j = 0; j < vector.width(); ++j) {
            if (i > 0 || j > 0) file << ", ";
            file << "[";
            for (std::ptrdiff_t k = 0; k < vector(i,j).size(); ++k) {
                if (k > 0) file << ", ";
                file << vector(i,j)[k];
            }
            file << "]";
        }
    }
    file << "]\n}\n";
}

void save_array2d_json(const Array2D<std::vector<bool>> &vector, std::ofstream& file, const std::string& name) {
    file << "\"" << name << "\": {\n";
    file << "  \"rows\": " << vector.height() << ",\n";
    file << "  \"cols\": " << vector.width() << ",\n";
    file << "  \"data\": [";
    
    for (std::ptrdiff_t i = 0; i < vector.height(); ++i) {
        for (std::ptrdiff_t j = 0; j < vector.width(); ++j) {
            if (i > 0 || j > 0) file << ", ";
            file << "[";
            for (std::ptrdiff_t k = 0; k < vector(i,j).size(); ++k) {
                if (k > 0) file << ", ";
                file << (vector(i,j)[k] ? "true" : "false");
            }
            file << "]";
        }
    }
    file << "]\n}\n";
}

void save_nd_vector_as_json(const Array2D<std::vector<ROI2D::difference_type>> &vector, const std::string &filename) {
    std::ofstream file(filename);
    file << "{\n";
    save_array2d_json(vector, file, "vector");
    file << "}\n";
    file.close();
}

void save_np_vector_as_json(const Array2D<std::vector<bool>> &vector, const std::string &filename) {
    std::ofstream file(filename);
    file << "{\n";
    save_array2d_json(vector, file, "vector");
    file << "}\n";
    file.close();
}

void save_nlinfo_vector_as_json(const std::vector<ROI2D::region_nlinfo> &vector, const std::string &filename) {
    std::ofstream file(filename);
    file << "{\n";
    for (std::size_t i = 0; i < vector.size(); ++i) {
        save_nlinfo_json(vector[i], file, "nlinfo" + std::to_string(i));
    }
    file << "}\n";
    file.close();
}

void add_interacting_nodes_bis(ROI2D::difference_type np_adj_p2, 
                            ROI2D::difference_type np_loaded_top,
                            ROI2D::difference_type np_loaded_bottom,
                            Array2D<std::vector<ROI2D::difference_type>> &overall_nodelist,
                            Array2D<std::vector<bool>> &overall_active_nodepairs,
                            std::stack<ROI2D::difference_type> &queue_np_idx) {
    typedef ROI2D::difference_type                          difference_type;
    
    // Make sure adjacent nodepair(s) position is in range.
    if (overall_nodelist.in_bounds(np_adj_p2)) {
        // Scans nodes from top to bottom
        for (difference_type np_adj_idx = 0; np_adj_idx < difference_type(overall_nodelist(np_adj_p2).size()); np_adj_idx += 2) {
            difference_type np_adj_top = overall_nodelist(np_adj_p2)[np_adj_idx];
            difference_type np_adj_bottom = overall_nodelist(np_adj_p2)[np_adj_idx + 1];
            if (np_loaded_bottom < np_adj_top) {
                return; // top node of adjacent nodepair is below bottom node of loaded nodepair
            }                 
            if (overall_active_nodepairs(np_adj_p2)[np_adj_idx/2] && np_loaded_top <= np_adj_bottom) {
                // Inactivate node pair, and then insert into the queue
                overall_active_nodepairs(np_adj_p2)[np_adj_idx/2] = false;
                queue_np_idx.push(np_adj_top);
                queue_np_idx.push(np_adj_bottom);
                queue_np_idx.push(np_adj_p2);
            } 
        }
    }
}

std::pair<std::vector<ROI2D::region_nlinfo>,bool> form_nlinfos_bis(const Array2D<bool> &mask, ROI2D::difference_type cutoff) {            
    if (cutoff < 0) {        
        throw std::invalid_argument("Attempted to form nlinfos with cutoff of: " + std::to_string(cutoff) + 
                                    ". cutoff must be an integer of value 0 or greater.");
    }

    // Form overall_nodelist and overall_active_nodepairs --------------------//
    Array2D<std::vector<ROI2D::difference_type>> overall_nodelist(1,mask.width());
    Array2D<std::vector<bool>> overall_active_nodepairs(1,mask.width());
    for (ROI2D::difference_type p2 = 0; p2 < mask.width(); ++p2) {
        bool in_nodepair = false;
        for (ROI2D::difference_type p1 = 0; p1 < mask.height(); ++p1) { 
            if (!in_nodepair && mask(p1,p2)) {
                in_nodepair = true;
                overall_nodelist(p2).push_back(p1); // sets top node
            }
            if (in_nodepair && (!mask(p1,p2) || p1 == mask.height()-1)) {
                in_nodepair = false;
                overall_nodelist(p2).push_back((p1 == mask.height()-1 && mask(p1,p2)) ? p1 : p1-1); // Sets bottom node
            }
        }

        // Update overall_active_nodepairs - this keeps track of which node
        // pairs have been analyzed when doing contiguous separation; set to 
        // true initially.
        overall_active_nodepairs(p2).resize(overall_nodelist(p2).size()/2, true);
    }

    save_nd_vector_as_json(overall_nodelist, "roi2d_test_output/overall_nodelist.json");
    save_np_vector_as_json(overall_active_nodepairs, "roi2d_test_output/overall_active_nodepairs.json");

    // Separate regions ------------------------------------------------------//   
    // Regions are made 4-way contiguous. Scan over columns and separate 
    // contiguous nodelists in overall_nodelist.
    std::vector<ROI2D::region_nlinfo> nlinfos; // This will get updated and returned
    bool removed = false;                      // Keeps track if regions are removed due to "cutoff" parameter
    for (ROI2D::difference_type p2_sweep = 0; p2_sweep < overall_nodelist.width(); ++p2_sweep) {
        // Find first active node pair
        ROI2D::difference_type np_init_idx = -1;
        for (ROI2D::difference_type np_idx = 0; np_idx < ROI2D::difference_type(overall_nodelist(p2_sweep).size()); np_idx += 2) {
            if (overall_active_nodepairs(p2_sweep)[np_idx/2]) {       // Test if nodepair is active
                overall_active_nodepairs(p2_sweep)[np_idx/2] = false; // Inactivate node pair
                np_init_idx = np_idx;                                 // Store nodepair idx
                break;
            }
        }

        // If there are no active node pairs, then continue to next column
        if (np_init_idx == -1) {
            continue;
        }

        std::cout << "p2_sweep: " << p2_sweep << " overall_active_nodepairs:" << std::endl;
        for (ROI2D::difference_type xxx = 0; xxx < ROI2D::difference_type(overall_nodelist(p2_sweep).size()); xxx += 2) {
            std::cout << overall_active_nodepairs(p2_sweep)[xxx/2] << " ";
        }
        std::cout << std::endl;
        std::cout << "p2_sweep: " << p2_sweep << " overall_nodelist:" << std::endl;
        for (ROI2D::difference_type xxx = 0; xxx < ROI2D::difference_type(overall_nodelist(p2_sweep).size()); xxx += 1) {
            std::cout << overall_nodelist(p2_sweep)[xxx] << " ";
        }
        std::cout << std::endl;

        // nlinfo_buf to be updated, then inserted into nlinfos
        ROI2D::region_nlinfo nlinfo_buf(mask.height()-1,    // Top bound of region                (this gets updated)
                                        0,                  // Bottom bound of region             (this gets updated)
                                        p2_sweep,           // Left bound of region               (this is correct)
                                        0,                  // Right bound of region              (this gets updated)
                                        p2_sweep,           // Left position of nodelist          (this is correct)
                                        0,                  // Right position of nodelist         (this gets updated)
                                        0,                  // h of nodelist                      (this gets updated)  
                                        0,                  // w of nodelist                      (this gets updated)     
                                        0);                 // Number of points in region         (this gets updated) 

        // Keep track of nodes with separate_nodelist
        Array2D<std::vector<ROI2D::difference_type>> separate_nodelist(1,mask.width());

        // Initialize queue and enter while loop - exit when queue is empty
        std::stack<ROI2D::difference_type> queue_np_idx;                       // Holds all nodepairs (along with their index) which need to be processed
        queue_np_idx.push(overall_nodelist(p2_sweep)[np_init_idx]);     // Top
        queue_np_idx.push(overall_nodelist(p2_sweep)[np_init_idx + 1]); // Bottom
        queue_np_idx.push(p2_sweep);                                    // position of nodepair

        std::cout << ">> p2_sweep: " << p2_sweep << " -> Queue size: " << queue_np_idx.size() << std::endl;
        std::cout << "Queue: ";
        std::cout << overall_nodelist(p2_sweep)[np_init_idx] << " " << overall_nodelist(p2_sweep)[np_init_idx + 1] << " " << p2_sweep << std::endl;
        while (!queue_np_idx.empty()) {
            // Pop nodepair and its position out of queue and compare 
            // it to adjacent nodepairs (left and right of np_loaded_p2)
            ROI2D::difference_type np_loaded_p2 = queue_np_idx.top(); queue_np_idx.pop();
            ROI2D::difference_type np_loaded_bottom = queue_np_idx.top(); queue_np_idx.pop();
            ROI2D::difference_type np_loaded_top = queue_np_idx.top(); queue_np_idx.pop();
            
            std::cout << "Queue size: " << queue_np_idx.size() << " np_loaded_p2: " << np_loaded_p2 << " np_loaded_top: " << np_loaded_top << " np_loaded_bottom: " << np_loaded_bottom << std::endl;

            // Compare to node pairs LEFT. Any node pairs which interact are added to the queue
            add_interacting_nodes_bis(np_loaded_p2 - 1, 
                                           np_loaded_top, 
                                           np_loaded_bottom,
                                           overall_nodelist, 
                                           overall_active_nodepairs, 
                                           queue_np_idx);
            
            std::cout << "Queue size: " << queue_np_idx.size() << std::endl;

            // Compare to node pairs RIGHT. Any node pairs which interact are added to the queue
            add_interacting_nodes_bis(np_loaded_p2 + 1, 
                                           np_loaded_top, 
                                           np_loaded_bottom, 
                                           overall_nodelist, 
                                           overall_active_nodepairs, 
                                           queue_np_idx);
            
            std::cout << "Queue size: " << queue_np_idx.size() << std::endl;

            // Update points
            nlinfo_buf.points += np_loaded_bottom - np_loaded_top + 1;

            // Update bounds - note that "right_nl" and "right" are the same
            if (np_loaded_top < nlinfo_buf.top) { nlinfo_buf.top = np_loaded_top; }                         // Top
            if (np_loaded_bottom > nlinfo_buf.bottom) { nlinfo_buf.bottom = np_loaded_bottom; }             // Bottom
            if (np_loaded_p2 > nlinfo_buf.right) { nlinfo_buf.right_nl = nlinfo_buf.right = np_loaded_p2; } // Right

            // Insert node pairs and then sort - usually very small so BST isn't
            // necessary.
            separate_nodelist(np_loaded_p2).push_back(np_loaded_top);
            separate_nodelist(np_loaded_p2).push_back(np_loaded_bottom);
            std::sort(separate_nodelist(np_loaded_p2).begin(), separate_nodelist(np_loaded_p2).end());
        }

        // Now finish setting nodelist and noderange for this region.
        // Find max nodes first so we can use it to set the correct height
        // for nodelist.
        ROI2D::difference_type max_nodes = 0;
        for (const auto &nodes : separate_nodelist) {
            if (ROI2D::difference_type(nodes.size()) > max_nodes) {
                max_nodes = nodes.size();
            }
        }

        // Set and fill nodelist and noderange
        nlinfo_buf.nodelist = Array2D<ROI2D::difference_type>(max_nodes, nlinfo_buf.right_nl - nlinfo_buf.left_nl + 1);
        nlinfo_buf.noderange = Array2D<ROI2D::difference_type>(1, nlinfo_buf.right_nl - nlinfo_buf.left_nl + 1);
        for (ROI2D::difference_type nl_idx = 0; nl_idx < nlinfo_buf.nodelist.width(); ++nl_idx) {
            ROI2D::difference_type p2 = nl_idx + nlinfo_buf.left_nl;
            // noderange:
            nlinfo_buf.noderange(nl_idx) = separate_nodelist(p2).size();
            // nodelist:
            for (ROI2D::difference_type np_idx = 0; np_idx < ROI2D::difference_type(separate_nodelist(p2).size()); ++np_idx) {
                nlinfo_buf.nodelist(np_idx,nl_idx) = separate_nodelist(p2)[np_idx];
            }
        }

        // Subtract one from p2_sweep in order to recheck the column to 
        // ensure all nodes are deactivated before proceeding
        --p2_sweep;

        // Make sure number of points in nlinfo is greater than or equal to
        // the cutoff
        if (nlinfo_buf.points >= cutoff) {      
            nlinfos.push_back(nlinfo_buf);
        } else {
            removed = true; // Parameter lets caller know regions were removed
        }
    }

    save_nlinfo_vector_as_json(nlinfos, "roi2d_test_output/nlinfos.json");

    return {std::move(nlinfos), removed};
}

int main() {
    try {
        std::cout << "Starting ROI2D reduce and form_union test..." << std::endl;
        
        // Create output file
        std::ofstream output_file("roi2d_test_output/roi2d_test_results.json");
        output_file << "{\n";
        output_file << "  \"test_info\": {\n";
        output_file << "    \"description\": \"ROI2D reduce and form_union function tests\",\n";
        output_file << "    \"version\": \"1.0\"\n";
        output_file << "  },\n";
        
        // Test 1: Simple rectangular ROI
        std::cout << "Test 1: Simple rectangular ROI..." << std::endl;
        Array2D<bool> mask1(100, 80);
        mask1() = false;
        // Create a 40x30 rectangle starting at (20, 15)
        for (std::ptrdiff_t i = 20; i < 60; ++i) {
            for (std::ptrdiff_t j = 15; j < 45; ++j) {
                mask1(i, j) = true;
            }
        }  
        
        ROI2D roi1(std::move(mask1));
        output_file << "  \"test1\": {\n";
        output_file << "    \"description\": \"Simple rectangular ROI\",\n";
        output_file << "    ";
        save_roi_json(roi1, output_file, "original");
        output_file << ",\n";
        
        // Test reduce with scalefactor 2
        ROI2D roi1_reduced2 = roi1.reduce(2);
        output_file << "    ";
        save_roi_json(roi1_reduced2, output_file, "reduced_sf2");
        output_file << ",\n";
        
        // Test reduce with scalefactor 3
        ROI2D roi1_reduced3 = roi1.reduce(3);
        output_file << "    ";
        save_roi_json(roi1_reduced3, output_file, "reduced_sf3");
        output_file << ",\n";
        
        // Test form_union with a mask that adds some pixels
        Array2D<bool> union_mask1(100, 80);
        union_mask1() = false;
        // Add pixels around the original rectangle
        for (std::ptrdiff_t i = 18; i < 62; ++i) {
            for (std::ptrdiff_t j = 13; j < 47; ++j) {
                union_mask1(i, j) = true;
            }
        }
        
        ROI2D roi1_union = roi1.form_union(union_mask1);
        output_file << "    ";
        save_mask_json(union_mask1, output_file, "union_mask");
        output_file << ",\n";
        output_file << "    ";
        save_roi_json(roi1_union, output_file, "union_result");
        output_file << "\n  },\n";
        
        // Test 2: Complex ROI with holes
        std::cout << "Test 2: Complex ROI with holes..." << std::endl;
        Array2D<bool> mask2(120, 100);
        mask2() = false;
        
        // Create outer rectangle
        for (std::ptrdiff_t i = 10; i < 110; ++i) {
            for (std::ptrdiff_t j = 10; j < 90; ++j) {
                mask2(i, j) = true;
            }
        }
        
        // Create holes
        for (std::ptrdiff_t i = 30; i < 50; ++i) {
            for (std::ptrdiff_t j = 30; j < 50; ++j) {
                mask2(i, j) = false;
            }
        }
        for (std::ptrdiff_t i = 70; i < 90; ++i) {
            for (std::ptrdiff_t j = 60; j < 80; ++j) {
                mask2(i, j) = false;
            }
        }
        
        ROI2D roi2(std::move(mask2));
        output_file << "  \"test2\": {\n";
        output_file << "    \"description\": \"Complex ROI with holes\",\n";
        output_file << "    ";
        save_roi_json(roi2, output_file, "original");
        output_file << ",\n";
        
        // Test reduce with scalefactor 2
        ROI2D roi2_reduced2 = roi2.reduce(2);
        output_file << "    ";
        save_roi_json(roi2_reduced2, output_file, "reduced_sf2");
        output_file << ",\n";
        
        // Test reduce with scalefactor 4
        ROI2D roi2_reduced4 = roi2.reduce(4);
        output_file << "    ";
        save_roi_json(roi2_reduced4, output_file, "reduced_sf4");
        output_file << "\n  },\n";
        
        // Test 3: Small ROI that might disappear with large scalefactor
        std::cout << "Test 3: Small ROI with large scalefactor..." << std::endl;
        Array2D<bool> mask3(50, 50);
        mask3() = false;
        
        // Create small 5x5 square
        for (std::ptrdiff_t i = 20; i < 25; ++i) {
            for (std::ptrdiff_t j = 20; j < 25; ++j) {
                mask3(i, j) = true;
            }
        }
        
        ROI2D roi3(std::move(mask3));
        output_file << "  \"test3\": {\n";
        output_file << "    \"description\": \"Small ROI with large scalefactor\",\n";
        output_file << "    ";
        save_roi_json(roi3, output_file, "original");
        output_file << ",\n";
        
        // Test reduce with scalefactor 2
        ROI2D roi3_reduced2 = roi3.reduce(2);
        output_file << "    ";
        save_roi_json(roi3_reduced2, output_file, "reduced_sf2");
        output_file << ",\n";
        
        // Test reduce with scalefactor 5 (should make it very small or empty)
        ROI2D roi3_reduced5 = roi3.reduce(5);
        output_file << "    ";
        save_roi_json(roi3_reduced5, output_file, "reduced_sf5");
        output_file << ",\n";
        
        // Test reduce with scalefactor 10 (should likely be empty)
        ROI2D roi3_reduced10 = roi3.reduce(10);
        output_file << "    ";
        save_roi_json(roi3_reduced10, output_file, "reduced_sf10");
        output_file << "\n  },\n";
        
        // Test 4: Multiple regions ROI
        std::cout << "Test 4: Multiple regions ROI..." << std::endl;
        Array2D<bool> mask4(80, 80);
        mask4() = false;
        
        // Create first region
        for (std::ptrdiff_t i = 10; i < 30; ++i) {
            for (std::ptrdiff_t j = 10; j < 30; ++j) {
                mask4(i, j) = true;
            }
        }
        
        // Create second region (disconnected)
        for (std::ptrdiff_t i = 50; i < 70; ++i) {
            for (std::ptrdiff_t j = 50; j < 70; ++j) {
                mask4(i, j) = true;
            }
        }
        
        ROI2D roi4(std::move(mask4));
        output_file << "  \"test4\": {\n";
        output_file << "    \"description\": \"Multiple regions ROI\",\n";
        output_file << "    ";
        save_roi_json(roi4, output_file, "original");
        output_file << ",\n";
        
        // Test reduce with scalefactor 2
        ROI2D roi4_reduced2 = roi4.reduce(2);
        output_file << "    ";
        save_roi_json(roi4_reduced2, output_file, "reduced_sf2");
        output_file << ",\n";
        
        // Test form_union to connect the regions
        Array2D<bool> union_mask4(80, 80);
        union_mask4() = false;
        
        // Fill the gap between regions
        for (std::ptrdiff_t i = 30; i < 50; ++i) {
            for (std::ptrdiff_t j = 30; j < 50; ++j) {
                union_mask4(i, j) = true;
            }
        }
        
        ROI2D roi4_union = roi4.form_union(union_mask4);
        output_file << "    ";
        save_mask_json(union_mask4, output_file, "union_mask");
        output_file << ",\n";
        output_file << "    ";
        save_roi_json(roi4_union, output_file, "union_result");
        output_file << "\n  }\n";
        
        output_file << "}\n";
        output_file.close();
        
        std::cout << "Test completed successfully! Results saved to roi2d_test_results.json" << std::endl;


        ////====== form nlinfos from mask ======
        Array2D<bool> maskx(100, 80);
        maskx() = false;
        // Create a 40x30 rectangle starting at (20, 15)
        for (std::ptrdiff_t i = 20; i < 60; ++i) {
            for (std::ptrdiff_t j = 15; j < 45; ++j) {
                maskx(i, j) = true;
            }
        }  

        // Create output file
        std::ofstream output_file_form_nlinfos("roi2d_test_output/roi2d_test_form_nlinfos.json");
        output_file_form_nlinfos << "{\n";
        output_file_form_nlinfos << "  \"test_info\": {\n";
        output_file_form_nlinfos << "    \"description\": \"ROI2D form nlinfos from mask function tests\",\n";
        output_file_form_nlinfos << "    \"version\": \"1.0\"\n";
        output_file_form_nlinfos << "  },\n";

        // save mask
        output_file_form_nlinfos << "    \"original\": {\n";
        save_mask_json(maskx, output_file_form_nlinfos, "mask");
        output_file_form_nlinfos << "\n  },\n";
        
        
        // form nlinfos from mask
        std::pair<std::vector<ROI2D::region_nlinfo>,bool> nlinfos_pair = form_nlinfos_bis(maskx, 0);
        output_file_form_nlinfos << "  \"test nlinfos\": {\n";
        output_file_form_nlinfos << "    \"description\": \"Simple rectangular ROI\",\n";
        output_file_form_nlinfos << "    \"remove\": " << (nlinfos_pair.second ? "true" : "false") << ",\n";
        output_file_form_nlinfos << "    \"nlinfos\": [\n";
        for (std::ptrdiff_t i = 0; i < nlinfos_pair.first.size(); ++i) {
            if (i > 0) output_file_form_nlinfos << ",\n";
            output_file_form_nlinfos << "     {\n";
            save_nlinfo_json(nlinfos_pair.first[i], output_file_form_nlinfos, "nlinfo");

            if (i == nlinfos_pair.first.size() - 1) {
                output_file_form_nlinfos << "\n    }";
            } else {
                output_file_form_nlinfos << "\n    },";
            }
        }
        output_file_form_nlinfos << "\n  ]\n";
        output_file_form_nlinfos << "}\n";
        output_file_form_nlinfos << "}\n";
        output_file_form_nlinfos.close();
        
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
