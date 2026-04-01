#include "ncorr.h"
#include "ncorr/io/binary_io.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace ncorr {

DIC_analysis_input DIC_analysis_input::load(std::ifstream &is) {
    DIC_analysis_input DIC_input;

    const auto num_images = io::read_pod<difference_type>(is);
    DIC_input.imgs.resize(num_images);
    for (auto &img : DIC_input.imgs) {
        img = Image2D::load(is);
    }

    DIC_input.roi = ROI2D::load(is);
    DIC_input.scalefactor = io::read_pod<difference_type>(is);
    DIC_input.interp_type = io::read_pod<INTERP>(is);
    DIC_input.subregion_type = io::read_pod<SUBREGION>(is);
    DIC_input.r = io::read_pod<difference_type>(is);
    DIC_input.num_threads = io::read_pod<difference_type>(is);
    DIC_input.cutoff_corrcoef = io::read_pod<double>(is);
    DIC_input.update_corrcoef = io::read_pod<double>(is);
    DIC_input.prctile_corrcoef = io::read_pod<double>(is);
    DIC_input.roi_update_mode = io::read_pod<ROI_UPDATE_MODE>(is);
    DIC_input.accumulation_mode = io::read_pod<ACCUMULATION_MODE>(is);
    DIC_input.save_disps_steps = io::read_pod<bool>(is);
    DIC_input.debug = io::read_pod<bool>(is);

    return DIC_input;
}

DIC_analysis_input DIC_analysis_input::load(const std::string &filename) {
    std::ifstream is(filename.c_str(), std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for loading DIC_analysis_input.");
    }

    auto DIC_input = DIC_analysis_input::load(is);
    is.close();
    return DIC_input;
}

void save(const DIC_analysis_input &DIC_input, std::ofstream &os) {
    typedef ROI2D::difference_type difference_type;

    io::write_pod(os, static_cast<difference_type>(DIC_input.imgs.size()));
    for (const auto &img : DIC_input.imgs) {
        save(img, os);
    }

    save(DIC_input.roi, os);
    io::write_pod(os, DIC_input.scalefactor);
    io::write_pod(os, DIC_input.interp_type);
    io::write_pod(os, DIC_input.subregion_type);
    io::write_pod(os, DIC_input.r);
    io::write_pod(os, DIC_input.num_threads);
    io::write_pod(os, DIC_input.cutoff_corrcoef);
    io::write_pod(os, DIC_input.update_corrcoef);
    io::write_pod(os, DIC_input.prctile_corrcoef);
    io::write_pod(os, DIC_input.roi_update_mode);
    io::write_pod(os, DIC_input.accumulation_mode);
    io::write_pod(os, DIC_input.save_disps_steps);
    io::write_pod(os, DIC_input.debug);
}

void save(const DIC_analysis_input &DIC_input, const std::string &filename) {
    std::ofstream os(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for saving DIC_analysis_input.");
    }

    save(DIC_input, os);
    os.close();
}

DIC_analysis_output DIC_analysis_output::load(std::ifstream &is) {
    DIC_analysis_output DIC_output;

    const auto num_disps = io::read_pod<difference_type>(is);
    DIC_output.disps.resize(num_disps);
    for (auto &disp : DIC_output.disps) {
        disp = Disp2D::load(is);
    }

    DIC_output.perspective_type = io::read_pod<PERSPECTIVE>(is);
    DIC_output.units = io::read_string<difference_type>(is);
    DIC_output.units_per_pixel = io::read_pod<double>(is);

    return DIC_output;
}

DIC_analysis_output DIC_analysis_output::load(const std::string &filename) {
    std::ifstream is(filename.c_str(), std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for loading DIC_analysis_output.");
    }

    auto DIC_output = DIC_analysis_output::load(is);
    is.close();
    return DIC_output;
}

void save(const DIC_analysis_output &DIC_output, std::ofstream &os) {
    typedef ROI2D::difference_type difference_type;

    io::write_pod(os, static_cast<difference_type>(DIC_output.disps.size()));
    for (const auto &disp : DIC_output.disps) {
        save(disp, os);
    }

    io::write_pod(os, DIC_output.perspective_type);
    io::write_string<difference_type>(os, DIC_output.units);
    io::write_pod(os, DIC_output.units_per_pixel);
}

void save(const DIC_analysis_output &DIC_output, const std::string &filename) {
    std::ofstream os(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for saving DIC_analysis_output.");
    }

    save(DIC_output, os);
    os.close();
}

DIC_analysis_step_data DIC_analysis_step_data::load(std::ifstream &is) {
    typedef ROI2D::difference_type difference_type;

    DIC_analysis_step_data step_data;

    const auto num_disps = io::read_pod<difference_type>(is);
    step_data.step_disps.resize(num_disps);
    for (auto &disp : step_data.step_disps) {
        disp = Disp2D::load(is);
    }

    const auto num_rois = io::read_pod<difference_type>(is);
    step_data.step_rois.resize(num_rois);
    for (auto &roi : step_data.step_rois) {
        roi = ROI2D::load(is);
    }

    const auto num_ref_idx = io::read_pod<difference_type>(is);
    step_data.step_ref_idx.resize(num_ref_idx);
    for (auto &idx : step_data.step_ref_idx) {
        idx = io::read_pod<difference_type>(is);
    }

    return step_data;
}

DIC_analysis_step_data DIC_analysis_step_data::load(const std::string &filename) {
    std::ifstream is(filename.c_str(), std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for loading DIC_analysis_step_data.");
    }

    auto step_data = DIC_analysis_step_data::load(is);
    is.close();
    return step_data;
}

void save(const DIC_analysis_step_data &step_data, std::ofstream &os) {
    typedef ROI2D::difference_type difference_type;

    io::write_pod(os, static_cast<difference_type>(step_data.step_disps.size()));
    for (const auto &disp : step_data.step_disps) {
        save(disp, os);
    }

    io::write_pod(os, static_cast<difference_type>(step_data.step_rois.size()));
    for (const auto &roi : step_data.step_rois) {
        save(roi, os);
    }

    io::write_pod(os, static_cast<difference_type>(step_data.step_ref_idx.size()));
    for (const auto &idx : step_data.step_ref_idx) {
        io::write_pod(os, idx);
    }
}

void save(const DIC_analysis_step_data &step_data, const std::string &filename) {
    std::ofstream os(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for saving DIC_analysis_step_data.");
    }

    save(step_data, os);
    os.close();
}

strain_analysis_input strain_analysis_input::load(std::ifstream &is) {
    strain_analysis_input strain_input;

    strain_input.DIC_input = DIC_analysis_input::load(is);
    strain_input.DIC_output = DIC_analysis_output::load(is);
    strain_input.subregion_type = io::read_pod<SUBREGION>(is);
    strain_input.r = io::read_pod<difference_type>(is);

    return strain_input;
}

strain_analysis_input strain_analysis_input::load(const std::string &filename) {
    std::ifstream is(filename.c_str(), std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for loading strain_analysis_input.");
    }

    auto strain_input = strain_analysis_input::load(is);
    is.close();
    return strain_input;
}

void save(const strain_analysis_input &strain_input, std::ofstream &os) {
    save(strain_input.DIC_input, os);
    save(strain_input.DIC_output, os);
    io::write_pod(os, strain_input.subregion_type);
    io::write_pod(os, strain_input.r);
}

void save(const strain_analysis_input &strain_input, const std::string &filename) {
    std::ofstream os(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for saving strain_analysis_input.");
    }

    save(strain_input, os);
    os.close();
}

strain_analysis_output strain_analysis_output::load(std::ifstream &is) {
    strain_analysis_output strain_output;

    const auto num_strains = io::read_pod<difference_type>(is);
    strain_output.strains.resize(num_strains);
    for (auto &strain : strain_output.strains) {
        strain = Strain2D::load(is);
    }

    return strain_output;
}

strain_analysis_output strain_analysis_output::load(const std::string &filename) {
    std::ifstream is(filename.c_str(), std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for loading strain_analysis_output.");
    }

    auto strain_output = strain_analysis_output::load(is);
    is.close();
    return strain_output;
}

void save(const strain_analysis_output &strain_output, std::ofstream &os) {
    typedef ROI2D::difference_type difference_type;

    io::write_pod(os, static_cast<difference_type>(strain_output.strains.size()));
    for (const auto &strain : strain_output.strains) {
        save(strain, os);
    }
}

void save(const strain_analysis_output &strain_output, const std::string &filename) {
    std::ofstream os(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for saving strain_analysis_output.");
    }

    save(strain_output, os);
    os.close();
}

} // namespace ncorr
