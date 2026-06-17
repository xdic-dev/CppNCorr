/**
 * @file config.cpp
 * @brief Implements the config-file tier of the override chain.
 *
 * load_config_file() starts from the compiled defaults already present in the
 * passed-in Config and overlays any keys found in the INI file. Each key name in
 * the file matches a Config field exactly (see config/default.cfg).
 */

#include "ncorr/config.h"
#include "ncorr/ini.h"

namespace ncorr {

bool load_config_file(const std::string& config_path, Config& out) {
    IniFile ini;
    if (!ini.load(config_path)) {
        return false;  // No file: leave `out` at its incoming/default values.
    }

    // Pyramid / search
    out.scalefactor = ini.get_int("scalefactor", out.scalefactor);

    // Subregion
    out.subregion_type = ini.get("subregion_type", out.subregion_type);
    out.subregion_radius = ini.get_int("subregion_radius", out.subregion_radius);

    // Interpolation
    out.interp_type = ini.get("interp_type", out.interp_type);

    // Strain
    out.strain_subregion_type = ini.get("strain_subregion_type", out.strain_subregion_type);
    out.strain_radius = ini.get_int("strain_radius", out.strain_radius);

    // DIC behaviour
    out.dic_config = ini.get("dic_config", out.dic_config);
    out.num_threads = ini.get_int("num_threads", out.num_threads);
    out.debug = ini.get_bool("debug", out.debug);

    // Perspective / units
    out.perspective_interp = ini.get("perspective_interp", out.perspective_interp);
    out.units = ini.get("units", out.units);
    out.units_per_pixel = ini.get_double("units_per_pixel", out.units_per_pixel);

    // Output / video
    out.alpha = ini.get_double("alpha", out.alpha);
    out.fps = ini.get_double("fps", out.fps);

    return true;
}

}  // namespace ncorr
