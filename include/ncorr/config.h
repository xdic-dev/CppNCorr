#pragma once
/**
 * @file config.h
 * @brief Compiled-default DIC parameters and the three-tier override chain.
 *
 * CppNCorr resolves tuneable parameters from three sources, highest priority
 * first:
 *   1. Direct CLI arguments         (applied by the caller, e.g. proxyncorr)
 *   2. A config file (INI format)   (see config/default.cfg)
 *   3. Compiled defaults            (the initial values of @ref Config below)
 *
 * The config file uses a minimal **INI** format (chosen over TOML to avoid
 * adding any third-party dependency; a tiny in-tree parser handles it — see
 * @ref IniFile in ini.h). Keys in the file MUST exactly match the field names of
 * @ref Config so there are no silent mismatches.
 */

#include <string>

namespace ncorr {

/**
 * @brief All tuneable ncorr parameters with their compiled-in defaults.
 *
 * The field names here are the canonical config keys. The shipped
 * config/default.cfg mirrors every field one-to-one.
 */
struct Config {
    // --- Pyramid / search --------------------------------------------------//
    /// Pyramid scale factor (downsampling level used for the seed search).
    int scalefactor = 3;

    // --- Subregion (correlation window) ------------------------------------//
    /// Subregion shape: "CIRCLE" or "SQUARE".
    std::string subregion_type = "CIRCLE";
    /// Subregion (correlation window) radius in pixels.
    int subregion_radius = 20;

    // --- Interpolation -----------------------------------------------------//
    /// Interpolation/order: NEAREST, LINEAR, CUBIC_KEYS[_PRECOMPUTE],
    /// QUINTIC_BSPLINE[_PRECOMPUTE].
    std::string interp_type = "QUINTIC_BSPLINE_PRECOMPUTE";

    // --- Strain ------------------------------------------------------------//
    /// Strain subregion shape: "CIRCLE" or "SQUARE".
    std::string strain_subregion_type = "CIRCLE";
    /// Strain window (subregion) radius in pixels.
    int strain_radius = 5;

    // --- DIC behaviour -----------------------------------------------------//
    /// Reference-update policy: NO_UPDATE, KEEP_MOST_POINTS, REMOVE_BAD_POINTS.
    std::string dic_config = "NO_UPDATE";
    /// Number of worker threads for parallel analysis.
    int num_threads = 4;
    /// Enable verbose debug output from the engine.
    bool debug = false;

    // --- Perspective / units ----------------------------------------------//
    /// Interpolation used by the perspective change step.
    std::string perspective_interp = "CUBIC_KEYS";
    /// Physical units label for displacement output.
    std::string units = "mm";
    /// Physical units per pixel scaling.
    double units_per_pixel = 0.2;

    // --- Output / video ----------------------------------------------------//
    /// Video overlay alpha (0..1).
    double alpha = 0.5;
    /// Output video frames-per-second.
    double fps = 15.0;
};

/**
 * @brief Load compiled defaults, then overlay values from an INI config file.
 *
 * Only keys present in the file override the corresponding @ref Config field;
 * unknown keys are ignored (optionally reported by the parser). Missing file is
 * treated as "use defaults".
 *
 * @param config_path Path to an INI config file (e.g. config/default.cfg).
 * @param out         Config to update in place (start from compiled defaults).
 * @return true if the file was found and parsed; false if it did not exist
 *         (in which case @p out is left at its incoming/default values).
 * @throws std::runtime_error on a malformed value (e.g. non-integer for an int).
 */
bool load_config_file(const std::string& config_path, Config& out);

}  // namespace ncorr
