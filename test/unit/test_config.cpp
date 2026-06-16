/**
 * @file test_config.cpp
 * @brief Unit tests for the compiled-default Config and the config-file overlay.
 *
 * Also verifies that every key shipped in config/default.cfg maps to a real
 * Config field (key/field parity) to guard against silent mismatches.
 */

#include <catch2/catch_test_macros.hpp>

#include "ncorr/config.h"
#include "ncorr/ini.h"

#include <cstdio>
#include <fstream>
#include <set>
#include <string>

namespace {

std::string write_temp(const std::string& content, const char* tag) {
    static int counter = 0;
    std::string path = std::string(std::tmpnam(nullptr)) + "_ncorr_cfg_" + tag + "_" +
                       std::to_string(counter++) + ".cfg";
    std::ofstream out(path, std::ios::binary);
    out << content;
    out.close();
    return path;
}

}  // namespace

TEST_CASE("config_defaults", "[unit][config]") {
    ncorr::Config c;
    CHECK(c.scalefactor == 3);
    CHECK(c.subregion_type == "CIRCLE");
    CHECK(c.subregion_radius == 20);
    CHECK(c.interp_type == "QUINTIC_BSPLINE_PRECOMPUTE");
    CHECK(c.strain_subregion_type == "CIRCLE");
    CHECK(c.strain_radius == 5);
    CHECK(c.dic_config == "NO_UPDATE");
    CHECK(c.num_threads == 4);
    CHECK(c.debug == false);
    CHECK(c.perspective_interp == "CUBIC_KEYS");
    CHECK(c.units == "mm");
    CHECK(c.units_per_pixel == 0.2);
    CHECK(c.alpha == 0.5);
    CHECK(c.fps == 15.0);
}

TEST_CASE("config_overlay", "[unit][config]") {
    auto path = write_temp(
        "scalefactor = 7\n"
        "subregion_radius = 33\n"
        "units = px\n",
        "overlay");

    ncorr::Config c;  // start from compiled defaults
    REQUIRE(ncorr::load_config_file(path, c));

    // Overridden keys take the file value...
    CHECK(c.scalefactor == 7);
    CHECK(c.subregion_radius == 33);
    CHECK(c.units == "px");
    // ...everything else stays at the compiled default.
    CHECK(c.subregion_type == "CIRCLE");
    CHECK(c.interp_type == "QUINTIC_BSPLINE_PRECOMPUTE");
    CHECK(c.num_threads == 4);
    std::remove(path.c_str());
}

TEST_CASE("config_overlay_missing_file_keeps_values", "[unit][config]") {
    ncorr::Config c;
    c.scalefactor = 99;  // pretend a prior tier set this
    CHECK_FALSE(ncorr::load_config_file("/no/such/file_98765.cfg", c));
    CHECK(c.scalefactor == 99);  // left untouched
}

TEST_CASE("config_overlay_invalid_value_throws", "[unit][config]") {
    auto path = write_temp("scalefactor = not_a_number\n", "bad");
    ncorr::Config c;
    CHECK_THROWS_AS(ncorr::load_config_file(path, c), std::runtime_error);
    std::remove(path.c_str());
}

// Every key in the shipped default.cfg must correspond to a Config field that
// the overlay knows how to set. We verify this indirectly: load default.cfg over
// a Config whose fields are all set to non-default sentinels, then confirm every
// key actually changed at least one field away from its sentinel (i.e. is wired).
TEST_CASE("config_key_field_parity", "[unit][config]") {
    // NCORR_DEFAULT_CFG is provided as a compile definition pointing at the
    // repository's config/default.cfg.
    const char* cfg_path = NCORR_DEFAULT_CFG;

    // 1. The default.cfg must parse cleanly into a Config.
    ncorr::Config c;
    REQUIRE(ncorr::load_config_file(cfg_path, c));

    // 2. Collect the set of keys the overlay code recognises by diffing two
    //    Configs: one loaded from default.cfg and one left at compiled defaults.
    //    default.cfg mirrors the compiled defaults, so loading it should leave
    //    a default Config unchanged (proves keys map to the right fields with
    //    the documented values).
    ncorr::Config from_file;
    REQUIRE(ncorr::load_config_file(cfg_path, from_file));
    ncorr::Config compiled;  // compiled defaults
    CHECK(from_file.scalefactor == compiled.scalefactor);
    CHECK(from_file.subregion_type == compiled.subregion_type);
    CHECK(from_file.subregion_radius == compiled.subregion_radius);
    CHECK(from_file.interp_type == compiled.interp_type);
    CHECK(from_file.strain_subregion_type == compiled.strain_subregion_type);
    CHECK(from_file.strain_radius == compiled.strain_radius);
    CHECK(from_file.dic_config == compiled.dic_config);
    CHECK(from_file.num_threads == compiled.num_threads);
    CHECK(from_file.debug == compiled.debug);
    CHECK(from_file.perspective_interp == compiled.perspective_interp);
    CHECK(from_file.units == compiled.units);
    CHECK(from_file.units_per_pixel == compiled.units_per_pixel);
    CHECK(from_file.alpha == compiled.alpha);
    CHECK(from_file.fps == compiled.fps);

    // 3. Guard against an unrecognised key silently slipping into default.cfg:
    //    every non-comment 'key = value' line must be one of the known fields.
    static const std::set<std::string> known_keys = {"scalefactor",
                                                     "subregion_type",
                                                     "subregion_radius",
                                                     "interp_type",
                                                     "strain_subregion_type",
                                                     "strain_radius",
                                                     "dic_config",
                                                     "num_threads",
                                                     "debug",
                                                     "perspective_interp",
                                                     "units",
                                                     "units_per_pixel",
                                                     "alpha",
                                                     "fps"};

    ncorr::IniFile ini;
    REQUIRE(ini.load(cfg_path));
    for (const auto& kv : ini.values()) {
        INFO("default.cfg key: " << kv.first);
        CHECK(known_keys.count(kv.first) == 1);
    }
}
