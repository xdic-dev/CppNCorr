/**
 * @file test_ini.cpp
 * @brief Unit tests for the header-only INI parser (include/ncorr/ini.h).
 *
 * Dependency-free: exercises parsing rules, typed getters and error handling
 * using temporary files written to the system temp directory.
 */

#include <catch2/catch_test_macros.hpp>

#include "ncorr/ini.h"

#include <cstdio>
#include <fstream>
#include <string>

namespace {

// Write `content` to a unique temp file and return its path. The file is left on
// disk for the duration of the test process (small, harmless); Catch reports
// per-section so collisions are avoided via a counter.
std::string write_temp_ini(const std::string& content) {
    static int counter = 0;
    std::string path =
        std::string(std::tmpnam(nullptr)) + "_ncorr_ini_" + std::to_string(counter++) + ".cfg";
    std::ofstream out(path, std::ios::binary);
    out << content;
    out.close();
    return path;
}

}  // namespace

TEST_CASE("ini_parse_basic", "[unit][ini]") {
    auto path = write_temp_ini("  key1 = value1 \nkey2=value2\n");
    ncorr::IniFile ini;
    REQUIRE(ini.load(path));
    CHECK(ini.get("key1") == "value1");
    CHECK(ini.get("key2") == "value2");
    CHECK(ini.has("key1"));
    CHECK_FALSE(ini.has("missing"));
    CHECK(ini.get("missing", "fallback") == "fallback");
    std::remove(path.c_str());
}

TEST_CASE("ini_parse_comments", "[unit][ini]") {
    auto path = write_temp_ini(
        "# full line comment\n"
        "; semicolon comment\n"
        "key = value   # inline comment\n"
        "key2 = val2 ; also inline\n");
    ncorr::IniFile ini;
    REQUIRE(ini.load(path));
    CHECK(ini.get("key") == "value");
    CHECK(ini.get("key2") == "val2");
    CHECK(ini.values().size() == 2);
    std::remove(path.c_str());
}

TEST_CASE("ini_parse_quoted_values", "[unit][ini]") {
    auto path = write_temp_ini(
        "a = \"  spaced value  \"\n"
        "b = 'has # hash and ; semi'\n");
    ncorr::IniFile ini;
    REQUIRE(ini.load(path));
    CHECK(ini.get("a") == "  spaced value  ");
    CHECK(ini.get("b") == "has # hash and ; semi");
    std::remove(path.c_str());
}

TEST_CASE("ini_parse_sections", "[unit][ini]") {
    auto path = write_temp_ini(
        "top = 1\n"
        "[dic]\n"
        "radius = 20\n");
    ncorr::IniFile ini;
    REQUIRE(ini.load(path));
    CHECK(ini.get("top") == "1");
    CHECK(ini.get("dic.radius") == "20");
    std::remove(path.c_str());
}

TEST_CASE("ini_parse_crlf", "[unit][ini]") {
    auto path = write_temp_ini("a = 1\r\nb = 2\r\n");
    ncorr::IniFile ini;
    REQUIRE(ini.load(path));
    CHECK(ini.get("a") == "1");
    CHECK(ini.get("b") == "2");
    std::remove(path.c_str());
}

TEST_CASE("ini_typed_getters", "[unit][ini]") {
    auto path = write_temp_ini(
        "i = 42\n"
        "d = 3.5\n"
        "t = true\n"
        "f = no\n"
        "on = ON\n"
        "one = 1\n"
        "zero = 0\n");
    ncorr::IniFile ini;
    REQUIRE(ini.load(path));
    CHECK(ini.get_int("i", -1) == 42);
    CHECK(ini.get_double("d", -1.0) == 3.5);
    CHECK(ini.get_bool("t", false) == true);
    CHECK(ini.get_bool("f", true) == false);
    CHECK(ini.get_bool("on", false) == true);
    CHECK(ini.get_bool("one", false) == true);
    CHECK(ini.get_bool("zero", true) == false);
    // Missing keys return the fallback.
    CHECK(ini.get_int("missing", 7) == 7);
    CHECK(ini.get_double("missing", 1.25) == 1.25);
    CHECK(ini.get_bool("missing", true) == true);
    std::remove(path.c_str());
}

TEST_CASE("ini_typed_getters_invalid", "[unit][ini]") {
    auto path = write_temp_ini(
        "i = not_an_int\n"
        "d = not_a_double\n"
        "b = maybe\n");
    ncorr::IniFile ini;
    REQUIRE(ini.load(path));
    CHECK_THROWS_AS(ini.get_int("i", 0), std::runtime_error);
    CHECK_THROWS_AS(ini.get_double("d", 0.0), std::runtime_error);
    CHECK_THROWS_AS(ini.get_bool("b", false), std::runtime_error);
    std::remove(path.c_str());
}

TEST_CASE("ini_missing_file", "[unit][ini]") {
    ncorr::IniFile ini;
    CHECK_FALSE(ini.load("/nonexistent/path/does_not_exist_12345.cfg"));
}
