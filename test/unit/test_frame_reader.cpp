/**
 * @file test_frame_reader.cpp
 * @brief Unit tests for the image-folder frame discovery helpers
 *        (include/ncorr/frame_reader.h).
 *
 * Uses a temporary directory populated with empty placeholder files so the
 * discovery / ordering / exclusion logic can be tested without real images.
 */

#include <catch2/catch_test_macros.hpp>

#include "ncorr/frame_reader.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

// Create a unique temporary directory and return its path (no trailing slash).
std::string make_temp_dir() {
    static int counter = 0;
    std::string base =
        std::string(std::tmpnam(nullptr)) + "_ncorr_frames_" + std::to_string(counter++);
    mkdir(base.c_str(), 0755);
    return base;
}

void touch(const std::string& path) {
    std::ofstream out(path);
    out << "x";
    out.close();
}

}  // namespace

TEST_CASE("has_image_extension", "[unit][frame_reader]") {
    using ncorr::has_image_extension;
    CHECK(has_image_extension("frame.png"));
    CHECK(has_image_extension("frame.tif"));
    CHECK(has_image_extension("frame.tiff"));
    CHECK(has_image_extension("frame.bmp"));
    CHECK(has_image_extension("frame.jpg"));
    CHECK(has_image_extension("frame.jpeg"));
    // Rejections.
    CHECK_FALSE(has_image_extension("frame.txt"));
    CHECK_FALSE(has_image_extension("frame.pn"));
    CHECK_FALSE(has_image_extension("noext"));
    // Length-edge: a name equal to just the extension is not a valid image name.
    CHECK_FALSE(has_image_extension(".png"));
    CHECK_FALSE(has_image_extension(""));
}

TEST_CASE("natural_less", "[unit][frame_reader]") {
    using ncorr::natural_less;
    CHECK(natural_less("frame_2.png", "frame_10.png"));
    CHECK_FALSE(natural_less("frame_10.png", "frame_2.png"));
    // Zero-padded names also order correctly.
    CHECK(natural_less("img_00.png", "img_01.png"));
    CHECK(natural_less("img_09.png", "img_10.png"));
    // Pure lexicographic fallback for non-digit prefixes.
    CHECK(natural_less("a.png", "b.png"));
}

TEST_CASE("discover_frames_sorted", "[unit][frame_reader]") {
    std::string dir = make_temp_dir();
    touch(dir + "/frame_10.png");
    touch(dir + "/frame_2.png");
    touch(dir + "/frame_1.png");

    auto frames = ncorr::discover_frames(dir, "", "");
    REQUIRE(frames.size() == 3);
    CHECK(frames[0] == dir + "/frame_1.png");
    CHECK(frames[1] == dir + "/frame_2.png");
    CHECK(frames[2] == dir + "/frame_10.png");
}

TEST_CASE("discover_frames_excludes_reserved", "[unit][frame_reader]") {
    std::string dir = make_temp_dir();
    touch(dir + "/frame_1.png");
    touch(dir + "/roi.png");                 // reserved
    touch(dir + "/ref.png");                 // reserved
    touch(dir + "/.hidden.png");             // hidden
    touch(dir + "/notes.txt");               // non-image
    touch(dir + "/custom_ref.png");          // excluded via ref_path basename
    mkdir((dir + "/subdir").c_str(), 0755);  // sub-directory, ignored

    auto frames = ncorr::discover_frames(dir, dir + "/custom_ref.png", "");
    REQUIRE(frames.size() == 1);
    CHECK(frames[0] == dir + "/frame_1.png");
}

TEST_CASE("discover_frames_empty", "[unit][frame_reader]") {
    std::string dir = make_temp_dir();
    auto frames = ncorr::discover_frames(dir, "", "");
    CHECK(frames.empty());
}

TEST_CASE("discover_frames_missing_folder", "[unit][frame_reader]") {
    CHECK_THROWS_AS(ncorr::discover_frames("/no/such/folder_abc_98765", "", ""),
                    std::runtime_error);
}
