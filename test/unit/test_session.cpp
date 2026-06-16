/**
 * @file test_session.cpp
 * @brief Unit tests for the in-memory NcorrSession API (stub contract).
 *
 * These tests pin the *contract* of the stubbed API: input validation, the
 * reference-required precondition, geometry checks, and the current
 * "not yet implemented" return. When section 3b (real in-memory DIC) lands,
 * the `session_stub_contract` test must be updated.
 */

#include <catch2/catch_test_macros.hpp>

#include "ncorr/session.h"

#include <cstdint>
#include <vector>

namespace {

// Build a small valid grayscale buffer kept alive by the returned vector.
ncorr::ImageBuffer make_buffer(std::vector<std::uint8_t>& storage, int w, int h, int ch = 1) {
    storage.assign(static_cast<size_t>(w) * h * ch, 0);
    return ncorr::ImageBuffer(storage.data(), w, h, ch);
}

}  // namespace

TEST_CASE("imagebuffer_valid", "[unit][session]") {
    std::vector<std::uint8_t> storage;
    auto good = make_buffer(storage, 4, 3);
    CHECK(good.valid());
    CHECK(good.size_bytes() == 12u);

    ncorr::ImageBuffer empty;
    CHECK_FALSE(empty.valid());

    std::uint8_t one = 0;
    ncorr::ImageBuffer zero_dim(&one, 0, 5, 1);
    CHECK_FALSE(zero_dim.valid());

    ncorr::ImageBuffer null_data(nullptr, 4, 4, 1);
    CHECK_FALSE(null_data.valid());
}

TEST_CASE("session_requires_reference", "[unit][session]") {
    ncorr::NcorrSession session;
    CHECK_FALSE(session.has_reference());
    std::vector<std::uint8_t> storage;
    auto def = make_buffer(storage, 8, 8);
    CHECK_THROWS_AS(session.process_frame(def), std::logic_error);
}

TEST_CASE("session_rejects_invalid_reference", "[unit][session]") {
    ncorr::NcorrSession session;
    ncorr::ImageBuffer bad;  // null / zero dims
    CHECK_THROWS_AS(session.set_reference(bad), std::invalid_argument);
    CHECK_FALSE(session.has_reference());
}

TEST_CASE("session_geometry_mismatch", "[unit][session]") {
    ncorr::NcorrSession session;
    std::vector<std::uint8_t> ref_storage;
    auto ref = make_buffer(ref_storage, 8, 8);
    session.set_reference(ref);
    CHECK(session.has_reference());

    std::vector<std::uint8_t> def_storage;
    auto def = make_buffer(def_storage, 8, 9);  // different height
    auto result = session.process_frame(def);
    CHECK_FALSE(result.valid);
    CHECK_FALSE(result.message.empty());
}

// STUB CONTRACT: update this test when in-memory DIC (section 3b) is implemented.
TEST_CASE("session_stub_contract", "[unit][session]") {
    ncorr::NcorrSession session;
    std::vector<std::uint8_t> ref_storage;
    auto ref = make_buffer(ref_storage, 8, 8);
    session.set_reference(ref);

    std::vector<std::uint8_t> def_storage;
    auto def = make_buffer(def_storage, 8, 8);  // matching geometry
    auto result = session.process_frame(def);

    CHECK_FALSE(result.valid);
    CHECK(result.message == "NcorrSession::process_frame not yet implemented");
    CHECK(result.width == 8);
    CHECK(result.height == 8);
}
