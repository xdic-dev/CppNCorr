#pragma once
/**
 * @file frame_reader.h
 * @brief Image-folder frame discovery helpers for the file-based DIC pipeline.
 *
 * These helpers were extracted out of @c proxyncorr.cpp so they can be reused
 * and unit-tested in isolation without compiling the whole driver. The behaviour
 * is intentionally identical to the original in-driver implementation.
 *
 * NAMING CONVENTION / EXPECTED LAYOUT (see @ref discover_frames):
 *   - One image per frame, plus (optionally) a ROI mask and a dedicated
 *     reference image.
 *   - Frames numbered so they sort in acquisition order. Both zero-padded
 *     ("frame_00.png", "frame_01.png", ...) and unpadded ("frame_2.png",
 *     "frame_10.png", ...) numbering work; ordering uses a natural
 *     (numeric-aware) comparison, not raw lexicographic order.
 *   - Files named "roi.png" and "ref.png" (case-insensitive) are reserved and
 *     excluded, as are files matching the supplied @c ref_path / @c roi_path
 *     basenames.
 *   - Hidden files (leading '.') and non-image extensions are ignored.
 *   - Supported extensions: .png .tif .tiff .bmp .jpg .jpeg
 *
 * The functions are @c inline so this header is self-contained: tests can
 * include it directly without linking the driver translation unit.
 */

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

namespace ncorr {

/**
 * @brief Supported image extensions (lowercase, including the leading dot).
 *
 * Mirrors the formats OpenCV's imread handles in this build. Keep in sync with
 * the user guide.
 */
inline const std::vector<std::string>& image_extensions() {
    static const std::vector<std::string> kImageExtensions = {".png", ".tif", ".tiff",
                                                              ".bmp", ".jpg", ".jpeg"};
    return kImageExtensions;
}

/**
 * @brief Test whether a (lowercased) filename ends with a supported image extension.
 *
 * Length-safe: never reads out of bounds for short names.
 *
 * @param lower_name Filename already converted to lowercase.
 * @return true if the name ends with one of @ref image_extensions().
 */
inline bool has_image_extension(const std::string& lower_name) {
    for (const std::string& ext : image_extensions()) {
        if (lower_name.size() > ext.size() &&
            lower_name.compare(lower_name.size() - ext.size(), ext.size(), ext) == 0) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Natural (human) ordering comparison for filenames.
 *
 * Compares runs of digits by numeric value so unpadded numeric frame names sort
 * correctly, e.g. "frame_2.png" < "frame_10.png". Falls back to lexicographic
 * for non-digit runs. Zero-padded names also sort correctly under this rule.
 *
 * @param a Left-hand filename.
 * @param b Right-hand filename.
 * @return true if @p a should sort before @p b.
 */
inline bool natural_less(const std::string& a, const std::string& b) {
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (std::isdigit(static_cast<unsigned char>(a[i])) &&
            std::isdigit(static_cast<unsigned char>(b[j]))) {
            // Compare two runs of digits by numeric value (skip leading zeros).
            size_t ai = i, bj = j;
            while (ai < a.size() && std::isdigit(static_cast<unsigned char>(a[ai]))) ++ai;
            while (bj < b.size() && std::isdigit(static_cast<unsigned char>(b[bj]))) ++bj;
            std::string da = a.substr(i, ai - i);
            std::string db = b.substr(j, bj - j);
            da.erase(0, da.find_first_not_of('0'));
            db.erase(0, db.find_first_not_of('0'));
            if (da.size() != db.size()) return da.size() < db.size();
            if (da != db) return da < db;
            i = ai;
            j = bj;
        } else {
            if (a[i] != b[j]) return a[i] < b[j];
            ++i;
            ++j;
        }
    }
    return a.size() < b.size();
}

/**
 * @brief Discover the deformed-frame image files inside @p folder.
 *
 * See the file-level documentation for the expected naming convention. Files
 * named "roi.png"/"ref.png" (case-insensitive) and any matching the supplied
 * @p roi_path / @p ref_path basenames are excluded; hidden files, sub-directories
 * and non-image extensions are skipped. The result is sorted with @ref natural_less.
 *
 * @param folder   Directory to scan.
 * @param ref_path Reference image path whose basename should be excluded (may be empty).
 * @param roi_path ROI image path whose basename should be excluded (may be empty).
 * @return Frame paths (folder + "/" + name) in natural order. Empty if the folder
 *         has no usable frames (the caller is expected to report that).
 * @throws std::runtime_error if @p folder cannot be opened.
 */
inline std::vector<std::string> discover_frames(const std::string& folder,
                                                const std::string& ref_path,
                                                const std::string& roi_path) {
    std::vector<std::string> frames;
    DIR* dir = opendir(folder.c_str());
    if (!dir) {
        throw std::runtime_error("Cannot open folder '" + folder + "': " + std::strerror(errno));
    }

    // Get basenames to exclude.
    std::string roi_basename = "";
    std::string ref_basename = "";
    if (!roi_path.empty()) {
        size_t pos = roi_path.find_last_of("/\\");
        roi_basename = (pos != std::string::npos) ? roi_path.substr(pos + 1) : roi_path;
    }
    if (!ref_path.empty()) {
        size_t pos = ref_path.find_last_of("/\\");
        ref_basename = (pos != std::string::npos) ? ref_path.substr(pos + 1) : ref_path;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;

        // Skip empty names (defensive) and hidden files / "." / ".." entries.
        if (name.empty() || name[0] == '.') continue;

        // Skip nested sub-directories: only regular files are valid frames.
        std::string full_path = folder + "/" + name;
        struct stat st;
        if (stat(full_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) continue;

        // Check for image extensions (case-insensitive, length-safe).
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        if (!has_image_extension(lower_name)) continue;

        // Skip reserved roi.png / ref.png by default.
        if (lower_name == "roi.png") continue;
        if (lower_name == "ref.png") continue;

        // Skip explicitly specified roi and ref files.
        if (!roi_basename.empty() && name == roi_basename) continue;
        if (!ref_basename.empty() && name == ref_basename) continue;

        frames.push_back(full_path);
    }
    closedir(dir);

    // Sort frames naturally (handles both padded and unpadded numbered files).
    std::sort(frames.begin(), frames.end(), natural_less);

    return frames;
}

}  // namespace ncorr
