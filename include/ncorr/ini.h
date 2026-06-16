#pragma once
/**
 * @file ini.h
 * @brief Tiny, dependency-free INI-style configuration parser (header-only).
 *
 * Written in-tree to avoid adding any external dependency (no TOML/INI library).
 * Format (deliberately minimal):
 *   - One `key = value` pair per line.
 *   - Blank lines are ignored.
 *   - Full-line comments start with '#' or ';'.
 *   - Inline comments: text after an unquoted '#' or ';' is stripped.
 *   - Optional `[section]` headers are parsed; keys may be addressed as
 *     "section.key" or, for the default (top) section, just "key".
 *   - Surrounding whitespace on keys and values is trimmed.
 *   - Values may be wrapped in matching single or double quotes to preserve
 *     leading/trailing spaces or embedded comment characters.
 *   - Keys are case-sensitive (to match C++ field names exactly).
 *
 * This is intended for small config files (a few dozen keys); it loads the whole
 * file into a map. It is not a general-purpose INI implementation.
 */

#include <cctype>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace ncorr {

/**
 * @brief In-memory representation of a parsed INI file.
 */
class IniFile {
public:
    /// Construct an empty INI store.
    IniFile() = default;

    /**
     * @brief Parse an INI file from disk.
     * @param path Path to the INI file.
     * @return true if the file was opened and parsed; false if it did not exist.
     * @throws std::runtime_error on a structurally invalid line.
     */
    bool load(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return false;  // missing file => caller falls back to defaults
        }
        values_.clear();
        std::string section;
        std::string line;
        int line_no = 0;
        while (std::getline(file, line)) {
            ++line_no;
            // Strip a trailing '\r' so CRLF files parse correctly.
            if (!line.empty() && line.back() == '\r') line.pop_back();

            std::string content = strip_inline_comment(line);
            content = trim(content);
            if (content.empty()) continue;

            // Section header: [name]
            if (content.front() == '[') {
                if (content.back() != ']') {
                    throw std::runtime_error("INI parse error at line " +
                        std::to_string(line_no) + ": malformed section header");
                }
                section = trim(content.substr(1, content.size() - 2));
                continue;
            }

            // key = value
            std::size_t eq = content.find('=');
            if (eq == std::string::npos) {
                throw std::runtime_error("INI parse error at line " +
                    std::to_string(line_no) + ": expected 'key = value'");
            }
            std::string key = trim(content.substr(0, eq));
            std::string value = unquote(trim(content.substr(eq + 1)));
            if (key.empty()) {
                throw std::runtime_error("INI parse error at line " +
                    std::to_string(line_no) + ": empty key");
            }
            std::string full_key = section.empty() ? key : (section + "." + key);
            values_[full_key] = value;
        }
        loaded_ = true;
        return true;
    }

    /// @return true if a value exists for @p key.
    bool has(const std::string& key) const {
        return values_.find(key) != values_.end();
    }

    /// @return the raw string value for @p key, or @p fallback if absent.
    std::string get(const std::string& key, const std::string& fallback = "") const {
        auto it = values_.find(key);
        return it == values_.end() ? fallback : it->second;
    }

    /// @return parsed int for @p key, or @p fallback if absent. Throws on bad value.
    int get_int(const std::string& key, int fallback) const {
        auto it = values_.find(key);
        if (it == values_.end()) return fallback;
        try {
            return std::stoi(it->second);
        } catch (const std::exception&) {
            throw std::runtime_error("INI value for '" + key +
                "' is not a valid integer: '" + it->second + "'");
        }
    }

    /// @return parsed double for @p key, or @p fallback if absent. Throws on bad value.
    double get_double(const std::string& key, double fallback) const {
        auto it = values_.find(key);
        if (it == values_.end()) return fallback;
        try {
            return std::stod(it->second);
        } catch (const std::exception&) {
            throw std::runtime_error("INI value for '" + key +
                "' is not a valid number: '" + it->second + "'");
        }
    }

    /// @return parsed bool for @p key, or @p fallback if absent.
    /// Accepts (case-insensitively) true/false, 1/0, yes/no, on/off.
    bool get_bool(const std::string& key, bool fallback) const {
        auto it = values_.find(key);
        if (it == values_.end()) return fallback;
        std::string v = to_lower(it->second);
        if (v == "true" || v == "1" || v == "yes" || v == "on") return true;
        if (v == "false" || v == "0" || v == "no" || v == "off") return false;
        throw std::runtime_error("INI value for '" + key +
            "' is not a valid boolean: '" + it->second + "'");
    }

    /// @return the full key/value map (for iteration / diagnostics).
    const std::map<std::string, std::string>& values() const { return values_; }

private:
    static std::string to_lower(std::string s) {
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

    static std::string trim(const std::string& s) {
        std::size_t b = s.find_first_not_of(" \t");
        if (b == std::string::npos) return "";
        std::size_t e = s.find_last_not_of(" \t");
        return s.substr(b, e - b + 1);
    }

    // Remove an inline comment introduced by an unquoted '#' or ';'.
    static std::string strip_inline_comment(const std::string& s) {
        bool in_single = false, in_double = false;
        for (std::size_t i = 0; i < s.size(); ++i) {
            char c = s[i];
            if (c == '\'' && !in_double) in_single = !in_single;
            else if (c == '"' && !in_single) in_double = !in_double;
            else if ((c == '#' || c == ';') && !in_single && !in_double) {
                return s.substr(0, i);
            }
        }
        return s;
    }

    // Strip a single matching pair of surrounding quotes, if present.
    static std::string unquote(const std::string& s) {
        if (s.size() >= 2 &&
            ((s.front() == '"' && s.back() == '"') ||
             (s.front() == '\'' && s.back() == '\''))) {
            return s.substr(1, s.size() - 2);
        }
        return s;
    }

    std::map<std::string, std::string> values_;
    bool loaded_ = false;
};

}  // namespace ncorr
