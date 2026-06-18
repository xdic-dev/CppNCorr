/**
 * @file log.cpp
 * @brief Implementation of the CppNCorr logging facility (see ncorr/log.h).
 */

#include "ncorr/log.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>

#if defined(_WIN32)
#include <io.h>
#define NCORR_ISATTY(fd) _isatty(fd)
#define NCORR_FILENO(f) _fileno(f)
#else
#include <unistd.h>
#define NCORR_ISATTY(fd) ::isatty(fd)
#define NCORR_FILENO(f) ::fileno(f)
#endif

namespace ncorr {
namespace log {

namespace {

/// Lowercase a copy of @p s (ASCII only).
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

/// Strip the directory portion of a path so log lines show just the file name.
const char* base_name(const char* path) {
    if (!path) return "";
    const char* base = path;
    for (const char* p = path; *p; ++p) {
        if (*p == '/' || *p == '\\') base = p + 1;
    }
    return base;
}

/// ANSI colour escape for a level (empty when colour is disabled by the caller).
const char* color_for(Level l) {
    switch (l) {
        case Level::Trace: return "\033[37m";  // grey
        case Level::Debug: return "\033[36m";  // cyan
        case Level::Info: return "\033[32m";   // green
        case Level::Warn: return "\033[33m";   // yellow
        case Level::Error: return "\033[31m";  // red
        default: return "";
    }
}
const char* color_reset() { return "\033[0m"; }

/// Format "YYYY-MM-DD HH:MM:SS.mmm" for the current wall-clock time.
std::string timestamp() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto t = system_clock::to_time_t(now);
    const auto ms =
        duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
    char out[40];
    std::snprintf(out, sizeof(out), "%s.%03d", buf, static_cast<int>(ms));
    return out;
}

/// Remove trailing CR/LF/space so callers can keep stray "<< std::endl".
std::string rtrim(const std::string& s) {
    std::size_t end = s.size();
    while (end > 0) {
        const char c = s[end - 1];
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t')
            --end;
        else
            break;
    }
    return s.substr(0, end);
}

}  // namespace

Level level_from_string(const std::string& s, Level fallback) {
    const std::string v = to_lower(s);
    if (v == "trace") return Level::Trace;
    if (v == "debug") return Level::Debug;
    if (v == "info") return Level::Info;
    if (v == "warn" || v == "warning") return Level::Warn;
    if (v == "error" || v == "err") return Level::Error;
    if (v == "off" || v == "none" || v == "silent") return Level::Off;
    return fallback;
}

const char* level_name(Level l) {
    switch (l) {
        case Level::Trace: return "TRACE";
        case Level::Debug: return "DEBUG";
        case Level::Info: return "INFO ";
        case Level::Warn: return "WARN ";
        case Level::Error: return "ERROR";
        case Level::Off: return "OFF  ";
    }
    return "?????";
}

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

Logger::Logger() {
    color_ = NCORR_ISATTY(NCORR_FILENO(stderr)) != 0;
}

void Logger::ensure_env() {
    if (!env_done_) {
        env_done_ = true;
        configure_from_env();
    }
}

void Logger::configure_from_env() {
    if (const char* lvl = std::getenv("NCORR_LOG_LEVEL")) {
        console_level_ = level_from_string(lvl, console_level_);
    }
    if (const char* con = std::getenv("NCORR_LOG_CONSOLE")) {
        const std::string v = to_lower(con);
        console_enabled_ = !(v == "0" || v == "false" || v == "off" || v == "no");
    }
    if (const char* path = std::getenv("NCORR_LOG_FILE")) {
        if (path[0] != '\0') {
            file_.open(path, std::ios::out | std::ios::app);
        }
    }
}

void Logger::set_console_level(Level l) {
    std::lock_guard<std::mutex> lk(mtx_);
    console_level_ = l;
}

void Logger::set_file_level(Level l) {
    std::lock_guard<std::mutex> lk(mtx_);
    file_level_ = l;
}

Level Logger::console_level() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return console_level_;
}

Level Logger::file_level() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return file_level_;
}

bool Logger::set_file(const std::string& path) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (file_.is_open()) file_.close();
    if (path.empty()) return true;
    file_.open(path, std::ios::out | std::ios::app);
    return file_.is_open();
}

void Logger::set_console_enabled(bool on) {
    std::lock_guard<std::mutex> lk(mtx_);
    console_enabled_ = on;
}

void Logger::set_verbose_format(bool on) {
    std::lock_guard<std::mutex> lk(mtx_);
    verbose_format_ = on;
}

bool Logger::enabled(Level l) {
    std::lock_guard<std::mutex> lk(mtx_);
    ensure_env();
    if (l == Level::Off) return false;
    const bool to_console = console_enabled_ && l >= console_level_;
    const bool to_file = file_.is_open() && l >= file_level_;
    return to_console || to_file;
}

void Logger::write(Level l, const char* file, int line, const std::string& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    ensure_env();
    if (l == Level::Off) return;

    const std::string text = rtrim(msg);
    const char* fname = base_name(file);

    if (console_enabled_ && l >= console_level_) {
        std::ostream& os = (l >= Level::Warn) ? std::cerr : std::cout;
        if (color_) os << color_for(l);
        os << "[" << level_name(l) << "]";
        if (color_) os << color_reset();
        if (verbose_format_) {
            os << " " << timestamp() << " " << fname << ":" << line;
        }
        os << " " << text << "\n";
    }

    if (file_.is_open() && l >= file_level_) {
        file_ << timestamp() << " [" << level_name(l) << "] " << fname << ":"
              << line << " " << text << "\n";
        file_.flush();
    }
}

void set_level(Level l) { Logger::instance().set_console_level(l); }

void set_debug(bool on) {
    if (on && Logger::instance().console_level() > Level::Debug) {
        Logger::instance().set_console_level(Level::Debug);
    }
}

bool set_file(const std::string& path) {
    return Logger::instance().set_file(path);
}

}  // namespace log
}  // namespace ncorr
