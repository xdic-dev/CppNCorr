#pragma once
/**
 * @file log.h
 * @brief Lightweight, dependency-free logging facility for CppNCorr.
 *
 * CppNCorr is a static library consumed by other projects (e.g. CPPxDIC), so
 * the logger is intentionally tiny and brings in no third-party dependency. It
 * provides:
 *   - Five severity levels (Trace, Debug, Info, Warn, Error) plus Off.
 *   - Independent thresholds for the console and an optional log file, so a file
 *     sink can capture full Debug detail while the console stays quiet.
 *   - Thread-safe emission (an internal mutex), safe to call from OpenMP
 *     parallel regions used by the DIC engine.
 *   - Stream-style macros (NLOG_INFO << ...) that short-circuit message
 *     construction when the level is disabled — important for the per-frame /
 *     per-iteration hot loops in ncorr.cpp.
 *
 * Because the library has no command line of its own, configuration is driven by
 * environment variables (applied lazily on first use) and by the engine's
 * existing @c debug flag:
 *   - @c NCORR_LOG_LEVEL   trace|debug|info|warn|error|off   (console threshold)
 *   - @c NCORR_LOG_FILE    path to a log file (full Debug detail is written)
 *   - @c NCORR_LOG_CONSOLE 0|1|true|false   (disable/enable console output)
 *
 * A parent application can also configure the logger programmatically via the
 * helpers below before invoking the engine.
 */

#include <fstream>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>

namespace ncorr {
namespace log {

/// Severity levels, ordered from most to least verbose.
enum class Level { Trace = 0, Debug = 1, Info = 2, Warn = 3, Error = 4, Off = 5 };

/// Parse a level name (case-insensitive): trace, debug, info, warn|warning,
/// error, off. Returns @p fallback if @p s is empty or unrecognized.
Level level_from_string(const std::string& s, Level fallback = Level::Info);

/// Short, fixed-width display name for a level (e.g. "INFO ").
const char* level_name(Level l);

/**
 * @brief Process-wide singleton logger. All members are thread-safe.
 */
class Logger {
  public:
    static Logger& instance();

    void set_console_level(Level l);
    void set_file_level(Level l);
    Level console_level() const;
    Level file_level() const;

    /// Open (or replace) the file sink. An empty path closes the file sink.
    /// @return false if @p path could not be opened (file sink left closed).
    bool set_file(const std::string& path);

    /// Enable or disable console output entirely.
    void set_console_enabled(bool on);

    /// Include timestamp + source location in console output too (the file sink
    /// always includes them). Off by default to keep the console readable.
    void set_verbose_format(bool on);

    /// Apply NCORR_LOG_LEVEL / NCORR_LOG_FILE / NCORR_LOG_CONSOLE. Runs once
    /// automatically on first use; calling again re-applies them.
    void configure_from_env();

    /// True if a message at @p l would reach any sink. Use to guard expensive
    /// message construction in hot loops.
    bool enabled(Level l);

    /// Emit a fully-formed message (trailing newlines are trimmed) at @p l.
    void write(Level l, const char* file, int line, const std::string& msg);

  private:
    Logger();
    void ensure_env();

    mutable std::mutex mtx_;
    Level console_level_ = Level::Info;
    Level file_level_ = Level::Debug;
    bool console_enabled_ = true;
    bool verbose_format_ = false;
    bool color_ = false;
    bool env_done_ = false;
    std::ofstream file_;
};

/// Set the console threshold (convenience wrapper).
void set_level(Level l);

/// Lower the console threshold to Debug when @p on is true (used to honour the
/// engine's existing @c debug flag). A no-op when @p on is false.
void set_debug(bool on);

/// Convenience: open a log file sink (full Debug detail).
bool set_file(const std::string& path);

/// True if a message at @p l would be emitted by any sink.
inline bool enabled(Level l) {
    return Logger::instance().enabled(l);
}

/**
 * @brief RAII stream builder; flushes its accumulated text to the logger when it
 *        is destroyed at the end of the full expression.
 */
class Stream {
  public:
    Stream(Level level, const char* file, int line) : level_(level), file_(file), line_(line) {}
    ~Stream() { Logger::instance().write(level_, file_, line_, oss_.str()); }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    template <typename T>
    Stream& operator<<(const T& v) {
        oss_ << v;
        return *this;
    }
    /// Support stream manipulators such as std::endl / std::setprecision.
    Stream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        oss_ << manip;
        return *this;
    }

  private:
    Level level_;
    const char* file_;
    int line_;
    std::ostringstream oss_;
};

/// Helper that turns the conditional logging expression back into a void
/// statement (glog idiom). @c operator& has lower precedence than @c << but
/// higher than @c ?:, so the macro is safe inside an unbraced if/else.
class Voidify {
  public:
    Voidify() = default;
    void operator&(Stream&) {}
};

}  // namespace log
}  // namespace ncorr

// Stream-style logging macros. When the level is disabled, the right-hand side
// (message construction) is never evaluated.
#define NLOG_AT(lvl)            \
    !::ncorr::log::enabled(lvl) \
        ? (void)0               \
        : ::ncorr::log::Voidify() & ::ncorr::log::Stream((lvl), __FILE__, __LINE__)

#define NLOG_TRACE NLOG_AT(::ncorr::log::Level::Trace)
#define NLOG_DEBUG NLOG_AT(::ncorr::log::Level::Debug)
#define NLOG_INFO NLOG_AT(::ncorr::log::Level::Info)
#define NLOG_WARN NLOG_AT(::ncorr::log::Level::Warn)
#define NLOG_ERROR NLOG_AT(::ncorr::log::Level::Error)
