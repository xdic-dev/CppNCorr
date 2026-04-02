#ifndef NCORR_IO_BINARY_IO_HPP
#define NCORR_IO_BINARY_IO_HPP

#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ncorr {
namespace io {

template <typename T>
inline T read_pod(std::ifstream &is) {
    static_assert(std::is_trivially_copyable<T>::value, "read_pod requires trivially copyable types.");
    T value{};
    is.read(reinterpret_cast<char*>(&value), std::streamsize(sizeof(T)));
    return value;
}

template <typename T>
inline void write_pod(std::ofstream &os, const T &value) {
    static_assert(std::is_trivially_copyable<T>::value, "write_pod requires trivially copyable types.");
    os.write(reinterpret_cast<const char*>(&value), std::streamsize(sizeof(T)));
}

template <typename SizeType>
inline std::string read_string(std::ifstream &is) {
    static_assert(std::is_integral<SizeType>::value, "read_string requires an integral size type.");
    const auto length = read_pod<SizeType>(is);
    if constexpr (std::is_signed<SizeType>::value) {
        if (length < 0) {
            throw std::invalid_argument("Encountered negative string length while reading binary data.");
        }
    }

    std::string value(static_cast<std::size_t>(length), '\0');
    if (!value.empty()) {
        is.read(value.data(), std::streamsize(length));
    }
    return value;
}

template <typename SizeType>
inline void write_string(std::ofstream &os, const std::string &value) {
    static_assert(std::is_integral<SizeType>::value, "write_string requires an integral size type.");
    const auto length = static_cast<SizeType>(value.size());
    write_pod(os, length);
    if (!value.empty()) {
        os.write(value.data(), std::streamsize(value.size()));
    }
}

} // namespace io
} // namespace ncorr

#endif
