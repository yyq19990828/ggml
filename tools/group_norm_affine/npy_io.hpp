#ifndef GGML_GROUP_NORM_AFFINE_PROBE_NPY_IO_HPP
#define GGML_GROUP_NORM_AFFINE_PROBE_NPY_IO_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace gngn {

struct NpyF32 {
    std::string path;
    std::vector<std::size_t> shape;
    std::vector<float> values;
};

inline std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

inline std::size_t checked_product(const std::vector<std::size_t> &shape) {
    std::size_t result = 1;
    for (const std::size_t dim : shape) {
        if (dim == 0 || result > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("invalid or overflowing NPY shape");
        }
        result *= dim;
    }
    return result;
}

inline NpyF32 read_f32_npy(const std::string &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open NPY: " + path);
    std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(stream)), {});
    const std::uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (raw.size() < 10 || std::memcmp(raw.data(), magic, sizeof(magic)) != 0) {
        throw std::runtime_error("invalid NPY magic: " + path);
    }
    const std::uint8_t major = raw[6];
    std::size_t header_start = 0;
    std::size_t header_size = 0;
    if (major == 1) {
        header_start = 10;
        header_size = static_cast<std::size_t>(raw[8]) |
                      (static_cast<std::size_t>(raw[9]) << 8U);
    } else if (major == 2 || major == 3) {
        if (raw.size() < 12) throw std::runtime_error("truncated NPY header: " + path);
        header_start = 12;
        header_size = static_cast<std::size_t>(raw[8]) |
                      (static_cast<std::size_t>(raw[9]) << 8U) |
                      (static_cast<std::size_t>(raw[10]) << 16U) |
                      (static_cast<std::size_t>(raw[11]) << 24U);
    } else {
        throw std::runtime_error("unsupported NPY version: " + path);
    }
    if (header_size > raw.size() - header_start) throw std::runtime_error("truncated NPY header: " + path);
    const std::string header(reinterpret_cast<const char *>(raw.data() + header_start), header_size);
    std::smatch match;
    if (!std::regex_search(header, match, std::regex("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]"))) {
        throw std::runtime_error("NPY descriptor missing: " + path);
    }
    const std::string descr = match[1].str();
    if (descr != "<f4" && descr != "=f4" && descr != "|f4") {
        throw std::runtime_error("NPY must contain little-endian F32: " + path);
    }
    if (!std::regex_search(header, match, std::regex("['\"]fortran_order['\"]\\s*:\\s*(True|False)")) ||
        match[1].str() != "False") {
        throw std::runtime_error("NPY must be C-contiguous: " + path);
    }
    if (!std::regex_search(header, match, std::regex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)"))) {
        throw std::runtime_error("NPY shape missing: " + path);
    }
    std::vector<std::size_t> shape;
    std::string dimensions = match[1].str();
    std::size_t begin = 0;
    while (begin < dimensions.size()) {
        const std::size_t comma = dimensions.find(',', begin);
        std::string token = trim(dimensions.substr(begin, comma == std::string::npos ? comma : comma - begin));
        if (!token.empty()) {
            const unsigned long long dim = std::stoull(token);
            if (dim == 0 || dim > std::numeric_limits<std::size_t>::max()) {
                throw std::runtime_error("invalid NPY dimension: " + path);
            }
            shape.push_back(static_cast<std::size_t>(dim));
        }
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    if (shape.empty()) throw std::runtime_error("scalar NPY is not supported: " + path);
    const std::size_t count = checked_product(shape);
    const std::size_t payload_offset = header_start + header_size;
    if (count > (std::numeric_limits<std::size_t>::max() / sizeof(float)) ||
        raw.size() - payload_offset != count * sizeof(float)) {
        throw std::runtime_error("NPY payload length disagrees with shape: " + path);
    }
    const std::uint16_t endian_test = 1;
    if (*reinterpret_cast<const std::uint8_t *>(&endian_test) != 1) {
        throw std::runtime_error("host byte order must be little-endian");
    }
    NpyF32 result;
    result.path = path;
    result.shape = std::move(shape);
    result.values.resize(count);
    std::memcpy(result.values.data(), raw.data() + payload_offset, count * sizeof(float));
    return result;
}

inline std::string shape_tuple(const std::vector<std::size_t> &shape) {
    std::string value = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) value += ", ";
        value += std::to_string(shape[i]);
    }
    if (shape.size() == 1) value += ",";
    value += ")";
    return value;
}

inline void write_f32_npy(const std::string &path,
                          const std::vector<std::size_t> &shape,
                          const std::vector<float> &values) {
    if (checked_product(shape) != values.size()) throw std::runtime_error("NPY output shape mismatch");
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_tuple(shape) + ", }";
    const std::size_t prefix_size = 10;
    const std::size_t padding = (16 - ((prefix_size + header.size() + 1) % 16)) % 16;
    header.append(padding, ' ');
    header.push_back('\n');
    if (header.size() > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("NPY v1 header exceeds 16-bit size");
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot create NPY: " + path);
    const std::uint8_t prefix[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
        static_cast<std::uint8_t>(header.size() & 0xffU),
        static_cast<std::uint8_t>((header.size() >> 8U) & 0xffU)};
    stream.write(reinterpret_cast<const char *>(prefix), sizeof(prefix));
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    stream.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!stream) throw std::runtime_error("failed writing NPY: " + path);
}

}  // namespace gngn

#endif  // GGML_GROUP_NORM_AFFINE_PROBE_NPY_IO_HPP
