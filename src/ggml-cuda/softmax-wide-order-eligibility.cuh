#ifndef GGML_CUDA_WIDE_SOFTMAX_ELIGIBILITY_CUH
#define GGML_CUDA_WIDE_SOFTMAX_ELIGIBILITY_CUH

#include <cstdint>
#include <limits>

namespace ggml_cuda_wide_softmax {

constexpr int32_t kIlp = 4;
constexpr int32_t kWarpSize = 32;
constexpr int32_t kMaxThreads = 1024;
constexpr uint64_t kAlignmentBytes = 16;

enum class Route : uint8_t {
    kUnsupported,
    kPersistent,
    kRegister,
    kSharedMemory,
    kRegularWide,
};

struct Request {
    int64_t width = 0;
    int64_t outer_rows = 0;
    int64_t inner_size = 1;
    uint64_t device_shared_memory_bytes = 0;
    uintptr_t input_address = 0;
    uintptr_t output_address = 0;
    float scale = 1.0f;
    float max_bias = 0.0f;
    bool input_f32 = true;
    bool output_f32 = true;
    bool input_contiguous = true;
    bool output_contiguous = true;
    bool mask_present = false;
    bool sinks_present = false;
    uint64_t input_available_bytes = std::numeric_limits<uint64_t>::max();
    uint64_t output_available_bytes = std::numeric_limits<uint64_t>::max();
};

struct Decision {
    Route route = Route::kUnsupported;
    int32_t block_threads = 0;
    int32_t registers_per_thread_estimate = 0;
    uint64_t reduction_shared_bytes = 0;
    bool can_use_shared_memory = false;
    const char* reason = "invalid_request";
};

inline int32_t source_block_size(int64_t width) {
    if (width <= 0) {
        return 0;
    }
    const int64_t max_block_size = width < kMaxThreads ? width : kMaxThreads;
    if (max_block_size % kWarpSize == 0) {
        return static_cast<int32_t>(max_block_size);
    }
    return static_cast<int32_t>((max_block_size / kWarpSize + 1) * kWarpSize);
}

inline Decision source_route(const Request& request) {
    Decision decision;
    if (request.width <= 0 || request.width > 2147483644LL || request.outer_rows <= 0 || request.inner_size <= 0) {
        decision.reason = "nonpositive_geometry";
        return decision;
    }
    if (!request.input_f32 || !request.output_f32) {
        decision.reason = "non_f32";
        return decision;
    }
    if (!request.input_contiguous || !request.output_contiguous) {
        decision.reason = "noncontiguous";
        return decision;
    }
    if (request.mask_present) {
        decision.reason = "masked";
        return decision;
    }
    if (request.sinks_present) {
        decision.reason = "sinks";
        return decision;
    }
    if (request.scale != 1.0f || request.max_bias != 0.0f) {
        decision.reason = "non_default_softmax_parameters";
        return decision;
    }
    if (request.inner_size != 1) {
        decision.reason = "inner_size_not_one";
        return decision;
    }
    if (request.input_address == 0 || request.output_address == 0) {
        decision.reason = "null_data";
        return decision;
    }
    if (request.input_address % alignof(float) != 0 || request.output_address % alignof(float) != 0) {
        decision.reason = "element_alignment";
        return decision;
    }
    const uint64_t width = static_cast<uint64_t>(request.width);
    const uint64_t rows = static_cast<uint64_t>(request.outer_rows);
    if (rows > std::numeric_limits<uint64_t>::max() / width) {
        decision.reason = "element_count_overflow";
        return decision;
    }
    const uint64_t elements = rows * width;
    if (elements > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        decision.reason = "byte_count_overflow";
        return decision;
    }
    const uint64_t required_bytes = elements * sizeof(float);
    if (required_bytes > request.input_available_bytes || required_bytes > request.output_available_bytes) {
        decision.reason = "buffer_range";
        return decision;
    }
    const uint64_t address_max = std::numeric_limits<uintptr_t>::max();
    if (required_bytes > address_max - request.input_address ||
        required_bytes > address_max - request.output_address) {
        decision.reason = "address_range";
        return decision;
    }

    decision.block_threads = source_block_size(request.width);
    if (decision.block_threads == 0) {
        decision.reason = "invalid_block_size";
        return decision;
    }
    decision.registers_per_thread_estimate = static_cast<int32_t>(
        (request.width + decision.block_threads - 1) / decision.block_threads);
    decision.reduction_shared_bytes = static_cast<uint64_t>(
        decision.block_threads / kWarpSize) * sizeof(float);

    if (request.width <= 2048 && request.width * static_cast<int64_t>(sizeof(float)) <= 8192) {
        decision.route = Route::kPersistent;
        decision.reason = "persistent_source_route";
        return decision;
    }
    if (decision.registers_per_thread_estimate < 10) {
        decision.route = Route::kRegister;
        decision.reason = "register_specialized_source_route";
        return decision;
    }

    const bool shared_bytes_suffice = request.device_shared_memory_bytes >= decision.reduction_shared_bytes &&
        static_cast<uint64_t>(request.width) <
            (request.device_shared_memory_bytes - decision.reduction_shared_bytes) / sizeof(float);
    const bool input_aligned = request.input_address % kAlignmentBytes == 0;
    const bool output_aligned = request.output_address % kAlignmentBytes == 0;
    const bool width_vectorized = request.width % kIlp == 0;
    decision.can_use_shared_memory = shared_bytes_suffice && input_aligned && output_aligned && width_vectorized;
    if (decision.can_use_shared_memory) {
        decision.route = Route::kSharedMemory;
        decision.reason = "shared_memory_source_route";
        return decision;
    }

    decision.route = Route::kRegularWide;
    if (!shared_bytes_suffice) {
        decision.reason = "regular_wide_shared_capacity";
    } else if (!input_aligned || !output_aligned) {
        decision.reason = "regular_wide_alignment";
    } else {
        decision.reason = "regular_wide_width_tail";
    }
    return decision;
}

inline const char* route_name(Route route) {
    switch (route) {
        case Route::kUnsupported: return "unsupported";
        case Route::kPersistent: return "persistent";
        case Route::kRegister: return "register";
        case Route::kSharedMemory: return "shared_memory";
        case Route::kRegularWide: return "regular_wide";
    }
    return "unknown";
}

} // namespace ggml_cuda_wide_softmax

#endif // GGML_CUDA_WIDE_SOFTMAX_ELIGIBILITY_CUH
