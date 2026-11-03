// Source-derived adaptation of PyTorch's generic regular F32 softmax CUDA path.
// Pinned source and license are recorded beside this isolated runtime overlay.

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "common.cuh"
#include "ggml.h"
#include "softmax-wide-order-eligibility.cuh"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr int kIlp = 4;
constexpr int kWarpSize = 32;
constexpr int kThreads = 1024;
constexpr std::uint64_t kAlignmentBytes = 16;
using ggml_cuda_wide_softmax::Request;
using ggml_cuda_wide_softmax::Route;

struct MaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return a < b ? b : a;
    }
};

struct AddOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return a + b;
    }
};

template <typename Op>
__device__ __forceinline__ float warp_reduce_down(float value, Op op) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = op(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

template <typename Op>
__device__ __forceinline__ float block_reduce_warp(float value, Op op, float identity, float* shared) {
    const int thread_id = static_cast<int>(threadIdx.x);
    const int lane = thread_id % kWarpSize;
    const int warp = thread_id / kWarpSize;
    value = warp_reduce_down(value, op);
    __syncthreads();
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = thread_id < (static_cast<int>(blockDim.x) / kWarpSize) ? shared[lane] : identity;
    if (warp == 0) {
        value = warp_reduce_down(value, op);
    }
    if (thread_id == 0) {
        shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ int32_t alignment_shift(const void* address, std::uint64_t alignment_bytes) {
    return static_cast<int32_t>(reinterpret_cast<std::uintptr_t>(address) % alignment_bytes / sizeof(float));
}

template <int Ilp>
__device__ __forceinline__ float ilp_reduce_max(const float* row, int32_t width, int32_t shift) {
    float value = -FLT_MAX;
    int32_t offset = static_cast<int32_t>(threadIdx.x);
    const float* data = row;
    int32_t size = width;
    if (shift > 0) {
        data -= shift;
        size += shift;
        if (offset >= shift && offset < size) {
            value = ::max(value, data[offset]);
        }
        size -= static_cast<int32_t>(blockDim.x) > size ? size : static_cast<int32_t>(blockDim.x);
        data += blockDim.x;
    }
    const int32_t last = size % (Ilp * static_cast<int32_t>(blockDim.x));
    for (; offset * Ilp < size - last; offset += static_cast<int32_t>(blockDim.x)) {
        const float4 vector = reinterpret_cast<const float4*>(data)[offset];
        value = ::max(value, vector.x);
        value = ::max(value, vector.y);
        value = ::max(value, vector.z);
        value = ::max(value, vector.w);
    }
    offset = size - last + static_cast<int32_t>(threadIdx.x);
    for (; offset < size; offset += static_cast<int32_t>(blockDim.x)) {
        value = ::max(value, data[offset]);
    }
    return value;
}

template <int Ilp>
__device__ __forceinline__ float ilp_reduce_sum_exp(
        const float* row, int32_t width, int32_t shift, float row_max) {
    float value = 0.0f;
    int32_t offset = static_cast<int32_t>(threadIdx.x);
    const float* data = row;
    int32_t size = width;
    if (shift > 0) {
        data -= shift;
        size += shift;
        if (offset >= shift && offset < size) {
            value = value + std::exp(data[offset] - row_max);
        }
        size -= static_cast<int32_t>(blockDim.x) > size ? size : static_cast<int32_t>(blockDim.x);
        data += blockDim.x;
    }
    const int32_t last = size % (Ilp * static_cast<int32_t>(blockDim.x));
    for (; offset * Ilp < size - last; offset += static_cast<int32_t>(blockDim.x)) {
        const float4 vector = reinterpret_cast<const float4*>(data)[offset];
        value = value + std::exp(vector.x - row_max);
        value = value + std::exp(vector.y - row_max);
        value = value + std::exp(vector.z - row_max);
        value = value + std::exp(vector.w - row_max);
    }
    offset = size - last + static_cast<int32_t>(threadIdx.x);
    for (; offset < size; offset += static_cast<int32_t>(blockDim.x)) {
        value = value + std::exp(data[offset] - row_max);
    }
    return value;
}

template <int Ilp>
__device__ __forceinline__ void write_results_vectorized(
        int32_t width,
        int32_t shift,
        const float* row,
        float* output,
        float row_max,
        float row_sum) {
    int32_t offset = static_cast<int32_t>(threadIdx.x);
    const float* input = row;
    int32_t size = width;
    if (shift > 0) {
        input -= shift;
        output -= shift;
        size += shift;
        if (offset >= shift && offset < size) {
            output[offset] = std::exp(input[offset] - row_max) / row_sum;
        }
        size -= static_cast<int32_t>(blockDim.x) > size ? size : static_cast<int32_t>(blockDim.x);
        input += blockDim.x;
        output += blockDim.x;
    }
    const int32_t last = size % (Ilp * static_cast<int32_t>(blockDim.x));
    for (; offset * Ilp < size - last; offset += static_cast<int32_t>(blockDim.x)) {
        const float4 in = reinterpret_cast<const float4*>(input)[offset];
        const float4 result = {
            std::exp(in.x - row_max) / row_sum,
            std::exp(in.y - row_max) / row_sum,
            std::exp(in.z - row_max) / row_sum,
            std::exp(in.w - row_max) / row_sum,
        };
        reinterpret_cast<float4*>(output)[offset] = result;
    }
    offset = size - last + static_cast<int32_t>(threadIdx.x);
    for (; offset < size; offset += static_cast<int32_t>(blockDim.x)) {
        output[offset] = std::exp(input[offset] - row_max) / row_sum;
    }
}

template <int Ilp>
__device__ __forceinline__ void write_results(
        int32_t width,
        const float* input,
        float* output,
        float row_max,
        float row_sum) {
    for (int32_t offset = static_cast<int32_t>(threadIdx.x); offset < width;
         offset += static_cast<int32_t>(blockDim.x)) {
        output[offset] = std::exp(input[offset] - row_max) / row_sum;
    }
}

template <int Ilp>
__global__ void torch_regular_f32_softmax(float* output, const float* input, int32_t width) {
    extern __shared__ unsigned char shared_bytes[];
    float* shared = reinterpret_cast<float*>(shared_bytes);

    input += static_cast<int64_t>(blockIdx.x) * width;
    output += static_cast<int64_t>(blockIdx.x) * width;

    const int32_t shift = alignment_shift(input, kAlignmentBytes);
    const int32_t output_shift = alignment_shift(output, kAlignmentBytes);

    const float thread_max = ilp_reduce_max<Ilp>(input, width, shift);
    const float row_max = block_reduce_warp(thread_max, MaxOp{}, -FLT_MAX, shared);

    const float thread_sum = ilp_reduce_sum_exp<Ilp>(input, width, shift, row_max);
    const float row_sum = block_reduce_warp(thread_sum, AddOp{}, 0.0f, shared);

    if (shift == output_shift) {
        write_results_vectorized<Ilp>(width, shift, input, output, row_max, row_sum);
    } else {
        write_results<Ilp>(width, input, output, row_max, row_sum);
    }
}

bool regular_wide_route(
        const ggml_backend_cuda_context& ctx,
        const ggml_tensor* src,
        const ggml_tensor* dst,
        float scale,
        float max_bias,
        std::uint64_t* rows_out,
        const char** reason) {
    if (src == nullptr || dst == nullptr || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        *reason = "type";
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        *reason = "noncontiguous";
        return false;
    }
    if (src->ne[0] != dst->ne[0] || src->ne[1] != dst->ne[1] ||
        src->ne[2] != dst->ne[2] || src->ne[3] != dst->ne[3]) {
        *reason = "shape";
        return false;
    }
    if (src->ne[0] <= 0 || src->ne[0] > 2147483644LL) {
        *reason = "width_range";
        return false;
    }
    if (src->data == nullptr || dst->data == nullptr) {
        *reason = "null_data";
        return false;
    }
    const auto input_address = reinterpret_cast<std::uintptr_t>(src->data);
    const auto output_address = reinterpret_cast<std::uintptr_t>(dst->data);
    if (input_address % alignof(float) != 0 || output_address % alignof(float) != 0) {
        *reason = "element_alignment";
        return false;
    }

    std::uint64_t rows = 1;
    for (int axis = 1; axis < GGML_MAX_DIMS; ++axis) {
        if (src->ne[axis] <= 0 || rows > std::numeric_limits<std::uint64_t>::max() /
                static_cast<std::uint64_t>(src->ne[axis])) {
            *reason = "row_overflow";
            return false;
        }
        rows *= static_cast<std::uint64_t>(src->ne[axis]);
    }
    if (rows == 0 || rows > static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max())) {
        *reason = "row_count";
        return false;
    }
    const std::uint64_t width = static_cast<std::uint64_t>(src->ne[0]);
    if (rows > std::numeric_limits<std::uint64_t>::max() / width) {
        *reason = "element_count_overflow";
        return false;
    }
    const std::uint64_t elements = rows * width;
    if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
        *reason = "byte_count_overflow";
        return false;
    }
    const std::uint64_t required_bytes = elements * sizeof(float);
    const std::uint64_t input_bytes = static_cast<std::uint64_t>(ggml_nbytes(src));
    const std::uint64_t output_bytes = static_cast<std::uint64_t>(ggml_nbytes(dst));
    if (input_bytes < required_bytes || output_bytes < required_bytes) {
        *reason = "buffer_range";
        return false;
    }
    const std::uintptr_t address_max = std::numeric_limits<std::uintptr_t>::max();
    if (required_bytes > address_max - input_address || required_bytes > address_max - output_address) {
        *reason = "address_range";
        return false;
    }

    const int device = ctx.device;
    const auto& cuda_info = ggml_cuda_info();
    if (device < 0 || device >= cuda_info.device_count) {
        *reason = "device";
        return false;
    }

    Request request;
    request.width = src->ne[0];
    request.outer_rows = static_cast<std::int64_t>(rows);
    request.inner_size = 1;
    request.device_shared_memory_bytes = cuda_info.devices[device].smpb;
    request.input_address = input_address;
    request.output_address = output_address;
    request.scale = scale;
    request.max_bias = max_bias;
    request.input_f32 = src->type == GGML_TYPE_F32;
    request.output_f32 = dst->type == GGML_TYPE_F32;
    request.input_contiguous = ggml_is_contiguous(src);
    request.output_contiguous = ggml_is_contiguous(dst);
    request.mask_present = dst->src[1] != nullptr;
    request.sinks_present = dst->src[2] != nullptr;
    request.input_available_bytes = input_bytes;
    request.output_available_bytes = output_bytes;

    const auto decision = ggml_cuda_wide_softmax::source_route(request);
    if (decision.route != Route::kRegularWide) {
        *reason = decision.reason;
        return false;
    }
    *reason = decision.reason;
    *rows_out = rows;
    return true;
}

bool trace_enabled() {
    const char* value = std::getenv("GGML_CUDA_SOFTMAX_PERSISTENT_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

} // namespace

bool ggml_cuda_softmax_wide_order_f32(
        ggml_backend_cuda_context& ctx, ggml_tensor* dst, float scale, float max_bias) {
    const ggml_tensor* src = dst == nullptr ? nullptr : dst->src[0];
    const char* reason = "unknown";
    std::uint64_t rows = 0;
    if (!regular_wide_route(ctx, src, dst, scale, max_bias, &rows, &reason)) {
        return false;
    }
    const int width = static_cast<int>(src->ne[0]);
    constexpr std::size_t shared_bytes = static_cast<std::size_t>(kThreads / kWarpSize) * sizeof(float);
    torch_regular_f32_softmax<kIlp><<<static_cast<unsigned int>(rows), kThreads, shared_bytes, ctx.stream()>>>(
        static_cast<float*>(dst->data), static_cast<const float*>(src->data), width);
    CUDA_CHECK(cudaGetLastError());

    if (trace_enabled()) {
        std::fprintf(stderr,
            "[ggml-cuda-softmax-wide] path=regular_wide rows=%llu width=%d grid_x=%llu block=%d shared=%zu\n",
            static_cast<unsigned long long>(rows), width,
            static_cast<unsigned long long>(rows), kThreads, shared_bytes);
    }
    return true;
}

#endif // !GGML_USE_HIP && !GGML_USE_MUSA
