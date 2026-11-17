// Persistent F32 softmax ordering adapted from PyTorch's licensed source path.
// See the adjacent candidate NOTICE.md and LICENSE.pytorch.txt.

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "common.cuh"
#include "ggml.h"

#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr int kCudaWarpSize = 32;
constexpr int kThreadsPerBlock = 128;
constexpr unsigned int kFullMask = 0xffffffffu;
constexpr std::uint64_t kConservativeGridXLimit = 65535;

__device__ __forceinline__ float shuffle_xor(float value, int offset, int width) {
    return __shfl_xor_sync(kFullMask, value, offset, width);
}

template <int WarpBatch, int WarpSize, typename Op>
__device__ __forceinline__ void warp_reduce(float * values, Op op) {
    for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int i = 0; i < WarpBatch; ++i) {
            const float other = shuffle_xor(values[i], offset, WarpSize);
            values[i] = op(values[i], other);
        }
    }
}

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

template <int Log2Elements>
__global__ void persistent_softmax_f32_kernel(
        float * dst, const float * src, std::uint64_t rows, int width, std::uint64_t blocks) {
    constexpr int next_power_of_two = 1 << Log2Elements;
    constexpr int warp_size = next_power_of_two < kCudaWarpSize ? next_power_of_two : kCudaWarpSize;
    constexpr int warp_iterations = next_power_of_two / warp_size;
    constexpr int warp_batch = next_power_of_two <= 128 ? 2 : 1;

    const std::uint64_t linear_block = static_cast<std::uint64_t>(blockIdx.x) +
            static_cast<std::uint64_t>(gridDim.x) * static_cast<std::uint64_t>(blockIdx.y);
    if (linear_block >= blocks) {
        return;
    }
    const std::uint64_t first_batch =
            (static_cast<std::uint64_t>(blockDim.y) * linear_block +
             static_cast<std::uint64_t>(threadIdx.y)) * static_cast<std::uint64_t>(warp_batch);
    if (first_batch >= rows) {
        return;
    }
    const std::uint64_t remaining = rows - first_batch;
    const int local_batches = remaining > static_cast<std::uint64_t>(warp_batch)
            ? warp_batch : static_cast<int>(remaining);
    const int local_index = static_cast<int>(threadIdx.x);

    float elements[warp_batch][warp_iterations];
#pragma unroll
    for (int i = 0; i < warp_batch; ++i) {
        const int batch_element_count = i >= local_batches ? 0 : width;
#pragma unroll
        for (int it = 0; it < warp_iterations; ++it) {
            const int element_index = local_index + it * warp_size;
            if (element_index < batch_element_count) {
                const std::uint64_t row = first_batch + static_cast<std::uint64_t>(i);
                const std::uint64_t element_offset = row * static_cast<std::uint64_t>(width) +
                        static_cast<std::uint64_t>(element_index);
                elements[i][it] = src[element_offset];
            } else {
                elements[i][it] = -CUDART_INF_F;
            }
        }
    }

    float max_value[warp_batch];
#pragma unroll
    for (int i = 0; i < warp_batch; ++i) {
        max_value[i] = elements[i][0];
#pragma unroll
        for (int it = 0; it < warp_iterations; ++it) {
            max_value[i] = max_value[i] > elements[i][it] ? max_value[i] : elements[i][it];
        }
    }
    warp_reduce<warp_batch, warp_size>(max_value, MaxOp{});

    float sum[warp_batch];
#pragma unroll
    for (int i = 0; i < warp_batch; ++i) {
        sum[i] = 0.0f;
#pragma unroll
        for (int it = 0; it < warp_iterations; ++it) {
            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
            sum[i] += elements[i][it];
        }
    }
    warp_reduce<warp_batch, warp_size>(sum, AddOp{});

#pragma unroll
    for (int i = 0; i < warp_batch; ++i) {
        if (i >= local_batches) {
            break;
        }
#pragma unroll
        for (int it = 0; it < warp_iterations; ++it) {
            const int element_index = local_index + it * warp_size;
            if (element_index < width) {
                const std::uint64_t row = first_batch + static_cast<std::uint64_t>(i);
                const std::uint64_t element_offset = row * static_cast<std::uint64_t>(width) +
                        static_cast<std::uint64_t>(element_index);
                dst[element_offset] = elements[i][it] / sum[i];
            } else {
                break;
            }
        }
    }
}

template <int Log2Elements>
__global__ void persistent_softmax_f32_masked_kernel(
        float * dst, const float * src, const half * mask,
        std::uint64_t rows, std::uint64_t width,
        std::uint64_t src_ne1, std::uint64_t src_ne2,
        std::uint64_t mask_ne2, std::uint64_t mask_ne3,
        std::uint64_t mask_nb1, std::uint64_t mask_nb2, std::uint64_t mask_nb3,
        std::uint64_t blocks) {
    constexpr int next_power_of_two = 1 << Log2Elements;
    constexpr int warp_size = next_power_of_two < kCudaWarpSize ? next_power_of_two : kCudaWarpSize;
    constexpr int warp_iterations = next_power_of_two / warp_size;
    constexpr int warp_batch = next_power_of_two <= 128 ? 2 : 1;

    const std::uint64_t linear_block = static_cast<std::uint64_t>(blockIdx.x) +
            static_cast<std::uint64_t>(gridDim.x) * static_cast<std::uint64_t>(blockIdx.y);
    if (linear_block >= blocks) {
        return;
    }
    const std::uint64_t first_batch =
            (static_cast<std::uint64_t>(blockDim.y) * linear_block +
             static_cast<std::uint64_t>(threadIdx.y)) * static_cast<std::uint64_t>(warp_batch);
    if (first_batch >= rows) {
        return;
    }
    const std::uint64_t remaining = rows - first_batch;
    const int local_batches = remaining > static_cast<std::uint64_t>(warp_batch)
            ? warp_batch : static_cast<int>(remaining);
    const int local_index = static_cast<int>(threadIdx.x);

    const half * mask_rows[warp_batch];
#pragma unroll
    for (int i = 0; i < warp_batch; ++i) {
        if (i < local_batches) {
            const std::uint64_t row = first_batch + static_cast<std::uint64_t>(i);
            const std::uint64_t i01 = row % src_ne1;
            const std::uint64_t i02 = (row / src_ne1) % src_ne2;
            const std::uint64_t i03 = row / (src_ne1 * src_ne2);
            const std::uint64_t mask_offset_bytes =
                    i01 * mask_nb1 + (i02 % mask_ne2) * mask_nb2 + (i03 % mask_ne3) * mask_nb3;
            mask_rows[i] = reinterpret_cast<const half *>(
                    reinterpret_cast<const char *>(mask) + mask_offset_bytes);
        } else {
            mask_rows[i] = nullptr;
        }
    }

    float elements[warp_batch][warp_iterations];
#pragma unroll
    for (int i = 0; i < warp_batch; ++i) {
        const int batch_element_count = i >= local_batches ? 0 : static_cast<int>(width);
#pragma unroll
        for (int it = 0; it < warp_iterations; ++it) {
            const int element_index = local_index + it * warp_size;
            if (element_index < batch_element_count) {
                const std::uint64_t row = first_batch + static_cast<std::uint64_t>(i);
                const std::uint64_t element_offset = row * width + static_cast<std::uint64_t>(element_index);
                const float score = src[element_offset] + __half2float(mask_rows[i][element_index]);
                elements[i][it] = score;
            } else {
                elements[i][it] = -CUDART_INF_F;
            }
        }
    }

    float max_value[warp_batch];
#pragma unroll
    for (int i = 0; i < warp_batch; ++i) {
        max_value[i] = elements[i][0];
#pragma unroll
        for (int it = 0; it < warp_iterations; ++it) {
            max_value[i] = max_value[i] > elements[i][it] ? max_value[i] : elements[i][it];
        }
    }
    warp_reduce<warp_batch, warp_size>(max_value, MaxOp{});

    float sum[warp_batch];
#pragma unroll
    for (int i = 0; i < warp_batch; ++i) {
        sum[i] = 0.0f;
#pragma unroll
        for (int it = 0; it < warp_iterations; ++it) {
            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
            sum[i] += elements[i][it];
        }
    }
    warp_reduce<warp_batch, warp_size>(sum, AddOp{});

#pragma unroll
    for (int i = 0; i < warp_batch; ++i) {
        if (i >= local_batches) {
            break;
        }
#pragma unroll
        for (int it = 0; it < warp_iterations; ++it) {
            const int element_index = local_index + it * warp_size;
            if (element_index < static_cast<int>(width)) {
                const std::uint64_t row = first_batch + static_cast<std::uint64_t>(i);
                const std::uint64_t element_offset = row * width + static_cast<std::uint64_t>(element_index);
                dst[element_offset] = elements[i][it] / sum[i];
            } else {
                break;
            }
        }
    }
}

int log2_ceil(int value) {
    int result = 0;
    while ((1 << result) < value) {
        ++result;
    }
    return result;
}

template <int Log2Elements>
bool launch_for_log2(
        const float * src, float * dst, std::uint64_t rows, int width, cudaStream_t stream,
        std::uint64_t * grid_x_out, std::uint64_t * grid_y_out) {
    constexpr int next_power_of_two = 1 << Log2Elements;
    constexpr int warp_size = next_power_of_two < kCudaWarpSize ? next_power_of_two : kCudaWarpSize;
    constexpr int warp_batch = next_power_of_two <= 128 ? 2 : 1;
    constexpr int warps_per_block = kThreadsPerBlock / warp_size;
    constexpr int rows_per_block = warps_per_block * warp_batch;
    const std::uint64_t blocks = 1 + (rows - 1) / static_cast<std::uint64_t>(rows_per_block);
    const std::uint64_t grid_x = blocks < kConservativeGridXLimit ? blocks : kConservativeGridXLimit;
    const std::uint64_t grid_y = 1 + (blocks - 1) / grid_x;
    if (grid_y > kConservativeGridXLimit) {
        return false;
    }

    persistent_softmax_f32_kernel<Log2Elements>
            <<<dim3(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y), 1),
               dim3(warp_size, warps_per_block, 1), 0, stream>>>(dst, src, rows, width, blocks);
    CUDA_CHECK(cudaGetLastError());
    *grid_x_out = grid_x;
    *grid_y_out = grid_y;
    return true;
}

template <int Log2Elements>
bool launch_masked_for_log2(
        const float * src, float * dst, const half * mask,
        std::uint64_t rows, int width,
        std::uint64_t src_ne1, std::uint64_t src_ne2,
        std::uint64_t mask_ne2, std::uint64_t mask_ne3,
        std::uint64_t mask_nb1, std::uint64_t mask_nb2, std::uint64_t mask_nb3,
        cudaStream_t stream, std::uint64_t * grid_x_out, std::uint64_t * grid_y_out) {
    constexpr int next_power_of_two = 1 << Log2Elements;
    constexpr int warp_size = next_power_of_two < kCudaWarpSize ? next_power_of_two : kCudaWarpSize;
    constexpr int warp_batch = next_power_of_two <= 128 ? 2 : 1;
    constexpr int warps_per_block = kThreadsPerBlock / warp_size;
    constexpr int rows_per_block = warps_per_block * warp_batch;
    const std::uint64_t blocks = 1 + (rows - 1) / static_cast<std::uint64_t>(rows_per_block);
    const std::uint64_t grid_x = blocks < kConservativeGridXLimit ? blocks : kConservativeGridXLimit;
    const std::uint64_t grid_y = 1 + (blocks - 1) / grid_x;
    if (grid_y > kConservativeGridXLimit) {
        return false;
    }

    persistent_softmax_f32_masked_kernel<Log2Elements>
            <<<dim3(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y), 1),
               dim3(warp_size, warps_per_block, 1), 0, stream>>>(
                    dst, src, mask, rows, static_cast<std::uint64_t>(width),
                    src_ne1, src_ne2, mask_ne2, mask_ne3,
                    mask_nb1, mask_nb2, mask_nb3, blocks);
    CUDA_CHECK(cudaGetLastError());
    *grid_x_out = grid_x;
    *grid_y_out = grid_y;
    return true;
}

bool launch_persistent_masked_softmax(
        const float * src, float * dst, const half * mask,
        std::uint64_t rows, int width,
        std::uint64_t src_ne1, std::uint64_t src_ne2,
        std::uint64_t mask_ne2, std::uint64_t mask_ne3,
        std::uint64_t mask_nb1, std::uint64_t mask_nb2, std::uint64_t mask_nb3,
        cudaStream_t stream, std::uint64_t * grid_x_out, std::uint64_t * grid_y_out) {
    switch (log2_ceil(width)) {
        case 0:  return launch_masked_for_log2<0>(src, dst, mask, rows, width, src_ne1, src_ne2, mask_ne2, mask_ne3, mask_nb1, mask_nb2, mask_nb3, stream, grid_x_out, grid_y_out);
        case 1:  return launch_masked_for_log2<1>(src, dst, mask, rows, width, src_ne1, src_ne2, mask_ne2, mask_ne3, mask_nb1, mask_nb2, mask_nb3, stream, grid_x_out, grid_y_out);
        case 2:  return launch_masked_for_log2<2>(src, dst, mask, rows, width, src_ne1, src_ne2, mask_ne2, mask_ne3, mask_nb1, mask_nb2, mask_nb3, stream, grid_x_out, grid_y_out);
        case 3:  return launch_masked_for_log2<3>(src, dst, mask, rows, width, src_ne1, src_ne2, mask_ne2, mask_ne3, mask_nb1, mask_nb2, mask_nb3, stream, grid_x_out, grid_y_out);
        case 4:  return launch_masked_for_log2<4>(src, dst, mask, rows, width, src_ne1, src_ne2, mask_ne2, mask_ne3, mask_nb1, mask_nb2, mask_nb3, stream, grid_x_out, grid_y_out);
        case 5:  return launch_masked_for_log2<5>(src, dst, mask, rows, width, src_ne1, src_ne2, mask_ne2, mask_ne3, mask_nb1, mask_nb2, mask_nb3, stream, grid_x_out, grid_y_out);
        case 6:  return launch_masked_for_log2<6>(src, dst, mask, rows, width, src_ne1, src_ne2, mask_ne2, mask_ne3, mask_nb1, mask_nb2, mask_nb3, stream, grid_x_out, grid_y_out);
        case 7:  return launch_masked_for_log2<7>(src, dst, mask, rows, width, src_ne1, src_ne2, mask_ne2, mask_ne3, mask_nb1, mask_nb2, mask_nb3, stream, grid_x_out, grid_y_out);
        case 8:  return launch_masked_for_log2<8>(src, dst, mask, rows, width, src_ne1, src_ne2, mask_ne2, mask_ne3, mask_nb1, mask_nb2, mask_nb3, stream, grid_x_out, grid_y_out);
        case 9:  return launch_masked_for_log2<9>(src, dst, mask, rows, width, src_ne1, src_ne2, mask_ne2, mask_ne3, mask_nb1, mask_nb2, mask_nb3, stream, grid_x_out, grid_y_out);
        case 10: return launch_masked_for_log2<10>(src, dst, mask, rows, width, src_ne1, src_ne2, mask_ne2, mask_ne3, mask_nb1, mask_nb2, mask_nb3, stream, grid_x_out, grid_y_out);
        case 11: return launch_masked_for_log2<11>(src, dst, mask, rows, width, src_ne1, src_ne2, mask_ne2, mask_ne3, mask_nb1, mask_nb2, mask_nb3, stream, grid_x_out, grid_y_out);
        default: GGML_ABORT("unsupported persistent softmax width");
    }
}

bool launch_persistent_softmax(
        const float * src, float * dst, std::uint64_t rows, int width, cudaStream_t stream,
        std::uint64_t * grid_x_out, std::uint64_t * grid_y_out) {
    switch (log2_ceil(width)) {
        case 0:  return launch_for_log2<0>(src, dst, rows, width, stream, grid_x_out, grid_y_out);
        case 1:  return launch_for_log2<1>(src, dst, rows, width, stream, grid_x_out, grid_y_out);
        case 2:  return launch_for_log2<2>(src, dst, rows, width, stream, grid_x_out, grid_y_out);
        case 3:  return launch_for_log2<3>(src, dst, rows, width, stream, grid_x_out, grid_y_out);
        case 4:  return launch_for_log2<4>(src, dst, rows, width, stream, grid_x_out, grid_y_out);
        case 5:  return launch_for_log2<5>(src, dst, rows, width, stream, grid_x_out, grid_y_out);
        case 6:  return launch_for_log2<6>(src, dst, rows, width, stream, grid_x_out, grid_y_out);
        case 7:  return launch_for_log2<7>(src, dst, rows, width, stream, grid_x_out, grid_y_out);
        case 8:  return launch_for_log2<8>(src, dst, rows, width, stream, grid_x_out, grid_y_out);
        case 9:  return launch_for_log2<9>(src, dst, rows, width, stream, grid_x_out, grid_y_out);
        case 10: return launch_for_log2<10>(src, dst, rows, width, stream, grid_x_out, grid_y_out);
        case 11: return launch_for_log2<11>(src, dst, rows, width, stream, grid_x_out, grid_y_out);
        default: GGML_ABORT("unsupported persistent softmax width");
    }
}

bool trace_enabled() {
    const char * value = std::getenv("GGML_CUDA_SOFTMAX_PERSISTENT_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void trace_fallback(const char * reason, const ggml_tensor * src, const ggml_tensor * dst) {
    if (trace_enabled()) {
        std::fprintf(stderr,
                "[ggml-cuda-softmax] path=legacy_fallback reason=%s width=%lld rows=%lld\n",
                reason, static_cast<long long>(src == nullptr ? 0 : src->ne[0]),
                static_cast<long long>(src == nullptr ? 0 : ggml_nrows(src)));
    }
    (void) dst;
}

bool checked_rows(const ggml_tensor * src, std::uint64_t * rows) {
    std::uint64_t product = 1;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src->ne[i] <= 0 ||
                product > std::numeric_limits<std::uint64_t>::max() /
                        static_cast<std::uint64_t>(src->ne[i])) {
            return false;
        }
        product *= static_cast<std::uint64_t>(src->ne[i]);
    }
    *rows = product;
    return true;
}

bool checked_add_product(std::uint64_t index, std::uint64_t stride, std::uint64_t * offset) {
    const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
    if (stride != 0 && index > max / stride) {
        return false;
    }
    const std::uint64_t term = index * stride;
    if (*offset > max - term) {
        return false;
    }
    *offset += term;
    return true;
}

bool checked_mask_access_range(const ggml_tensor * src, const ggml_tensor * mask) {
    if (mask->nb[0] != sizeof(half)) {
        return false;
    }

    std::uint64_t max_offset = 0;
    if (!checked_add_product(static_cast<std::uint64_t>(src->ne[1] - 1),
                             static_cast<std::uint64_t>(mask->nb[1]), &max_offset) ||
        !checked_add_product(static_cast<std::uint64_t>(mask->ne[2] - 1),
                             static_cast<std::uint64_t>(mask->nb[2]), &max_offset) ||
        !checked_add_product(static_cast<std::uint64_t>(mask->ne[3] - 1),
                             static_cast<std::uint64_t>(mask->nb[3]), &max_offset)) {
        return false;
    }

    const std::uint64_t row_bytes = static_cast<std::uint64_t>(mask->ne[0]) * sizeof(half);
    if (max_offset > std::numeric_limits<std::uint64_t>::max() - row_bytes) {
        return false;
    }
    const std::uint64_t max_byte_access = max_offset + row_bytes;
    return max_byte_access <= static_cast<std::uint64_t>(ggml_nbytes(mask));
}

bool checked_f32_access_range(const ggml_tensor * src, const ggml_tensor * dst, std::uint64_t rows) {
    const std::uint64_t width = static_cast<std::uint64_t>(src->ne[0]);
    if (width == 0 || rows > std::numeric_limits<std::uint64_t>::max() / width) {
        return false;
    }
    const std::uint64_t elements = rows * width;
    if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
        return false;
    }
    const std::uint64_t bytes = elements * sizeof(float);
    return bytes <= static_cast<std::uint64_t>(ggml_nbytes(src)) &&
            bytes <= static_cast<std::uint64_t>(ggml_nbytes(dst));
}

bool valid_mask_shape(const ggml_tensor * src, const ggml_tensor * mask) {
    if (mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask)) {
        return false;
    }
    if (mask->ne[0] != src->ne[0] || mask->ne[1] < src->ne[1] ||
        mask->ne[2] <= 0 || mask->ne[3] <= 0) {
        return false;
    }
    if (src->ne[2] % mask->ne[2] != 0 || src->ne[3] % mask->ne[3] != 0) {
        return false;
    }
    return checked_mask_access_range(src, mask);
}

} // namespace

bool ggml_cuda_softmax_order_f32(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, float scale, float max_bias) {
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * mask_tensor = dst->src[1];
    const char * reason = nullptr;

    if (src == nullptr || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        reason = "type";
    } else if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        reason = "noncontiguous";
    } else if (src->ne[0] != dst->ne[0] || src->ne[1] != dst->ne[1] ||
               src->ne[2] != dst->ne[2] || src->ne[3] != dst->ne[3]) {
        reason = "shape";
    } else if (src->ne[0] <= 0 || src->ne[0] > 2048) {
        reason = "width";
    } else if (scale != 1.0f || max_bias != 0.0f) {
        reason = "scale_or_max_bias";
    } else if (dst->src[2] != nullptr) {
        reason = "sink";
    } else if (src->data == nullptr || dst->data == nullptr ||
               (mask_tensor != nullptr && mask_tensor->data == nullptr)) {
        reason = "null_data";
    } else if (mask_tensor != nullptr && mask_tensor->type != GGML_TYPE_F16) {
        reason = "mask_type";
    } else if (mask_tensor != nullptr && !ggml_is_contiguous(mask_tensor)) {
        reason = "mask_noncontiguous";
    }

    std::uint64_t rows = 0;
    if (reason == nullptr && !checked_rows(src, &rows)) {
        reason = "row_overflow";
    }
    if (reason == nullptr && (rows == 0 || rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))) {
        reason = "row_count";
    }
    if (reason == nullptr && !checked_f32_access_range(src, dst, rows)) {
        reason = "tensor_range";
    }
    if (reason == nullptr && mask_tensor != nullptr && !valid_mask_shape(src, mask_tensor)) {
        reason = "mask_shape_or_range";
    }
    if (reason == nullptr) {
        const int width = static_cast<int>(src->ne[0]);
        std::uint64_t grid_x = 0;
        std::uint64_t grid_y = 0;
        if (mask_tensor == nullptr) {
            if (!launch_persistent_softmax(
                        static_cast<const float *>(src->data), static_cast<float *>(dst->data),
                        rows, width, ctx.stream(), &grid_x, &grid_y)) {
                reason = "grid_xy";
            } else {
                if (trace_enabled()) {
                    std::fprintf(stderr,
                            "[ggml-cuda-softmax] path=persistent rows=%llu width=%d grid_x=%llu grid_y=%llu\n",
                            static_cast<unsigned long long>(rows), width,
                            static_cast<unsigned long long>(grid_x), static_cast<unsigned long long>(grid_y));
                }
                return true;
            }
        } else {
            if (!launch_persistent_masked_softmax(
                        static_cast<const float *>(src->data), static_cast<float *>(dst->data),
                        static_cast<const half *>(mask_tensor->data), rows, width,
                        static_cast<std::uint64_t>(src->ne[1]),
                        static_cast<std::uint64_t>(src->ne[2]),
                        static_cast<std::uint64_t>(mask_tensor->ne[2]),
                        static_cast<std::uint64_t>(mask_tensor->ne[3]),
                        static_cast<std::uint64_t>(mask_tensor->nb[1]),
                        static_cast<std::uint64_t>(mask_tensor->nb[2]),
                        static_cast<std::uint64_t>(mask_tensor->nb[3]),
                        ctx.stream(), &grid_x, &grid_y)) {
                reason = "grid_xy";
            } else {
                if (trace_enabled()) {
                    std::fprintf(stderr,
                            "[ggml-cuda-softmax] path=persistent_masked rows=%llu width=%d grid_x=%llu grid_y=%llu mask=F16\n",
                            static_cast<unsigned long long>(rows), width,
                            static_cast<unsigned long long>(grid_x), static_cast<unsigned long long>(grid_y));
                }
                return true;
            }
        }
    }

    trace_fallback(reason, src, dst);
    return false;
}

#endif // !GGML_USE_HIP && !GGML_USE_MUSA
