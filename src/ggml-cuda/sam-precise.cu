// SPDX-License-Identifier: MIT AND BSD-3-Clause
// SAM's precise, bounded-workspace CUDA operators for pinned GGML.
#include "sam-precise.cuh"

#include <cmath>
#include <cfloat>
#include <algorithm>
#include <climits>
#include <cstdint>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// Match SDPA math's operand-side scaling and block reduction order. The 5184
// key DART rows use six registers per thread at 1024 threads, as in ATen's
// cunn_SoftMaxForwardReg. Short rows use the same persistent warp ordering.
__device__ __forceinline__ float sam_exp(float x) { return std::exp(x); }

struct SamMax {
    __device__ __forceinline__ float operator()(float a, float b) const { return a < b ? b : a; }
    __device__ __forceinline__ float combine(float a, float b) const { return a < b ? b : a; }
    __device__ __forceinline__ float warp_shfl_down(float v, int offset) const {
        return __shfl_down_sync(0xffffffffu, v, offset);
    }
};
struct SamAdd {
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
    __device__ __forceinline__ float combine(float a, float b) const { return a + b; }
    __device__ __forceinline__ float warp_shfl_down(float v, int offset) const {
        return __shfl_down_sync(0xffffffffu, v, offset);
    }
};
template<class Op> __device__ __forceinline__ float sam_warp_reduce(float v, Op op) {
    for (int offset = 16; offset; offset >>= 1) v = op.combine(v, op.warp_shfl_down(v, offset));
    return v;
}
template<class Op> __device__ __forceinline__ float sam_block_reduce(
        float v, Op op, float identity, float * shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = sam_warp_reduce(v, op);
    __syncthreads();
    if (!lane) shared[warp] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? shared[lane] : identity;
    if (!warp) v = sam_warp_reduce(v, op);
    if (threadIdx.x == 0) shared[0] = v;
    __syncthreads();
    return shared[0];
}
static __global__ void sam_attention_persistent_softmax(float * scores, const char * mask,
        int keys, int first_query, size_t mask_row_stride, int first_head,
        int mask_heads, size_t mask_head_stride, size_t score_head_stride) {
    const int query = blockIdx.x;
    float * row = scores + size_t(blockIdx.y) * score_head_stride + int64_t(query) * keys;
    const half * mask_row = mask ? reinterpret_cast<const half *>(mask +
        size_t(first_query + query) * mask_row_stride +
        size_t((first_head + blockIdx.y) % mask_heads) * mask_head_stride) : nullptr;
    float values[64];
    float maximum = -FLT_MAX;
    for (int i = 0; i < (keys + 31) / 32; ++i) {
        const int key = threadIdx.x + i * 32;
        values[i] = key < keys ? row[key] + (mask_row ? __half2float(mask_row[key]) : 0.0f) : -INFINITY;
        maximum = SamMax{}(maximum, values[i]);
    }
    for (int offset = 16; offset; offset >>= 1) maximum = SamMax{}(maximum,
        __shfl_xor_sync(0xffffffffu, maximum, offset, 32));
    float sum = 0.0f;
    for (int i = 0; i < (keys + 31) / 32; ++i) {
        values[i] = sam_exp(values[i] - maximum);
        sum += values[i];
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset, 32);
    for (int i = 0; i < (keys + 31) / 32; ++i) {
        const int key = threadIdx.x + i * 32;
        if (key < keys) row[key] = sum > 0.0f ? values[i] / sum : 0.0f;
    }
}


template<int Regs> static __global__ void sam_attention_register_softmax(
        float * scores, const char * mask, int keys, int first_query,
        size_t mask_row_stride, int first_head, int mask_heads,
        size_t mask_head_stride, size_t score_head_stride) {
    const int query = blockIdx.x;
    float * row = scores + size_t(blockIdx.y) * score_head_stride + int64_t(query) * keys;
    const half * mask_row = mask ? reinterpret_cast<const half *>(mask +
        size_t(first_query + query) * mask_row_stride +
        size_t((first_head + blockIdx.y) % mask_heads) * mask_head_stride) : nullptr;
    __shared__ float partial[32];
    float values[Regs];
    float local_max = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < Regs; ++i) {
        const int key = threadIdx.x + i * blockDim.x;
        if (key < keys) {
            values[i] = row[key] + (mask_row ? __half2float(mask_row[key]) : 0.0f);
            local_max = SamMax{}(local_max, values[i]);
        } else values[i] = -INFINITY;
    }
    const float maximum = sam_block_reduce(local_max, SamMax{}, -FLT_MAX, partial);
    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < Regs; ++i) {
        const int key = threadIdx.x + i * blockDim.x;
        if (key < keys) {
            values[i] = sam_exp(values[i] - maximum);
            local_sum += values[i];
        }
    }
    const float sum = sam_block_reduce(local_sum, SamAdd{}, 0.0f, partial);
    #pragma unroll
    for (int i = 0; i < Regs; ++i) {
        const int key = threadIdx.x + i * blockDim.x;
        if (key < keys) row[key] = sum > 0.0f ? values[i] / sum : 0.0f;
    }
}

static __global__ void sam_scale_k(const float * src, float * dst, int dimensions,
        int keys, int heads, size_t key_stride, size_t head_stride, float factor) {
    const int64_t count = int64_t(dimensions) * keys * heads;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
            i += int64_t(blockDim.x) * gridDim.x) {
        const int d = int(i % dimensions);
        const int key = int((i / dimensions) % keys);
        const int head = int(i / (int64_t(dimensions) * keys));
        dst[i] = src[size_t(head) * head_stride + size_t(key) * key_stride + d] * factor;
    }
}
static __global__ void sam_scale_q(const float * src, float * dst, int dimensions,
        int queries, int heads, size_t query_stride, size_t head_stride, float factor) {
    const int64_t count = int64_t(dimensions) * queries * heads;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
            i += int64_t(blockDim.x) * gridDim.x) {
        const int d = int(i % dimensions);
        const int query = int((i / dimensions) % queries);
        const int head = int(i / (int64_t(dimensions) * queries));
        dst[i] = src[size_t(head) * head_stride + size_t(query) * query_stride + d] * factor;
    }
}

static __global__ void sam_f32_short_dot_kernel(const float * lhs, const float * rhs,
                                                float * dst, int k) {
    __shared__ double partial[256];
    double sum = 0.0;
    for (int index = threadIdx.x; index < k; index += blockDim.x) {
        sum += double(lhs[index]) * double(rhs[index]);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) dst[0] = __double2float_rn(partial[0]);
}

void ggml_cuda_sam_f32_short_dot(const float * lhs, const float * rhs, float * dst,
                                int k, cudaStream_t stream) {
    GGML_ASSERT(k >= 1 && k <= 1024);
    sam_f32_short_dot_kernel<<<1, 256, 0, stream>>>(lhs, rhs, dst, k);
    CUDA_CHECK(cudaGetLastError());
}
#endif

static __global__ void sam_window_f32(const float * src, float * dst, int64_t count,
        int64_t channels, int64_t width, int64_t height, int64_t window, int64_t windows_x,
        bool partition) {
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += int64_t(gridDim.x) * blockDim.x) {
        const int64_t channel = index % channels;
        const int64_t pixel = index / channels;
        if (partition) {
            const int64_t x = pixel % window;
            const int64_t y = (pixel / window) % window;
            const int64_t tile = pixel / (window * window);
            const int64_t source_x = (tile % windows_x) * window + x;
            const int64_t source_y = (tile / windows_x) * window + y;
            dst[index] = source_x < width && source_y < height
                ? src[(source_y * width + source_x) * channels + channel] : 0.0f;
        } else {
            const int64_t x = pixel % width;
            const int64_t y = pixel / width;
            const int64_t tile = (y / window) * windows_x + x / window;
            dst[index] = src[((tile * window + y % window) * window + x % window)
                * channels + channel];
        }
    }
}

bool ggml_cuda_sam_window_supported(const ggml_tensor * dst) {
    const auto * src = dst->src[0];
    if (!src || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const bool partition = dst->op == GGML_OP_WIN_PART;
    const int64_t window = ggml_get_op_params_i32(dst, partition ? 2 : 0);
    const auto * image = partition ? src : dst;
    const auto * tiles = partition ? dst : src;
    if (window <= 0 || image->ne[3] != 1 || image->ne[0] != tiles->ne[0] ||
            tiles->ne[1] != window || tiles->ne[2] != window) {
        return false;
    }
    const int64_t nx = image->ne[1] / window + (image->ne[1] % window != 0);
    const int64_t ny = image->ne[2] / window + (image->ne[2] % window != 0);
    return tiles->ne[3] == nx * ny && (!partition ||
        (ggml_get_op_params_i32(dst, 0) == nx && ggml_get_op_params_i32(dst, 1) == ny));
}

void ggml_cuda_sam_window(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_sam_window_supported(dst));
    const bool partition = dst->op == GGML_OP_WIN_PART;
    const auto * image = partition ? dst->src[0] : dst;
    const int64_t window = ggml_get_op_params_i32(dst, partition ? 2 : 0);
    const int64_t nx = image->ne[1] / window + (image->ne[1] % window != 0);
    const int64_t count = ggml_nelements(dst);
    if (count == 0) return;
    const unsigned blocks = unsigned(std::min<int64_t>((count + 255) / 256, 65535));
    sam_window_f32<<<blocks, 256, 0, ctx.stream()>>>((const float *) dst->src[0]->data,
        (float *) dst->data, count, image->ne[0], image->ne[1], image->ne[2], window, nx, partition);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_sam_f32_attention_requested(const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(dst);
    return false;
#else
    return ggml_flash_attn_ext_get_prec(dst) == GGML_PREC_F32 &&
        dst->src[0]->type == GGML_TYPE_F32 && dst->src[1]->type == GGML_TYPE_F32 &&
        dst->src[2]->type == GGML_TYPE_F32 &&
        (dst->src[0]->ne[0] == 32 || dst->src[0]->ne[0] == 64);
#endif
}

bool ggml_cuda_sam_f32_attention_supported(const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(dst);
    return false;
#else
    if (!ggml_cuda_sam_f32_attention_requested(dst) || !ggml_is_contiguous(dst) || dst->nb[2] / sizeof(float) > INT_MAX || dst->src[4]) {
        return false;
    }
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    if (k->ne[0] != q->ne[0] || v->ne[0] != q->ne[0] || k->ne[1] != v->ne[1] ||
        k->ne[2] != v->ne[2] || k->ne[3] != v->ne[3] ||
        q->ne[2] % k->ne[2] != 0 || q->ne[3] % k->ne[3] != 0) return false;
    for (const auto * src : {q, k, v}) {
        if (src->nb[0] != sizeof(float) || src->ne[1] <= 0 || src->ne[1] > INT_MAX ||
            src->nb[1] / sizeof(float) > INT_MAX) return false;
        for (int axis = 1; axis < GGML_MAX_DIMS; ++axis) {
            if (src->nb[axis] % sizeof(float)) return false;
        }
    }
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) return false;
    const auto * mask = dst->src[3];
    return !mask || (mask->type == GGML_TYPE_F16 && mask->nb[0] == sizeof(half) &&
        mask->ne[0] == k->ne[1] && mask->ne[1] >= q->ne[1] &&
        (mask->ne[2] == 1 || mask->ne[2] == q->ne[2]) &&
        (mask->ne[3] == 1 || mask->ne[3] == q->ne[3]));
#endif
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static __global__ void sam_attention_softmax(float * scores, const char * mask,
        int keys, int first_query, size_t mask_row_stride, int first_head,
        int mask_heads, size_t mask_head_stride, size_t score_head_stride) {
    const int query = blockIdx.x;
    float * row = scores + size_t(blockIdx.y) * score_head_stride + int64_t(query) * keys;
    const half * mask_row = mask ? (const half *) (mask +
        size_t(first_query + query) * mask_row_stride +
        size_t((first_head + blockIdx.y) % mask_heads) * mask_head_stride) : nullptr;
    __shared__ float reduction[256];
    float maximum = -INFINITY;
    for (int key = threadIdx.x; key < keys; key += blockDim.x) {
        const float score = row[key] + (mask_row ? __half2float(mask_row[key]) : 0.0f);
        row[key] = score;
        maximum = fmaxf(maximum, score);
    }
    reduction[threadIdx.x] = maximum;
    __syncthreads();
    for (int offset = 128; offset; offset /= 2) {
        if (threadIdx.x < offset) reduction[threadIdx.x] =
            fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
        __syncthreads();
    }
    maximum = reduction[0];
    __syncthreads();
    float sum = 0.0f;
    for (int key = threadIdx.x; key < keys; key += blockDim.x) {
        const float value = maximum == -INFINITY ? 0.0f : expf(row[key] - maximum);
        row[key] = value;
        sum += value;
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = 128; offset; offset /= 2) {
        if (threadIdx.x < offset) reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        __syncthreads();
    }
    const float reciprocal = reduction[0] > 0.0f ? 1.0f / reduction[0] : 0.0f;
    for (int key = threadIdx.x; key < keys; key += blockDim.x) row[key] *= reciprocal;
}

#endif

void ggml_cuda_sam_f32_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_sam_f32_attention_supported(dst));
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * mask = dst->src[3];
    const int dimensions = int(q->ne[0]);
    const int keys = int(k->ne[1]);
    // Preserve head batches: single-head GEMM changes F32 reduction order.
    // Shrink query tiles to bound scores by 32 MiB (or one key row).
    const size_t scratch_budget = 32 * 1024 * 1024;
    const size_t row_bytes = size_t(keys) * sizeof(float);
    const int max_heads = q->ne[2] == k->ne[2]
        ? int(std::min<int64_t>({q->ne[2], 8,
            int64_t(std::max<size_t>(1, scratch_budget / row_bytes))})) : 1;
    const int tile = int(std::min<int64_t>({q->ne[1], 1024,
        int64_t(std::max<size_t>(1, scratch_budget / (row_bytes * max_heads)))}));
    const size_t score_head_stride = size_t(keys) * tile;
    ggml_cuda_pool_alloc<float> scores(ctx.pool(), score_head_stride * max_heads);
    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));
    const float scale_factor = std::sqrt(std::fabs(scale));
    const float key_scale_factor = scale_factor;
    const float query_scale_factor = std::copysign(scale_factor, scale);
    ggml_cuda_pool_alloc<float> scaled_keys(ctx.pool(), size_t(dimensions) * keys * k->ne[2]);
    ggml_cuda_pool_alloc<float> scaled_queries(ctx.pool(), size_t(dimensions) * tile * max_heads);
    const float zero = 0.0f, one = 1.0f;
    const int64_t head_ratio = q->ne[2] / k->ne[2];
    const int64_t batch_ratio = q->ne[3] / k->ne[3];
    for (int64_t batch = 0; batch < q->ne[3]; ++batch) {
        if (batch == 0 || (k->nb[3] != 0 && batch % batch_ratio == 0)) {
            const int64_t kv_batch = batch / batch_ratio;
            const char * raw_keys = (const char *) k->data + kv_batch * k->nb[3];
            const int64_t key_count = int64_t(dimensions) * keys * k->ne[2];
            sam_scale_k<<<unsigned(std::min<int64_t>((key_count + 255) / 256, 65535)), 256, 0, ctx.stream()>>>(
                (const float *) raw_keys, scaled_keys.get(), dimensions, keys, int(k->ne[2]),
                int(k->nb[1] / sizeof(float)), int(k->nb[2] / sizeof(float)), key_scale_factor);
            CUDA_CHECK(cudaGetLastError());
        }
        for (int64_t head = 0; head < q->ne[2]; head += max_heads) {
            const int heads = int(std::min<int64_t>(max_heads, q->ne[2] - head));
            const float * key_data = scaled_keys.get() + size_t(head / head_ratio) * dimensions * keys;
            const char * value_data = (const char *) v->data + (batch / batch_ratio) * v->nb[3] +
                (head / head_ratio) * v->nb[2];
            const char * mask_data = mask ? (const char *) mask->data +
                (batch % mask->ne[3]) * mask->nb[3] : nullptr;
            for (int64_t first = 0; first < q->ne[1]; first += tile) {
                const int queries = int(std::min<int64_t>(tile, q->ne[1] - first));
                const char * query_data = (const char *) q->data + batch * q->nb[3] + head * q->nb[2] + first * q->nb[1];
                const int64_t query_count = int64_t(dimensions) * queries * heads;
                float * query_pack = scaled_queries.get();
                sam_scale_q<<<unsigned(std::min<int64_t>((query_count + 255) / 256, 65535)), 256, 0, ctx.stream()>>>(
                    (const float *) query_data, query_pack, dimensions, queries, heads,
                    int(q->nb[1] / sizeof(float)), int(q->nb[2] / sizeof(float)), query_scale_factor);
                CUDA_CHECK(cudaGetLastError());
                if (heads == 1) {
                    CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                        keys, queries, dimensions, &one, key_data, CUDA_R_32F, dimensions,
                        query_pack, CUDA_R_32F, dimensions, &zero,
                        scores.get(), CUDA_R_32F, keys, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                } else {
                    CUBLAS_CHECK(cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                        keys, queries, dimensions, &one, key_data, CUDA_R_32F, dimensions, size_t(dimensions) * keys,
                        query_pack, CUDA_R_32F, dimensions, size_t(dimensions) * queries, &zero,
                        scores.get(), CUDA_R_32F, keys, score_head_stride, heads,
                        CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                }
                const dim3 softmax_grid(queries, heads);
                if (keys <= 2048) {
                    sam_attention_persistent_softmax<<<softmax_grid, 32, 0, ctx.stream()>>>(scores.get(), mask_data,
                        keys, int(first), mask ? mask->nb[1] : 0, int(head), mask ? int(mask->ne[2]) : 1,
                        mask ? mask->nb[2] : 0, score_head_stride);
                } else if (keys <= 9216) {
                    if (keys <= 3072) sam_attention_register_softmax<3><<<softmax_grid, 1024, 0, ctx.stream()>>>(scores.get(), mask_data, keys, int(first), mask ? mask->nb[1] : 0, int(head), mask ? int(mask->ne[2]) : 1, mask ? mask->nb[2] : 0, score_head_stride);
                    else if (keys <= 4096) sam_attention_register_softmax<4><<<softmax_grid, 1024, 0, ctx.stream()>>>(scores.get(), mask_data, keys, int(first), mask ? mask->nb[1] : 0, int(head), mask ? int(mask->ne[2]) : 1, mask ? mask->nb[2] : 0, score_head_stride);
                    else if (keys <= 5120) sam_attention_register_softmax<5><<<softmax_grid, 1024, 0, ctx.stream()>>>(scores.get(), mask_data, keys, int(first), mask ? mask->nb[1] : 0, int(head), mask ? int(mask->ne[2]) : 1, mask ? mask->nb[2] : 0, score_head_stride);
                    else if (keys <= 6144) sam_attention_register_softmax<6><<<softmax_grid, 1024, 0, ctx.stream()>>>(scores.get(), mask_data, keys, int(first), mask ? mask->nb[1] : 0, int(head), mask ? int(mask->ne[2]) : 1, mask ? mask->nb[2] : 0, score_head_stride);
                    else if (keys <= 7168) sam_attention_register_softmax<7><<<softmax_grid, 1024, 0, ctx.stream()>>>(scores.get(), mask_data, keys, int(first), mask ? mask->nb[1] : 0, int(head), mask ? int(mask->ne[2]) : 1, mask ? mask->nb[2] : 0, score_head_stride);
                    else if (keys <= 8192) sam_attention_register_softmax<8><<<softmax_grid, 1024, 0, ctx.stream()>>>(scores.get(), mask_data, keys, int(first), mask ? mask->nb[1] : 0, int(head), mask ? int(mask->ne[2]) : 1, mask ? mask->nb[2] : 0, score_head_stride);
                    else sam_attention_register_softmax<9><<<softmax_grid, 1024, 0, ctx.stream()>>>(scores.get(), mask_data, keys, int(first), mask ? mask->nb[1] : 0, int(head), mask ? int(mask->ne[2]) : 1, mask ? mask->nb[2] : 0, score_head_stride);
                } else sam_attention_softmax<<<softmax_grid, 256, 0, ctx.stream()>>>(scores.get(), mask_data,
                    keys, int(first), mask ? mask->nb[1] : 0, int(head), mask ? int(mask->ne[2]) : 1,
                    mask ? mask->nb[2] : 0, score_head_stride);
                CUDA_CHECK(cudaGetLastError());
                char * output = (char *) dst->data + batch * dst->nb[3] + head * dst->nb[1] + first * dst->nb[2];
                if (heads == 1) {
                    CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                        dimensions, queries, keys, &one, value_data, CUDA_R_32F, int(v->nb[1] / sizeof(float)),
                        scores.get(), CUDA_R_32F, keys, &zero, output, CUDA_R_32F, int(dst->nb[2] / sizeof(float)),
                        CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                } else {
                    CUBLAS_CHECK(cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                        dimensions, queries, keys, &one, value_data, CUDA_R_32F, int(v->nb[1] / sizeof(float)), v->nb[2] / sizeof(float),
                        scores.get(), CUDA_R_32F, keys, score_head_stride, &zero,
                        output, CUDA_R_32F, int(dst->nb[2] / sizeof(float)), dst->nb[1] / sizeof(float), heads,
                        CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                }
            }
        }
    }
#else
    GGML_UNUSED(ctx);
#endif
}
