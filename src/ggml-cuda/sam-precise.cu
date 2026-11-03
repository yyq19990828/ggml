// SPDX-License-Identifier: MIT
// SAM's precise, bounded-workspace CUDA operators for pinned GGML.
#include "sam-precise.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
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
    const int tile = int(std::min<int64_t>(q->ne[1], 1024));
    const size_t score_head_stride = size_t(keys) * tile;
    // Batch heads while keeping score scratch bounded by 32 MiB or one head.
    // GQA retains its original head mapping and single-head workspace.
    const size_t scratch_budget = 32 * 1024 * 1024;
    const int max_heads = q->ne[2] == k->ne[2]
        ? int(std::min<int64_t>({q->ne[2], 8,
            int64_t(std::max<size_t>(1, scratch_budget / (score_head_stride * sizeof(float))))})) : 1;
    ggml_cuda_pool_alloc<float> scores(ctx.pool(), score_head_stride * max_heads);
    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));
    const float zero = 0.0f, one = 1.0f;
    const int64_t head_ratio = q->ne[2] / k->ne[2];
    const int64_t batch_ratio = q->ne[3] / k->ne[3];
    for (int64_t batch = 0; batch < q->ne[3]; ++batch) {
        for (int64_t head = 0; head < q->ne[2]; head += max_heads) {
            const int heads = int(std::min<int64_t>(max_heads, q->ne[2] - head));
            const char * key_data = (const char *) k->data + (batch / batch_ratio) * k->nb[3] +
                (head / head_ratio) * k->nb[2];
            const char * value_data = (const char *) v->data + (batch / batch_ratio) * v->nb[3] +
                (head / head_ratio) * v->nb[2];
            const char * mask_data = mask ? (const char *) mask->data +
                (batch % mask->ne[3]) * mask->nb[3] : nullptr;
            for (int64_t first = 0; first < q->ne[1]; first += tile) {
                const int queries = int(std::min<int64_t>(tile, q->ne[1] - first));
                const char * query_data = (const char *) q->data + batch * q->nb[3] + head * q->nb[2] + first * q->nb[1];
                if (heads == 1) {
                    CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                        keys, queries, dimensions, &scale, key_data, CUDA_R_32F, int(k->nb[1] / sizeof(float)),
                        query_data, CUDA_R_32F, int(q->nb[1] / sizeof(float)), &zero,
                        scores.get(), CUDA_R_32F, keys, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                } else {
                    CUBLAS_CHECK(cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                        keys, queries, dimensions, &scale, key_data, CUDA_R_32F, int(k->nb[1] / sizeof(float)), k->nb[2] / sizeof(float),
                        query_data, CUDA_R_32F, int(q->nb[1] / sizeof(float)), q->nb[2] / sizeof(float), &zero,
                        scores.get(), CUDA_R_32F, keys, score_head_stride, heads,
                        CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                }
                sam_attention_softmax<<<dim3(queries, heads), 256, 0, ctx.stream()>>>(scores.get(), mask_data,
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
