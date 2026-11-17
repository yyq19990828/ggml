#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "unary.cuh"

namespace {

static __global__ void log_f32_precise_kernel(const float * src, float * dst, const int64_t count) {
    ggml_cuda_pdl_lc();
    const int64_t index = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    ggml_cuda_pdl_sync();
    dst[index] = logf(src[index]);
}

static __global__ void log2_f32_precise_kernel(const float * src, float * dst, const int64_t count) {
    ggml_cuda_pdl_lc();
    const int64_t index = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (index >= count) return;
    ggml_cuda_pdl_sync();
    dst[index] = log2f(src[index]);
}

} // namespace

void ggml_cuda_op_log_f32_precise(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));

    const int64_t count = ggml_nelements(src);
    const int blocks = static_cast<int>((count + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE);
    const ggml_cuda_kernel_launch_params params(
        dim3(blocks), CUDA_NEG_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(log_f32_precise_kernel, params,
                            static_cast<const float *>(src->data),
                            static_cast<float *>(dst->data), count);
}

void ggml_cuda_op_log2_f32_precise(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));

    const int64_t count = ggml_nelements(src);
    if (count == 0) return;
    const int blocks = static_cast<int>((count + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE);
    const ggml_cuda_kernel_launch_params params(dim3(blocks), CUDA_NEG_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(log2_f32_precise_kernel, params,
                            static_cast<const float *>(src->data),
                            static_cast<float *>(dst->data), count);
}

#endif // !GGML_USE_HIP && !GGML_USE_MUSA
