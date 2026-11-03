#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "unary.cuh"

namespace {

static __global__ void sigmoid_f32_precise_kernel(const float * src, float * dst, const int64_t count) {
    ggml_cuda_pdl_lc();
    const int64_t index = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    ggml_cuda_pdl_sync();

    const float neg = -src[index];
    const float exponential = expf(neg);
    const float denominator = 1.0f + exponential;
    dst[index] = 1.0f / denominator;
}

} // namespace

void ggml_cuda_op_sigmoid_f32_precise(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));

    const int64_t count = ggml_nelements(src);
    const int blocks = static_cast<int>((count + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE);
    const ggml_cuda_kernel_launch_params params(
        dim3(blocks), CUDA_NEG_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(sigmoid_f32_precise_kernel, params,
                            static_cast<const float *>(src->data),
                            static_cast<float *>(dst->data), count);
}

#endif // !GGML_USE_HIP && !GGML_USE_MUSA
