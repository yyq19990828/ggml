#include "unary.cuh"

namespace {

template <bool IsSin>
static __global__ void nonfast_sincos_f32_kernel(const float* source, float* destination,
                                                const int64_t count) {
    ggml_cuda_pdl_lc();
    const int64_t index = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    ggml_cuda_pdl_sync();
    if constexpr (IsSin) {
        destination[index] = sinf(source[index]);
    } else {
        destination[index] = cosf(source[index]);
    }
}

template <bool IsSin>
static void launch_nonfast_sincos_f32(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    const ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));

    const int64_t count = ggml_nelements(src);
    const int blocks = static_cast<int>((count + CUDA_SIN_BLOCK_SIZE - 1) / CUDA_SIN_BLOCK_SIZE);
    const ggml_cuda_kernel_launch_params params(
        dim3(blocks), CUDA_SIN_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(nonfast_sincos_f32_kernel<IsSin>, params,
                            static_cast<const float*>(src->data),
                            static_cast<float*>(dst->data), count);
}

} // namespace

void ggml_cuda_op_sin_f32_nonfast(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    launch_nonfast_sincos_f32<true>(ctx, dst);
}

void ggml_cuda_op_cos_f32_nonfast(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    launch_nonfast_sincos_f32<false>(ctx, dst);
}
