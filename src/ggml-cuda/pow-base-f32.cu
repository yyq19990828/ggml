#include "pow-base-f32.cuh"

#include "ggml-impl.h"

#include <cmath>

#define CUDA_POW_BASE_F32_BLOCK_SIZE 256

static __global__ void pow_base_f32_kernel(
        const float * exponent,
        float * output,
        float base,
        int64_t count) {
    ggml_cuda_pdl_lc();
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    ggml_cuda_pdl_sync();

    for (int64_t i = tid; i < count; i += stride) {
        output[i] = powf(base, exponent[i]);
    }
}

void ggml_cuda_op_pow_base_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float base = ggml_get_op_params_f32(dst, 0);

    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
    GGML_ASSERT(std::isfinite(base) && base > 0.0f);

    const int64_t count = ggml_nelements(dst);
    if (count == 0) {
        return;
    }

    const int64_t blocks = count / CUDA_POW_BASE_F32_BLOCK_SIZE +
        (count % CUDA_POW_BASE_F32_BLOCK_SIZE != 0);
    const ggml_cuda_kernel_launch_params launch_params(
        MIN(0x7fffffff, blocks), CUDA_POW_BASE_F32_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(pow_base_f32_kernel, launch_params,
        static_cast<const float *>(src->data),
        static_cast<float *>(dst->data), base, count);
}
