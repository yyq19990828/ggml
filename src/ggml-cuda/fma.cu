#include "fma.cuh"

#define CUDA_FMA_BLOCK_SIZE 256

static __global__ void fma_f32_ieee_rn(
        const float * a,
        const float * b,
        const float * c,
        float * dst,
        int64_t nelements) {
    ggml_cuda_pdl_lc();
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        // This intrinsic ignores the build-wide -ftz=true implied by -use_fast_math.
        dst[i] = __fmaf_ieee_rn(a[i], b[i], c[i]);
    }
}

void ggml_cuda_op_fma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const ggml_tensor * c = dst->src[2];

    GGML_ASSERT(a && b && c);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && c->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, b) && ggml_are_same_shape(a, c) && ggml_are_same_shape(a, dst));
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(c) && ggml_is_contiguous(dst));

    const int64_t nelements = ggml_nelements(dst);
    if (nelements == 0) {
        return;
    }
    const int64_t nblocks = (nelements + CUDA_FMA_BLOCK_SIZE - 1) / CUDA_FMA_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params(
            MIN(0x7fffffff, nblocks), CUDA_FMA_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(fma_f32_ieee_rn, launch_params,
            static_cast<const float *>(a->data),
            static_cast<const float *>(b->data),
            static_cast<const float *>(c->data),
            static_cast<float *>(dst->data), nelements);
}
