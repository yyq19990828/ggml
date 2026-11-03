#include "fma.cuh"

#define CUDA_FMA_BLOCK_SIZE 256

struct ggml_cuda_fma_layout {
    int64_t ne[12];
    int64_t nb[12];
};

template <bool dense>
static __global__ void fma_f32_ieee_rn(
        const float * a, const float * b, const float * c, float * dst,
        int64_t nelements, ggml_cuda_fma_layout layout) {
    ggml_cuda_pdl_lc();
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        if constexpr (dense) {
            dst[i] = __fmaf_ieee_rn(a[i], b[i], c[i]);
        } else {
            int64_t q = i, off[3] = { 0, 0, 0 };
            for (int d = 0; d < 4; ++d) {
                const int64_t coord = q % layout.ne[d];
                q /= layout.ne[d];
                off[0] += coord * layout.nb[d];
                off[1] += (coord % layout.ne[4 + d]) * layout.nb[4 + d];
                off[2] += (coord % layout.ne[8 + d]) * layout.nb[8 + d];
            }
            dst[i] = __fmaf_ieee_rn(
                    *reinterpret_cast<const float *>(reinterpret_cast<const char *>(a) + off[0]),
                    *reinterpret_cast<const float *>(reinterpret_cast<const char *>(b) + off[1]),
                    *reinterpret_cast<const float *>(reinterpret_cast<const char *>(c) + off[2]));
        }
    }
}

void ggml_cuda_op_fma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const ggml_tensor * c = dst->src[2];
    GGML_ASSERT(a && b && c);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && c->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, dst) && ggml_can_repeat(b, dst) && ggml_can_repeat(c, dst));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t nelements = ggml_nelements(dst);
    if (nelements == 0) return;
    const int64_t nblocks = (nelements + CUDA_FMA_BLOCK_SIZE - 1) / CUDA_FMA_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params(MIN(0x7fffffff, nblocks), CUDA_FMA_BLOCK_SIZE, 0, ctx.stream());
    const ggml_tensor * src[3] = { a, b, c };
    ggml_cuda_fma_layout layout = {};
    bool dense = true;
    for (int k = 0; k < 3; ++k) {
        dense &= ggml_are_same_shape(src[k], dst) && ggml_is_contiguous(src[k]);
        for (int d = 0; d < 4; ++d) {
            layout.ne[k*4+d] = src[k]->ne[d];
            layout.nb[k*4+d] = src[k]->nb[d];
        }
    }
    if (dense) {
        ggml_cuda_kernel_launch(fma_f32_ieee_rn<true>, launch_params,
                static_cast<const float *>(a->data), static_cast<const float *>(b->data),
                static_cast<const float *>(c->data), static_cast<float *>(dst->data), nelements, layout);
    } else {
        ggml_cuda_kernel_launch(fma_f32_ieee_rn<false>, launch_params,
                static_cast<const float *>(a->data), static_cast<const float *>(b->data),
                static_cast<const float *>(c->data), static_cast<float *>(dst->data), nelements, layout);
    }
}
