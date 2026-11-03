#ifndef GGML_CUDA_MS_DEFORM_ATTN_CUH
#define GGML_CUDA_MS_DEFORM_ATTN_CUH

#include "ggml-cuda/common.cuh"

void ggml_cuda_op_ms_deform_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#endif // GGML_CUDA_MS_DEFORM_ATTN_CUH
