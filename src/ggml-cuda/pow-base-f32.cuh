#ifndef GGML_CUDA_POW_BASE_F32_CUH
#define GGML_CUDA_POW_BASE_F32_CUH

#include "common.cuh"

void ggml_cuda_op_pow_base_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#endif // GGML_CUDA_POW_BASE_F32_CUH
