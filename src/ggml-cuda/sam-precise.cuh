#ifndef GGML_CUDA_SAM_PRECISE_CUH
#define GGML_CUDA_SAM_PRECISE_CUH

#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
void ggml_cuda_sam_f32_short_dot(const float * lhs, const float * rhs, float * dst,
                                int k, cudaStream_t stream);
#endif

bool ggml_cuda_sam_window_supported(const ggml_tensor * dst);
void ggml_cuda_sam_window(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_sam_f32_attention_requested(const ggml_tensor * dst);
bool ggml_cuda_sam_f32_attention_supported(const ggml_tensor * dst);
void ggml_cuda_sam_f32_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#endif // GGML_CUDA_SAM_PRECISE_CUH
