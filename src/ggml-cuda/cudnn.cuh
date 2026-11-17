#ifndef SAM_CPP_GGML_CUDA_CUDNN_CUH
#define SAM_CPP_GGML_CUDA_CUDNN_CUH

#include "common.cuh"

#ifdef GGML_CUDA_CUDNN
bool ggml_cuda_cudnn_supports(const ggml_tensor * dst);
void ggml_cuda_cudnn_conv2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_cudnn_release(ggml_backend_cuda_context & ctx);
int ggml_cuda_cudnn_version();
#endif

#endif // SAM_CPP_GGML_CUDA_CUDNN_CUH
