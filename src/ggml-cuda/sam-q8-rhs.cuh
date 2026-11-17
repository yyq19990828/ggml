#ifndef GGML_CUDA_SAM_Q8_RHS_CUH
#define GGML_CUDA_SAM_Q8_RHS_CUH
#include "ggml-sam.h"
static constexpr int32_t GGML_SAM_Q8_RHS = 0x53413801;
bool ggml_cuda_sam_q8_rhs_supported(int device, const ggml_tensor * op);
bool ggml_cuda_sam_q8_rhs_configure(ggml_tensor * op, int device);
bool ggml_cuda_sam_q8_rhs_query(const ggml_tensor * op, ggml_sam_q8_rhs_info * info);
void ggml_cuda_sam_q8_rhs_install_capture(ggml_sam_q8_rhs_capture, void *);
#endif
