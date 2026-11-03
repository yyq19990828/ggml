#ifndef SAM_CPP_BUILD_GGML_ORDERED_F32_CONVOLUTION_PROTOTYPE_001_SOURCE_002_GGML_SRC_INCLUDE_GGML_MUL_MAT_REDUCTION_HPP
#define SAM_CPP_BUILD_GGML_ORDERED_F32_CONVOLUTION_PROTOTYPE_001_SOURCE_002_GGML_SRC_INCLUDE_GGML_MUL_MAT_REDUCTION_HPP

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opt-in reduction policy for GGML_OP_MUL_MAT. DEFAULT preserves the backend's
// existing reduction algorithm. F32_CANONICAL_K_FMA requests, for every output,
// an accumulator initialized to +0 followed by logical-K-ascending IEEE RN FMA
// with denormals preserved; any bias remains a separate GGML operation.
//
// The candidate currently implements this policy only on CUDA for the declared
// 2D, batch-one F32/layout subset. Callers must check backend supports_op before
// dispatch and must not silently fall back to the default arithmetic when a
// non-default policy is unsupported.
enum ggml_mul_mat_reduction_policy {
    GGML_MUL_MAT_REDUCTION_DEFAULT = 0,
    GGML_MUL_MAT_REDUCTION_F32_CANONICAL_K_FMA = 1,
};

// Capability handshake exposed by the CUDA registry through
// ggml_backend_reg_get_proc_address. Hosts using a non-default policy must
// resolve this exact proc and require return value 1; an older CUDA provider
// may ignore op_params[4] and must fail closed.
#define GGML_BACKEND_CUDA_MUL_MAT_REDUCTION_POLICY_VERSION_PROC_NAME \
    "ggml_backend_cuda_mul_mat_reduction_policy_version"
typedef int (*ggml_backend_cuda_mul_mat_reduction_policy_version_fn)(void);

// Sets op_params[4] only. Returns false for a null/non-MUL_MAT tensor or an
// unknown policy value; callers must check this result.
static inline bool ggml_mul_mat_set_reduction_policy(
        struct ggml_tensor * tensor,
        int32_t policy) {
    if (tensor == NULL || tensor->op != GGML_OP_MUL_MAT ||
            (policy != GGML_MUL_MAT_REDUCTION_DEFAULT &&
             policy != GGML_MUL_MAT_REDUCTION_F32_CANONICAL_K_FMA)) {
        return false;
    }

    tensor->op_params[4] = (int32_t) policy;
    return true;
}

// Reads op_params[4] without narrowing it to the enum. A null or non-MUL_MAT
// tensor is treated as DEFAULT; unknown stored values are returned as raw
// int32_t tags so backend validation can reject them safely.
static inline int32_t ggml_mul_mat_get_reduction_policy(
        const struct ggml_tensor * tensor) {
    if (tensor == NULL || tensor->op != GGML_OP_MUL_MAT) {
        return GGML_MUL_MAT_REDUCTION_DEFAULT;
    }

    return tensor->op_params[4];
}

#ifdef __cplusplus
}
#endif

#endif // SAM_CPP_BUILD_GGML_ORDERED_F32_CONVOLUTION_PROTOTYPE_001_SOURCE_002_GGML_SRC_INCLUDE_GGML_MUL_MAT_REDUCTION_HPP
