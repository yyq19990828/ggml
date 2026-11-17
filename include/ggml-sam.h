#ifndef GGML_SAM_EXTENSION_H
#define GGML_SAM_EXTENSION_H
#include "ggml.h"
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Optional private registry ABI, version 1. Ordinary inference needs no observer.
struct ggml_sam_q8_rhs_info {
    uint32_t version;
    uint32_t path; // 1: actual Q8_0 MMQ D4 dispatch.
    uint64_t launches;
    uint64_t payload_bytes;
    uint64_t allocation_bytes;
};
typedef bool (*ggml_sam_q8_rhs_configure)(struct ggml_tensor *, int device);
typedef bool (*ggml_sam_q8_rhs_query)(const struct ggml_tensor *, struct ggml_sam_q8_rhs_info *);
// Install/remove only while all CUDA graphs are idle; callback may copy bytes
// on the borrowed CUDA stream, cannot throw/reenter GGML/retain pool pointers.
typedef void (*ggml_sam_q8_rhs_capture)(const struct ggml_tensor *, const void * device_data,
    size_t payload_bytes, size_t allocation_bytes, void * stream, void * user);
typedef void (*ggml_sam_q8_rhs_install_capture)(ggml_sam_q8_rhs_capture, void *);
#ifdef __cplusplus
}
#endif
#endif
