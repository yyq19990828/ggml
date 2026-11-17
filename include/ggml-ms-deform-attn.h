#ifndef GGML_MS_DEFORM_ATTN_H
#define GGML_MS_DEFORM_ATTN_H

#include "ggml.h"

#define GGML_MS_DEFORM_ATTN_VERSION 1
#define GGML_BACKEND_CUDA_MS_DEFORM_ATTN_VERSION_PROC_NAME "ggml_backend_cuda_ms_deform_attn_version"

typedef int (*ggml_backend_cuda_ms_deform_attn_version_fn)(void);

#ifdef __cplusplus
extern "C" {
#endif

// values: [C, heads, spatial_size, batch], F32
// spatial_shapes: [2, levels], I32, with each column storing (height, width)
// level_start_index: [levels], I32
// Metadata contract: positive level extents, exact prefix starts, and total area == values.spatial_size.
// The CUDA kernel validates this contract independently and writes NaN for invalid metadata.
// sampling_locations: [2, points, levels, heads * queries * batch], F32
//   physical order follows Torch [batch, queries, heads, levels, points, xy]
// attention_weights: [points, levels, heads, queries * batch], F32
// output: [C, heads, queries, batch], F32
GGML_API struct ggml_tensor * ggml_ms_deform_attn(
        struct ggml_context * ctx,
        struct ggml_tensor * values,
        struct ggml_tensor * spatial_shapes,
        struct ggml_tensor * level_start_index,
        struct ggml_tensor * sampling_locations,
        struct ggml_tensor * attention_weights);

#ifdef __cplusplus
}
#endif

#endif // GGML_MS_DEFORM_ATTN_H
