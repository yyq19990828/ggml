/*!
**************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************
* Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
* Copyright (c) 2018 Microsoft
**************************************************************************
*/

/*
 * Adapted from GroundingDINO revision 856dde20aee659246248e20734ef9ba5214f5e44:
 * groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_im2col_cuda.cuh
 */

#include "ms-deform-attn.cuh"

#include "ggml-ms-deform-attn.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define GGML_CUDA_MSDA_BLOCK_SIZE 256

static __device__ __forceinline__ int64_t ggml_cuda_msda_value_offset(
        int64_t channel,
        int64_t head,
        int64_t spatial,
        int64_t batch,
        int64_t channels,
        int64_t heads,
        int64_t spatial_size) {
    return channel + channels * (head + heads * (spatial + spatial_size * batch));
}

static __global__ void ggml_cuda_ms_deform_attn_kernel(
        const float * values,
        const int32_t * spatial_shapes,
        const int32_t * level_start_index,
        const float * sampling_locations,
        const float * attention_weights,
        float * output,
        int64_t channels,
        int64_t heads,
        int64_t spatial_size,
        int64_t batch,
        int64_t queries,
        int64_t levels,
        int64_t points,
        int64_t count) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t index = tid; index < count; index += stride) {
        int64_t remainder = index;
        const int64_t channel = remainder % channels;
        remainder /= channels;
        const int64_t head = remainder % heads;
        remainder /= heads;
        const int64_t query = remainder % queries;
        const int64_t batch_index = remainder / queries;

        int64_t validated_spatial = 0;
        bool metadata_valid = true;
        for (int64_t level = 0; level < levels; ++level) {
            const int64_t height = spatial_shapes[2 * level];
            const int64_t width = spatial_shapes[2 * level + 1];
            const int64_t start = level_start_index[level];
            if (height <= 0 || width <= 0 || start != validated_spatial || validated_spatial > spatial_size) {
                metadata_valid = false;
                break;
            }
            const int64_t area = height * width;
            if (area > spatial_size - validated_spatial) {
                metadata_valid = false;
                break;
            }
            validated_spatial += area;
        }
        if (!metadata_valid || validated_spatial != spatial_size) {
            output[index] = __int_as_float(0x7fc00000);
            continue;
        }

        float accumulated = 0.0f;
        const int64_t sampling_row = head + heads * (query + queries * batch_index);
        for (int64_t level = 0; level < levels; ++level) {
            const int64_t height = spatial_shapes[2 * level];
            const int64_t width = spatial_shapes[2 * level + 1];
            const int64_t level_start = level_start_index[level];
            for (int64_t point = 0; point < points; ++point) {
                const int64_t location_index = 2 * (point + points * (level + levels * sampling_row));
                const float loc_w = sampling_locations[location_index];
                const float loc_h = sampling_locations[location_index + 1];
                const int64_t weight_index = point + points * (level + levels * sampling_row);
                const float weight = attention_weights[weight_index];

                const float h_im = loc_h * height - 0.5;
                const float w_im = loc_w * width - 0.5;
                if (h_im > -1.0f && w_im > -1.0f && h_im < static_cast<float>(height) && w_im < static_cast<float>(width)) {
                    const int64_t h_low = static_cast<int64_t>(floorf(h_im));
                    const int64_t w_low = static_cast<int64_t>(floorf(w_im));
                    const int64_t h_high = h_low + 1;
                    const int64_t w_high = w_low + 1;
                    const float lh = h_im - static_cast<float>(h_low);
                    const float lw = w_im - static_cast<float>(w_low);
                    const float hh = 1.0f - lh;
                    const float hw = 1.0f - lw;

                    float v1 = 0.0f;
                    if (h_low >= 0 && w_low >= 0) {
                        v1 = values[ggml_cuda_msda_value_offset(channel, head, level_start + h_low * width + w_low,
                            batch_index, channels, heads, spatial_size)];
                    }
                    float v2 = 0.0f;
                    if (h_low >= 0 && w_high <= width - 1) {
                        v2 = values[ggml_cuda_msda_value_offset(channel, head, level_start + h_low * width + w_high,
                            batch_index, channels, heads, spatial_size)];
                    }
                    float v3 = 0.0f;
                    if (h_high <= height - 1 && w_low >= 0) {
                        v3 = values[ggml_cuda_msda_value_offset(channel, head, level_start + h_high * width + w_low,
                            batch_index, channels, heads, spatial_size)];
                    }
                    float v4 = 0.0f;
                    if (h_high <= height - 1 && w_high <= width - 1) {
                        v4 = values[ggml_cuda_msda_value_offset(channel, head, level_start + h_high * width + w_high,
                            batch_index, channels, heads, spatial_size)];
                    }

                    const float w1 = hh * hw;
                    const float w2 = hh * lw;
                    const float w3 = lh * hw;
                    const float w4 = lh * lw;
                    const float sampled = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                    accumulated += sampled * weight;
                }
            }
        }
        output[index] = accumulated;
    }
}

void ggml_cuda_op_ms_deform_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * values = dst->src[0];
    const ggml_tensor * spatial_shapes = dst->src[1];
    const ggml_tensor * level_start_index = dst->src[2];
    const ggml_tensor * sampling_locations = dst->src[3];
    const ggml_tensor * attention_weights = dst->src[4];

    GGML_ASSERT(values != nullptr && spatial_shapes != nullptr && level_start_index != nullptr &&
        sampling_locations != nullptr && attention_weights != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && values->type == GGML_TYPE_F32 &&
        sampling_locations->type == GGML_TYPE_F32 && attention_weights->type == GGML_TYPE_F32);
    GGML_ASSERT(spatial_shapes->type == GGML_TYPE_I32 && level_start_index->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(values) && ggml_is_contiguous(spatial_shapes) &&
        ggml_is_contiguous(level_start_index) && ggml_is_contiguous(sampling_locations) && ggml_is_contiguous(attention_weights));

    const int64_t channels = values->ne[0];
    const int64_t heads = values->ne[1];
    const int64_t spatial_size = values->ne[2];
    const int64_t batch = values->ne[3];
    const int64_t queries = dst->ne[2];
    const int64_t levels = spatial_shapes->ne[1];
    const int64_t points = sampling_locations->ne[1];
    const int64_t count = ggml_nelements(dst);
    if (count == 0) {
        return;
    }

    const int64_t blocks = count / GGML_CUDA_MSDA_BLOCK_SIZE + (count % GGML_CUDA_MSDA_BLOCK_SIZE != 0);
    const char * trace = std::getenv("GGML_CUDA_MS_DEFORM_ATTN_TRACE");
    if (trace != nullptr && std::strcmp(trace, "1") == 0) {
        std::fprintf(stderr,
            "ggml_ms_deform_attn version=%d B=%lld Q=%lld H=%lld C=%lld L=%lld P=%lld S=%lld\n",
            GGML_MS_DEFORM_ATTN_VERSION,
            static_cast<long long>(batch), static_cast<long long>(queries), static_cast<long long>(heads),
            static_cast<long long>(channels), static_cast<long long>(levels), static_cast<long long>(points),
            static_cast<long long>(spatial_size));
    }
    const ggml_cuda_kernel_launch_params launch_params(
        MIN(0x7fffffff, blocks), GGML_CUDA_MSDA_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(ggml_cuda_ms_deform_attn_kernel, launch_params,
        static_cast<const float *>(values->data),
        static_cast<const int32_t *>(spatial_shapes->data),
        static_cast<const int32_t *>(level_start_index->data),
        static_cast<const float *>(sampling_locations->data),
        static_cast<const float *>(attention_weights->data),
        static_cast<float *>(dst->data),
        channels, heads, spatial_size, batch, queries, levels, points, count);
}
