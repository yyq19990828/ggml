// Private isolated GGML GroupNorm-affine candidate implementation.
//
// The arithmetic order was selected from a source investigation of the pinned
// PyTorch revision. Candidate 002 and its source/license provenance are
// recorded under build/ggml-group-norm-affine-prototype-001/candidate-002/.

#include "common.cuh"
#include "norm.cuh"

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxThreadsPerGroup = 512;
constexpr int kThreadsPerAffineBlock = 256;

struct WelfordState {
    float mean;
    float m2;
    int64_t count;
    float count_f;
};

__device__ __forceinline__ WelfordState welford_update(WelfordState state, float value) {
    const int64_t new_count = state.count + 1;
    const float new_count_f = static_cast<float>(new_count);
    const float delta = value - state.mean;
    const float new_mean = state.mean + delta / new_count_f;
    const float new_delta = value - new_mean;
    return {new_mean, state.m2 + delta * new_delta, new_count, new_count_f};
}

__device__ __forceinline__ WelfordState welford_combine(WelfordState left, WelfordState right) {
    if (left.count_f == 0.0f) return right;
    if (right.count_f == 0.0f) return left;
    const float delta = right.mean - left.mean;
    const float count = left.count_f + right.count_f;
    const float right_fraction = right.count_f / count;
    const float mean = left.mean + delta * right_fraction;
    const float m2 = left.m2 + right.m2 + delta * delta * left.count_f * right_fraction;
    return {mean, m2, -1, count};
}

__device__ __forceinline__ WelfordState warp_reduce(WelfordState state) {
    const int lane = threadIdx.x & (kWarpSize - 1);
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const WelfordState other = {
            __shfl_down_sync(0xffffffffU, state.mean, offset),
            __shfl_down_sync(0xffffffffU, state.m2, offset),
            __shfl_down_sync(0xffffffffU, state.count, offset),
            __shfl_down_sync(0xffffffffU, state.count_f, offset),
        };
        if (lane + offset < kWarpSize) {
            state = welford_combine(state, other);
        }
    }
    return state;
}

__global__ void group_moments_kernel(int64_t group_elements,
                                     float epsilon,
                                     const float * input,
                                     float * mean,
                                     float * rstd) {
    const int64_t group_index = static_cast<int64_t>(blockIdx.x);
    const int64_t input_offset = group_index * group_elements;

    WelfordState state{0.0f, 0.0f, 0, 0.0f};
    for (int64_t j = static_cast<int64_t>(threadIdx.x); j < group_elements; j += blockDim.x) {
        state = welford_update(state, input[input_offset + j]);
    }

    __shared__ WelfordState warp_states[kMaxThreadsPerGroup / kWarpSize];
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int num_warps = blockDim.x / kWarpSize;

    if (blockDim.x == kWarpSize) {
        state = warp_reduce(state);
        if (threadIdx.x == 0) {
            const float variance = state.m2 / state.count_f;
            mean[group_index] = state.mean;
            rstd[group_index] = rsqrtf(variance + epsilon);
        }
        return;
    }

    state = warp_reduce(state);
    __syncthreads();
    if (lane == 0) {
        warp_states[warp] = state;
    }
    __syncthreads();

    if (warp == 0) {
        WelfordState combined = lane < num_warps
            ? warp_states[lane]
            : WelfordState{0.0f, 0.0f, 0, 0.0f};
        combined = warp_reduce(combined);
        if (lane == 0) {
            const float variance = combined.m2 / combined.count_f;
            mean[group_index] = combined.mean;
            rstd[group_index] = rsqrtf(variance + epsilon);
        }
    }
}

__global__ void group_affine_parameters_kernel(int64_t channel_instances,
                                               int64_t channels,
                                               int64_t spatial,
                                               int groups,
                                               const float * mean,
                                               const float * rstd,
                                               const float * gamma,
                                               const float * beta,
                                               float * scale,
                                               float * bias) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= channel_instances) return;

    const int64_t channels_per_group = channels / groups;
    const int64_t group_index = (index / channels) * groups + (index % channels) / channels_per_group;
    const int64_t channel = index % channels;
    const float a = rstd[group_index] * gamma[channel];
    const float product = __fmul_rn(-a, mean[group_index]);
    scale[index] = a;
    bias[index] = __fadd_rn(product, beta[channel]);
}

__global__ void group_apply_affine_kernel(int64_t elements,
                                          int64_t spatial,
                                          const float * input,
                                          const float * scale,
                                          const float * bias,
                                          float * output) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const int64_t channel_instance = index / spatial;
    output[index] = __fmaf_rn(scale[channel_instance], input[index], bias[channel_instance]);
}

} // namespace

void ggml_cuda_op_group_norm_affine(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor * group_norm,
                                    const ggml_tensor * mul,
                                    const ggml_tensor * add) {
    GGML_ASSERT(group_norm != nullptr && mul != nullptr && add != nullptr);
    const ggml_tensor * input = group_norm->src[0];
    const ggml_tensor * gamma = mul->src[1];
    const ggml_tensor * beta = add->src[1];
    GGML_ASSERT(input != nullptr && gamma != nullptr && beta != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && group_norm->type == GGML_TYPE_F32 &&
                mul->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 &&
                gamma->type == GGML_TYPE_F32 && beta->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(group_norm) &&
                ggml_is_contiguous(mul) && ggml_is_contiguous(add) &&
                ggml_is_contiguous(gamma) && ggml_is_contiguous(beta));

    int32_t groups = 0;
    float epsilon = 0.0f;
    memcpy(&groups, group_norm->op_params, sizeof(groups));
    memcpy(&epsilon, reinterpret_cast<const char *>(group_norm->op_params) + sizeof(groups), sizeof(epsilon));

    const int64_t width = input->ne[0];
    const int64_t height = input->ne[1];
    const int64_t channels = input->ne[2];
    const int64_t batches = input->ne[3];
    GGML_ASSERT(width > 0 && height > 0 && channels > 0 && batches > 0);
    GGML_ASSERT(groups > 0 && groups <= channels && channels % groups == 0);
    GGML_ASSERT(epsilon >= 0.0f && std::isfinite(epsilon));
    GGML_ASSERT(gamma->ne[0] == 1 && gamma->ne[1] == 1 && gamma->ne[2] == channels && gamma->ne[3] == 1);
    GGML_ASSERT(beta->ne[0] == 1 && beta->ne[1] == 1 && beta->ne[2] == channels && beta->ne[3] == 1);
    GGML_ASSERT(ggml_are_same_shape(input, group_norm) && ggml_are_same_shape(group_norm, mul) &&
                ggml_are_same_shape(mul, add));

    const int64_t spatial = width * height;
    const int64_t channel_instances = batches * channels;
    const int64_t elements = ggml_nelements(input);
    const int64_t group_elements = spatial * (channels / groups);
    const int64_t group_count = batches * groups;
    GGML_ASSERT(spatial > 0 && channel_instances > 0 && elements > 0 && group_elements > 0 && group_count > 0);

    const int64_t group_blocks = group_count;
    const int threads_per_group = group_elements < 512 ? kWarpSize : kMaxThreadsPerGroup;
    const int64_t affine_blocks = (channel_instances + kThreadsPerAffineBlock - 1) / kThreadsPerAffineBlock;
    const int64_t output_blocks = (elements + kThreadsPerAffineBlock - 1) / kThreadsPerAffineBlock;
    GGML_ASSERT(group_blocks <= std::numeric_limits<int>::max());
    GGML_ASSERT(affine_blocks <= std::numeric_limits<int>::max());
    GGML_ASSERT(output_blocks <= std::numeric_limits<int>::max());

    ggml_cuda_pool_alloc<float> mean_alloc(ctx.pool(), static_cast<size_t>(group_count));
    ggml_cuda_pool_alloc<float> rstd_alloc(ctx.pool(), static_cast<size_t>(group_count));
    ggml_cuda_pool_alloc<float> scale_alloc(ctx.pool(), static_cast<size_t>(channel_instances));
    ggml_cuda_pool_alloc<float> bias_alloc(ctx.pool(), static_cast<size_t>(channel_instances));

    const float * input_d = static_cast<const float *>(input->data);
    const float * gamma_d = static_cast<const float *>(gamma->data);
    const float * beta_d = static_cast<const float *>(beta->data);
    float * output_d = static_cast<float *>(add->data);
    cudaStream_t stream = ctx.stream();

    group_moments_kernel<<<static_cast<unsigned int>(group_blocks), threads_per_group, 0, stream>>>(
        group_elements, epsilon, input_d, mean_alloc.get(), rstd_alloc.get());
    CUDA_CHECK(cudaGetLastError());
    group_affine_parameters_kernel<<<static_cast<unsigned int>(affine_blocks), kThreadsPerAffineBlock, 0, stream>>>(
        channel_instances, channels, spatial, groups, mean_alloc.get(), rstd_alloc.get(),
        gamma_d, beta_d, scale_alloc.get(), bias_alloc.get());
    CUDA_CHECK(cudaGetLastError());
    group_apply_affine_kernel<<<static_cast<unsigned int>(output_blocks), kThreadsPerAffineBlock, 0, stream>>>(
        elements, spatial, input_d, scale_alloc.get(), bias_alloc.get(), output_d);
    CUDA_CHECK(cudaGetLastError());
}
