// Isolated LayerNorm -> MUL -> ADD diagnostic kernel.
// The numerical order follows the independently implemented and separately
// accepted F32 primitive in cuda-layernorm-order-prototype-001. Attribution
// and the upstream license are retained in the candidate directory.
#include "common.cuh"
#include "norm.cuh"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kVectorWidth = 4;

struct WelfordState {
    float mean;
    float m2;
    float count;
};

__device__ __forceinline__ WelfordState online_update(float value, WelfordState state) {
    const float delta = value - state.mean;
    const float new_count = state.count + 1.0f;
    const float new_mean = state.mean + delta * (1.0f / new_count);
    const float new_m2 = state.m2 + delta * (value - new_mean);
    return {new_mean, new_m2, new_count};
}

// Argument order mirrors the pinned ATen call cuWelfordCombine(dataB, dataA).
__device__ __forceinline__ WelfordState combine_states(WelfordState data_b, WelfordState data_a) {
    const float delta = data_b.mean - data_a.mean;
    const float count = data_a.count + data_b.count;
    if (count > 0.0f) {
        const float coefficient = 1.0f / count;
        const float n_a = data_a.count * coefficient;
        const float n_b = data_b.count * coefficient;
        const float mean = n_a * data_a.mean + n_b * data_b.mean;
        const float m2 = data_a.m2 + data_b.m2 + delta * delta * data_a.count * n_b;
        return {mean, m2, count};
    }
    return {0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ WelfordState warp_reduce(WelfordState state) {
    constexpr unsigned int kFullMask = 0xffffffffu;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const WelfordState other{
            __shfl_down_sync(kFullMask, state.mean, offset, kWarpSize),
            __shfl_down_sync(kFullMask, state.m2, offset, kWarpSize),
            __shfl_down_sync(kFullMask, state.count, offset, kWarpSize),
        };
        state = combine_states(state, other);
    }
    return state;
}

__device__ __forceinline__ WelfordState row_statistics(const float * row, int width, float * scratch) {
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int thread_in_block = lane + warp * kWarpSize;
    const int vector_count = width / kVectorWidth;
    const float4 * row_vectors = reinterpret_cast<const float4 *>(row);

    WelfordState state{0.0f, 0.0f, 0.0f};
    for (int vector_index = thread_in_block; vector_index < vector_count;
         vector_index += kThreadsPerBlock) {
        const float4 values = row_vectors[vector_index];
        state = online_update(values.x, state);
        state = online_update(values.y, state);
        state = online_update(values.z, state);
        state = online_update(values.w, state);
    }

    state = warp_reduce(state);
    float * mean_m2_scratch = scratch;
    float * count_scratch = scratch + kWarpsPerBlock;
    for (int offset = kWarpsPerBlock / 2; offset > 0; offset >>= 1) {
        if (lane == 0 && warp >= offset && warp < 2 * offset) {
            const int write_warp = warp - offset;
            mean_m2_scratch[2 * write_warp] = state.mean;
            mean_m2_scratch[2 * write_warp + 1] = state.m2;
            count_scratch[write_warp] = state.count;
        }
        __syncthreads();
        if (lane == 0 && warp < offset) {
            const WelfordState other{
                mean_m2_scratch[2 * warp],
                mean_m2_scratch[2 * warp + 1],
                count_scratch[warp],
            };
            state = combine_states(state, other);
        }
        __syncthreads();
    }
    if (lane == 0 && warp == 0) {
        mean_m2_scratch[0] = state.mean;
        mean_m2_scratch[1] = state.m2 / static_cast<float>(width);
    }
    __syncthreads();
    return {mean_m2_scratch[0], mean_m2_scratch[1], 0.0f};
}

__global__ void norm_affine_f32_kernel(
    const float * input,
    const float * gamma,
    const float * beta,
    float * output,
    int width,
    float epsilon) {
    extern __shared__ float scratch[];
    const int row_index = static_cast<int>(blockIdx.x);
    const float * row = input + static_cast<std::size_t>(row_index) * width;
    const WelfordState stats = row_statistics(row, width, scratch);
    const float inverse_stddev = rsqrtf(stats.m2 + epsilon);

    const int thread_in_block = static_cast<int>(threadIdx.x) +
                                static_cast<int>(threadIdx.y) * kWarpSize;
    const int vector_count = width / kVectorWidth;
    const float4 * input_vectors = reinterpret_cast<const float4 *>(row);
    const float4 * gamma_vectors = reinterpret_cast<const float4 *>(gamma);
    const float4 * beta_vectors = reinterpret_cast<const float4 *>(beta);
    float4 * output_vectors = reinterpret_cast<float4 *>(
        output + static_cast<std::size_t>(row_index) * width);

    for (int vector_index = thread_in_block; vector_index < vector_count;
         vector_index += kThreadsPerBlock) {
        const float4 x = input_vectors[vector_index];
        const float4 weight = gamma_vectors[vector_index];
        const float4 bias = beta_vectors[vector_index];
        float4 y;
        y.x = weight.x * (inverse_stddev * (x.x - stats.mean)) + bias.x;
        y.y = weight.y * (inverse_stddev * (x.y - stats.mean)) + bias.y;
        y.z = weight.z * (inverse_stddev * (x.z - stats.mean)) + bias.z;
        y.w = weight.w * (inverse_stddev * (x.w - stats.mean)) + bias.w;
        output_vectors[vector_index] = y;
    }
}

}  // namespace

void ggml_cuda_op_norm_affine(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * norm,
                              const ggml_tensor * mul,
                              const ggml_tensor * add) {
    const ggml_tensor * input = norm->src[0];
    const ggml_tensor * gamma = mul->src[1];
    const ggml_tensor * beta = add->src[1];
    float epsilon = 0.0f;
    memcpy(&epsilon, norm->op_params, sizeof(epsilon));

    const int width = static_cast<int>(input->ne[0]);
    const int rows = static_cast<int>(ggml_nelements(input) / input->ne[0]);
    const size_t shared_bytes = kWarpsPerBlock * 3 / 2 * sizeof(float);
    const dim3 threads(kWarpSize, kWarpsPerBlock, 1);
    norm_affine_f32_kernel<<<static_cast<unsigned int>(rows), threads, shared_bytes, ctx.stream()>>>(
        static_cast<const float *>(input->data),
        static_cast<const float *>(gamma->data),
        static_cast<const float *>(beta->data),
        static_cast<float *>(add->data),
        width,
        epsilon);
    CUDA_CHECK(cudaGetLastError());

    if (std::getenv("GGML_CUDA_NORM_AFFINE_TRACE") != nullptr) {
        static std::atomic<uint64_t> invocation_count{0};
        const uint64_t invocation = ++invocation_count;
        std::fprintf(stderr,
                     "GGML_CUDA_NORM_AFFINE_FUSED invocation=%" PRIu64 " rows=%d width=%d eps=%.9g\n",
                     invocation, rows, width, static_cast<double>(epsilon));
        std::fflush(stderr);
    }
}
