#include "cudnn.cuh"

#ifdef GGML_CUDA_CUDNN
#include <cudnn.h>
#include <array>
#include <climits>

static_assert(CUDNN_VERSION == 91002, "cuDNN Winograd requires the verified 9.10.2 headers");

#define GGML_CUDNN_CHECK(call) do { \
    const cudnnStatus_t status_ = (call); \
    if (status_ != CUDNN_STATUS_SUCCESS) { \
        GGML_ABORT("cuDNN: %s failed: %s (%s:%d)", #call, cudnnGetErrorString(status_), __FILE__, __LINE__); \
    } \
} while (0)

// Fixed F32 NCHW, 3x3, stride/dilation one and padding one. Device and
// stream are owned by the enclosing context, so only N,C,H,W,K vary.
struct ggml_cuda_cudnn_plan {
    std::array<int, 5> shape = {};
    cudnnTensorDescriptor_t input = nullptr;
    cudnnTensorDescriptor_t output = nullptr;
    cudnnFilterDescriptor_t filter = nullptr;
    size_t workspace_bytes = 0;

    ggml_cuda_cudnn_plan() = default;
    ggml_cuda_cudnn_plan(const ggml_cuda_cudnn_plan &) = delete;
    ggml_cuda_cudnn_plan & operator=(const ggml_cuda_cudnn_plan &) = delete;

    ~ggml_cuda_cudnn_plan() {
        if (input)  GGML_CUDNN_CHECK(cudnnDestroyTensorDescriptor(input));
        if (output) GGML_CUDNN_CHECK(cudnnDestroyTensorDescriptor(output));
        if (filter) GGML_CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter));
    }

    void configure(const std::array<int, 5> & dimensions, cudnnHandle_t handle,
                   cudnnConvolutionDescriptor_t convolution) {
        if (!input) {
            GGML_CUDNN_CHECK(cudnnCreateTensorDescriptor(&input));
            GGML_CUDNN_CHECK(cudnnCreateTensorDescriptor(&output));
            GGML_CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter));
        }
        const int n = dimensions[0], c = dimensions[1], h = dimensions[2];
        const int w = dimensions[3], k = dimensions[4];
        GGML_CUDNN_CHECK(cudnnSetTensor4dDescriptor(input, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
        GGML_CUDNN_CHECK(cudnnSetTensor4dDescriptor(output, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, k, h, w));
        GGML_CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, 3, 3));
        const cudnnStatus_t status = cudnnGetConvolutionForwardWorkspaceSize(
            handle, input, filter, convolution, output, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, &workspace_bytes);
        if (status != CUDNN_STATUS_SUCCESS) {
            GGML_ABORT("cuDNN F32 Winograd N=%d C=%d H=%d W=%d K=%d is unavailable: %s",
                       n, c, h, w, k, cudnnGetErrorString(status));
        }
        shape = dimensions;
    }
};

struct ggml_cuda_cudnn_context {
    cudnnHandle_t handle = nullptr;
    cudnnConvolutionDescriptor_t convolution = nullptr;
    // Reuse descriptors on eviction; neither the cache nor its host allocations
    // grows with the number of input shapes in a long-lived backend.
    std::array<ggml_cuda_cudnn_plan, 32> plans;
    size_t count = 0;
    size_t next = 0;

    explicit ggml_cuda_cudnn_context(cudaStream_t stream) {
        const size_t version = cudnnGetVersion();
        if (version != 91002) {
            GGML_ABORT("cuDNN runtime version %zu differs from verified 91002", version);
        }
        GGML_CUDNN_CHECK(cudnnCreate(&handle));
        GGML_CUDNN_CHECK(cudnnSetStream(handle, stream));
        GGML_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convolution));
        GGML_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convolution, 1, 1, 1, 1, 1, 1,
                                                       CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        GGML_CUDNN_CHECK(cudnnSetConvolutionGroupCount(convolution, 1));
        GGML_CUDNN_CHECK(cudnnSetConvolutionMathType(convolution, CUDNN_FMA_MATH));
    }

    ~ggml_cuda_cudnn_context() {
        GGML_CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convolution));
        GGML_CUDNN_CHECK(cudnnDestroy(handle));
    }

    ggml_cuda_cudnn_plan & plan(const std::array<int, 5> & shape) {
        for (size_t i = 0; i < count; ++i) {
            if (plans[i].shape == shape) return plans[i];
        }
        auto & entry = plans[next];
        entry.configure(shape, handle, convolution);
        next = (next + 1) % plans.size();
        if (count < plans.size()) ++count;
        return entry;
    }
};

bool ggml_cuda_cudnn_supports(const ggml_tensor * dst) {
    if (dst->op != GGML_OP_CONV_2D || !dst->src[0] || !dst->src[1]) return false;
    const auto * kernel = dst->src[0];
    const auto * input = dst->src[1];
    if (kernel->type != GGML_TYPE_F32 || input->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(kernel) || !ggml_is_contiguous(input) || !ggml_is_contiguous(dst)) return false;
    const int32_t * p = dst->op_params;
    if (kernel->ne[0] != 3 || kernel->ne[1] != 3 || p[0] != 1 || p[1] != 1 ||
        p[2] != 1 || p[3] != 1 || p[4] != 1 || p[5] != 1 || p[6] != 0) return false;
    if (kernel->ne[2] != input->ne[2] || dst->ne[0] != input->ne[0] ||
        dst->ne[1] != input->ne[1] || dst->ne[2] != kernel->ne[3] || dst->ne[3] != input->ne[3]) return false;
    // The legacy descriptor API represents dimensions and dense strides as int.
    for (const auto * tensor : {kernel, input, dst}) {
        for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
            if (tensor->ne[dimension] <= 0 || tensor->ne[dimension] > INT_MAX) return false;
        }
        if (ggml_nelements(tensor) > INT_MAX) return false;
    }
    return true;
}

void ggml_cuda_cudnn_conv2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_cudnn_supports(dst));
    auto * & resource = ctx.cudnn[ctx.device][ctx.curr_stream_no];
    if (!resource) {
        ggml_cuda_set_device(ctx.device);
        resource = new ggml_cuda_cudnn_context(ctx.stream());
    }
    const auto * input = dst->src[1];
    const auto * kernel = dst->src[0];
    auto & plan = resource->plan({int(input->ne[3]), int(input->ne[2]), int(input->ne[1]),
                                 int(input->ne[0]), int(kernel->ne[3])});
    // Pool reuse is stream-local, so the asynchronous convolution completes
    // before a later operation on this stream can reuse its workspace.
    ggml_cuda_pool_alloc<char> workspace;
    if (plan.workspace_bytes) workspace.alloc(ctx.pool(), plan.workspace_bytes);
    const float alpha = 1.0f, beta = 0.0f;
    GGML_CUDNN_CHECK(cudnnConvolutionForward(resource->handle, &alpha, plan.input, input->data,
        plan.filter, kernel->data, resource->convolution, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        workspace.get(), plan.workspace_bytes, &beta, plan.output, dst->data));
}

void ggml_cuda_cudnn_release(ggml_backend_cuda_context & ctx) {
    // The backend destructor has synchronized streams and released captured
    // graphs. Handles must be released before their owning CUDA streams.
    for (int device = 0; device < GGML_CUDA_MAX_DEVICES; ++device) {
        for (int stream = 0; stream < GGML_CUDA_MAX_STREAMS; ++stream) {
            auto * & resource = ctx.cudnn[device][stream];
            if (resource) {
                ggml_cuda_set_device(device);
                delete resource;
                resource = nullptr;
            }
        }
    }
    ggml_cuda_set_device(ctx.device);
}

int ggml_cuda_cudnn_version() {
    return static_cast<int>(cudnnGetVersion());
}
#endif // GGML_CUDA_CUDNN
