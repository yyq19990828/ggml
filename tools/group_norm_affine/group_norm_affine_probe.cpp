#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "npy_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string backend = "cuda";
    std::string mode = "fused";
    std::string case_id;
    std::string expose;
    std::string affine_layout = "channel";
    std::string affine_storage = "direct";
    bool intervening_compute = false;
    std::string expected_route;
    std::filesystem::path input_path;
    std::filesystem::path gamma_path;
    std::filesystem::path beta_path;
    std::filesystem::path expected_path;
    std::filesystem::path output_dir;
    int groups = 0;
    int device = 0;
    float epsilon = 1.0e-5f;
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

Options parse(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) fail(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--backend") options.backend = value("--backend");
        else if (arg == "--mode") options.mode = value("--mode");
        else if (arg == "--case-id") options.case_id = value("--case-id");
        else if (arg == "--expose") options.expose = value("--expose");
        else if (arg == "--affine-layout") options.affine_layout = value("--affine-layout");
        else if (arg == "--affine-storage") options.affine_storage = value("--affine-storage");
        else if (arg == "--intervening-compute") options.intervening_compute = true;
        else if (arg == "--expect-route") options.expected_route = value("--expect-route");
        else if (arg == "--input") options.input_path = value("--input");
        else if (arg == "--gamma") options.gamma_path = value("--gamma");
        else if (arg == "--beta") options.beta_path = value("--beta");
        else if (arg == "--expected") options.expected_path = value("--expected");
        else if (arg == "--out") options.output_dir = value("--out");
        else if (arg == "--groups") options.groups = std::stoi(value("--groups"));
        else if (arg == "--device") options.device = std::stoi(value("--device"));
        else if (arg == "--epsilon") options.epsilon = std::stof(value("--epsilon"));
        else if (arg == "--help") {
            std::cout << "--backend cpu|cuda --mode fused|unfused --case-id ID --input NPY "
                         "--gamma NPY --beta NPY [--expected NPY] --groups N --epsilon F32 "
                         "--out DIR [--expose norm|mul] [--affine-layout channel|channel-fastest] "
                         "[--affine-storage direct|weight-reshape|nonweight-reshape|computed-operand] "
                         "[--intervening-compute] "
                         "[--expect-route fused|fallback] [--device N]\n";
            std::exit(0);
        } else {
            fail("unknown argument: " + arg);
        }
    }
    return options;
}

void require_fresh_directory(const std::filesystem::path & path) {
    if (std::filesystem::exists(path)) {
        if (!std::filesystem::is_directory(path) || !std::filesystem::is_empty(path)) {
            fail("output directory must be fresh and empty: " + path.string());
        }
    } else {
        std::filesystem::create_directories(path);
    }
}

void require_finite(const std::vector<float> & values, const std::string & name) {
    for (float value : values) {
        if (!std::isfinite(value)) fail(name + " contains a non-finite F32 value");
    }
}

struct Compare {
    std::size_t count = 0;
    std::size_t raw_unequal = 0;
    double max_abs = 0.0;
    double squared_error = 0.0;
};

Compare compare(const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) fail("comparison vector sizes differ");
    Compare result;
    result.count = actual.size();
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0) ++result.raw_unequal;
        const double error = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        result.max_abs = std::max(result.max_abs, error);
        result.squared_error += error * error;
    }
    return result;
}

std::string comparison_json(const Compare & result) {
    const double rmse = result.count == 0 ? 0.0 : std::sqrt(result.squared_error / result.count);
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\"count\":" << result.count
           << ",\"raw_unequal\":" << result.raw_unequal
           << ",\"max_abs\":" << result.max_abs
           << ",\"rmse\":" << rmse << "}";
    return output.str();
}

std::string json_escape(const std::string & value) {
    std::string result;
    for (unsigned char c : value) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (c < 0x20) fail("control byte in JSON value");
                result += static_cast<char>(c);
        }
    }
    return result;
}

void tensor_get_exact(const ggml_tensor * tensor, std::vector<float> & values) {
    if (ggml_nbytes(tensor) != values.size() * sizeof(float)) {
        fail("readback tensor byte count mismatch");
    }
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
}

int run(const Options & options) {
    if (options.backend != "cpu" && options.backend != "cuda") fail("backend must be cpu or cuda");
    if (options.mode != "fused" && options.mode != "unfused") fail("mode must be fused or unfused");
    if (options.case_id.empty() || options.input_path.empty() || options.gamma_path.empty() ||
        options.beta_path.empty() || options.output_dir.empty() || options.groups <= 0 ||
        !std::isfinite(options.epsilon) || options.epsilon < 0.0f) {
        fail("case-id, input, gamma, beta, output, valid groups and epsilon are required");
    }
    if (!options.expose.empty() && options.expose != "norm" && options.expose != "mul") {
        fail("expose must be norm or mul");
    }
    if (options.affine_layout != "channel" && options.affine_layout != "channel-fastest") {
        fail("affine-layout must be channel or channel-fastest");
    }
    if (options.affine_storage != "direct" && options.affine_storage != "weight-reshape" &&
        options.affine_storage != "nonweight-reshape" && options.affine_storage != "computed-operand") {
        fail("affine-storage must be direct, weight-reshape, nonweight-reshape, or computed-operand");
    }
    if (options.affine_storage != "direct" && options.affine_layout != "channel") {
        fail("reshape affine storage requires channel layout");
    }
    if (!options.expected_route.empty() && options.expected_route != "fused" && options.expected_route != "fallback") {
        fail("expected route must be fused or fallback");
    }

    gngn::NpyF32 input = gngn::read_f32_npy(options.input_path.string());
    gngn::NpyF32 gamma = gngn::read_f32_npy(options.gamma_path.string());
    gngn::NpyF32 beta = gngn::read_f32_npy(options.beta_path.string());
    if (input.shape.size() != 4) fail("input NPY must have NCHW rank four");
    const std::size_t batches = input.shape[0];
    const std::size_t channels = input.shape[1];
    const std::size_t height = input.shape[2];
    const std::size_t width = input.shape[3];
    if (gamma.values.size() != channels || beta.values.size() != channels) {
        fail("gamma and beta must each contain one value per channel");
    }
    if (options.groups > static_cast<int>(channels) || channels % static_cast<std::size_t>(options.groups) != 0) {
        fail("groups must be positive, no greater than channels, and divide channels exactly");
    }
    if (options.affine_layout == "channel-fastest" && width != channels) {
        fail("channel-fastest affine control requires W == C");
    }
    require_finite(input.values, "input");
    require_finite(gamma.values, "gamma");
    require_finite(beta.values, "beta");

    gngn::NpyF32 expected;
    if (!options.expected_path.empty()) {
        expected = gngn::read_f32_npy(options.expected_path.string());
        if (expected.shape != input.shape) fail("expected NPY shape differs from input");
        require_finite(expected.values, "expected comparison reference");
    }
    require_fresh_directory(options.output_dir);

    if (options.backend == "cuda") {
        setenv("GGML_CUDA_GROUP_NORM_AFFINE_TRACE", "1", 1);
        setenv("GGML_CUDA_GROUP_NORM_AFFINE_DISABLE", options.mode == "unfused" ? "1" : "0", 1);
    }

    ggml_init_params params{32u * 1024u * 1024u, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) fail("ggml_init failed");

    ggml_context * weight_ctx = nullptr;
    ggml_tensor * gamma_base = nullptr;
    ggml_tensor * beta_base = nullptr;
    ggml_tensor * gamma_source = nullptr;
    ggml_tensor * beta_source = nullptr;

    ggml_tensor * input_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
        static_cast<int64_t>(width), static_cast<int64_t>(height),
        static_cast<int64_t>(channels), static_cast<int64_t>(batches));
    ggml_tensor * gamma_tensor = nullptr;
    ggml_tensor * beta_tensor = nullptr;
    const bool use_weight_views = options.affine_storage != "direct";
    if (!use_weight_views) {
        gamma_tensor = options.affine_layout == "channel"
            ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, static_cast<int64_t>(channels), 1)
            : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, static_cast<int64_t>(channels), 1, 1, 1);
        beta_tensor = options.affine_layout == "channel"
            ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, static_cast<int64_t>(channels), 1)
            : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, static_cast<int64_t>(channels), 1, 1, 1);
    } else {
        ggml_init_params weight_params{2u * 1024u * 1024u, nullptr, true};
        weight_ctx = ggml_init(weight_params);
        if (weight_ctx == nullptr) {
            ggml_free(ctx);
            fail("weight ggml_init failed");
        }
        gamma_base = ggml_new_tensor_1d(weight_ctx, GGML_TYPE_F32, static_cast<int64_t>(channels));
        beta_base = ggml_new_tensor_1d(weight_ctx, GGML_TYPE_F32, static_cast<int64_t>(channels));
        gamma_source = gamma_base;
        beta_source = beta_base;
        if (options.affine_storage == "computed-operand") {
            gamma_source = ggml_scale(weight_ctx, gamma_base, 1.0f);
        }
        gamma_tensor = ggml_reshape_4d(ctx, gamma_source, 1, 1, static_cast<int64_t>(channels), 1);
        beta_tensor = ggml_reshape_4d(ctx, beta_source, 1, 1, static_cast<int64_t>(channels), 1);
    }
    ggml_tensor * group_norm = ggml_group_norm(ctx, input_tensor, options.groups, options.epsilon);
    ggml_tensor * mul_source = options.intervening_compute ? ggml_scale(ctx, group_norm, 1.0f) : group_norm;
    ggml_tensor * mul = ggml_mul(ctx, mul_source, gamma_tensor);
    ggml_tensor * add = ggml_add(ctx, mul, beta_tensor);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    if (options.expose == "norm") ggml_set_output(group_norm);
    if (options.expose == "mul") ggml_set_output(mul);
    ggml_build_forward_expand(graph, add);

    ggml_backend_t backend = options.backend == "cpu"
        ? ggml_backend_cpu_init()
        : ggml_backend_cuda_init(options.device);
    if (backend == nullptr) {
        ggml_free(ctx);
        if (weight_ctx != nullptr) ggml_free(weight_ctx);
        fail("backend initialization failed");
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (!ggml_backend_supports_op(backend, node)) {
            ggml_backend_free(backend);
            ggml_free(ctx);
            if (weight_ctx != nullptr) ggml_free(weight_ctx);
            fail(std::string("backend rejects graph op: ") + ggml_op_name(node->op));
        }
    }
    ggml_backend_buffer_t weight_buffer = nullptr;
    if (weight_ctx != nullptr) {
        weight_buffer = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);
        if (weight_buffer == nullptr) {
            ggml_backend_free(backend);
            ggml_free(ctx);
            ggml_free(weight_ctx);
            fail("weight context backend allocation failed");
        }
        if (options.affine_storage == "weight-reshape") {
            ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        if (weight_buffer != nullptr) ggml_backend_buffer_free(weight_buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        if (weight_ctx != nullptr) ggml_free(weight_ctx);
        fail("backend tensor allocation failed");
    }

    const std::size_t input_bytes = input.values.size() * sizeof(float);
    const std::size_t affine_bytes = gamma.values.size() * sizeof(float);
    ggml_backend_tensor_set(input_tensor, input.values.data(), 0, input_bytes);
    if (weight_ctx != nullptr) {
        ggml_backend_tensor_set(gamma_base, gamma.values.data(), 0, affine_bytes);
        ggml_backend_tensor_set(beta_base, beta.values.data(), 0, affine_bytes);
    } else {
        ggml_backend_tensor_set(gamma_tensor, gamma.values.data(), 0, affine_bytes);
        ggml_backend_tensor_set(beta_tensor, beta.values.data(), 0, affine_bytes);
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        if (weight_buffer != nullptr) ggml_backend_buffer_free(weight_buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        if (weight_ctx != nullptr) ggml_free(weight_ctx);
        fail("backend graph compute failed");
    }

    std::vector<float> input_readback(input.values.size());
    std::vector<float> gamma_readback(gamma.values.size());
    std::vector<float> beta_readback(beta.values.size());
    std::vector<float> output(static_cast<std::size_t>(ggml_nelements(add)));
    tensor_get_exact(input_tensor, input_readback);
    tensor_get_exact(weight_ctx != nullptr ? gamma_base : gamma_tensor, gamma_readback);
    tensor_get_exact(weight_ctx != nullptr ? beta_base : beta_tensor, beta_readback);
    tensor_get_exact(add, output);
    require_finite(output, "output");
    const bool input_rawexact = std::memcmp(input.values.data(), input_readback.data(), input_bytes) == 0;
    const bool gamma_rawexact = std::memcmp(gamma.values.data(), gamma_readback.data(), affine_bytes) == 0;
    const bool beta_rawexact = std::memcmp(beta.values.data(), beta_readback.data(), affine_bytes) == 0;
    if (!input_rawexact || !gamma_rawexact || !beta_rawexact) fail("input or affine device readback changed bytes");

    gngn::write_f32_npy((options.output_dir / "input.readback.npy").string(), input.shape, input_readback);
    gngn::write_f32_npy((options.output_dir / "gamma.readback.npy").string(), {channels}, gamma_readback);
    gngn::write_f32_npy((options.output_dir / "beta.readback.npy").string(), {channels}, beta_readback);
    gngn::write_f32_npy((options.output_dir / "output.npy").string(), input.shape, output);
    if (options.expose == "norm") {
        std::vector<float> values(static_cast<std::size_t>(ggml_nelements(group_norm)));
        tensor_get_exact(group_norm, values);
        gngn::write_f32_npy((options.output_dir / "group_norm.npy").string(), input.shape, values);
    }
    if (options.expose == "mul") {
        std::vector<float> values(static_cast<std::size_t>(ggml_nelements(mul)));
        tensor_get_exact(mul, values);
        gngn::write_f32_npy((options.output_dir / "affine_mul.npy").string(), input.shape, values);
    }

    std::ofstream metadata(options.output_dir / "run.json", std::ios::out | std::ios::trunc);
    if (!metadata) fail("cannot write run metadata");
    metadata << std::setprecision(9)
             << "{\n  \"schema_version\": 1,\n"
             << "  \"status\": \"PROBE_COMPLETE\",\n"
             << "  \"case_id\": \"" << json_escape(options.case_id) << "\",\n"
             << "  \"backend\": \"" << json_escape(options.backend) << "\",\n"
             << "  \"requested_mode\": \"" << json_escape(options.mode) << "\",\n"
             << "  \"expected_route\": \"" << json_escape(options.expected_route) << "\",\n"
             << "  \"affine_layout\": \"" << json_escape(options.affine_layout) << "\",\n"
             << "  \"affine_storage\": \"" << json_escape(options.affine_storage) << "\",\n"
             << "  \"intervening_compute\": " << (options.intervening_compute ? "true" : "false") << ",\n"
             << "  \"affine_view_source_buffer_usage\": \""
             << (options.affine_storage == "weight-reshape" ? "WEIGHTS" :
                 options.affine_storage == "direct" ? "not_applicable" : "ANY") << "\",\n"
             << "  \"shape_nchw\": [" << batches << ", " << channels << ", " << height << ", " << width << "],\n"
             << "  \"ggml_ne_whcn\": [" << width << ", " << height << ", " << channels << ", " << batches << "],\n"
             << "  \"groups\": " << options.groups << ",\n"
             << "  \"epsilon\": " << options.epsilon << ",\n"
             << "  \"device_readback_rawexact\": {\"input\": true, \"gamma\": true, \"beta\": true},\n"
             << "  \"expected_uploaded\": false,\n"
             << "  \"exposed_intermediate\": \"" << json_escape(options.expose) << "\"";
    if (!expected.values.empty()) {
        metadata << ",\n  \"official_comparison\": " << comparison_json(compare(output, expected.values));
    }
    metadata << "\n}\n";
    if (!metadata) fail("failed writing run metadata");

    ggml_backend_buffer_free(buffer);
    if (weight_buffer != nullptr) ggml_backend_buffer_free(weight_buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    if (weight_ctx != nullptr) ggml_free(weight_ctx);
    std::cout << "PASS " << options.case_id << " backend=" << options.backend
              << " mode=" << options.mode << " shape=" << gngn::shape_tuple(input.shape)
              << " output=" << options.output_dir << "\n";
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
