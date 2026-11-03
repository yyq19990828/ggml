#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string backend;
    std::string case_id;
    std::filesystem::path exponent_path;
    std::filesystem::path position_path;
    std::filesystem::path output_dir;
    float base = 10000.0f;
    int device = 0;
    bool pow_only = false;
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

std::vector<float> read_f32(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        fail("cannot open F32 input: " + path.string());
    }
    const std::streamoff bytes = stream.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        fail("F32 input has invalid byte length: " + path.string());
    }
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    stream.seekg(0, std::ios::beg);
    stream.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!stream) {
        fail("short read from F32 input: " + path.string());
    }
    return values;
}

void write_f32(const std::filesystem::path & path, const std::vector<float> & values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        fail("cannot create output: " + path.string());
    }
    stream.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!stream) {
        fail("failed writing output: " + path.string());
    }
}

void tensor_get_exact(ggml_tensor * tensor, std::vector<float> & values) {
    const std::size_t bytes = values.size() * sizeof(float);
    ggml_backend_tensor_get(tensor, values.data(), 0, bytes);
}

Options parse(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                fail("missing value after " + arg);
            }
            return argv[++i];
        };
        if (arg == "--backend") {
            options.backend = value();
        } else if (arg == "--case-id") {
            options.case_id = value();
        } else if (arg == "--exponents") {
            options.exponent_path = value();
        } else if (arg == "--positions") {
            options.position_path = value();
        } else if (arg == "--out") {
            options.output_dir = value();
        } else if (arg == "--base") {
            const std::string text = value();
            char * end = nullptr;
            options.base = std::strtof(text.c_str(), &end);
            if (end == text.c_str() || end == nullptr || *end != '\0') {
                fail("invalid F32 base: " + text);
            }
        } else if (arg == "--device") {
            options.device = std::stoi(value());
        } else if (arg == "--pow-only") {
            options.pow_only = true;
        } else if (arg == "--invalid-base-zero") {
            options.backend = "invalid-base-zero";
        } else if (arg == "--invalid-base-negative") {
            options.backend = "invalid-base-negative";
        } else if (arg == "--invalid-base-nan") {
            options.backend = "invalid-base-nan";
        } else if (arg == "--invalid-base-inf") {
            options.backend = "invalid-base-inf";
        } else if (arg == "--invalid-type") {
            options.backend = "invalid-type";
        } else if (arg == "--invalid-noncontiguous") {
            options.backend = "invalid-noncontiguous";
        } else if (arg == "--support-shape-negative") {
            options.backend = "support-shape-negative";
        } else if (arg == "--help") {
            std::cout << "--backend cpu|cuda --case-id ID --exponents FILE --positions FILE "
                         "--base F32 --out FRESH_DIR [--device N]\n"
                         "--invalid-base-zero|negative|nan|inf, --invalid-type, "
                         "--invalid-noncontiguous, --support-shape-negative\n";
            std::exit(0);
        } else {
            fail("unknown argument: " + arg);
        }
    }
    return options;
}

ggml_context * new_context(std::size_t mem_size = 1u << 20) {
    ggml_init_params params{};
    params.mem_size = mem_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fail("ggml_init failed");
    }
    return ctx;
}

int invalid_contract_case(const std::string & mode) {
    ggml_context * ctx = new_context();
    ggml_tensor * input = nullptr;
    float base = 2.0f;
    if (mode == "invalid-base-zero") {
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        base = 0.0f;
    } else if (mode == "invalid-base-negative") {
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        base = -2.0f;
    } else if (mode == "invalid-base-nan") {
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        base = std::numeric_limits<float>::quiet_NaN();
    } else if (mode == "invalid-base-inf") {
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        base = std::numeric_limits<float>::infinity();
    } else if (mode == "invalid-type") {
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 4);
    } else if (mode == "invalid-noncontiguous") {
        ggml_tensor * contiguous = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
        input = ggml_transpose(ctx, contiguous);
    } else {
        fail("unknown invalid contract case: " + mode);
    }

    // Each mode is launched as a child by the Python preflight. The constructor
    // is expected to terminate this process through GGML_ASSERT.
    (void) ggml_pow_base_f32(ctx, input, base);
    ggml_free(ctx);
    return 0;
}

int support_shape_negative() {
    ggml_context * ctx = new_context();
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * output = ggml_pow_base_f32(ctx, input, 2.0f);
    output->ne[0] = 3;
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fail("CPU backend initialization failed");
    }
    const bool accepted = ggml_backend_supports_op(backend, output);
    ggml_backend_free(backend);
    ggml_free(ctx);
    if (accepted) {
        fail("CPU support predicate accepted mismatched input/output shapes");
    }
    std::cout << "support-shape-negative: rejected\n";
    return 0;
}

int run_probe(const Options & options) {
    if (options.backend != "cpu" && options.backend != "cuda") {
        fail("backend must be cpu or cuda");
    }
    if (options.case_id.empty() || options.exponent_path.empty() ||
        options.position_path.empty() || options.output_dir.empty()) {
        fail("case-id, exponents, positions, and out are required");
    }
    if (!std::isfinite(options.base) || options.base <= 0.0f) {
        fail("base must be finite and positive");
    }
    if (std::filesystem::exists(options.output_dir) &&
        !std::filesystem::is_empty(options.output_dir)) {
        fail("output directory must be fresh and empty: " + options.output_dir.string());
    }
    std::filesystem::create_directories(options.output_dir);

    std::vector<float> exponents = read_f32(options.exponent_path);
    std::vector<float> positions = read_f32(options.position_path);
    if (exponents.empty() || exponents.size() != positions.size()) {
        fail("exponent and position vectors must be nonempty and have equal length");
    }
    for (float exponent : exponents) {
        if (!std::isfinite(exponent)) {
            fail("all exponent inputs must be finite");
        }
    }
    const int64_t count = static_cast<int64_t>(exponents.size());
    ggml_context * ctx = new_context(2u << 20);
    ggml_tensor * exponent_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
    ggml_tensor * position_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
    ggml_set_name(exponent_tensor, "input.exponent");
    ggml_set_name(position_tensor, "input.position");
    ggml_set_input(exponent_tensor);
    ggml_set_input(position_tensor);

    ggml_tensor * power = ggml_pow_base_f32(ctx, exponent_tensor, options.base);
    ggml_set_name(power, "output.pow_base_f32");
    ggml_set_output(power);

    ggml_tensor * phase = nullptr;
    ggml_tensor * sine = nullptr;
    ggml_tensor * cosine = nullptr;
    if (!options.pow_only) {
        phase = ggml_div(ctx, position_tensor, power);
        ggml_set_name(phase, "output.phase");
        sine = ggml_sin(ctx, phase);
        ggml_set_name(sine, "output.sin");
        cosine = ggml_cos(ctx, phase);
        ggml_set_name(cosine, "output.cos");
        ggml_set_output(phase);
        ggml_set_output(sine);
        ggml_set_output(cosine);
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    if (options.pow_only) {
        ggml_build_forward_expand(graph, power);
    } else {
        ggml_build_forward_expand(graph, sine);
        ggml_build_forward_expand(graph, cosine);
    }

    ggml_backend_t backend = options.backend == "cpu"
        ? ggml_backend_cpu_init()
        : ggml_backend_cuda_init(options.device);
    if (!backend) {
        fail("backend initialization failed");
    }
    if (!ggml_backend_supports_op(backend, power)) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        fail("selected backend rejects the power node");
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        fail("backend tensor allocation failed");
    }

    const std::size_t bytes = exponents.size() * sizeof(float);
    ggml_backend_tensor_set(exponent_tensor, exponents.data(), 0, bytes);
    ggml_backend_tensor_set(position_tensor, positions.data(), 0, bytes);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        fail("backend graph compute failed");
    }

    std::vector<float> exponent_readback(exponents.size());
    std::vector<float> position_readback(positions.size());
    std::vector<float> power_values(exponents.size());
    std::vector<float> phase_values(options.pow_only ? 0 : exponents.size());
    std::vector<float> sine_values(options.pow_only ? 0 : exponents.size());
    std::vector<float> cosine_values(options.pow_only ? 0 : exponents.size());
    tensor_get_exact(exponent_tensor, exponent_readback);
    tensor_get_exact(position_tensor, position_readback);
    tensor_get_exact(power, power_values);
    if (!options.pow_only) {
        tensor_get_exact(phase, phase_values);
        tensor_get_exact(sine, sine_values);
        tensor_get_exact(cosine, cosine_values);
    }
    if (std::memcmp(exponents.data(), exponent_readback.data(), bytes) != 0 ||
        std::memcmp(positions.data(), position_readback.data(), bytes) != 0) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        fail("backend input readback differs from uploaded bytes");
    }

    write_f32(options.output_dir / "input.exponent.readback.f32", exponent_readback);
    write_f32(options.output_dir / "input.position.readback.f32", position_readback);
    write_f32(options.output_dir / "output.pow.f32", power_values);
    std::vector<std::string> output_names = {"output.pow.f32"};
    if (!options.pow_only) {
        write_f32(options.output_dir / "output.phase.f32", phase_values);
        write_f32(options.output_dir / "output.sin.f32", sine_values);
        write_f32(options.output_dir / "output.cos.f32", cosine_values);
        output_names.insert(output_names.end(), {"output.phase.f32", "output.sin.f32", "output.cos.f32"});
    }

    std::ofstream meta(options.output_dir / "run.json");
    meta << "{\n"
         << "  \"schema\": \"ggml-position-power-probe-run-v1\",\n"
         << "  \"case_id\": \"" << options.case_id << "\",\n"
         << "  \"backend\": \"" << options.backend << "\",\n"
         << "  \"device\": " << options.device << ",\n"
         << "  \"base_f32\": " << std::setprecision(9) << options.base << ",\n"
         << "  \"count\": " << count << ",\n"
         << "  \"pow_only\": " << (options.pow_only ? "true" : "false") << ",\n"
         << "  \"input_readback_memcmp\": true,\n"
         << "  \"reference_values_loaded\": false,\n"
         << "  \"outputs\": [";
    for (std::size_t i = 0; i < output_names.size(); ++i) {
        meta << (i ? ", " : "") << "\"" << output_names[i] << "\"";
    }
    meta << "]\n}\n";
    if (!meta) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        fail("cannot write run metadata");
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::cout << "PASS " << options.case_id << " backend=" << options.backend
              << " count=" << count << " outputs=" << options.output_dir << "\n";
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        Options options = parse(argc, argv);
        if (options.backend.rfind("invalid-", 0) == 0) {
            return invalid_contract_case(options.backend);
        }
        if (options.backend == "support-shape-negative") {
            return support_shape_negative();
        }
        return run_probe(options);
    } catch (const std::exception & ex) {
        std::cerr << "FAIL: " << ex.what() << "\n";
        return 2;
    }
}
