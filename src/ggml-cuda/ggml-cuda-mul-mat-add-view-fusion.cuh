#ifndef GGML_CUDA_GGML_CUDA_MUL_MAT_ADD_VIEW_FUSION_CUH
#define GGML_CUDA_GGML_CUDA_MUL_MAT_ADD_VIEW_FUSION_CUH

#include "ggml-impl.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ggml_cuda_mul_mat_add_view_fusion {

constexpr int max_bias_view_chain = 8;

enum class RejectReason : uint8_t {
    none,
    invalid_graph,
    graph_pattern,
    add_pattern,
    view_chain,
    view_consumer,
    view_layout,
    view_range,
    bias_shape,
    base_not_immutable_weight,
    output_alias,
    subgraph_not_fusable,
};

struct Match {
    int mm_index = -1;
    int add_index = -1;
    int view_count = 0;
    int view_indices[max_bias_view_chain] = {};
    const ggml_tensor * bias = nullptr;
    const ggml_tensor * bias_base = nullptr;
};

inline bool reject(RejectReason * reason, RejectReason value) {
    if (reason != nullptr) {
        *reason = value;
    }
    return false;
}

inline int graph_index(const ggml_cgraph * graph, const ggml_tensor * tensor) {
    if (graph == nullptr || tensor == nullptr) {
        return -1;
    }

    int found = -1;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == tensor) {
            if (found != -1) {
                return -1;
            }
            found = i;
        }
    }
    return found;
}

inline bool tensor_byte_range(const ggml_tensor * tensor, uintptr_t * begin, uintptr_t * end) {
    if (tensor == nullptr || tensor->data == nullptr || tensor->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(tensor)) {
        return false;
    }

    const size_t bytes = ggml_nbytes(tensor);
    const uintptr_t address = reinterpret_cast<uintptr_t>(tensor->data);
    if (bytes > std::numeric_limits<uintptr_t>::max() - address) {
        return false;
    }

    *begin = address;
    *end = address + bytes;
    return true;
}

inline bool ranges_disjoint(const ggml_tensor * a, const ggml_tensor * b) {
    uintptr_t a_begin = 0;
    uintptr_t a_end = 0;
    uintptr_t b_begin = 0;
    uintptr_t b_end = 0;
    if (!tensor_byte_range(a, &a_begin, &a_end) || !tensor_byte_range(b, &b_begin, &b_end)) {
        return false;
    }
    return a_end <= b_begin || b_end <= a_begin;
}

template<typename IsImmutableWeight>
inline bool match(const ggml_cgraph * graph, int mm_index, IsImmutableWeight is_immutable_weight,
                  Match * match_out, RejectReason * reason = nullptr) {
    if (reason != nullptr) {
        *reason = RejectReason::none;
    }
    if (match_out != nullptr) {
        *match_out = Match{};
    }
    if (graph == nullptr || match_out == nullptr || mm_index < 0 || mm_index >= graph->n_nodes ||
            graph->nodes[mm_index] == nullptr) {
        return reject(reason, RejectReason::invalid_graph);
    }

    const ggml_tensor * mm = graph->nodes[mm_index];
    if (mm->op != GGML_OP_MUL_MAT || mm->src[0] == nullptr || mm->src[1] == nullptr ||
            mm->src[2] != nullptr || mm->view_src != nullptr || mm->type != GGML_TYPE_F32 ||
            ggml_get_op_params_i32(mm, 0) != GGML_PREC_F32) {
        return reject(reason, RejectReason::graph_pattern);
    }

    int add_index = -1;
    for (int i = mm_index + 1; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        if (node == nullptr) {
            return reject(reason, RejectReason::invalid_graph);
        }
        const bool consumes_mm = node->src[0] == mm || node->src[1] == mm;
        if (node->op == GGML_OP_ADD && consumes_mm) {
            add_index = i;
            break;
        }
        if (node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE) {
            return reject(reason, RejectReason::graph_pattern);
        }
    }
    if (add_index < 0) {
        return reject(reason, RejectReason::add_pattern);
    }

    const ggml_tensor * add = graph->nodes[add_index];
    if ((add->src[0] == mm) == (add->src[1] == mm) || add->src[2] != nullptr ||
            add->view_src != nullptr || add->type != GGML_TYPE_F32 || !ggml_are_same_shape(mm, add)) {
        return reject(reason, RejectReason::add_pattern);
    }
    const ggml_tensor * bias = add->src[0] == mm ? add->src[1] : add->src[0];
    if (bias == nullptr || (bias->op != GGML_OP_VIEW && bias->op != GGML_OP_RESHAPE)) {
        // Direct bias leaves remain on the already-qualified adjacent path.
        return reject(reason, RejectReason::view_chain);
    }

    Match candidate{};
    candidate.mm_index = mm_index;
    candidate.add_index = add_index;
    candidate.bias = bias;

    const ggml_tensor * base = bias;
    while (base != nullptr && (base->op == GGML_OP_VIEW || base->op == GGML_OP_RESHAPE)) {
        if (candidate.view_count >= max_bias_view_chain) {
            return reject(reason, RejectReason::view_chain);
        }
        const int index = graph_index(graph, base);
        if (index <= mm_index || index >= add_index || base->src[0] == nullptr ||
                base->src[1] != nullptr || base->src[2] != nullptr ||
                (base->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || ggml_node_get_use_count(graph, index) != 1) {
            return reject(reason, RejectReason::view_consumer);
        }
        candidate.view_indices[candidate.view_count++] = index;
        base = base->src[0];
    }

    if (candidate.view_count == 0 || base == nullptr || base->op != GGML_OP_NONE ||
            base->type != GGML_TYPE_F32 || !ggml_is_contiguous(base) || !is_immutable_weight(base)) {
        return reject(reason, RejectReason::base_not_immutable_weight);
    }
    candidate.bias_base = base;

    if (bias->type != GGML_TYPE_F32 || bias->ne[0] != mm->ne[0] || bias->ne[1] != 1 ||
            bias->ne[2] != 1 || bias->ne[3] != 1 || !ggml_is_contiguous(bias) ||
            bias->nb[0] != sizeof(float)) {
        return reject(reason, RejectReason::bias_shape);
    }

    const size_t base_bytes = ggml_nbytes(base);
    for (int i = 0; i < candidate.view_count; ++i) {
        const ggml_tensor * view = graph->nodes[candidate.view_indices[i]];
        const ggml_tensor * parent = view->src[0];
        if (view->view_src != base || parent == nullptr || view->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(view) || view->nb[0] != sizeof(float) ||
                view->view_offs % sizeof(float) != 0) {
            return reject(reason, RejectReason::view_layout);
        }

        const size_t parent_offset = parent == base ? 0 : parent->view_offs;
        const size_t parent_bytes = ggml_nbytes(parent);
        if ((parent != base && parent->view_src != base) || view->view_offs < parent_offset ||
                view->view_offs - parent_offset > parent_bytes ||
                ggml_nbytes(view) > parent_bytes - (view->view_offs - parent_offset) ||
                view->view_offs > base_bytes || ggml_nbytes(view) > base_bytes - view->view_offs) {
            return reject(reason, RejectReason::view_range);
        }

        if ((parent->data == nullptr) != (view->data == nullptr)) {
            return reject(reason, RejectReason::view_layout);
        }
        if (parent->data != nullptr) {
            const size_t offset_from_parent = view->view_offs - parent_offset;
            const uintptr_t parent_address = reinterpret_cast<uintptr_t>(parent->data);
            if (offset_from_parent > std::numeric_limits<uintptr_t>::max() - parent_address ||
                    reinterpret_cast<uintptr_t>(view->data) != parent_address + offset_from_parent) {
                return reject(reason, RejectReason::view_layout);
            }
        }
    }

    if (add_index - mm_index - 1 != candidate.view_count) {
        return reject(reason, RejectReason::graph_pattern);
    }
    for (int index = mm_index + 1; index < add_index; ++index) {
        bool found = false;
        for (int i = 0; i < candidate.view_count; ++i) {
            if (candidate.view_indices[i] == index) {
                found = true;
                break;
            }
        }
        if (!found) {
            return reject(reason, RejectReason::graph_pattern);
        }
    }

    const int selected_nodes[] = { mm_index, add_index };
    const enum ggml_op selected_ops[] = { GGML_OP_MUL_MAT, GGML_OP_ADD };
    const int output_nodes[] = { add_index };
    if (!ggml_can_fuse_subgraph_ext(graph, selected_nodes, 2, selected_ops, output_nodes, 1)) {
        return reject(reason, RejectReason::subgraph_not_fusable);
    }
    if (!ranges_disjoint(add, base)) {
        return reject(reason, RejectReason::output_alias);
    }

    *match_out = candidate;
    return true;
}

} // namespace ggml_cuda_mul_mat_add_view_fusion

#endif // GGML_CUDA_GGML_CUDA_MUL_MAT_ADD_VIEW_FUSION_CUH
