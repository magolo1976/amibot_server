#pragma once

#include "llmcomms-core.h"

#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// llmcomms_adapter_cvec
//

struct llmcomms_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const llmcomms_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llmcomms_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

using llmcomms_adapter_cvec_ptr = std::shared_ptr<llmcomms_adapter_cvec>;

//
// llmcomms_adapter_lora
//

struct llmcomms_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llmcomms_adapter_lora_weight() = default;
    llmcomms_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llmcomms_adapter_lora {
    llmcomms_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llmcomms_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llmcomms_token> alora_invocation_tokens;

    explicit llmcomms_adapter_lora(llmcomms_model * model) : model(model) {}
    ~llmcomms_adapter_lora() = default;

    llmcomms_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using llmcomms_adapter_loras = std::unordered_map<llmcomms_adapter_lora *, float>;
using llmcomms_adapter_loras_ptr = std::unique_ptr<llmcomms_adapter_loras>;
