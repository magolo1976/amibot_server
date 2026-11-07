#include "models.h"

llmcomms_rope_params llmcomms_model_hunyuan_dense::get_rope_params(const llmcomms_cparams & cparams, int il) const {
    auto res = llmcomms_model::get_rope_params(cparams, il);

    res.k_norm = layers[il].attn_k_norm;

    return res;
}

std::unique_ptr<llm_graph_context> llmcomms_model_hunyuan_dense::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

