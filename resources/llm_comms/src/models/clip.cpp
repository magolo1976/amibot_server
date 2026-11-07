#include "models.h"

// Stub to allow llmcomms-quantize to open mmproj GGUFs

[[noreturn]]
void llmcomms_model_clip::load_arch_hparams(llmcomms_model_loader &) {
    GGML_ABORT("CLIP is a quant-only stub; load_arch_hparams should not be called");
}

[[noreturn]]
void llmcomms_model_clip::load_arch_tensors(llmcomms_model_loader &) {
    GGML_ABORT("CLIP is a quant-only stub; load_arch_tensors should not be called");
}

[[noreturn]]
std::unique_ptr<llm_graph_context> llmcomms_model_clip::build_arch_graph(const llm_graph_params &) const {
    GGML_ABORT("CLIP has no inference graph via llmcomms_model dispatch; runtime lives in tools/mtmd/clip.cpp");
}
