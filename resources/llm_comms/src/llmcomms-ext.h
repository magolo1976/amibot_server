#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llmcomms-core.h"
#include "llmcomms-arch.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

//
// llama architecture queries
//

// all the LLM architectures known to the library
LLMCOMMS_API std::vector<llm_arch> llmcomms_arch_all();

// the canonical name of an architecture (e.g. "llama"), or "unknown"
LLMCOMMS_API const char * llmcomms_arch_name(llm_arch arch);

// the architecture with the given canonical name, or LLM_ARCH_UNKNOWN
LLMCOMMS_API llm_arch llmcomms_arch_from_string(const std::string & name);

// the arch-resolved GGUF key of a KV id, e.g. LLM_KV_CONTEXT_LENGTH -> "llama.context_length"
LLMCOMMS_API std::string llmcomms_arch_kv_name(llm_arch arch, llm_kv key);

LLMCOMMS_API bool llmcomms_arch_supports_sm_tensor(llm_arch arch);

LLMCOMMS_API bool llmcomms_arch_supports_mixed_batch(llm_arch arch);

// FIXME temporary function for better error messages
LLMCOMMS_API bool llmcomms_arch_supports_save(llm_arch arch);

// Reserve a new compute graph. It is valid until the next call to llmcomms_graph_reserve.
LLMCOMMS_API struct ggml_cgraph * llmcomms_graph_reserve(
        struct llmcomms_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLMCOMMS_API ggml_type llmcomms_ftype_get_default_type(llmcomms_ftype ftype);

struct quantize_state_impl;

LLMCOMMS_API quantize_state_impl * llmcomms_quant_init(
        const llmcomms_model * model,
        const llmcomms_model_quantize_params * params);

LLMCOMMS_API void llmcomms_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llmcomms_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llmcomms_model_free().
LLMCOMMS_API llmcomms_model * llmcomms_quant_model_from_metadata(const llmcomms_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLMCOMMS_API bool llmcomms_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llmcomms_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLMCOMMS_API void llmcomms_quant_compute_types(
        quantize_state_impl * qs,
        llmcomms_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llmcomms_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llmcomms_device_memory_data {
    int64_t total;
    int64_t free;
    llmcomms_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llmcomms_memory_breakdown = std::map<ggml_backend_buffer_type_t, llmcomms_memory_breakdown_data>;

LLMCOMMS_API int32_t llmcomms_model_n_expert (const struct llmcomms_model * model);
LLMCOMMS_API int32_t llmcomms_model_n_devices(const struct llmcomms_model * model);

LLMCOMMS_API ggml_backend_dev_t llmcomms_model_get_device(const struct llmcomms_model * model, int i);

LLMCOMMS_API llmcomms_memory_breakdown llmcomms_get_memory_breakdown(const struct llmcomms_context * ctx);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLMCOMMS_API std::vector<ggml_backend_buffer_t> llmcomms_memory_get_buffers(const llmcomms_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLMCOMMS_API void llmcomms_model_set_attn_rot(struct llmcomms_model * model, bool value);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLMCOMMS_API void llmcomms_set_embeddings_nextn(struct llmcomms_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLMCOMMS_API void llmcomms_set_nextn_layer_offset(struct llmcomms_context * ctx, int32_t offset);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llmcomms_decision_order {
    LLMCOMMS_DECISION_ORDER_NONE            = 0, // not read by the head
    LLMCOMMS_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLMCOMMS_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLMCOMMS_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLMCOMMS_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLMCOMMS_API bool llmcomms_batch_ext_set_decision_order(struct llmcomms_batch_ext * batch, int32_t idx, enum llmcomms_decision_order order);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLMCOMMS_API const char * llmcomms_model_decision_label(const struct llmcomms_model * model, uint32_t i);

// mirrors:
// LLMCOMMS_API float * llmcomms_get_embeddings(struct llmcomms_context * ctx);
LLMCOMMS_API float * llmcomms_get_embeddings_nextn(struct llmcomms_context * ctx);

// LLMCOMMS_API float * llmcomms_get_embeddings_ith(struct llmcomms_context * ctx, int32_t i);
LLMCOMMS_API float * llmcomms_get_embeddings_nextn_ith(struct llmcomms_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLMCOMMS_API void llmcomms_set_embeddings_layer_inp(struct llmcomms_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLMCOMMS_API float * llmcomms_get_embeddings(struct llmcomms_context * ctx);
LLMCOMMS_API float * llmcomms_get_embeddings_layer_inp(struct llmcomms_context * ctx, uint32_t lid);

// Write the enabled layer inputs to a device buffer [n_layer_inp*n_embd, n_batch] instead of host memory.
// Row r holds the token at batch index r, layers in ascending layer id order.
LLMCOMMS_API void llmcomms_set_embeddings_layer_inp_dev(struct llmcomms_context * ctx, bool value);

// Whether decode accepts entries added with llmcomms_batch_ext_add_embd_row()
// (needs a DFlash/DSpark decoder context whose ctx_other has a layer input device buffer).
LLMCOMMS_API bool llmcomms_supports_embd_row(const struct llmcomms_context * ctx);

// Add an entry whose embedding is row `row` of the layer input device buffer of ctx_other.
// All entries of the batch must be of this kind. Positions are set with llmcomms_batch_ext_set_pos(), n_pos_per_embd of them.
// Returns the batch index (>= 0), or the llmcomms_batch_ext_add() error codes.
LLMCOMMS_API int32_t llmcomms_batch_ext_add_embd_row(struct llmcomms_batch_ext * batch, llmcomms_seq_id seq_id, int32_t row);

LLMCOMMS_API llmcomms_context * llmcomms_get_ctx_other(struct llmcomms_context * ctx);

//
// model/context data extraction
//

LLMCOMMS_API int32_t llmcomms_model_dflash_selector_top_k(const struct llmcomms_model * model);

// returns pointer to the target-model layer indices
LLMCOMMS_API const int32_t * llmcomms_model_target_layer_ids  (const struct llmcomms_model * model);
// returns the number of extracted layers from target model
LLMCOMMS_API uint32_t        llmcomms_model_target_layer_ids_n(const struct llmcomms_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLMCOMMS_API uint32_t llmcomms_model_get_tok_embd(const struct llmcomms_model * model, float * out);

// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLMCOMMS_API uint32_t llmcomms_model_get_tok_embd_rows(const struct llmcomms_model * model, const llmcomms_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLMCOMMS_API uint32_t llmcomms_model_n_tokens_delay(const struct llmcomms_model * model);

// each sequence is paired with an unconditional one at n_seq_max / 2 + seq_id, in a unified KV cache
LLMCOMMS_API bool llmcomms_model_classifier_free_guidance(const struct llmcomms_model * model);
