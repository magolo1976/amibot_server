#ifndef LLMCOMMS_H
#define LLMCOMMS_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include "gguf.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef LLMCOMMS_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLMCOMMS_BUILD
#            define LLMCOMMS_API __declspec(dllexport)
#        else
#            define LLMCOMMS_API __declspec(dllimport)
#        endif
#    else
#        define LLMCOMMS_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLMCOMMS_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define LLMCOMMS_DEFAULT_SEED 0xFFFFFFFF

#define LLMCOMMS_TOKEN_NULL -1

#define LLMCOMMS_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
#define LLMCOMMS_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define LLMCOMMS_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define LLMCOMMS_SESSION_MAGIC   LLMCOMMS_FILE_MAGIC_GGSN
#define LLMCOMMS_SESSION_VERSION 11

#define LLMCOMMS_STATE_SEQ_MAGIC   LLMCOMMS_FILE_MAGIC_GGSQ
#define LLMCOMMS_STATE_SEQ_VERSION 4

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llmcomms_vocab;
    struct llmcomms_model;
    struct llmcomms_context;
    struct llmcomms_sampler;

    typedef struct llmcomms_memory_i * llmcomms_memory_t;

    typedef int32_t llmcomms_pos;
    typedef int32_t llmcomms_token;
    typedef int32_t llmcomms_seq_id;

    enum llmcomms_vocab_type {
        LLMCOMMS_VOCAB_TYPE_NONE   = 0, // For models without vocab
        LLMCOMMS_VOCAB_TYPE_SPM    = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLMCOMMS_VOCAB_TYPE_BPE    = 2, // GPT-2 tokenizer based on byte-level BPE
        LLMCOMMS_VOCAB_TYPE_WPM    = 3, // BERT tokenizer based on WordPiece
        LLMCOMMS_VOCAB_TYPE_UGM    = 4, // T5 tokenizer based on Unigram
        LLMCOMMS_VOCAB_TYPE_RWKV   = 5, // RWKV tokenizer based on greedy tokenization
        LLMCOMMS_VOCAB_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
        LLMCOMMS_VOCAB_TYPE_TEST   = 7, // Dummy tokenizer for testing: rolling hash of fixed-size chunks -> tokens, tokens -> hex
        LLMCOMMS_VOCAB_TYPE_PLAMO3 = 8, // PLaMo-3 tokenizer with pre-segmentation and dynamic programming
    };

    enum llmcomms_rope_type {
        LLMCOMMS_ROPE_TYPE_NONE   = -1,
        LLMCOMMS_ROPE_TYPE_NORM   = 0,
        LLMCOMMS_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
        LLMCOMMS_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
        LLMCOMMS_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE,
        LLMCOMMS_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
    };

    enum llmcomms_token_type { //TODO: remove, required until per token attributes are available from GGUF file
        LLMCOMMS_TOKEN_TYPE_UNDEFINED    = 0,
        LLMCOMMS_TOKEN_TYPE_NORMAL       = 1,
        LLMCOMMS_TOKEN_TYPE_UNKNOWN      = 2,
        LLMCOMMS_TOKEN_TYPE_CONTROL      = 3,
        LLMCOMMS_TOKEN_TYPE_USER_DEFINED = 4,
        LLMCOMMS_TOKEN_TYPE_UNUSED       = 5,
        LLMCOMMS_TOKEN_TYPE_BYTE         = 6,
    };

    enum llmcomms_token_attr {
        LLMCOMMS_TOKEN_ATTR_UNDEFINED    = 0,
        LLMCOMMS_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        LLMCOMMS_TOKEN_ATTR_UNUSED       = 1 << 1,
        LLMCOMMS_TOKEN_ATTR_NORMAL       = 1 << 2,
        LLMCOMMS_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        LLMCOMMS_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        LLMCOMMS_TOKEN_ATTR_BYTE         = 1 << 5,
        LLMCOMMS_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        LLMCOMMS_TOKEN_ATTR_LSTRIP       = 1 << 7,
        LLMCOMMS_TOKEN_ATTR_RSTRIP       = 1 << 8,
        LLMCOMMS_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    // model file types
    enum llmcomms_ftype {
        LLMCOMMS_FTYPE_ALL_F32              = 0,
        LLMCOMMS_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        // LLMCOMMS_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // LLMCOMMS_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // LLMCOMMS_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        LLMCOMMS_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
        //LLMCOMMS_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
        //LLMCOMMS_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
        //LLMCOMMS_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
        LLMCOMMS_FTYPE_MOSTLY_TQ1_0         = 36, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_TQ2_0         = 37, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_MXFP4_MOE     = 38, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_NVFP4         = 39, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q1_0          = 40, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_Q2_0          = 41, // except 1d tensors
        LLMCOMMS_FTYPE_MOSTLY_F8_E4M3       = 42, // except 1d tensors

        LLMCOMMS_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    // Get the model file type (quantization) as a string, e.g. "Q8_0" or "Q4_K - Medium"
    LLMCOMMS_API const char * llmcomms_ftype_name(enum llmcomms_ftype ftype);

    enum llmcomms_rope_scaling_type {
        LLMCOMMS_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LLMCOMMS_ROPE_SCALING_TYPE_NONE        = 0,
        LLMCOMMS_ROPE_SCALING_TYPE_LINEAR      = 1,
        LLMCOMMS_ROPE_SCALING_TYPE_YARN        = 2,
        LLMCOMMS_ROPE_SCALING_TYPE_LONGROPE    = 3,
        LLMCOMMS_ROPE_SCALING_TYPE_MAX_VALUE   = LLMCOMMS_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum llmcomms_pooling_type {
        LLMCOMMS_POOLING_TYPE_UNSPECIFIED = -1,
        LLMCOMMS_POOLING_TYPE_NONE = 0,
        LLMCOMMS_POOLING_TYPE_MEAN = 1,
        LLMCOMMS_POOLING_TYPE_CLS  = 2,
        LLMCOMMS_POOLING_TYPE_LAST = 3,
        LLMCOMMS_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
    };

    enum llmcomms_attention_type {
        LLMCOMMS_ATTENTION_TYPE_UNSPECIFIED = -1,
        LLMCOMMS_ATTENTION_TYPE_CAUSAL      = 0,
        LLMCOMMS_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum llmcomms_flash_attn_type {
        LLMCOMMS_FLASH_ATTN_TYPE_AUTO     = -1,
        LLMCOMMS_FLASH_ATTN_TYPE_DISABLED = 0,
        LLMCOMMS_FLASH_ATTN_TYPE_ENABLED  = 1,
    };

    LLMCOMMS_API const char * llmcomms_flash_attn_type_name(enum llmcomms_flash_attn_type flash_attn_type);

    enum llmcomms_split_mode {
        LLMCOMMS_SPLIT_MODE_NONE   = 0, // single GPU
        LLMCOMMS_SPLIT_MODE_LAYER  = 1, // split layers and KV across GPUs
        LLMCOMMS_SPLIT_MODE_ROW    = 2, // split layers and KV across GPUs, use tensor parallelism if supported
        LLMCOMMS_SPLIT_MODE_TENSOR = 3,
    };

    enum llmcomms_load_mode {
        LLMCOMMS_LOAD_MODE_AUTO       = -1, // auto-detect based on device capabilities
        LLMCOMMS_LOAD_MODE_NONE       =  0, // no special loading mode
        LLMCOMMS_LOAD_MODE_MMAP       =  1, // memory map the model
        LLMCOMMS_LOAD_MODE_MLOCK      =  2, // force system to keep model in RAM rather than swapping or compressing
        LLMCOMMS_LOAD_MODE_MMAP_MLOCK =  3, // mmap + force system to keep model in RAM rather than swapping or compressing
        LLMCOMMS_LOAD_MODE_DIRECT_IO  =  4, // use direct I/O if available
    };

    LLMCOMMS_API const char * llmcomms_load_mode_name(enum llmcomms_load_mode load_mode);
    LLMCOMMS_API enum llmcomms_load_mode llmcomms_load_mode_from_str(const char * str);

    enum llmcomms_lazy_mode {
        LLMCOMMS_LAZY_MODE_OFF  = 0, // always read the whole tensor up front
        LLMCOMMS_LAZY_MODE_AUTO = 1, // lazy only for marked tensors larger than 4 GiB (requires mmap)
        LLMCOMMS_LAZY_MODE_ON   = 2, // read the rows of tensors marked by the arch on demand (requires mmap)
    };

    enum llmcomms_context_type {
        LLMCOMMS_CONTEXT_TYPE_DEFAULT = 0,
        LLMCOMMS_CONTEXT_TYPE_MTP     = 1,
    };

    // TODO: simplify (https://github.com/ggml-org/llama.cpp/pull/9294#pullrequestreview-2286561979)
    typedef struct llmcomms_token_data {
        llmcomms_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } llmcomms_token_data;

    typedef struct llmcomms_token_data_array {
        // TODO: consider SoA
        // NOTE: this pointer can be modified by the samplers
        llmcomms_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;      // note: do not assume the data is sorted - always check this flag
    } llmcomms_token_data_array;

    typedef bool (*llmcomms_progress_callback)(float progress, void * user_data);

    // Input data for llmcomms_encode/llmcomms_decode
    // A llmcomms_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by llmcomms_encode/llmcomms_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL:
    //               - if embeddings: all tokens are output
    //               - if not:        only the last token is output
    //            )
    //
    typedef struct llmcomms_batch {
        int32_t n_tokens;

        llmcomms_token  *  token;
        float        *  embd;
        llmcomms_pos    *  pos;
        int32_t      *  n_seq_id;
        llmcomms_seq_id ** seq_id;
        int8_t       *  logits;   // TODO: rename this to "output"
    } llmcomms_batch;

    enum llmcomms_model_kv_override_type {
        LLMCOMMS_KV_OVERRIDE_TYPE_INT,
        LLMCOMMS_KV_OVERRIDE_TYPE_FLOAT,
        LLMCOMMS_KV_OVERRIDE_TYPE_BOOL,
        LLMCOMMS_KV_OVERRIDE_TYPE_STR,
    };

    enum llmcomms_model_meta_key {
        LLMCOMMS_MODEL_META_KEY_SAMPLING_SEQUENCE,
        LLMCOMMS_MODEL_META_KEY_SAMPLING_TOP_K,
        LLMCOMMS_MODEL_META_KEY_SAMPLING_TOP_P,
        LLMCOMMS_MODEL_META_KEY_SAMPLING_MIN_P,
        LLMCOMMS_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY,
        LLMCOMMS_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD,
        LLMCOMMS_MODEL_META_KEY_SAMPLING_TEMP,
        LLMCOMMS_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N,
        LLMCOMMS_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT,
        LLMCOMMS_MODEL_META_KEY_SAMPLING_MIROSTAT,
        LLMCOMMS_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU,
        LLMCOMMS_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA,
    };

    enum llmcomms_process_type {
        LLMCOMMS_PROCESS_TYPE_ENCODE,
        LLMCOMMS_PROCESS_TYPE_DECODE,
    };

    struct llmcomms_model_kv_override {
        enum llmcomms_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct llmcomms_model_tensor_buft_override {
        const char * pattern;
        ggml_backend_buffer_type_t buft;
    };

    struct llmcomms_model_params {
        // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ggml_backend_dev_t * devices;

        // NULL-terminated list of buffer types to use for tensors that match a pattern
        const struct llmcomms_model_tensor_buft_override * tensor_buft_overrides;

        int32_t n_gpu_layers; // number of layers to store in VRAM, a negative value means all layers
        enum llmcomms_split_mode split_mode; // how to split the model across multiple GPUs
        enum llmcomms_load_mode  load_mode;  // how to load the model

        enum llmcomms_lazy_mode lazy_mode; // on-demand reading of tensors marked by the arch

        // the GPU that is used for the entire model when split_mode is LLMCOMMS_SPLIT_MODE_NONE
        int32_t main_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: llmcomms_max_devices()
        const float * tensor_split;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        llmcomms_progress_callback progress_callback;

        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // override key-value pairs of the model meta data
        const struct llmcomms_model_kv_override * kv_overrides;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only;      // only load the vocabulary, no weights
        bool check_tensors;   // validate model tensor data
        bool use_extra_bufts; // use extra buffer types (used for weight repacking)
        bool no_host;         // bypass host buffer allowing extra buffers to be used
        bool no_alloc;        // only load metadata and simulate memory allocations
        bool load_mtp;        // whether to load MTP layers
    };

    struct llmcomms_sampler_seq_config {
        llmcomms_seq_id           seq_id;
        struct llmcomms_sampler * sampler;
    };

    // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
    //       https://github.com/ggml-org/llama.cpp/pull/7544
    struct llmcomms_context_params {
        uint32_t n_ctx;                 // text context, 0 = from model
        uint32_t n_batch;               // logical maximum batch size that can be submitted to llmcomms_decode
        uint32_t n_ubatch;              // physical maximum batch size
        uint32_t n_seq_max;             // max number of sequences (i.e. distinct states for recurrent models)
        uint32_t n_rs_seq;              // number of recurrent-state snapshots per seq for rollback (0 = no rollback) [EXPERIMENTAL]
        uint32_t n_outputs_max;         // max outputs in a ubatch (0 = n_batch)
        uint32_t n_outputs_max_per_seq; // max outputs per sequence (0 = n_outputs_max)
        int32_t  n_threads;             // number of threads to use for generation
        int32_t  n_threads_batch;       // number of threads to use for batch processing

        enum llmcomms_context_type      ctx_type;          // set the context type (e.g. MTP)
        enum llmcomms_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum llmcomms_rope_scaling_type`
        enum llmcomms_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
        enum llmcomms_attention_type    attention_type;    // attention type to use for embeddings
        enum llmcomms_flash_attn_type   flash_attn_type;   // when to enable Flash Attention

        // ref: https://github.com/ggml-org/llama.cpp/pull/2054
        float    rope_freq_base;   // RoPE base frequency, 0 = from model
        float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor; // YaRN magnitude scaling factor
        float    yarn_beta_fast;   // YaRN low correction dim
        float    yarn_beta_slow;   // YaRN high correction dim
        uint32_t yarn_orig_ctx;    // YaRN original context size
        float    defrag_thold;     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
        enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]

        size_t moe_cache_size; // device cache in bytes for the experts kept in host memory, split among the devices like the layers, 0 = disabled [EXPERIMENTAL]

        // Abort callback
        // if it returns true, execution of llmcomms_decode() will be aborted
        // currently works only with CPU execution
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        bool embeddings;  // if true, extract embeddings (together with logits, except for a pooling type other than NONE)
        bool offload_kqv; // offload the KQV ops (including the KV cache) to GPU
        bool no_perf;     // measure performance timings
        bool op_offload;  // offload host tensor operations to device
        bool swa_full;    // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
                          // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
                          //       ref: https://github.com/ggml-org/llama.cpp/pull/13845#issuecomment-2924800573
        bool kv_unified;  // use a unified buffer across the input sequences when computing the attention
                          // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
                          // ref: https://github.com/ggml-org/llama.cpp/pull/14363

        // [EXPERIMENTAL]
        // backend sampler chain configuration (make sure the caller keeps the sampler chains alive)
        // note: the samplers must be sampler chains (i.e. use llmcomms_sampler_chain_init)
        struct llmcomms_sampler_seq_config * samplers;
        size_t                            n_samplers;

        // a source/target/parent context
        // can be utilized in various ways, for example by sharing results or llmcomms_memory between 2 contexts
        struct llmcomms_context * ctx_other;
    };

    struct llmcomms_model_tensor_override {
        const char * pattern;
        enum ggml_type type;
    };

    struct llmcomms_model_imatrix_data {
        const char * name;
        const float * data;
        size_t size;
    };

    // model quantization parameters
    typedef struct llmcomms_model_quantize_params {
        int32_t nthread;                                            // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum llmcomms_ftype ftype;                                     // quantize to this llmcomms_ftype
        enum ggml_type output_tensor_type;                          // output tensor type
        enum ggml_type token_embedding_type;                        // token embeddings tensor type
        bool allow_requantize;                                      // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor;                                // quantize output.weight
        bool only_copy;                                             // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bool pure;                                                  // quantize all tensors to the default type
        bool keep_split;                                            // quantize to the same number of shards
        bool dry_run;                                               // calculate and show the final quantization size without performing quantization
        const struct llmcomms_model_imatrix_data * imatrix;            // pointer to importance matrix data
        const struct llmcomms_model_kv_override * kv_overrides;        // pointer to kv overrides
        const struct llmcomms_model_tensor_override * tt_overrides;    // pointer to tensor overrides
        const int32_t * prune_layers;                               // pointer to layer indices to prune
        size_t max_buf_size;                                        // max bytes of tensor rows kept in memory at once, 0 = default (8 GiB)
    } llmcomms_model_quantize_params;

    typedef struct llmcomms_logit_bias {
        llmcomms_token token;
        float bias;
    } llmcomms_logit_bias;

    typedef struct llmcomms_sampler_chain_params {
        bool no_perf; // whether to measure performance timings
    } llmcomms_sampler_chain_params;

    // used in chat template
    typedef struct llmcomms_chat_message {
        const char * role;
        const char * content;
    } llmcomms_chat_message;

    // lora adapter
    struct llmcomms_adapter_lora;

    LLMCOMMS_API const char * llmcomms_version(void);

    // Helpers for getting default parameters
    // TODO: update API to start accepting pointers to params structs (https://github.com/ggml-org/llama.cpp/discussions/9172)
    LLMCOMMS_API struct llmcomms_model_params          llmcomms_model_default_params(void);
    LLMCOMMS_API struct llmcomms_context_params        llmcomms_context_default_params(void);
    LLMCOMMS_API struct llmcomms_sampler_chain_params  llmcomms_sampler_chain_default_params(void);
    LLMCOMMS_API struct llmcomms_model_quantize_params llmcomms_model_quantize_default_params(void);

    // Initialize the llama + ggml backend
    // Call once at the start of the program
    LLMCOMMS_API void llmcomms_backend_init(void);

    // Call once at the end of the program - currently only used for MPI
    LLMCOMMS_API void llmcomms_backend_free(void);

    // Optional: enable numa optimizations
    // TODO: deprecate and make part of llmcomms_backend_init()
    LLMCOMMS_API void llmcomms_numa_init(enum ggml_numa_strategy numa);

    // Optional: an auto threadpool gets created in ggml if not passed explicitly
    LLMCOMMS_API void llmcomms_attach_threadpool(
            struct llmcomms_context * ctx,
               ggml_threadpool_t   threadpool,
               ggml_threadpool_t   threadpool_batch);

    LLMCOMMS_API void llmcomms_detach_threadpool(struct llmcomms_context * ctx);

    typedef void (*llmcomms_model_set_tensor_data_t)(struct ggml_tensor * tensor, void * userdata);

    // Create a new model from GGUF metadata as well as a function to set the tensor data
    //   - tensors are created as GGML_TYPE_F32 by default,
    //     override by adding a tensor with the same name but a different name to the context
    LLMCOMMS_API struct llmcomms_model * llmcomms_model_init_from_user(
                    struct gguf_context * metadata,
          llmcomms_model_set_tensor_data_t   set_tensor_data,    // function to initialize tensor data with
                                   void * set_tensor_data_ud, // userdata for function
              struct llmcomms_model_params   params);

    DEPRECATED(LLMCOMMS_API struct llmcomms_model * llmcomms_load_model_from_file(
                             const char * path_model,
              struct llmcomms_model_params   params),
            "use llmcomms_model_load_from_file instead");

    // Load a model from a file
    // If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
    // If the split file name does not follow this pattern, use llmcomms_model_load_from_splits
    LLMCOMMS_API struct llmcomms_model * llmcomms_model_load_from_file(
                             const char * path_model,
              struct llmcomms_model_params   params);

    // Load a model from an open FILE pointer
    // The GGUF is read from the current position, so it can be embedded in a larger file
    // mmap needs the GGUF data section at a file offset to be aligned to the CPU tensor alignment (32 bytes)
    LLMCOMMS_API struct llmcomms_model * llmcomms_model_load_from_file_ptr(
                                   FILE * file,
              struct llmcomms_model_params   params);

    // Load a model from multiple splits (support custom naming scheme)
    // The paths must be in the correct order
    LLMCOMMS_API struct llmcomms_model * llmcomms_model_load_from_splits(
                             const char ** paths,
                                 size_t    n_paths,
              struct llmcomms_model_params    params);

    LLMCOMMS_API void llmcomms_model_save_to_file(
            const struct llmcomms_model * model,
                        const char * path_model);

    DEPRECATED(LLMCOMMS_API void llmcomms_free_model(struct llmcomms_model * model),
            "use llmcomms_model_free instead");

    LLMCOMMS_API void llmcomms_model_free(struct llmcomms_model * model);

    LLMCOMMS_API struct llmcomms_context * llmcomms_init_from_model(
                     struct llmcomms_model * model,
            struct llmcomms_context_params   params);

    DEPRECATED(LLMCOMMS_API struct llmcomms_context * llmcomms_new_context_with_model(
                     struct llmcomms_model * model,
            struct llmcomms_context_params   params),
            "use llmcomms_init_from_model instead");

    // Frees all allocated memory
    LLMCOMMS_API void llmcomms_free(struct llmcomms_context * ctx);

    LLMCOMMS_API int64_t llmcomms_time_us(void);

    LLMCOMMS_API size_t llmcomms_max_devices(void);
    LLMCOMMS_API size_t llmcomms_max_parallel_sequences(void);
    LLMCOMMS_API size_t llmcomms_max_tensor_buft_overrides(void);

    LLMCOMMS_API bool llmcomms_supports_mmap       (void);
    LLMCOMMS_API bool llmcomms_supports_mlock      (void);
    LLMCOMMS_API bool llmcomms_supports_gpu_offload(void);
    LLMCOMMS_API bool llmcomms_supports_rpc        (void);

    // NOTE: After creating a llmcomms_context, it is recommended to query the actual values using these functions
    //       In some cases the requested values via llmcomms_context_params may differ from the actual values used by the context
    //       ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    LLMCOMMS_API uint32_t llmcomms_n_ctx      (const struct llmcomms_context * ctx);
    LLMCOMMS_API uint32_t llmcomms_n_ctx_seq  (const struct llmcomms_context * ctx);
    LLMCOMMS_API uint32_t llmcomms_n_batch    (const struct llmcomms_context * ctx);
    LLMCOMMS_API uint32_t llmcomms_n_ubatch   (const struct llmcomms_context * ctx);
    LLMCOMMS_API uint32_t llmcomms_n_seq_max  (const struct llmcomms_context * ctx);
    LLMCOMMS_API uint32_t llmcomms_n_rs_seq   (const struct llmcomms_context * ctx);

    DEPRECATED(LLMCOMMS_API int32_t llmcomms_n_ctx_train(const struct llmcomms_model * model), "use llmcomms_model_n_ctx_train instead");
    DEPRECATED(LLMCOMMS_API int32_t llmcomms_n_embd     (const struct llmcomms_model * model), "use llmcomms_model_n_embd instead");
    DEPRECATED(LLMCOMMS_API int32_t llmcomms_n_layer    (const struct llmcomms_model * model), "use llmcomms_model_n_layer instead");
    DEPRECATED(LLMCOMMS_API int32_t llmcomms_n_head     (const struct llmcomms_model * model), "use llmcomms_model_n_head instead");

    DEPRECATED(LLMCOMMS_API int32_t llmcomms_n_vocab    (const struct llmcomms_vocab * vocab), "use llmcomms_vocab_n_tokens instead");

    LLMCOMMS_API const struct llmcomms_model * llmcomms_get_model   (const struct llmcomms_context * ctx);
    LLMCOMMS_API           llmcomms_memory_t   llmcomms_get_memory  (const struct llmcomms_context * ctx);
    LLMCOMMS_API  enum llmcomms_pooling_type   llmcomms_pooling_type(const struct llmcomms_context * ctx); // TODO: rename to llmcomms_get_pooling_type

    LLMCOMMS_API const struct llmcomms_vocab * llmcomms_model_get_vocab(const struct llmcomms_model * model);
    LLMCOMMS_API enum llmcomms_rope_type       llmcomms_model_rope_type(const struct llmcomms_model * model);

    LLMCOMMS_API int32_t llmcomms_model_n_ctx_train  (const struct llmcomms_model * model);
    LLMCOMMS_API int32_t llmcomms_model_n_embd       (const struct llmcomms_model * model);
    LLMCOMMS_API int32_t llmcomms_model_n_embd_inp   (const struct llmcomms_model * model);
    LLMCOMMS_API int32_t llmcomms_model_n_embd_out   (const struct llmcomms_model * model);
    LLMCOMMS_API int32_t llmcomms_model_n_layer      (const struct llmcomms_model * model);
    LLMCOMMS_API int32_t llmcomms_model_n_layer_nextn(const struct llmcomms_model * model);
    LLMCOMMS_API int32_t llmcomms_model_n_head       (const struct llmcomms_model * model);
    LLMCOMMS_API int32_t llmcomms_model_n_head_kv    (const struct llmcomms_model * model);
    LLMCOMMS_API int32_t llmcomms_model_n_swa        (const struct llmcomms_model * model);

    // Get the model's RoPE frequency scaling factor
    LLMCOMMS_API float llmcomms_model_rope_freq_scale_train(const struct llmcomms_model * model);

    // Returns the number of classifier outputs (only valid for classifier models)
    // Undefined behavior for non-classifier models
    LLMCOMMS_API uint32_t llmcomms_model_n_cls_out(const struct llmcomms_model * model);

    // Returns label of classifier output by index (<n_cls_out). Returns nullptr if no label provided
    LLMCOMMS_API const char * llmcomms_model_cls_label(const struct llmcomms_model * model, uint32_t i);

    LLMCOMMS_API enum llmcomms_vocab_type llmcomms_vocab_type(const struct llmcomms_vocab * vocab);

    LLMCOMMS_API int32_t llmcomms_vocab_n_tokens(const struct llmcomms_vocab * vocab);

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LLMCOMMS_API int32_t llmcomms_model_meta_val_str(const struct llmcomms_model * model, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LLMCOMMS_API int32_t llmcomms_model_meta_count(const struct llmcomms_model * model);

    // Get sampling metadata key name. Returns nullptr if the key is invalid
    LLMCOMMS_API const char * llmcomms_model_meta_key_str(enum llmcomms_model_meta_key key);

    // Get metadata key name by index
    LLMCOMMS_API int32_t llmcomms_model_meta_key_by_index(const struct llmcomms_model * model, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LLMCOMMS_API int32_t llmcomms_model_meta_val_str_by_index(const struct llmcomms_model * model, int32_t i, char * buf, size_t buf_size);

    // Get a string describing the model type
    LLMCOMMS_API int32_t llmcomms_model_desc(const struct llmcomms_model * model, char * buf, size_t buf_size);

    // Get the model file type (quantization), e.g. LLMCOMMS_FTYPE_MOSTLY_Q8_0
    LLMCOMMS_API enum llmcomms_ftype llmcomms_model_ftype(const struct llmcomms_model * model);

    // Returns the total size of all the tensors in the model in bytes
    LLMCOMMS_API uint64_t llmcomms_model_size(const struct llmcomms_model * model);

    // Get the default chat template. Returns nullptr if not available
    // If name is NULL, returns the default chat template
    LLMCOMMS_API const char * llmcomms_model_chat_template(const struct llmcomms_model * model, const char * name);

    // Returns the total number of parameters in the model
    LLMCOMMS_API uint64_t llmcomms_model_n_params(const struct llmcomms_model * model);

    // Returns true if the model contains an encoder that requires llmcomms_encode() call
    LLMCOMMS_API bool llmcomms_model_has_encoder(const struct llmcomms_model * model);

    // Returns true if the model contains a decoder that requires llmcomms_decode() call
    LLMCOMMS_API bool llmcomms_model_has_decoder(const struct llmcomms_model * model);

    // For encoder-decoder models, this function returns id of the token that must be provided
    // to the decoder to start generating output sequence. For other models, it returns -1.
    LLMCOMMS_API llmcomms_token llmcomms_model_decoder_start_token(const struct llmcomms_model * model);

    // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    LLMCOMMS_API bool llmcomms_model_is_recurrent(const struct llmcomms_model * model);

    // Returns true if the model is hybrid (like Jamba, Granite, etc.)
    LLMCOMMS_API bool llmcomms_model_is_hybrid(const struct llmcomms_model * model);

    // Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
    LLMCOMMS_API bool llmcomms_model_is_diffusion(const struct llmcomms_model * model);

    // Returns 0 on success
    LLMCOMMS_API uint32_t llmcomms_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const llmcomms_model_quantize_params * params);

    //
    // Adapters
    //

    // Load a LoRA adapter from file
    // The adapter is valid as long as the associated model is not freed
    LLMCOMMS_API struct llmcomms_adapter_lora * llmcomms_adapter_lora_init(
            struct llmcomms_model * model,
            const char * path_lora);

    // Load a LoRA adapter from an open FILE pointer, reading from its current position
    LLMCOMMS_API struct llmcomms_adapter_lora * llmcomms_adapter_lora_init_from_file_ptr(
            struct llmcomms_model * model,
            FILE * file);

    // Functions to access the adapter's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LLMCOMMS_API int32_t llmcomms_adapter_meta_val_str(const struct llmcomms_adapter_lora * adapter, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LLMCOMMS_API int32_t llmcomms_adapter_meta_count(const struct llmcomms_adapter_lora * adapter);

    // Get metadata key name by index
    LLMCOMMS_API int32_t llmcomms_adapter_meta_key_by_index(const struct llmcomms_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LLMCOMMS_API int32_t llmcomms_adapter_meta_val_str_by_index(const struct llmcomms_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Manually free a LoRA adapter
    // NOTE: loaded adapters that are not manually freed will be freed when the associated model is deleted
    LLMCOMMS_API void llmcomms_adapter_lora_free(struct llmcomms_adapter_lora * adapter);

    // Get the invocation tokens if the current lora is an alora
    LLMCOMMS_API uint64_t            llmcomms_adapter_get_alora_n_invocation_tokens(const struct llmcomms_adapter_lora * adapter);
    LLMCOMMS_API const llmcomms_token * llmcomms_adapter_get_alora_invocation_tokens  (const struct llmcomms_adapter_lora * adapter);

    // The following functions operate on a llmcomms_context, hence the naming: llmcomms_verb_...

    // Set LoRa adapters on the context. Will only modify if the adapters currently in context are different.
    LLMCOMMS_API int32_t llmcomms_set_adapters_lora(
            struct llmcomms_context * ctx,
            struct llmcomms_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales);

    // Apply a loaded control vector to a llmcomms_context, or if data is NULL, clear
    // the currently loaded vector.
    // n_embd should be the size of a single layer's control, and data should point
    // to an n_embd x n_layers buffer starting from layer 1.
    // il_start and il_end are the layer range the vector should apply to (both inclusive)
    // See llmcomms_control_vector_load in common to load a control vector.
    LLMCOMMS_API int32_t llmcomms_set_adapter_cvec(
            struct llmcomms_context * ctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    //
    // Memory
    //

    // Clear the memory contents
    // If data == true, the data buffers will also be cleared together with the metadata
    LLMCOMMS_API void llmcomms_memory_clear(
            llmcomms_memory_t mem,
                      bool data);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence [TAG_LLMCOMMS_SEQ_ID_NEG]
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    LLMCOMMS_API bool llmcomms_memory_seq_rm(
            llmcomms_memory_t mem,
              llmcomms_seq_id seq_id,
                 llmcomms_pos p0,
                 llmcomms_pos p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLMCOMMS_API void llmcomms_memory_seq_cp(
            llmcomms_memory_t mem,
              llmcomms_seq_id seq_id_src,
              llmcomms_seq_id seq_id_dst,
                 llmcomms_pos p0,
                 llmcomms_pos p1);

    // Removes all tokens that do not belong to the specified sequence
    LLMCOMMS_API void llmcomms_memory_seq_keep(
            llmcomms_memory_t mem,
              llmcomms_seq_id seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLMCOMMS_API void llmcomms_memory_seq_add(
            llmcomms_memory_t mem,
              llmcomms_seq_id seq_id,
                 llmcomms_pos p0,
                 llmcomms_pos p1,
                 llmcomms_pos delta);

    // Integer division of the positions by factor of `d > 1`
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLMCOMMS_API void llmcomms_memory_seq_div(
            llmcomms_memory_t mem,
              llmcomms_seq_id seq_id,
                 llmcomms_pos p0,
                 llmcomms_pos p1,
                       int d);

    // Returns the smallest position present in the memory for the specified sequence
    // This is typically non-zero only for SWA caches
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    LLMCOMMS_API llmcomms_pos llmcomms_memory_seq_pos_min(
            llmcomms_memory_t mem,
              llmcomms_seq_id seq_id);

    // Returns the largest position present in the memory for the specified sequence
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    LLMCOMMS_API llmcomms_pos llmcomms_memory_seq_pos_max(
            llmcomms_memory_t mem,
              llmcomms_seq_id seq_id);

    // Check if the memory supports shifting
    LLMCOMMS_API bool llmcomms_memory_can_shift(llmcomms_memory_t mem);

    //
    // State / sessions
    //

    // Returns the *actual* size in bytes of the state
    // (logits, embedding and memory)
    // Only use when saving the state, not when restoring it, otherwise the size may be too small.
    LLMCOMMS_API size_t llmcomms_state_get_size(struct llmcomms_context * ctx);
    LLMCOMMS_API DEPRECATED(size_t llmcomms_get_state_size(struct llmcomms_context * ctx),
        "use llmcomms_state_get_size instead");

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    LLMCOMMS_API size_t llmcomms_state_get_data(
            struct llmcomms_context * ctx,
                         uint8_t * dst,
                          size_t   size);
    LLMCOMMS_API DEPRECATED(size_t llmcomms_copy_state_data(
            struct llmcomms_context * ctx,
                         uint8_t * dst),
        "use llmcomms_state_get_data instead");

    // Set the state reading from the specified address
    // Returns the number of bytes read
    LLMCOMMS_API size_t llmcomms_state_set_data(
            struct llmcomms_context * ctx,
                   const uint8_t * src,
                          size_t   size);
    LLMCOMMS_API DEPRECATED(size_t llmcomms_set_state_data(
            struct llmcomms_context * ctx,
                   const uint8_t * src),
        "use llmcomms_state_set_data instead");

    // Save/load session file
    LLMCOMMS_API bool llmcomms_state_load_file(
            struct llmcomms_context * ctx,
                      const char * path_session,
                     llmcomms_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);
    LLMCOMMS_API DEPRECATED(bool llmcomms_load_session_file(
            struct llmcomms_context * ctx,
                      const char * path_session,
                     llmcomms_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out),
        "use llmcomms_state_load_file instead");

    LLMCOMMS_API bool llmcomms_state_save_file(
            struct llmcomms_context * ctx,
                      const char * path_session,
               const llmcomms_token * tokens,
                          size_t   n_token_count);
    LLMCOMMS_API DEPRECATED(bool llmcomms_save_session_file(
            struct llmcomms_context * ctx,
                      const char * path_session,
               const llmcomms_token * tokens,
                          size_t   n_token_count),
        "use llmcomms_state_save_file instead");

    // Get the exact size needed to copy the state of a single sequence
    LLMCOMMS_API size_t llmcomms_state_seq_get_size(
            struct llmcomms_context * ctx,
                    llmcomms_seq_id   seq_id);

    // Copy the state of a single sequence into the specified buffer
    LLMCOMMS_API size_t llmcomms_state_seq_get_data(
            struct llmcomms_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llmcomms_seq_id   seq_id);

    // Copy the sequence data (originally copied with `llmcomms_state_seq_get_data`) into the specified sequence
    // Returns:
    //  - Positive: Ok
    //  - Zero: Failed to load
    LLMCOMMS_API size_t llmcomms_state_seq_set_data(
            struct llmcomms_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llmcomms_seq_id   dest_seq_id);

    LLMCOMMS_API size_t llmcomms_state_seq_save_file(
            struct llmcomms_context * ctx,
                      const char * filepath,
                    llmcomms_seq_id   seq_id,
               const llmcomms_token * tokens,
                          size_t   n_token_count);

    // If tokens_out is NULL, only the token count is reported through n_token_count_out and no state is loaded
    LLMCOMMS_API size_t llmcomms_state_seq_load_file(
            struct llmcomms_context * ctx,
                      const char * filepath,
                    llmcomms_seq_id   dest_seq_id,
                     llmcomms_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

#define LLMCOMMS_STATE_SEQ_FLAGS_NONE 0

// for backwards-compat
#define LLMCOMMS_STATE_SEQ_FLAGS_SWA_ONLY 1

// work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
#define LLMCOMMS_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

// Keeps the tensor data on device buffers (i.e. not accessible in host memory, but faster save/load).
// Getting the state for a seq_id with this flag invalidates all prior states gotten for that seq_id with this flag.
#define LLMCOMMS_STATE_SEQ_FLAGS_ON_DEVICE 2

    typedef uint32_t llmcomms_state_seq_flags;

    LLMCOMMS_API size_t llmcomms_state_seq_get_size_ext(
            struct llmcomms_context * ctx,
                    llmcomms_seq_id   seq_id,
           llmcomms_state_seq_flags   flags);

    LLMCOMMS_API size_t llmcomms_state_seq_get_data_ext(
            struct llmcomms_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llmcomms_seq_id   seq_id,
           llmcomms_state_seq_flags   flags);

    LLMCOMMS_API size_t llmcomms_state_seq_set_data_ext(
            struct llmcomms_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llmcomms_seq_id   dest_seq_id,
           llmcomms_state_seq_flags   flags);

    //
    // Decoding
    //

    // Return batch for single sequence of tokens
    // The sequence ID will be fixed to 0
    // The position of the tokens will be tracked automatically by llmcomms_decode
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    LLMCOMMS_API struct llmcomms_batch llmcomms_batch_get_one(
                  llmcomms_token * tokens,
                      int32_t   n_tokens);

    // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    // Each token can be assigned up to n_seq_max sequence ids
    // The batch has to be freed with llmcomms_batch_free()
    // If embd != 0, llmcomms_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    // Otherwise, llmcomms_batch.token will be allocated to store n_tokens llmcomms_token
    // The rest of the llmcomms_batch members are allocated with size n_tokens
    // All members are left uninitialized
    LLMCOMMS_API struct llmcomms_batch llmcomms_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    // Frees a batch of tokens allocated with llmcomms_batch_init()
    LLMCOMMS_API void llmcomms_batch_free(struct llmcomms_batch batch);

    // Process a batch of tokens.
    // In contrast to llmcomms_decode() - this call does not use KV cache.
    // For encode-decoder contexts, processes the batch using the encoder.
    // Can store the encoder output internally for later use by the decoder's cross-attention layers.
    //   0 - success
    // < 0 - error. the memory state is restored to the state before this call
    LLMCOMMS_API int32_t llmcomms_encode(
            struct llmcomms_context * ctx,
              struct llmcomms_batch   batch);

    // Process a batch of tokens.
    // Requires the context to have a memory.
    // For encode-decoder contexts, processes the batch using the decoder.
    // Positive return values does not mean a fatal error, but rather a warning.
    // Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the context
    //   To handle this correctly, query the memory state using llmcomms_memory_seq_pos_min() and llmcomms_memory_seq_pos_max()
    // Upon other return values, the memory state is restored to the state before this call
    //    0 - success
    //    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    //    2 - aborted     (processed ubatches will remain in the context's memory)
    //   -1 - invalid input batch
    // < -1 - fatal error (processed ubatches will remain in the context's memory)
    LLMCOMMS_API int32_t llmcomms_decode(
            struct llmcomms_context * ctx,
              struct llmcomms_batch   batch);

    //
    // Extended batch API
    //

    struct llmcomms_batch_ext;

    struct llmcomms_embd {
        const float * data;
        size_t n_rows; // number of embedding rows in data
        size_t n_embd; // size of one row
    };

    LLMCOMMS_API struct llmcomms_batch_ext * llmcomms_batch_ext_init (struct llmcomms_context * ctx);
    LLMCOMMS_API void                     llmcomms_batch_ext_free (struct llmcomms_batch_ext * batch);
    LLMCOMMS_API void                     llmcomms_batch_ext_clear(struct llmcomms_batch_ext * batch);

    // Add an input token to the batch, with default values:
    //     id = LLMCOMMS_TOKEN_NULL
    //     embd = nullptr
    //     pos = not set, the caller must set it with llmcomms_batch_ext_set_pos()
    // Returns the batch index (>= 0)
    // On error:
    //     -1: batch is full
    //     -2: token is invalid (id == LLMCOMMS_TOKEN_NULL or invalid embd)
    //     -3: invalid sequence id
    LLMCOMMS_API int32_t llmcomms_batch_ext_add      (struct llmcomms_batch_ext * batch, llmcomms_seq_id seq_id);

    // Add an input token to the batch, with a specified token ID or token embedding
    LLMCOMMS_API int32_t llmcomms_batch_ext_add_token(struct llmcomms_batch_ext * batch, llmcomms_seq_id seq_id, llmcomms_token id);
    LLMCOMMS_API int32_t llmcomms_batch_ext_add_embd (struct llmcomms_batch_ext * batch, llmcomms_seq_id seq_id, struct llmcomms_embd embd);

    // Add the token at index idx in the batch to another sequence id. The position will stays the same.
    // Note: this should be called before other _set() functions
    LLMCOMMS_API bool llmcomms_batch_ext_add_seq(
                                struct llmcomms_batch_ext * batch,
                                               int32_t   idx,
                                          llmcomms_seq_id   seq_id);

    // Set the token embedding for the token at index idx in the batch
    // use it after llmcomms_batch_ext_add_token() to have an entry with both a token id and an embedding
    LLMCOMMS_API bool llmcomms_batch_ext_set_embd_token(
                                struct llmcomms_batch_ext * batch,
                                               int32_t   idx,
                                     struct llmcomms_embd   embd);

    // Set the "state" embedding for the token at index idx in the batch
    // "state" here means extra hidden state carried over from a previous stage, e.g.:
    //   - MTP: state from N layers of the target model
    //   - Qwen3 VL (deepstack): state from N layers of the vision encoder
    //   - LiliCorr: context embeddings for the transformers
    // Returns false if the context does not take a state embedding (MTP and LiliCorr contexts do)
    LLMCOMMS_API bool llmcomms_batch_ext_set_embd_state(
                                struct llmcomms_batch_ext * batch,
                                               int32_t   idx,
                                     struct llmcomms_embd   embd);

    // Set if output embeddings should be available for the token at index idx in the batch
    // Note: for now, this is equivalent to setting the output logits
    LLMCOMMS_API bool llmcomms_batch_ext_set_output_embd(
                                struct llmcomms_batch_ext * batch,
                                               int32_t  idx,
                                                  bool  value);

    // Set output logits for the token at index idx in the batch
    // Note: for now, this is equivalent to setting the output embd
    LLMCOMMS_API bool llmcomms_batch_ext_set_output_logits(
                                struct llmcomms_batch_ext * batch,
                                               int32_t  idx,
                                                  bool  value);

    // Set custom position for the token at index idx in the batch
    // For M-RoPE models:
    //     - Embedding tokens must have n_pos_per_embd positions per token, in order [t, y, x, z]; t is also the KV cache position
    //     - Text token only requires one single position per token
    LLMCOMMS_API bool llmcomms_batch_ext_set_pos(
                                struct llmcomms_batch_ext * batch,
                                               int32_t   idx,
                                       const llmcomms_pos * pos);

    // TODO: implement get_embeddings() and get_logits() for llmcomms_batch_ext

    // Return values are the same as llmcomms_decode()
    LLMCOMMS_API int32_t llmcomms_process(
                                struct llmcomms_context * ctx,
                             enum llmcomms_process_type   type,
                              struct llmcomms_batch_ext * batch);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    LLMCOMMS_API void llmcomms_set_n_threads(struct llmcomms_context * ctx, int32_t n_threads, int32_t n_threads_batch);

    // Get the number of threads used for generation of a single token.
    LLMCOMMS_API int32_t llmcomms_n_threads(struct llmcomms_context * ctx);

    // Get the number of threads used for prompt and batch processing (multiple token).
    LLMCOMMS_API int32_t llmcomms_n_threads_batch(struct llmcomms_context * ctx);

    // Set whether the context outputs embeddings or not
    // TODO: rename to avoid confusion with llmcomms_get_embeddings()
    LLMCOMMS_API void llmcomms_set_embeddings(struct llmcomms_context * ctx, bool embeddings);

    // Set whether to use causal attention or not
    // If set to true, the model will only attend to the past tokens
    LLMCOMMS_API void llmcomms_set_causal_attn(struct llmcomms_context * ctx, bool causal_attn);

    // Returns whether the context is currently using causal attention
    LLMCOMMS_API bool llmcomms_get_causal_attn(const struct llmcomms_context * ctx);

    // Set whether the model is in warmup mode or not
    // If true, all model tensors are activated during llmcomms_decode() to load and cache their weights.
    //
    // note: using this can cause extra graph reallocations because it changes the graph topology with MoE models,
    //       so it is generally not recommended to use in practice. will be removed in the future
    DEPRECATED(LLMCOMMS_API void llmcomms_set_warmup(struct llmcomms_context * ctx, bool warmup),
            "user code should do warmup runs manually [TAG_LLMCOMMS_GRAPH_NO_WARMUP]");

    // Set abort callback
    LLMCOMMS_API void llmcomms_set_abort_callback(struct llmcomms_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Wait until all computations are finished
    // This is automatically done when using one of the functions below to obtain the computation results
    // and is not necessary to call it explicitly in most cases
    LLMCOMMS_API void llmcomms_synchronize(struct llmcomms_context * ctx);

    // Token logits obtained from the last call to llmcomms_decode()
    // The logits for which llmcomms_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which llmcomms_batch.logits[i] != 0
    // Cols: n_vocab
    // TODO: deprecate in favor of llmcomms_get_logits_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    LLMCOMMS_API float * llmcomms_get_logits(struct llmcomms_context * ctx);

    // Logits for the ith token. For positive indices, Equivalent to:
    // llmcomms_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    // Negative indices can be used to access logits in reverse order, -1 is the last logit.
    // returns NULL for invalid ids.
    LLMCOMMS_API float * llmcomms_get_logits_ith(struct llmcomms_context * ctx, int32_t i);

    // Get all output token embeddings.
    // when pooling_type == LLMCOMMS_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which llmcomms_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    // TODO: deprecate in favor of llmcomms_get_embeddings_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    LLMCOMMS_API float * llmcomms_get_embeddings(struct llmcomms_context * ctx);

    // Get the embeddings for the ith token. For positive indices, Equivalent to:
    // llmcomms_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    // Negative indices can be used to access embeddings in reverse order, -1 is the last embedding.
    // shape: [n_embd] (1-dimensional)
    // returns NULL for invalid ids.
    LLMCOMMS_API float * llmcomms_get_embeddings_ith(struct llmcomms_context * ctx, int32_t i);

    // Get the embeddings for a sequence id
    // Returns NULL if pooling_type is LLMCOMMS_POOLING_TYPE_NONE
    // when pooling_type == LLMCOMMS_POOLING_TYPE_RANK, returns float[n_cls_out] with the rank(s) of the sequence
    // otherwise: float[n_embd] (1-dimensional)
    LLMCOMMS_API float * llmcomms_get_embeddings_seq(struct llmcomms_context * ctx, llmcomms_seq_id seq_id);

    //
    // backend sampling API [EXPERIMENTAL]
    // note: use only if the llmcomms_context was created with at least one llmcomms_sampler_seq_config
    //

    // Get the backend sampled token for the ith token.
    // With multiple outputs, sampler state advances when the token is accepted,
    // not when it is read through this function.
    // When accepting multiple outputs, accept a contiguous prefix in output order.
    // Returns LLMCOMMS_TOKEN_NULL if no token was sampled.
    LLMCOMMS_API llmcomms_token llmcomms_get_sampled_token_ith(struct llmcomms_context * ctx, int32_t i);

    // Get the backend sampled probabilities for the ith token
    // The index matches llmcomms_get_sampled_token_ith().
    // Returns NULL if no probabilities were generated.
    LLMCOMMS_API float *  llmcomms_get_sampled_probs_ith      (struct llmcomms_context * ctx, int32_t i);
    LLMCOMMS_API uint32_t llmcomms_get_sampled_probs_count_ith(struct llmcomms_context * ctx, int32_t i);

    // Get the backend sampled logits for the ith token
    // Returns NULL if no logits were sampled.
    LLMCOMMS_API float *  llmcomms_get_sampled_logits_ith      (struct llmcomms_context * ctx, int32_t i);
    LLMCOMMS_API uint32_t llmcomms_get_sampled_logits_count_ith(struct llmcomms_context * ctx, int32_t i);

    // Get the backend sampled candidates (token ids) for the ith token
    // These are needed to map probability/logit indices to vocab token ids.
    // Returns NULL if no candidates were sampled.
    LLMCOMMS_API llmcomms_token * llmcomms_get_sampled_candidates_ith      (struct llmcomms_context * ctx, int32_t i);
    LLMCOMMS_API uint32_t      llmcomms_get_sampled_candidates_count_ith(struct llmcomms_context * ctx, int32_t i);

    //
    // Vocab
    //

    LLMCOMMS_API const char * llmcomms_vocab_get_text(const struct llmcomms_vocab * vocab, llmcomms_token token);

    LLMCOMMS_API float llmcomms_vocab_get_score(const struct llmcomms_vocab * vocab, llmcomms_token token);

    LLMCOMMS_API enum llmcomms_token_attr llmcomms_vocab_get_attr(const struct llmcomms_vocab * vocab, llmcomms_token token);

    // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    LLMCOMMS_API bool llmcomms_vocab_is_eog(const struct llmcomms_vocab * vocab, llmcomms_token token);

    // Identify if Token Id is a control token or a render-able token
    LLMCOMMS_API bool llmcomms_vocab_is_control(const struct llmcomms_vocab * vocab, llmcomms_token token);

    // Special tokens
    LLMCOMMS_API llmcomms_token llmcomms_vocab_bos(const struct llmcomms_vocab * vocab); // beginning-of-sentence
    LLMCOMMS_API llmcomms_token llmcomms_vocab_eos(const struct llmcomms_vocab * vocab); // end-of-sentence
    LLMCOMMS_API llmcomms_token llmcomms_vocab_eot(const struct llmcomms_vocab * vocab); // end-of-turn
    LLMCOMMS_API llmcomms_token llmcomms_vocab_sep(const struct llmcomms_vocab * vocab); // sentence separator
    LLMCOMMS_API llmcomms_token llmcomms_vocab_nl (const struct llmcomms_vocab * vocab); // next-line
    LLMCOMMS_API llmcomms_token llmcomms_vocab_pad(const struct llmcomms_vocab * vocab); // padding
    LLMCOMMS_API llmcomms_token llmcomms_vocab_mask(const struct llmcomms_vocab * vocab); // mask

    LLMCOMMS_API bool llmcomms_vocab_get_add_bos(const struct llmcomms_vocab * vocab);
    LLMCOMMS_API bool llmcomms_vocab_get_add_eos(const struct llmcomms_vocab * vocab);
    LLMCOMMS_API bool llmcomms_vocab_get_add_sep(const struct llmcomms_vocab * vocab);

    // model-specific suppress tokens (gguf key: tokenizer.ggml.suppress_tokens)
    LLMCOMMS_API const llmcomms_token * llmcomms_vocab_get_suppress_tokens(const struct llmcomms_vocab * vocab, int32_t * n_suppress_tokens);

    LLMCOMMS_API llmcomms_token llmcomms_vocab_fim_pre(const struct llmcomms_vocab * vocab);
    LLMCOMMS_API llmcomms_token llmcomms_vocab_fim_suf(const struct llmcomms_vocab * vocab);
    LLMCOMMS_API llmcomms_token llmcomms_vocab_fim_mid(const struct llmcomms_vocab * vocab);
    LLMCOMMS_API llmcomms_token llmcomms_vocab_fim_pad(const struct llmcomms_vocab * vocab);
    LLMCOMMS_API llmcomms_token llmcomms_vocab_fim_rep(const struct llmcomms_vocab * vocab);
    LLMCOMMS_API llmcomms_token llmcomms_vocab_fim_sep(const struct llmcomms_vocab * vocab);

    DEPRECATED(LLMCOMMS_API const char * llmcomms_token_get_text(const struct llmcomms_vocab * vocab, llmcomms_token token), "use llmcomms_vocab_get_text instead");
    DEPRECATED(LLMCOMMS_API float llmcomms_token_get_score(const struct llmcomms_vocab * vocab, llmcomms_token token), "use llmcomms_vocab_get_score instead");
    DEPRECATED(LLMCOMMS_API enum llmcomms_token_attr llmcomms_token_get_attr(const struct llmcomms_vocab * vocab, llmcomms_token token), "use llmcomms_vocab_get_attr instead");
    DEPRECATED(LLMCOMMS_API bool llmcomms_token_is_eog(const struct llmcomms_vocab * vocab, llmcomms_token token), "use llmcomms_vocab_is_eog instead");
    DEPRECATED(LLMCOMMS_API bool llmcomms_token_is_control(const struct llmcomms_vocab * vocab, llmcomms_token token), "use llmcomms_vocab_is_control instead");
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_token_bos(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_bos instead");
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_token_eos(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_eos instead");
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_token_eot(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_eot instead");
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_token_cls(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_cls instead");
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_token_sep(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_sep instead");
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_token_nl (const struct llmcomms_vocab * vocab), "use llmcomms_vocab_nl instead");
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_token_pad(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_pad instead");
    DEPRECATED(LLMCOMMS_API bool llmcomms_add_bos_token(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_get_add_bos instead");
    DEPRECATED(LLMCOMMS_API bool llmcomms_add_eos_token(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_get_add_eos instead");
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_token_fim_pre(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_fim_pre instead");
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_token_fim_suf(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_fim_suf instead");
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_token_fim_mid(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_fim_mid instead");
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_token_fim_pad(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_fim_pad instead");
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_token_fim_rep(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_fim_rep instead");
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_token_fim_sep(const struct llmcomms_vocab * vocab), "use llmcomms_vocab_fim_sep instead");

    // CLS is equivalent to BOS
    DEPRECATED(LLMCOMMS_API llmcomms_token llmcomms_vocab_cls(const struct llmcomms_vocab * vocab), // classification
            "use llmcomms_vocab_bos instead");

    //
    // Tokenization
    //
    // The API is thread-safe.
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
    /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    ///                      as plaintext. Does not insert a leading space.
    LLMCOMMS_API int32_t llmcomms_tokenize(
        const struct llmcomms_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     llmcomms_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    // @param special If true, special tokens are rendered in the output.
    LLMCOMMS_API int32_t llmcomms_token_to_piece(
              const struct llmcomms_vocab * vocab,
                           llmcomms_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);

    /// @details Convert the provided tokens into text (inverse of llmcomms_tokenize()).
    /// @param text The char pointer must be large enough to hold the resulting text.
    /// @return Returns the number of chars/bytes on success, no more than text_len_max.
    /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    /// @param unparse_special If true, special tokens are rendered in the output.
    LLMCOMMS_API int32_t llmcomms_detokenize(
        const struct llmcomms_vocab * vocab,
               const llmcomms_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

    //
    // Chat templates
    //

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    ///
    /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llmcomms_chat_apply_template
    /// @param tmpl A Jinja template to use for this chat.
    /// @param chat Pointer to a list of multiple llmcomms_chat_message
    /// @param n_msg Number of llmcomms_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    LLMCOMMS_API int32_t llmcomms_chat_apply_template(
                            const char * tmpl,
       const struct llmcomms_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);

    // Get list of built-in chat templates
    LLMCOMMS_API int32_t llmcomms_chat_builtin_templates(const char ** output, size_t len);

    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = llmcomms_sampler_chain_default_params();
    //
    //    llmcomms_sampler * smpl = llmcomms_sampler_chain_init(sparams);
    //
    //    llmcomms_sampler_chain_add(smpl, llmcomms_sampler_init_top_k(50));
    //    llmcomms_sampler_chain_add(smpl, llmcomms_sampler_init_top_p(0.9, 1));
    //    llmcomms_sampler_chain_add(smpl, llmcomms_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    llmcomms_sampler_chain_add(smpl, llmcomms_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        llmcomms_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const llmcomms_token id = llmcomms_sampler_sample(smpl, ctx, -1);
    //
    //        ...
    //    }
    //
    //    llmcomms_sampler_free(smpl);
    //

    typedef void * llmcomms_sampler_context_t;

    struct llmcomms_sampler_data {
        struct ggml_tensor * logits;
        struct ggml_tensor * probs;
        struct ggml_tensor * sampled;
        struct ggml_tensor * candidates;
    };

    // user code can implement the interface below in order to create custom llmcomms_sampler
    struct llmcomms_sampler_i {
        const char *           (*name)  (const struct llmcomms_sampler * smpl);                                 // can be NULL
        void                   (*accept)(      struct llmcomms_sampler * smpl, llmcomms_token token);              // can be NULL
        void                   (*apply) (      struct llmcomms_sampler * smpl, llmcomms_token_data_array * cur_p); // required
        void                   (*reset) (      struct llmcomms_sampler * smpl);                                 // can be NULL
        struct llmcomms_sampler * (*clone) (const struct llmcomms_sampler * smpl);                                 // can be NULL if ctx is NULL
        void                   (*free)  (      struct llmcomms_sampler * smpl);                                 // can be NULL if ctx is NULL

        // [EXPERIMENTAL]
        // backend sampling interface:

        // return true if the backend supports all ops needed by the sampler and can handle up to n_outputs_max_per_seq outputs per sequence
        // note: call once per sampler
        bool (*backend_init)(
                struct llmcomms_sampler       * smpl,
                ggml_backend_buffer_type_t   buft,
                uint32_t                     n_outputs_max_per_seq);

        // call after .backend_apply()
        void (*backend_accept)(
                struct llmcomms_sampler * smpl,
                struct ggml_context  * ctx,
                struct ggml_cgraph   * gf,
                struct ggml_tensor   * selected_token);

        // call after .backend_init()
        void (*backend_apply)(
                struct llmcomms_sampler      * smpl,
                struct ggml_context       * ctx,
                struct ggml_cgraph        * gf,
                struct llmcomms_sampler_data * data);

        // called before graph execution to set inputs for the current ubatch
        void (*backend_set_input)(struct llmcomms_sampler * smpl);

        // called before rebuilding a sampling graph to clear any internal sampler state
        void (*backend_reset)(struct llmcomms_sampler * smpl);

        // copy mutable state from src into dst while keeping dst's references to the current sampling graph
        // src and dst must have the same type and configuration
        void (*copy_state)(const struct llmcomms_sampler * src, struct llmcomms_sampler * dst);
    };

    struct llmcomms_sampler {
        struct llmcomms_sampler_i * iface;

        llmcomms_sampler_context_t ctx;
    };

    // [EXPERIMENTAL]
    // attach a sampler to the context
    // note: prefer initializing the context with llmcomms_context_params.samplers when possible
    LLMCOMMS_API bool llmcomms_set_sampler(struct llmcomms_context * ctx, llmcomms_seq_id seq_id, struct llmcomms_sampler * smpl);

    // mirror of llmcomms_sampler_i:
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init  (      struct llmcomms_sampler_i * iface, llmcomms_sampler_context_t ctx);
    LLMCOMMS_API const char *           llmcomms_sampler_name  (const struct llmcomms_sampler * smpl);
    LLMCOMMS_API void                   llmcomms_sampler_accept(      struct llmcomms_sampler * smpl, llmcomms_token token);
    LLMCOMMS_API void                   llmcomms_sampler_apply (      struct llmcomms_sampler * smpl, llmcomms_token_data_array * cur_p);
    LLMCOMMS_API void                   llmcomms_sampler_reset (      struct llmcomms_sampler * smpl);
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_clone (const struct llmcomms_sampler * smpl);
    LLMCOMMS_API void                   llmcomms_sampler_copy  (const struct llmcomms_sampler * src, struct llmcomms_sampler * dst);
    // important: do not free if the sampler has been added to a llmcomms_sampler_chain (via llmcomms_sampler_chain_add)
    LLMCOMMS_API void                   llmcomms_sampler_free  (      struct llmcomms_sampler * smpl);

    // llmcomms_sampler_chain
    // a type of llmcomms_sampler that can chain multiple samplers one after another

    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_chain_init(struct llmcomms_sampler_chain_params params);

    // important: takes ownership of the sampler object and will free it when llmcomms_sampler_free is called
    LLMCOMMS_API void                   llmcomms_sampler_chain_add(      struct llmcomms_sampler * chain, struct llmcomms_sampler * smpl);

    // return NULL if:
    //   - the sampler is NULL
    //   - the sampler is not a llmcomms_sampler_chain
    //   - the index is out of bounds, unless i == -1
    //   - if i == -1, returns the chain itself (can be used to check if the sampler is a chain)
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_chain_get(      struct llmcomms_sampler * chain, int32_t i);

    // the total number of samplers in the chain
    LLMCOMMS_API int32_t                llmcomms_sampler_chain_n  (const struct llmcomms_sampler * chain);

    // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_chain_remove(   struct llmcomms_sampler * chain, int32_t i);

    // available samplers:

    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_greedy(void);

    /// seed == LLMCOMMS_DEFAULT_SEED to use a random seed.
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_dist(uint32_t seed);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    /// Setting k <= 0 makes this a noop
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_top_k      (int32_t k);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_top_p      (float   p, size_t min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_min_p      (float   p, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_typical    (float   p, size_t min_keep);

    /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_temp       (float   t);

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_top_n_sigma(float   n);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llmcomms_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llmcomms_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_mirostat_v2(
                            uint32_t   seed,
                               float   tau,
                               float   eta);

    /// @details Initializes a GBNF grammar, see grammars/README.md for details.
    /// @param vocab The vocabulary that this grammar will be used with.
    /// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
    /// @param grammar_root The name of the start symbol for the grammar.
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_grammar(
            const struct llmcomms_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root);

    DEPRECATED(LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_grammar_lazy(
            const struct llmcomms_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root,
                         const char ** trigger_words,
                                size_t num_trigger_words,
                   const llmcomms_token * trigger_tokens,
                                size_t num_trigger_tokens),
        "use llmcomms_sampler_init_grammar_lazy_patterns instead");


    /// @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
    /// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    /// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_grammar_lazy_patterns(
        const struct llmcomms_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llmcomms_token * trigger_tokens,
                            size_t num_trigger_tokens);


    /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_penalties(
                             int32_t   n_vocab,
                             int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty)
                               float   penalty_repeat,   // must be > 0.0, 1.0 = disabled
                               float   penalty_freq,     // must be finite, 0.0 = disabled
                               float   penalty_present); // must be finite, 0.0 = disabled

    /// maximum number of distinct DRY sequence breakers, duplicates not counted
    /// processing each breaker scans the entire vocabulary, so the init cost is O(num_breakers x n_vocab)
    #define LLMCOMMS_DRY_MAX_SEQ_BREAKERS 64

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_dry(
            const struct llmcomms_vocab *  vocab,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n, // last n tokens to penalize (0 = disable penalty)
                          const char ** seq_breakers,
                              size_t    num_breakers);

    /// adaptive-p: select tokens near a configurable target probability over time.
    ///
    /// the adaptive-p sampler transforms the token probability distribution to favor tokens
    /// that fall near a user-configurable probability target.
    ///
    /// internally, the sampler maintains an exponential moving average of the *ORIGINAL*
    /// probabilities of selected tokens at each sampling step. it uses this EMA to compute an
    /// adapted target probability at each sampling step, thus maintaining the desired target
    /// probability over time.
    ///
    /// adaptive-p selects a token ID rather than just mutating candidates, so it must be last
    /// in the sampler chain (like mirostat, dist, greedy).
    ///
    /// only mild truncation before this sampler is recommended. we suggest applying min-p
    /// before adaptive-p as the only other active sampler in the chain.
    ///
    /// @param target select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    /// @param decay  EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
    /// @param seed   RNG seed
    ///
    /// ref: https://github.com/ggml-org/llama.cpp/pull/17927
    ///
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_adaptive_p(
                               float   target,
                               float   decay,
                            uint32_t   seed);

    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const llmcomms_logit_bias * logit_bias);

    // this sampler is meant to be used for fill-in-the-middle infilling
    // it's supposed to be used after top_k + top_p sampling
    //
    // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    // 2. combine probs of tokens that have the same prefix
    //
    // example:
    //
    // - before:
    //   "hel":   0.5
    //   "hell":  0.2
    //   "hello": 0.1
    //   "dummy": 0.1
    //
    // - after:
    //   "hel":   0.8
    //   "dummy": 0.1
    //
    // 3. discard non-EOG tokens with low prob
    // 4. if no tokens are left -> pick EOT
    //
    LLMCOMMS_API struct llmcomms_sampler * llmcomms_sampler_init_infill(const struct llmcomms_vocab * vocab);

    // Returns the seed used by the sampler if applicable, LLMCOMMS_DEFAULT_SEED otherwise
    LLMCOMMS_API uint32_t llmcomms_sampler_get_seed(const struct llmcomms_sampler * smpl);

    /// @details Sample and accept a token from the idx-th output of the last evaluation
    // For multiple outputs from one sampler, call this function in output order without gaps.
    //
    // Shorthand for:
    //    const auto * logits = llmcomms_get_logits_ith(ctx, idx);
    //    llmcomms_token_data_array cur_p = { ... init from logits ... };
    //    llmcomms_sampler_apply(smpl, &cur_p);
    //    auto token = cur_p.data[cur_p.selected].id;
    //    llmcomms_sampler_accept(smpl, token);
    //    return token;
    // Returns the sampled token
    LLMCOMMS_API llmcomms_token llmcomms_sampler_sample(struct llmcomms_sampler * smpl, struct llmcomms_context * ctx, int32_t idx);

    // TODO: extend in the future
    //LLMCOMMS_API void llmcomms_decode_with_sampler(struct llmcomms_context * ctx, struct llmcomms_sampler * smpl, struct llmcomms_batch batch, ...);

    //
    // Model split
    //

    /// @details Build a split GGUF final path for this chunk.
    ///          llmcomms_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    LLMCOMMS_API int32_t llmcomms_split_path(char * split_path, size_t maxlen, const char * path_prefix, int32_t split_no, int32_t split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          llmcomms_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    LLMCOMMS_API int32_t llmcomms_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int32_t split_no, int32_t split_count);

    // Print system information
    LLMCOMMS_API const char * llmcomms_print_system_info(void);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    // The logger state is global so these functions are NOT thread safe.
    LLMCOMMS_API void llmcomms_log_get(ggml_log_callback * log_callback, void ** user_data);
    LLMCOMMS_API void llmcomms_log_set(ggml_log_callback   log_callback, void *  user_data);

    //
    // Performance utils
    //
    // NOTE: Used by llama.cpp examples/tools, avoid using in third-party apps. Instead, do your own performance measurements.
    //

    struct llmcomms_perf_context_data {
        // ms == milliseconds
        double t_start_ms;  // absolute start time
        double t_load_ms;   // time needed for loading the model
        double t_p_eval_ms; // time needed for processing the prompt
        double t_eval_ms;   // time needed for generating tokens

        int32_t n_p_eval;   // number of prompt tokens
        int32_t n_eval;     // number of generated tokens
        int32_t n_reused;   // number of times a ggml compute graph had been reused
        int32_t n_reallocs_same; // number of graph reallocations during inference with an unchanged graph size
        int32_t n_reallocs_diff; // number of graph reallocations during inference with a changed graph size
    };

    struct llmcomms_perf_sampler_data {
        double t_sample_ms; // time needed for sampling in ms

        int32_t n_sample;   // number of sampled tokens
    };

    LLMCOMMS_API struct llmcomms_perf_context_data llmcomms_perf_context      (const struct llmcomms_context * ctx);
    LLMCOMMS_API void                           llmcomms_perf_context_print(const struct llmcomms_context * ctx);
    LLMCOMMS_API void                           llmcomms_perf_context_reset(      struct llmcomms_context * ctx);

    // NOTE: the following work only with samplers constructed via llmcomms_sampler_chain_init
    LLMCOMMS_API struct llmcomms_perf_sampler_data llmcomms_perf_sampler      (const struct llmcomms_sampler * chain);
    LLMCOMMS_API void                           llmcomms_perf_sampler_print(const struct llmcomms_sampler * chain);
    LLMCOMMS_API void                           llmcomms_perf_sampler_reset(      struct llmcomms_sampler * chain);

    //
    // training
    //

    // function that returns whether or not a given tensor contains trainable parameters
    typedef bool (*llmcomms_opt_param_filter)(const struct ggml_tensor * tensor, void * userdata);

    // always returns true
    LLMCOMMS_API bool llmcomms_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata);

    struct llmcomms_opt_params {
        uint32_t n_ctx_train; // assumed context size post training, use context size specified in llmcomms_context if 0

        llmcomms_opt_param_filter param_filter; // callback for determining which tensors contain trainable parameters
        void * param_filter_ud;              // userdata for determining which tensors contain trainable parameters

        ggml_opt_get_optimizer_params get_opt_pars; // callback for calculating optimizer parameters
        void * get_opt_pars_ud;                     // userdata for calculating optimizer parameters

        enum ggml_opt_optimizer_type optimizer_type;
    };

    LLMCOMMS_API void llmcomms_opt_init(struct llmcomms_context * lctx, struct llmcomms_model * model, struct llmcomms_opt_params lopt_params);

    LLMCOMMS_API void llmcomms_opt_epoch(
            struct llmcomms_context    * lctx,
            ggml_opt_dataset_t        dataset,
            ggml_opt_result_t         result_train,
            ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml_opt_epoch_callback   callback_train,
            ggml_opt_epoch_callback   callback_eval);

#ifdef __cplusplus
}
#endif

#endif // LLMCOMMS_H
