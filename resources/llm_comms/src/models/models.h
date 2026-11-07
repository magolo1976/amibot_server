#pragma once

#include "llmcomms-model.h"
#include "llmcomms-graph.h"
#include "llmcomms-model-loader.h"

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>
#include <map>

class llmcomms_memory_hybrid_idx_context;

// ref: https://github.com/ggml-org/llama.cpp/pull/28068
static inline ggml_tensor * build_gdn_l2_norm(ggml_context * ctx, ggml_tensor * x, float eps) {
    const float n = x->ne[0];

    return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps/n), 1.0f/sqrtf(n));
}

//
// base classes
//

struct llm_build_mamba_base : public llm_graph_context {
    llm_build_mamba_base(const llm_graph_params & params);

    virtual ~llm_build_mamba_base() = default;

    ggml_tensor * build_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llmcomms_ubatch & ubatch, int il);
    ggml_tensor * build_mamba2_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llmcomms_ubatch & ubatch, int il) const;

};

struct llm_build_delta_net_base : public llm_graph_context {
    llm_build_delta_net_base(const llm_graph_params & params);

    virtual ~llm_build_delta_net_base() = default;

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                int           il);

    // use the ggml_gated_delta_net fused operator (K=1; state has shape [S_v, S_v, H_v, n_seqs])
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_fused(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // choose one of two implementations above based on the number of tokens
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // read conv state from cache, concat with qkv_mixed, write back (single slot or per-token)
    // qkv_mixed: (qkv_dim, n_seq_tokens, n_seqs); returns conv_input: (kernel_size + n_seq_tokens - 1, channels, n_seqs)
    ggml_tensor * build_conv_state(
            llm_graph_input_rs * inp,
            ggml_tensor *        conv_states_all,
            ggml_tensor *        qkv_mixed,
            int64_t              conv_kernel_size,
            int64_t              conv_channels,
            int                  il);

    // run delta-net attention and write the new recurrent state(s) back to ssm_states_all
    // s: (head_v_dim, head_v_dim, num_v_heads, n_seqs); returns output: (head_v_dim, num_v_heads, n_seq_tokens, n_seqs)
    ggml_tensor * build_recurrent_attn(
            llm_graph_input_rs * inp,
            ggml_tensor *        ssm_states_all,
            ggml_tensor *        q,
            ggml_tensor *        k,
            ggml_tensor *        v,
            ggml_tensor *        g,
            ggml_tensor *        b,
            ggml_tensor *        s,
            int                  il);
};

struct llm_build_lfm_base : public llm_graph_context {
    llm_build_lfm_base(const llm_graph_params & params);

    virtual ~llm_build_lfm_base() = default;

    ggml_tensor * build_lfm_shortconv(const llmcomms_memory_recurrent_context * mctx_recr,
                                      llm_graph_input_rs * inp_recr,
                                      ggml_tensor * cur,
                                      int il);

    // no recurrent state (decision graphs): causal over the ubatch from a zero state
    ggml_tensor * build_lfm_shortconv(ggml_tensor * cur,
                                      int il);

    ggml_tensor * build_lfm_dense_ffn(ggml_tensor * cur,
                                      int il);

    ggml_tensor * build_lfm_moe_ffn(ggml_tensor * cur,
                                    int il);
};

struct llm_build_rwkv6_base : public llm_graph_context {
    llm_build_rwkv6_base(const llm_graph_params & params);

    virtual ~llm_build_rwkv6_base() = default;

    ggml_tensor * build_rwkv6_channel_mix(const llmcomms_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;

    ggml_tensor * build_rwkv6_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       const llmcomms_ubatch & ubatch,
                                       int                  il) const;
};

// Base class for RWKV7-related models
struct llm_build_rwkv7_base : public llm_graph_context {
    llm_build_rwkv7_base(const llm_graph_params & params);

    virtual ~llm_build_rwkv7_base() = default;

    // RWKV7-specific graph building methods
    ggml_tensor * build_rwkv7_channel_mix(const llmcomms_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;
    ggml_tensor * build_rwkv7_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       ggml_tensor *&       first_layer_value,
                                       const llmcomms_ubatch & ubatch,
                                       int                  il) const;
};

//
// models
//

struct llmcomms_model_llama : public llmcomms_model_base {
    llmcomms_model_llama(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool embed>
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_llama4 : public llmcomms_model_base {
    llmcomms_model_llama4(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_llmcomms_embed : public llmcomms_model_llama {
    llmcomms_model_llmcomms_embed(const struct llmcomms_model_params & params) : llmcomms_model_llama(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llmcomms_model_llama

    template <bool embed>
    using graph = llmcomms_model_llama::graph<embed>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_maincoder : public llmcomms_model_base {
    llmcomms_model_maincoder(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_maion_coder : public llmcomms_model_base {
    llmcomms_model_maion_coder(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_talkie : public llmcomms_model_base {
    llmcomms_model_talkie(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_deci : public llmcomms_model_base {
    llmcomms_model_deci(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_baichuan : public llmcomms_model_base {
    llmcomms_model_baichuan(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_falcon : public llmcomms_model_base {
    llmcomms_model_falcon(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_grok : public llmcomms_model_base {
    llmcomms_model_grok(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_starcoder : public llmcomms_model_base {
    llmcomms_model_starcoder(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_refact : public llmcomms_model_base {
    llmcomms_model_refact(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_bert : public llmcomms_model_base {
    llmcomms_model_bert(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_jina_bert_v2 : public llmcomms_model_base {
    llmcomms_model_jina_bert_v2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    using graph = llmcomms_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_jina_bert_v3 : public llmcomms_model_base {
    llmcomms_model_jina_bert_v3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    using graph = llmcomms_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_nomic_bert : public llmcomms_model_base {
    llmcomms_model_nomic_bert(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    using graph = llmcomms_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_nomic_bert_moe : public llmcomms_model_base {
    llmcomms_model_nomic_bert_moe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    using graph = llmcomms_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_modern_bert : public llmcomms_model_base {
    llmcomms_model_modern_bert(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

        ggml_tensor * build_decision_head(
                ggml_tensor * inp,
                llm_graph_input_attn_no_cache * inp_attn,
                ggml_tensor * inp_out_ids);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_neo_bert : public llmcomms_model_base {
    llmcomms_model_neo_bert(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_eurobert : public llmcomms_model_base {
    llmcomms_model_eurobert(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_bloom : public llmcomms_model_base {
    llmcomms_model_bloom(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// Quant-only stub for mmproj GGUFs
// none of these are ever called, they only exist to satisfy the llmcomms_model_base interface
struct llmcomms_model_clip : public llmcomms_model_base {
    llmcomms_model_clip(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}

    [[noreturn]]
    void load_arch_hparams(llmcomms_model_loader & ml) override;

    [[noreturn]]
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    [[noreturn]]
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_mpt : public llmcomms_model_base {
    llmcomms_model_mpt(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_stablelm : public llmcomms_model_base {
    llmcomms_model_stablelm(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_mellum : public llmcomms_model_base {
    llmcomms_model_mellum(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    template <bool iswa>
    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_nanbeige : public llmcomms_model_base {
    llmcomms_model_nanbeige(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    int  n_loops = 1;
    int  n_layer_phys = 0;
    bool skip_loop_final_norm = false;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_qwen : public llmcomms_model_base {
    llmcomms_model_qwen(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_qwen2 : public llmcomms_model_base {
    llmcomms_model_qwen2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_dream : public llmcomms_model_base {
    llmcomms_model_dream(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_llada : public llmcomms_model_base {
    llmcomms_model_llada(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_llada_moe : public llmcomms_model_base {
    llmcomms_model_llada_moe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_rnd1 : public llmcomms_model_base {
    llmcomms_model_rnd1(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_qwen2vl : public llmcomms_model_base {
    llmcomms_model_qwen2vl(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_qwen2moe : public llmcomms_model_base {
    llmcomms_model_qwen2moe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_sensenova_u1 : public llmcomms_model_base {
    llmcomms_model_sensenova_u1(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    float rope_freq_base_spatial = 10000.0f;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_qwen3 : public llmcomms_model_base {
    llmcomms_model_qwen3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    protected:
        // the helpers alone, a derived graph builds its own body
        struct no_build {};
        graph(const llm_graph_params & params, no_build) : llm_graph_context(params) {}

        // the layers and the final norm: [n_embd, n_outputs]
        ggml_tensor * build_backbone();
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// Qwen3 with the decision head of vllm-sr Decision 2.0: one score per option
struct llmcomms_model_vllm_decision2 : public llmcomms_model_qwen3 {
    llmcomms_model_vllm_decision2(const struct llmcomms_model_params & params) : llmcomms_model_qwen3(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    ggml_tensor * option_norm     = nullptr;
    ggml_tensor * option_norm_b   = nullptr;
    ggml_tensor * query_norm      = nullptr;
    ggml_tensor * query_norm_b    = nullptr;
    ggml_tensor * key             = nullptr;
    ggml_tensor * query           = nullptr;
    ggml_tensor * option_mlp      = nullptr;
    ggml_tensor * option_mlp_b    = nullptr;
    ggml_tensor * query_mlp       = nullptr;
    ggml_tensor * scorer_out      = nullptr;

    class input_decision;

    struct graph : public llmcomms_model_qwen3::graph {
        graph(const llm_graph_params & params);
    private:
        // hidden: [n_embd, n_outputs], returns one score per option: [1, n_options]
        ggml_tensor * build_head(ggml_tensor * hidden, input_decision * inp);

        const llmcomms_model_vllm_decision2 & model;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_qwen3moe : public llmcomms_model_base {
    llmcomms_model_qwen3moe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_flint : public llmcomms_model_base {
    llmcomms_model_flint(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_qwen3vl : public llmcomms_model_base {
    llmcomms_model_qwen3vl(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_qwen3vlmoe : public llmcomms_model_base {
    llmcomms_model_qwen3vlmoe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_qwen3tts : public llmcomms_model_qwen3vl {
    llmcomms_model_qwen3tts(const struct llmcomms_model_params & params) : llmcomms_model_qwen3vl(params) {}
};

struct llmcomms_model_phi2 : public llmcomms_model_base {
    llmcomms_model_phi2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_phi3 : public llmcomms_model_base {
    llmcomms_model_phi3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_phimoe : public llmcomms_model_base {
    llmcomms_model_phimoe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool iswa>
    using graph = llmcomms_model_phi3::graph<iswa>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_plamo : public llmcomms_model_base {
    llmcomms_model_plamo(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_plamo2 : public llmcomms_model_base {
    llmcomms_model_plamo2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llm_graph_params & params);
        private:
            ggml_tensor * build_plamo2_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llmcomms_ubatch & ubatch, int il);
            ggml_tensor * build_plamo2_attn_layer(llm_graph_input_attn_kv * inp, ggml_tensor * inp_pos, ggml_tensor * cur,
                                                    int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_plamo3 : public llmcomms_model_base {
    llmcomms_model_plamo3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_gpt2 : public llmcomms_model_base {
    llmcomms_model_gpt2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_pockettts : public llmcomms_model_base {
    llmcomms_model_pockettts(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_codeshell : public llmcomms_model_base {
    llmcomms_model_codeshell(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_orion : public llmcomms_model_base {
    llmcomms_model_orion(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_internlm2 : public llmcomms_model_base {
    llmcomms_model_internlm2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_minicpm3 : public llmcomms_model_base {
    llmcomms_model_minicpm3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_gemma : public llmcomms_model_base {
    llmcomms_model_gemma(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_gemma2 : public llmcomms_model_base {
    llmcomms_model_gemma2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_gemma3 : public llmcomms_model_base {
    llmcomms_model_gemma3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_gemma3n : public llmcomms_model_base {
    llmcomms_model_gemma3n(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        const int64_t n_embd_head;
        const int64_t n_embd_altup;
        const int64_t n_altup;
        const int     i_altup_act;
        const int     n_layer_sparsity = 10; // number of layers using activation sparsity
        const float   f_sparsity_std_mul = 1.6448533535003662f; // std_multiplier = normal_dist.icdf(0.95)

        graph(const llm_graph_params & params);
        ggml_tensor * calc_magnitude(ggml_tensor * x);

        // TODO: refactor in common "per-layer" functionality [TAG_PER_LAYER]
        ggml_tensor * build_inp_per_layer();
        ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);

        ggml_tensor * gaussian_topk(ggml_tensor * x);
        ggml_tensor * altup_compute_router_modalities(ggml_tensor * x, int il);
        ggml_tensor * altup_predict(ggml_tensor * cur, int il);
        ggml_tensor * laurel(ggml_tensor * cur, int il);
        ggml_tensor * altup_correct(ggml_tensor * predictions, ggml_tensor * activated, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_gemma4 : public llmcomms_model_base {
    llmcomms_model_gemma4(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        const int64_t n_embd_per_layer;

        graph(const llm_graph_params & params);

        // TODO: refactor in common "per-layer" functionality [TAG_PER_LAYER]
        ggml_tensor * build_inp_per_layer();
        ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_gemma4_assistant : public llmcomms_model_base {
    llmcomms_model_gemma4_assistant(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_gemma_embedding : public llmcomms_model_base {
    llmcomms_model_gemma_embedding(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_gemma_embedding2 : public llmcomms_model_base {
    llmcomms_model_gemma_embedding2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

        ggml_tensor * build_inp_per_layer(ggml_tensor * inpL);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_starcoder2 : public llmcomms_model_base {
    llmcomms_model_starcoder2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_mamba : public llmcomms_model_base {
    llmcomms_model_mamba(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_mamba2 : public llmcomms_model_base {
    llmcomms_model_mamba2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    using graph = llmcomms_model_mamba::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_maple : public llmcomms_model_base {
    llmcomms_model_maple(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_jamba : public llmcomms_model_base {
    llmcomms_model_jamba(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_xverse : public llmcomms_model_base {
    llmcomms_model_xverse(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_command_r : public llmcomms_model_base {
    llmcomms_model_command_r(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_cohere2 : public llmcomms_model_base {
    llmcomms_model_cohere2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_cohere2moe : public llmcomms_model_base {
    llmcomms_model_cohere2moe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_dbrx : public llmcomms_model_base {
    llmcomms_model_dbrx(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_olmo : public llmcomms_model_base {
    llmcomms_model_olmo(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_olmo2 : public llmcomms_model_base {
    llmcomms_model_olmo2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_olmoe : public llmcomms_model_base {
    llmcomms_model_olmoe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_muse_spark : public llmcomms_model_base {
    llmcomms_model_muse_spark(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_muse_glimmer : public llmcomms_model_base {
    llmcomms_model_muse_glimmer(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_openelm : public llmcomms_model_base {
    llmcomms_model_openelm(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_gptneox : public llmcomms_model_base {
    llmcomms_model_gptneox(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_arctic : public llmcomms_model_base {
    llmcomms_model_arctic(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_deepseek : public llmcomms_model_base {
    llmcomms_model_deepseek(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_deepseek2 : public llmcomms_model_base {
    llmcomms_model_deepseek2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_deepseek32 : public llmcomms_model_base {
    llmcomms_model_deepseek32(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_dots3note : public llmcomms_model_base {
    llmcomms_model_dots3note(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// YaRN attention factor that cancels ggml's rope mscale (the DeepSeek-V4 reference applies none)
static inline float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

struct llmcomms_model_deepseek4 : public llmcomms_model_base {
    llmcomms_model_deepseek4(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        struct no_build {};
        graph(const llm_graph_params & params, no_build) : llm_graph_context(params) {}
        graph(const llm_graph_params & params);

        // collapse the hc streams with per-stream weights
        ggml_tensor * build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * weights,
                int il) const;

        // mean over the hyper-connection streams: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
        ggml_tensor * build_hc_mean(ggml_tensor * x) const;

        // all outputs of the hc mix builders:
        //   build_hc_pre   - cur is the collapsed input, post / comb feed build_hc_post
        //   build_hc_mixes - pre is the pre weights, post / comb feed build_hc_post
        struct hc_pre {
            ggml_tensor * cur;
            ggml_tensor * post;
            ggml_tensor * comb;
        };

        struct hc_mixes {
            ggml_tensor * pre;
            ggml_tensor * post;
            ggml_tensor * comb;
        };

        hc_pre build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base,
                int il) const;

        hc_mixes build_hc_mixes(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base,
                int il) const;

        ggml_tensor * build_hc_post(
                ggml_tensor * x,
                ggml_tensor * residual,
                ggml_tensor * post,
                ggml_tensor * comb,
                int il) const;

        ggml_tensor * build_hc_sinkhorn(
                ggml_tensor * comb,
                int il) const;

        ggml_tensor * build_hc_head(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base) const;

        ggml_tensor * build_attention(
                llm_graph_input_dsv4 * inp_dsv4,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_attention(
                llm_graph_input_attn_k_iswa * inp_mtp,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_attention_impl(
                llm_graph_input_dsv4 * inp_dsv4,
                llm_graph_input_attn_k_iswa * inp_mtp,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_hca_compressed_kv_from_state(
                ggml_tensor * kv_state,
                ggml_tensor * score_state,
                ggml_tensor * state_read_idxs,
                ggml_tensor * comp_pos,
                ggml_tensor * norm,
                int64_t n_embd_head,
                const char * name,
                int il) const;

        ggml_tensor * build_overlap_compressed_kv_from_state(
                ggml_tensor * kv_state,
                ggml_tensor * score_state,
                ggml_tensor * state_read_idxs,
                ggml_tensor * comp_pos,
                ggml_tensor * norm,
                int64_t ratio,
                int64_t n_embd_head,
                const char * name,
                int il) const;

        ggml_tensor * build_lid_top_k(
                llm_graph_input_dsv4 * inp_dsv4,
                ggml_tensor * qr,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_top_k_mask(
                ggml_tensor * kq_mask,
                ggml_tensor * top_k,
                const char * name,
                int il) const;

        ggml_tensor * build_csa_lid_attention(
                llm_graph_input_dsv4 * inp_dsv4,
                llm_graph_input_dsv4_raw * inp_attn,
                ggml_tensor * q,
                ggml_tensor * kv,
                ggml_tensor * qr,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                ggml_tensor * sinks,
                float kq_scale,
                int il) const;

        ggml_tensor * build_hca_attention(
                llm_graph_input_dsv4 * inp_dsv4,
                llm_graph_input_dsv4_raw * inp_attn,
                ggml_tensor * q,
                ggml_tensor * kv,
                ggml_tensor * sinks,
                float kq_scale,
                int il) const;

        ggml_tensor * build_raw_attention(
                llm_graph_input_dsv4_raw * inp_attn,
                ggml_tensor * q,
                ggml_tensor * kv,
                ggml_tensor * sinks,
                float kq_scale,
                int il) const;

    };

    struct graph_mtp : public graph {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llmcomms_model_xing4_0 : public llmcomms_model_base {
    llmcomms_model_xing4_0(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llmcomms_model & model, const llm_graph_params & params);

        ggml_tensor * build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base,
                ggml_tensor ** post,
                ggml_tensor ** comb,
                int il) const;

        ggml_tensor * build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * weights,
                int il) const;

        ggml_tensor * build_hc_sinkhorn(
                ggml_tensor * comb,
                int il) const;

        ggml_tensor * build_hc_post(
                ggml_tensor * x,
                ggml_tensor * residual,
                ggml_tensor * post,
                ggml_tensor * comb,
                int il) const;
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llmcomms_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llmcomms_model_deepseek41 : public llmcomms_model_deepseek4 {
    llmcomms_model_deepseek41(const struct llmcomms_model_params & params) : llmcomms_model_deepseek4(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    // Engram: the hash constants are hparams.ple_* (one module per Engram layer), hashed on the host (ggml has no int64 mul/xor/mod)
    int engram_index(int il) const;

    void engram_hash(int e, const llmcomms_ubatch & ubatch, const std::vector<llmcomms_token> & prev, std::vector<int32_t> & out) const;

    struct graph : public llmcomms_model_deepseek4::graph {
        graph(const llm_graph_params & params);
        // helpers only (no graph built): base of graph_dsv41
        graph(const llm_graph_params & params, no_build) : llmcomms_model_deepseek4::graph(params, no_build{}) {}

        ggml_tensor * build_engram(
                const llmcomms_model_deepseek41 & model,
                ggml_tensor * x,
                ggml_tensor * rows,
                ggml_tensor * keep,
                int il) const;

        ggml_tensor * build_attention41(
                const llmcomms_model_deepseek41 & model,
                llm_graph_input_dsv4 * inp_dsv4,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                ggml_tensor * cand_pin,
                int il);

        std::vector<ggml_tensor *> topk_of; // top-k of each index source layer
        std::vector<ggml_tensor *> comp_row_of; // [n_layer] new compressed rows of each kv source (for mirrors)
        ggml_tensor * cand_sel    = nullptr; // candidate blocks of the candidate source layer
        int64_t       cand_n_comp = 0;       // compressed rows the candidate source scored

        // sparse path (LLMCOMMS_DSV41_SPARSE=0 disables): fused indexer, FA bounded to window + top-k, decode gather
        bool sparse = true;
        // gather path, shared by layers with the same top-k: selected K rows per (storage layer, top-k source),
        // the sorted top-k and the mask per top-k source (one input copy per split)
        std::map<std::pair<int32_t, int32_t>, ggml_tensor *> gather_k_of;
        std::map<int32_t, ggml_tensor *> gather_sel_of;
        std::map<int32_t, ggml_tensor *> gather_mask_of;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_deepseek2ocr : public llmcomms_model_base {
    llmcomms_model_deepseek2ocr(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    using graph = llmcomms_model_deepseek2::graph;

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_glm_dsa : public llmcomms_model_base {
    llmcomms_model_glm_dsa(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params_indexer(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_eagle3 : public llmcomms_model_base {
    llmcomms_model_eagle3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

        ggml_tensor * build_inp_embd_enc() const;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_dflash : public llmcomms_model_base {
    llmcomms_model_dflash(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

        ggml_tensor * build_inp_embd_enc() const;
    };

    struct graph_dsv4 : public llmcomms_model_deepseek4::graph {
        graph_dsv4(const llm_graph_params & params);
    };

    // DeepSeek-V4.1 DSpark: single-pass mHC (lagged pre-mix, no hc_head), no per-head q norm
    struct graph_dsv41 : public llmcomms_model_deepseek41::graph {
        graph_dsv41(const llm_graph_params & params);
    };

    // Mistral Large 4 DSpark: dense MLA stages, non-causal, rnn markov head
    struct graph_mistral_large4 : public llm_graph_context {
        graph_mistral_large4(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_mistral_large4 : public llmcomms_model_base {
    llmcomms_model_mistral_large4(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params_indexer(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_mistral4 : public llmcomms_model_deepseek2 {
    llmcomms_model_mistral4(const struct llmcomms_model_params & params) : llmcomms_model_deepseek2(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llmcomms_model_deepseek2

    using graph = llmcomms_model_deepseek2::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_chatglm : public llmcomms_model_base {
    llmcomms_model_chatglm(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_glm4 : public llmcomms_model_base {
    llmcomms_model_glm4(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_glm4_moe : public llmcomms_model_base {
    llmcomms_model_glm4_moe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_bitnet : public llmcomms_model_base {
    llmcomms_model_bitnet(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_t5 : public llmcomms_model_base {
    llmcomms_model_t5(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_t5encoder : public llmcomms_model_base {
    llmcomms_model_t5encoder(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    using graph = llmcomms_model_t5::graph<true>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_jais : public llmcomms_model_base {
    llmcomms_model_jais(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_jais2 : public llmcomms_model_base {
    llmcomms_model_jais2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_nemotron : public llmcomms_model_base {
    llmcomms_model_nemotron(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_nemotron_h : public llmcomms_model_base {
    llmcomms_model_nemotron_h(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llm_graph_params & params);
        ggml_tensor * build_ffn_layer(ggml_tensor * cur, int il);
        ggml_tensor * build_attention_layer(ggml_tensor * cur, llm_graph_input_attn_kv * inp_attn,
            int64_t n_embd_head, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_nemotron_h_moe : public llmcomms_model_nemotron_h {
    llmcomms_model_nemotron_h_moe(const struct llmcomms_model_params & params) : llmcomms_model_nemotron_h(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llmcomms_model_nemotron_h

    using graph = llmcomms_model_nemotron_h::graph;

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_metis16 : public llmcomms_model_base {
    llmcomms_model_metis16(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    uint32_t n_pass       = 1;
    uint32_t n_stream     = 1;
    uint32_t n_mem        = 0;
    uint32_t n_route      = 0;
    uint32_t n_layer_phys = 0;
    float    cont_limit   = 0.0f;

    struct graph : public llm_build_mamba_base {
        graph(const llm_graph_params & params);

        // memory bank of the current forward: keys [n_mem, n_entry, n_tokens], values [n_entry, n_mem, n_tokens]
        ggml_tensor * bank_k  = nullptr;
        ggml_tensor * bank_vt = nullptr;

        ggml_tensor * stream_mean(ggml_tensor * x) const;
        ggml_tensor * hc_read(ggml_tensor * x, ggml_tensor * mix, ggml_tensor * read, ggml_tensor * norm, int p, ggml_tensor ** mixed, int il) const;
        ggml_tensor * hc_write(ggml_tensor * mixed, ggml_tensor * write, ggml_tensor * update, int p) const;
        ggml_tensor * mem_read(const llmcomms_model_metis16 & model, ggml_tensor ** x) const;
        void          mem_write(const llmcomms_model_metis16 & model, ggml_tensor * x, ggml_tensor * summary,
                                ggml_tensor * entropy, ggml_tensor * confidence, ggml_tensor * cont, int entry);
        ggml_tensor * route_features(const llmcomms_model_metis16 & model, ggml_tensor * state, ggml_tensor * summary,
                                     ggml_tensor * diff, ggml_tensor * history) const;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_exaone : public llmcomms_model_base {
    llmcomms_model_exaone(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_exaone4 : public llmcomms_model_base {
    llmcomms_model_exaone4(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_exaone_moe : public llmcomms_model_base {
    llmcomms_model_exaone_moe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_rwkv6 : public llmcomms_model_base {
    llmcomms_model_rwkv6(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_rwkv6_base {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_rwkv6qwen2 : public llmcomms_model_base {
    llmcomms_model_rwkv6qwen2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_rwkv6_base {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_rwkv7 : public llmcomms_model_base {
    llmcomms_model_rwkv7(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_rwkv7_base {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_arwkv7 : public llmcomms_model_base {
    llmcomms_model_arwkv7(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_rwkv7_base {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_granite : public llmcomms_model_base {
    llmcomms_model_granite(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

    private:
        ggml_tensor * build_attention_layer(
                  ggml_tensor             * cur,
                  ggml_tensor             * inp_pos,
                  llm_graph_input_attn_kv * inp_attn,
            const int64_t                 n_embd_head,
            const int                     il);

        ggml_tensor * build_layer_ffn(
                  ggml_tensor       * cur,
                  ggml_tensor       * inpSA,
            const int                 il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_granite_moe : public llmcomms_model_base {
    llmcomms_model_granite_moe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    using graph = llmcomms_model_granite::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_granite_switch : public llmcomms_model_base {
    llmcomms_model_granite_switch(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    uint32_t n_adapters    = 0;
    uint32_t max_lora_rank = 0;
    float    router_gain   = 15.0f;

    std::unordered_map<llmcomms_token, int32_t>     adapter_token_to_slot;
    std::unordered_map<llmcomms_token, llmcomms_token> adapter_token_to_substitute;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

    private:
        ggml_tensor * build_switched_lora_delta(
                  ggml_tensor * lora_a,
                  ggml_tensor * lora_b,
                  ggml_tensor * cur,
                  ggml_tensor * ids);

        ggml_tensor * build_switched_lora_mm(
                  ggml_tensor * w,
                  ggml_tensor * lora_a,
                  ggml_tensor * lora_b,
                  ggml_tensor * cur,
                  ggml_tensor * ids);

        ggml_tensor * build_attention_layer(
                  ggml_tensor             * cur,
                  ggml_tensor             * inp_pos,
                  ggml_tensor             * adapter_ids,
                  llm_graph_input_attn_kv * inp_attn,
            const int64_t                 n_embd_head,
            const int                     il);

        ggml_tensor * build_layer_ffn(
                  ggml_tensor       * cur,
                  ggml_tensor       * inpSA,
                  ggml_tensor       * adapter_ids,
            const int                 il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_minicpm : public llmcomms_model_base {
    llmcomms_model_minicpm(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    using graph = llmcomms_model_granite::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_granite_hybrid : public llmcomms_model_base {
    llmcomms_model_granite_hybrid(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llm_graph_params & params);
        ggml_tensor * build_layer_ffn(ggml_tensor * cur, ggml_tensor * inpSA, const int il);
        ggml_tensor * build_attention_layer(ggml_tensor * cur, ggml_tensor * inp_pos, llm_graph_input_attn_kv * inp_attn,
            const int64_t n_embd_head, const int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_granite_swa : public llmcomms_model_base {
    llmcomms_model_granite_swa(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

    private:
        ggml_tensor * build_attention_layer(
                  ggml_tensor                  * cur,
                  ggml_tensor                  * inp_pos,
                  llm_graph_input_attn_kv_iswa * inp_attn,
            const int64_t                        n_embd_head,
            const int                            il);

        ggml_tensor * build_layer_ffn(
                  ggml_tensor * cur,
                  ggml_tensor * inpSA,
            const int           il);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_chameleon : public llmcomms_model_base {
    llmcomms_model_chameleon(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_wavtokenizer_dec : public llmcomms_model_base {
    llmcomms_model_wavtokenizer_dec(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_plm : public llmcomms_model_base {
    llmcomms_model_plm(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_limite : public llmcomms_model_base {
    llmcomms_model_limite(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_hrm_text : public llmcomms_model_base {
    llmcomms_model_hrm_text(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

        ggml_tensor * build_stack(
                llm_graph_input_attn_kv * inp_attn,
                        ggml_tensor * inp_pos,
                        ggml_tensor * cur,
                                int   slot_base) const;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_bailingmoe : public llmcomms_model_base {
    llmcomms_model_bailingmoe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_bailingmoe2 : public llmcomms_model_base {
    llmcomms_model_bailingmoe2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_bailingmoe3 : public llmcomms_model_base {
    llmcomms_model_bailingmoe3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);

        // causal conv1d over one of Q/K/V, qkv selects which third of the conv state row is used
        ggml_tensor * build_causal_conv1d(llm_graph_input_rs * inp_rs, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
                                          int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                                          int64_t d_conv, int64_t head_dim, int64_t n_head, int64_t n_seq_tokens, int64_t n_seqs);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_seed_oss : public llmcomms_model_base {
    llmcomms_model_seed_oss(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_dots1 : public llmcomms_model_base {
    llmcomms_model_dots1(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_arcee : public llmcomms_model_base {
    llmcomms_model_arcee(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_afmoe : public llmcomms_model_base {
    llmcomms_model_afmoe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_beam : public llmcomms_model_base {
    llmcomms_model_beam(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llmcomms_model_ricursive : public llmcomms_model_base {
    llmcomms_model_ricursive(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_laguna : public llmcomms_model_base {
    llmcomms_model_laguna(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_ernie4_5 : public llmcomms_model_base {
    llmcomms_model_ernie4_5(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_ernie4_5_moe : public llmcomms_model_ernie4_5 {
    llmcomms_model_ernie4_5_moe(const struct llmcomms_model_params & params) : llmcomms_model_ernie4_5(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llmcomms_model_ernie4_5

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_paddleocr : public llmcomms_model_ernie4_5 {
    llmcomms_model_paddleocr(const struct llmcomms_model_params & params) : llmcomms_model_ernie4_5(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llmcomms_model_ernie4_5

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_hunyuan_moe : public llmcomms_model_base {
    llmcomms_model_hunyuan_moe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_hy_v3 : public llmcomms_model_base {
    llmcomms_model_hy_v3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_hy_v4 : public llmcomms_model_base {
    llmcomms_model_hy_v4(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

        // iHC (independent Hyper-Connections): pre reduces the hc streams to one and returns the
        // per-stream post gates, post writes the sublayer output back into the streams, head
        // collapses the streams before the final norm.
        struct hc_pre {
            ggml_tensor * cur;   // the collapsed stream
            ggml_tensor * post;  // per-stream post gates, feeds build_hc_post
        };

        hc_pre build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base,
                int il) const;

        ggml_tensor * build_hc_post(
                ggml_tensor * x,
                ggml_tensor * residual,
                ggml_tensor * post,
                int il) const;

        ggml_tensor * build_hc_head(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base) const;

        ggml_tensor * build_attention(
                llm_graph_input_attn_k * inp_attn,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                float kq_scale,
                int il) const;

        // DSA lightning indexer: top-k KV positions for this layer. Only "full" layers compute
        // it, "shared" layers reuse the last preceding full layer result through last_top_k.
        ggml_tensor * build_indexer_top_k(
                llm_graph_input_attn_k_dsa * inp_attn_dsa,
                ggml_tensor * cur,
                ggml_tensor * qr,
                ggml_tensor * inp_pos,
                int il) const;

        // the attention output and the top-k to carry to the following "shared" layers
        struct attn_dsa {
            ggml_tensor * cur;
            ggml_tensor * top_k;
        };

        attn_dsa build_attention_dsa(
                llm_graph_input_attn_k_dsa * inp_attn_dsa,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                ggml_tensor * last_top_k,
                float kq_scale,
                int il) const;
    };

    llmcomms_rope_params get_rope_params_indexer(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_hunyuan_vl : public llmcomms_model_base {
    llmcomms_model_hunyuan_vl(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_hunyuan_dense : public llmcomms_model_hunyuan_vl {
    llmcomms_model_hunyuan_dense(const struct llmcomms_model_params & params) : llmcomms_model_hunyuan_vl(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llmcomms_model_hunyuan_vl

    using graph = llmcomms_model_hunyuan_vl::graph;

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_smollm3 : public llmcomms_model_base {
    llmcomms_model_smollm3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_openai_moe : public llmcomms_model_base {
    llmcomms_model_openai_moe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_falcon_h1 : public llmcomms_model_base {
    llmcomms_model_falcon_h1(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_lfm2 : public llmcomms_model_base {
    llmcomms_model_lfm2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_build_lfm_base {
        graph(const llm_graph_params & params);
    };

    // non-causal trunk without memory, then the decision head
    struct graph_decision : public llm_graph_context {
        graph_decision(const llm_graph_params & params);

        ggml_tensor * build_decision_head(
                ggml_tensor * inp,
                llm_graph_input_attn_no_cache * inp_attn,
                ggml_tensor * inp_out_ids);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_lfm2moe : public llmcomms_model_base {
    llmcomms_model_lfm2moe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool iswa>
    using graph = llmcomms_model_lfm2::graph<iswa>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_lfm3 : public llmcomms_model_base {
    llmcomms_model_lfm3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    // decision: d1, the prompt read in one batch, no memory
    template <bool decision>
    struct graph : public llm_build_lfm_base {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_smallthinker : public llmcomms_model_base {
    llmcomms_model_smallthinker(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_grovemoe : public llmcomms_model_base {
    llmcomms_model_grovemoe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_apertus : public llmcomms_model_base {
    llmcomms_model_apertus(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_minimax_01 : public llmcomms_model_base {
    llmcomms_model_minimax_01(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_minimax_m2 : public llmcomms_model_base {
    llmcomms_model_minimax_m2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct msa_params {
    int blk;
    int topk_blocks;
    int local;
};

struct llmcomms_model_minimax_m3 : public llmcomms_model_base {
    llmcomms_model_minimax_m3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;
    msa_params msa_p;
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

        ggml_tensor * build_attn_msa_fa(
                ggml_tensor * q_cur,   // [D, HQ, S] f32
                ggml_tensor * k,       // [D, n_keys, 1, C]  C = HKV or HKV*n_stream
                ggml_tensor * v,       // [D, n_keys, 1, C]
                ggml_tensor * mask,    // [n_keys, R, 1, C] f16, R = HQ*T/(Gp*C)
                int64_t Gp, float kq_scale, int il) const;
    };
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_cogvlm : public llmcomms_model_base {
    llmcomms_model_cogvlm(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_pangu_embed : public llmcomms_model_base {
    llmcomms_model_pangu_embed(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_qwen3next : public llmcomms_model_base {
    llmcomms_model_qwen3next(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_qwen35 : public llmcomms_model_base {
    llmcomms_model_qwen35(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// Qwen3.5 backbone evaluated without memory, with a joint decision head on top of its hidden states
struct llmcomms_model_clef : public llmcomms_model_qwen35 {
    llmcomms_model_clef(const struct llmcomms_model_params & params) : llmcomms_model_qwen35(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct attn {
        ggml_tensor * wq = nullptr;
        ggml_tensor * bq = nullptr;
        ggml_tensor * wk = nullptr;
        ggml_tensor * bk = nullptr;
        ggml_tensor * wv = nullptr;
        ggml_tensor * bv = nullptr;
        ggml_tensor * wo = nullptr;
        ggml_tensor * bo = nullptr;
    };

    struct norm {
        ggml_tensor * w = nullptr;
        ggml_tensor * b = nullptr;
    };

    // the first n_layer_routing blocks have no self attention
    struct head_layer {
        norm self_norm;
        attn self_attn;
        norm cross_norm;
        norm cross_norm_kv; // routing blocks only
        attn cross_attn;
        norm ffn_norm;
        ggml_tensor * ffn_up     = nullptr;
        ggml_tensor * ffn_up_b   = nullptr;
        ggml_tensor * ffn_down   = nullptr;
        ggml_tensor * ffn_down_b = nullptr;
    };

    uint32_t n_layer_routing = 0;
    uint32_t n_layer_joint   = 0;
    uint32_t n_head_decision = 0;

    std::vector<head_layer> head_layers; // routing blocks, then joint blocks

    norm hidden_norm;
    norm option_summary_norm;
    norm field_norm;
    norm option_norm;

    ggml_tensor * proj_memory          = nullptr;
    ggml_tensor * proj_question        = nullptr;
    ggml_tensor * proj_option_question = nullptr;
    ggml_tensor * proj_global          = nullptr;
    ggml_tensor * proj_option_context  = nullptr;
    ggml_tensor * proj_option_lexical  = nullptr;
    ggml_tensor * scales               = nullptr; // prior scale, joint scale, residual gate
    ggml_tensor * scorer               = nullptr;
    ggml_tensor * scorer_b             = nullptr;
    ggml_tensor * scorer_out           = nullptr;
    ggml_tensor * scorer_out_b         = nullptr;

    class input_decision;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);
    private:
        llm_graph_input_attn_no_cache * build_attn_inp_causal();

        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_no_cache * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
                    ggml_tensor * cur,
                            int   il);

        // hidden: [n_embd, n_tokens], returns one score per option: [1, n_options]
        ggml_tensor * build_head(
                    ggml_tensor * hidden,
                 input_decision * inp);

        ggml_tensor * build_head_norm(ggml_tensor * cur, const norm & n);
        ggml_tensor * build_head_attn(ggml_tensor * q, ggml_tensor * kv, const attn & a);
        ggml_tensor * build_head_ffn (ggml_tensor * cur, const head_layer & layer);

        const llmcomms_model_clef & model;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_qwen4exp : public llmcomms_model_base {
    llmcomms_model_qwen4exp(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}

    class llm_graph_input_kpool;

    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);
    protected:
        // the helpers alone, graph_mtp builds its own body
        struct no_build {};
        graph(const llm_graph_params & params, no_build) :
            llm_build_delta_net_base(params) {}

        // HC replaces every layer norm: residual is [n_embd, hc, n_tokens]
        struct hc_mix {
            ggml_tensor * cur;     // the mixed [n_embd, T] stream
            ggml_tensor * inject;  // [hc, T] scatter weights, null when the inject weights are null
        };

        hc_mix build_hc_mix(
                    ggml_tensor * x,
                    ggml_tensor * w_norm,
                    ggml_tensor * w_down,
                    ggml_tensor * w_up,
                    ggml_tensor * w_inject,
                    ggml_tensor * w_down_inject, // [w_down | w_inject] in one weight, can be null
                            int   il);

        ggml_tensor * build_hc_combine(
                    ggml_tensor * residual,
                    ggml_tensor * block_out,
                    ggml_tensor * inject,
                            int   il);

        ggml_tensor * build_layer_attn(
              llm_graph_input_attn_kv * inp_attn,
  const llmcomms_memory_hybrid_idx_context * mctx_hyb,
          llm_graph_input_kpool * inp_kpool,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int   il);

        // dense self-attention over the cells the QSA mask keeps
        ggml_tensor * build_attn_qsa(
        llm_graph_input_attn_kv * inp,
                    ggml_tensor * q_cur,
                    ggml_tensor * k_cur,
                    ggml_tensor * v_cur,
                    ggml_tensor * sel,
                        int64_t   n_sel,
                          float   kq_scale,
                            int   il);

        // the QSA layers share one set of k-pool inputs, see llmcomms_memory_hybrid_idx
        llm_graph_input_kpool * build_inp_kpool(const llmcomms_memory_hybrid_idx_context * mctx_hyb);

        // QSA: the additive mask [n_kv, n_tokens] of the top blocks and the tail
        ggml_tensor * build_qsa_sel(
  const llmcomms_memory_hybrid_idx_context * mctx_hyb,
          llm_graph_input_kpool * inp_kpool,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // build_rs writes the state tensor in place, so one gather per cache tensor is reused
        std::map<ggml_tensor *, ggml_tensor *> rs_rows;

        // one conv history per cache tensor: delta-net and PLE each have their own
        ggml_tensor * build_conv_state_at(
             llm_graph_input_rs * inp,
                    ggml_tensor * conv_states_all,
                    ggml_tensor * x,
                        int64_t   state_cols,
                        int64_t   channels,
                            int   il);

        ggml_tensor * build_inp_ple(
  const llmcomms_memory_hybrid_idx_context * mctx_hyb);

        ggml_tensor * build_ple(
             llm_graph_input_rs * inp,
                    ggml_tensor * emb,
                    ggml_tensor * hidden,
                            int   il);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);
    };

    // MTP draft head: one QSA block after the trunk, fed by the trunk's hc-wide residual
    struct graph_mtp : public graph {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_qwen35moe : public llmcomms_model_base {
    llmcomms_model_qwen35moe(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_clover_moe : public llmcomms_model_base {
    llmcomms_model_clover_moe(const llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(llm_graph_input_attn_kv * inp_attn, ggml_tensor * cur, ggml_tensor * inp_pos, int il);
        ggml_tensor * build_layer_attn_linear(llm_graph_input_rs * inp, ggml_tensor * cur, int il);
        ggml_tensor * build_layer_ffn(ggml_tensor * cur, int il);
        ggml_tensor * build_norm_gated(ggml_tensor * input, ggml_tensor * weights, ggml_tensor * gate, int il);
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(ggml_tensor * input, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_mistral3 : public llmcomms_model_base {
    llmcomms_model_mistral3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    class input_t_cond; // voxtral realtime: time conditioning of the ada rms norm

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_mimo2 : public llmcomms_model_base {
    llmcomms_model_mimo2(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_kimi_k3 : public llmcomms_model_base {
    llmcomms_model_kimi_k3(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);

        // Cross-layer residual attention (K3's `_apply_attn_res`).
        ggml_tensor * resi_stack = nullptr;

        void          res_push(ggml_tensor * cur, int64_t n_embd, int64_t n_tokens);
        ggml_tensor * res_mix(ggml_tensor * cur, ggml_tensor * score_w,
                              int64_t n_tokens, int il);

        ggml_tensor * build_kda_layer(ggml_tensor * cur, const llmcomms_layer & layer,
                                      llm_graph_input_rs * inp_rs,
                                      int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
                                      int64_t d_inner, int64_t n_seq_tokens, int64_t n_seqs, int il);

        // causal conv1d over one of Q/K/V, qkv selects which third of the conv state row is used
        ggml_tensor * build_kda_conv1d(llm_graph_input_rs * inp_rs, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
                                       int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                                       int64_t d_conv, int64_t head_dim, int64_t n_head_kda, int64_t n_seq_tokens, int64_t n_seqs);

        ggml_tensor * build_mla_layer(ggml_tensor * cur, const llmcomms_layer & layer,
                                      llm_graph_input_attn_k  * inp_attn_k,
                                      llm_graph_input_attn_kv * inp_attn_kv,
                                      int64_t n_embd_head_k_mla, int64_t n_embd_head_v_mla,
                                      int64_t kv_lora_rank, int64_t n_embd_head_qk_rope,
                                      int64_t n_embd_head_qk_nope, float kq_scale, int il);

        ggml_tensor * build_latent_moe(ggml_tensor * cur, const llmcomms_layer & layer,
                                       int64_t n_embd_latent, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// Kimi-K3.1: Kimi-K3 with a lightning indexer and top-k sparse attention on the full-attention layers
struct llmcomms_model_kimi_k31 : public llmcomms_model_base {
    llmcomms_model_kimi_k31(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);

        // Cross-layer residual attention (kimi-k3's `_apply_attn_res`).
        ggml_tensor * resi_stack = nullptr;

        void          res_push(ggml_tensor * cur, int64_t n_embd, int64_t n_tokens);
        ggml_tensor * res_mix(ggml_tensor * cur, ggml_tensor * score_w,
                              int64_t n_tokens, int il);

        ggml_tensor * build_kda_conv1d(llm_graph_input_rs * inp_rs,
                                       ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
                                       int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                                       int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
                                       int64_t n_seq_tokens, int64_t n_seqs);

        ggml_tensor * build_kda_layer(ggml_tensor * cur, const llmcomms_layer & layer,
                                      llm_graph_input_rs * inp_rs,
                                      int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
                                      int64_t d_inner, int64_t n_seq_tokens, int64_t n_seqs, int il);

        struct dsa_out {
            ggml_tensor * out;
            ggml_tensor * sel;
        };

        dsa_out build_dsa_layer(ggml_tensor * cur, const llmcomms_layer & layer,
                                const llmcomms_kv_cache_context * mctx_idx,
                                llm_graph_input_attn_k * inp_attn,
                                ggml_tensor * prev_sel,
                                int64_t n_embd_head_k_mla, int64_t kv_lora_rank,
                                int64_t n_embd_head_qk_rope,
                                int64_t n_embd_head_qk_nope, float kq_scale, int il);

        ggml_tensor * build_latent_moe(ggml_tensor * cur, const llmcomms_layer & layer,
                                       int64_t n_embd_latent, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_glm5_next : public llmcomms_model_base {
    llmcomms_model_glm5_next(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    // k-pool indexer inputs on top of the generic hybrid input
    class llm_graph_input_kpool;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);

        // build the helpers without the trunk, so graph_mtp can reuse them
        struct no_build {};
        graph(const llm_graph_params & params, no_build);

        // collapse the hc streams with per-stream weights
        ggml_tensor * build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * weights,
                int il) const;

        // mean over the hyper-connection streams: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
        ggml_tensor * build_hc_mean(ggml_tensor * x) const;

        // all outputs of the hc mix builder: cur is the collapsed input, post / comb feed build_hc_post
        struct hc_pre {
            ggml_tensor * cur;
            ggml_tensor * post;
            ggml_tensor * comb;
        };

        hc_pre build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base,
                int il) const;

        ggml_tensor * build_hc_post(
                ggml_tensor * x,
                ggml_tensor * residual,
                ggml_tensor * post,
                ggml_tensor * comb,
                int il) const;

        ggml_tensor * build_hc_sinkhorn(
                ggml_tensor * comb,
                int il) const;

        llm_graph_input_kpool * build_inp_kpool(const llmcomms_memory_hybrid_idx_context * mctx_hyb);

        ggml_tensor * build_kda_layer(ggml_tensor * cur, const llmcomms_layer & layer,
                                      llm_graph_input_rs * inp_rs,
                                      int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
                                      int64_t d_inner, int64_t n_seq_tokens, int64_t n_seqs, int il);

        // causal conv1d over one of Q/K/V, qkv selects which third of the conv state row is used
        ggml_tensor * build_kda_conv1d(llm_graph_input_rs * inp_rs, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
                                       int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                                       int64_t d_conv, int64_t head_dim, int64_t n_head_kda, int64_t n_seq_tokens, int64_t n_seqs);

        ggml_tensor * build_kpool_select(ggml_tensor * cur, ggml_tensor * qr, const llmcomms_layer & layer,
                                         const llmcomms_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_kpool * inp_kpool, int il);

        // attention output (cur) and the selection to carry to the next shared layer (sel)
        struct dsa_out {
            ggml_tensor * cur;
            ggml_tensor * sel;
        };

        dsa_out build_dsa_layer(ggml_tensor * cur, const llmcomms_layer & layer,
                                const llmcomms_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_attn_k * inp_attn,
                                llm_graph_input_kpool * inp_kpool, ggml_tensor * prev_sel, int il);

    };

    // the NextN / MTP draft head: one DSA block appended after the trunk
    struct graph_mtp : public graph {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_kimi_linear : public llmcomms_model_base {
    llmcomms_model_kimi_linear(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);

        std::pair<ggml_tensor *, ggml_tensor *> build_kda_autoregressive(
                    ggml_tensor * q,
                    ggml_tensor * k,
                    ggml_tensor * v,
                    ggml_tensor * gk,
                    ggml_tensor * beta,
                    ggml_tensor * state,
                            int   il);

        std::pair<ggml_tensor *, ggml_tensor *> build_kda_chunking(
                    ggml_tensor * q,
                    ggml_tensor * k,
                    ggml_tensor * v,
                    ggml_tensor * gk,
                    ggml_tensor * beta,
                    ggml_tensor * state,
                    ggml_tensor * causal_mask,
                    ggml_tensor * identity,
                    ggml_tensor * diag_mask,
                            int   il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_step35 : public llmcomms_model_base {
    llmcomms_model_step35(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    llmcomms_rope_params get_rope_params(const llmcomms_cparams & cparams, int il) const override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_spark2_5 : public llmcomms_model_base {
    llmcomms_model_spark2_5(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_k2_horizon : public llmcomms_model_base {
    llmcomms_model_k2_horizon(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

        ggml_tensor * build_routed_value(const llmcomms_layer & layer, ggml_tensor * cur, int il) const;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llmcomms_model_kolibri1 : public llmcomms_model_base {
    llmcomms_model_kolibri1(const struct llmcomms_model_params & params) : llmcomms_model_base(params) {}
    void load_arch_hparams(llmcomms_model_loader & ml) override;
    void load_arch_tensors(llmcomms_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
