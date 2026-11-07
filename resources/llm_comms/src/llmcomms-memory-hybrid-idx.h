#pragma once

#include "llmcomms-memory-hybrid.h"

#include <array>
#include <limits>
#include <memory>
#include <vector>

//
// llmcomms_memory_hybrid_idx
//

// llmcomms_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llmcomms_memory_hybrid_idx : public llmcomms_memory_hybrid {
public:
    llmcomms_memory_hybrid_idx(
        const llmcomms_model & model,
                     bool   no_alloc,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_swa,
           llmcomms_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    // Defined out of line because kpool_layout is incomplete here.
    ~llmcomms_memory_hybrid_idx();

    //
    // llmcomms_memory_i
    //

    llmcomms_memory_context_ptr init_batch(
            llmcomms_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llmcomms_memory_context_ptr init_full() override;

    llmcomms_memory_context_ptr init_update(llmcomms_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1) override;
    void seq_cp  (llmcomms_seq_id seq_id_src, llmcomms_seq_id seq_id_dst, llmcomms_pos p0, llmcomms_pos p1) override;
    void seq_keep(llmcomms_seq_id seq_id)                                                          override;
    void seq_add (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1, llmcomms_pos shift) override;
    void seq_div (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llmcomms_io_write_i & io, llmcomms_seq_id seq_id = -1, llmcomms_state_seq_flags flags = 0) const override;
    void state_read (llmcomms_io_read_i  & io, llmcomms_seq_id seq_id = -1, llmcomms_state_seq_flags flags = 0)       override;

    //
    // llmcomms_memory_hybrid_idx specific API
    //

    llmcomms_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // The model's indexer pool size.
    uint32_t get_kpool() const { return hparams_idx.indexer_kpool; }

    // Whether pools are kpool consecutive cells in sequence order (qwen4exp) instead of kpool consecutive positions.
    bool get_kpool_by_order() const { return hparams_idx.indexer_kpool_by_order; }

    // Which cells of a sequence make up which pool of kpool consecutive positions (or cells, in order mode).
    // It is kept here because it outlives the batch: pools are fixed by the positions relative to the
    // sequence's first one, so a ubatch only ever appends to it. Sequence edits drop it, see mem_idx_stale.
    struct kpool_layout;

    const kpool_layout & kpool_layout_update();
    const kpool_layout & kpool_layout_get() const;

    // The pooled keys persist in the idx cache across batches. A sequence edit can regroup the pools
    // from some position on, which stales every pooled key at or after it. POS_CLEAN means none.
    using stale_pos_t = std::array<llmcomms_pos, LLMCOMMS_MAX_SEQ>;

    static constexpr llmcomms_pos POS_CLEAN = std::numeric_limits<llmcomms_pos>::max();

    static stale_pos_t stale_pos_clean() {
        stale_pos_t res;
        res.fill(POS_CLEAN);
        return res;
    }

    const stale_pos_t & mem_idx_stale_get() const { return mem_idx_stale; }
    void mem_idx_stale_clear() { mem_idx_stale.fill(POS_CLEAN); }

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llmcomms_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llmcomms_kv_cache keeps a reference to what it is given
    llmcomms_hparams hparams_idx;

    const std::unique_ptr<llmcomms_kv_cache> mem_idx;

    // unique_ptr because kpool_layout is incomplete here
    std::unique_ptr<kpool_layout> kpool_lay;

    // seq_id < 0 stales every sequence, p0 < 0 stales the sequence from its first position
    void mem_idx_stale_set(llmcomms_seq_id seq_id, llmcomms_pos p0);

    // the position an edit at p0 stales the sequence from
    llmcomms_pos mem_idx_stale_pos(llmcomms_seq_id seq_id, llmcomms_pos p0) const;

    stale_pos_t mem_idx_stale = stale_pos_clean();
};

class llmcomms_memory_hybrid_idx_context : public llmcomms_memory_hybrid_context {
public:
    class kpool_access {
    public:
        ggml_tensor * gather_key_gate(ggml_tensor * idxs) const;
        ggml_tensor * scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const;
        ggml_tensor * gather_pooled(ggml_tensor * idxs) const;

    private:
        friend class llmcomms_memory_hybrid_idx_context;

        kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd);

        ggml_context * ctx;
        ggml_tensor  * key_gate;
        ggml_tensor  * pooled;
    };

    using slot_info_vec_t = llmcomms_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llmcomms_memory_hybrid_idx_context(llmcomms_memory_status status);

    // used to create a full-cache context
    explicit llmcomms_memory_hybrid_idx_context(llmcomms_memory_hybrid_idx * mem);

    // used to create an update context
    llmcomms_memory_hybrid_idx_context(
            llmcomms_memory_hybrid_idx * mem,
                      llmcomms_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llmcomms_memory_hybrid_idx_context(
            llmcomms_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llmcomms_ubatch>   ubatches);

    ~llmcomms_memory_hybrid_idx_context(); // Defined out of line because kpool_state is incomplete here.

    //
    // llmcomms_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llmcomms_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llmcomms_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // glm5-next and qwen4exp, complete pools of kpool cells per sequence, scored as whole pools.
    uint32_t get_n_kpool    () const; // Padded pool count, where the last pool is always unused.
    uint32_t get_n_kpool_new() const; // Pools to re-pool this ubatch, padded to a stable bound, never below 1.
    kpool_access get_kpool_access(ggml_context * ctx, int32_t il, int64_t n_embd) const;
    // sel_mask (F32 [n_sel, 1, 1, n_tokens], can be null): 0 for the live selection slots, -inf for the dead ones
    // new_pool_pos (I32 [4*n_new]): M-RoPE position of each new pool's first member, for pooled keys rotated at pooling time
    void set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
                         ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
                         const llmcomms_ubatch * ubatch, ggml_tensor * new_pool_pos = nullptr) const;

private:
    llmcomms_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // the indexer cells of each ubatch, kept for pools in cache order (qwen4exp): token s*n + i of ubatch u
    // sits in cell idxs[s][i] of stream strm[s] of sinfos_kpool[u], and several cells can share a position
    const slot_info_vec_t sinfos_kpool;

    // null unless the model has an indexer
    const llmcomms_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // Which pools of the layout this ubatch must re-pool. The layout itself belongs to the memory.
    struct kpool_state;
    kpool_state kpool_build_sizes() const;
    void kpool_build_state(const llmcomms_ubatch & ubatch);
    const kpool_state & kpool_cur() const;

    // unique_ptr because kpool_state is incomplete here.
    std::unique_ptr<kpool_state> kpool_st;

    // The ubatch kpool_st was built for, guards against reads before apply.
    size_t i_kpool = SIZE_MAX;

    // Whether this context tracks k-pool states.
    bool kpool_track() const;

    // Positions each sequence must re-pool from, cleared only after the first ubatch succeeds
    llmcomms_memory_hybrid_idx::stale_pos_t mem_idx_stale_batch = llmcomms_memory_hybrid_idx::stale_pos_clean();
};
