#pragma once

#include "llmcomms-batch.h"
#include "llmcomms-graph.h"
#include "llmcomms-kv-cells.h"
#include "llmcomms-memory.h"

#include <unordered_map>
#include <vector>

struct llmcomms_cparams;
struct llmcomms_hparams;
struct llmcomms_model;
struct llmcomms_context;

//
// llmcomms_kv_cache
//

class llmcomms_kv_cache : public llmcomms_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llmcomms_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llmcomms_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llmcomms_memory_params`
    llmcomms_kv_cache(
            const llmcomms_model & model,
                         bool   no_alloc,
          const llmcomms_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_swa,
               llmcomms_swa_type   swa_type,
               llmcomms_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
        // a model can hold more than one cache, so the tensor names have to stay unique
                 const char *   name_tag = "",
        // lightning indexer keys are rotated with the indexer RoPE parameters
                       bool   is_indexer = false);

    ~llmcomms_kv_cache() = default;

    //
    // llmcomms_memory_i
    //

    llmcomms_memory_context_ptr init_batch(
            llmcomms_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llmcomms_memory_context_ptr init_full() override;

    llmcomms_memory_context_ptr init_update(llmcomms_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1) override;
    void seq_cp  (llmcomms_seq_id seq_id_src, llmcomms_seq_id seq_id_dst, llmcomms_pos p0, llmcomms_pos p1) override;
    void seq_keep(llmcomms_seq_id seq_id)                                                          override;
    void seq_add (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1, llmcomms_pos shift) override;
    void seq_div (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1, int d) override;

    llmcomms_pos seq_pos_min(llmcomms_seq_id seq_id) const override;
    llmcomms_pos seq_pos_max(llmcomms_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llmcomms_io_write_i & io, llmcomms_seq_id seq_id = -1, llmcomms_state_seq_flags flags = 0) const override;
    void state_read (llmcomms_io_read_i  & io, llmcomms_seq_id seq_id = -1, llmcomms_state_seq_flags flags = 0) override;

    //
    // llmcomms_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_seq_max() const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;

    const llmcomms_kv_cells & get_cells(llmcomms_seq_id seq_id) const;

    // The stream holding seq_id's cells.
    uint32_t get_stream(llmcomms_seq_id seq_id) const;

    // state_read, plus the cells the restored tokens were placed in
    // a cache that mirrors another one (the qwen4exp indexer) must not search for its own cells: two searches agree only by luck
    //   sinfos_out: if set, filled with the layout used; a stream with no cells leaves an empty entry
    //   sinfos_in : if set, the layout to use instead of searching. one entry per stream, cell count must match the blob
    void state_read_sinfo(
            llmcomms_io_read_i & io,
               llmcomms_seq_id   seq_id,
      llmcomms_state_seq_flags   flags,
            slot_info_vec_t * sinfos_out,
      const slot_info_vec_t * sinfos_in);

    // undo a state_read() of seq_id (-1 for the whole cache) that another memory module failed to complete
    void state_clear(llmcomms_seq_id seq_id);

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // attention over the cells of each sequence of the ubatch instead of the whole cache (see ggml_flash_attn_ext_rows)
    // the ubatch splits into n_slices groups of tokens with one sequence each
    // if it cannot (f.ex. interleaved or coupled sequences), it is a single slice over the cells of all its sequences
    // TODO: allow slices with coupled sequences
    static uint32_t get_n_kv_slices(const llmcomms_ubatch & ubatch);

    // padded length of the longest cell list of the slices; 0 if kv_rows are not used (multiple streams)
    uint32_t get_n_kv_rows(const llmcomms_ubatch & ubatch) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llmcomms_ubatch> & ubatches);

    bool update(llmcomms_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llmcomms_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llmcomms_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llmcomms_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llmcomms_ubatch & ubatch) const;

    std::vector<ggml_tensor *> get_k_rot() const;
    std::vector<ggml_tensor *> get_v_rot() const;

    void set_input_k_idxs(ggml_tensor * dst, const llmcomms_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llmcomms_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_attn_mask (const llm_attn_mask & attn_mask, const llmcomms_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llmcomms_ubatch * ubatch) const;

    // true if llmcomms_kv_cell_ext holds information that has to survive a state save/restore
    bool has_cell_ext() const;

    // for every token of the ubatch, the ids of the n tokens that precede it in its sequence
    // example for M-RoPE image case: tokens A B X X X C, where X is a 3-token image at pos 2 spanning positions 2..4:
    //   tok: A B X X X C
    //   pos: 0 1 2 2 2 5
    //   prev, n=2: A -> [NULL, NULL], B -> [NULL, A], 3rd X -> [X, X], C -> [X, X]
    // note: used by n-gram input embeddings
    void get_prev_tokens(const llmcomms_ubatch & ubatch, uint32_t n, std::vector<llmcomms_token> & res) const;

private:
    const llmcomms_model & model;
    const llmcomms_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        ggml_tensor * k_rot = nullptr;
        ggml_tensor * v_rot = nullptr;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // SWA
    const uint32_t n_swa = 0;

    const bool is_indexer = false;

    // env: LLMCOMMS_ATTN_ROT_DISABLE
    uint32_t n_rot_k = 0;
    uint32_t n_rot_v = 0;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // keep constants separate from mutable KV buffers
    llmcomms_memory_bufs attn_rot_bufs;

    void init_rotation(const char * name_tag);

    // env: LLMCOMMS_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llmcomms_swa_type swa_type = LLMCOMMS_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    llmcomms_memory_bufs ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llmcomms_kv_cache * other;

    std::shared_ptr<llmcomms_kv_cells_vec> v_cells_impl;

    llmcomms_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llmcomms_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * k,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llmcomms_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llmcomms_io_write_i & io, const cell_ranges_t & cr, llmcomms_seq_id seq_id = -1) const;
    void state_write_data(llmcomms_io_write_i & io, const cell_ranges_t & cr) const;

    // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
    bool state_read_meta(llmcomms_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llmcomms_seq_id dest_seq_id = -1, const slot_info * sinfo_in = nullptr);
    bool state_read_data(llmcomms_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);

    void state_clear(llmcomms_seq_id seq_id, uint32_t strm, const slot_info & sinfo);
};

class llmcomms_kv_cache_context : public llmcomms_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llmcomms_kv_cache::slot_info_vec_t;
    using stream_copy_info = llmcomms_kv_cache::stream_copy_info;

    // used for errors
    llmcomms_kv_cache_context(llmcomms_memory_status status);

    // used to create a full-cache context
    llmcomms_kv_cache_context(
            llmcomms_kv_cache * kv);

    // used to create an update context
    llmcomms_kv_cache_context(
            llmcomms_kv_cache * kv,
            llmcomms_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llmcomms_kv_cache_context(
            llmcomms_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llmcomms_ubatch> ubatches);

    virtual ~llmcomms_kv_cache_context();

    //
    // llmcomms_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llmcomms_memory_status  get_status() const override;
    const llmcomms_ubatch & get_ubatch() const override;

    //
    // llmcomms_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;
    uint32_t get_n_kv_rows() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llmcomms_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llmcomms_ubatch & ubatch) const;

    std::vector<ggml_tensor *> get_k_rot() const;
    std::vector<ggml_tensor *> get_v_rot() const;

    void set_input_k_idxs(ggml_tensor * dst, const llmcomms_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llmcomms_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_attn_mask (const llm_attn_mask & attn_mask, const llmcomms_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llmcomms_ubatch * ubatch) const;

    // see llmcomms_kv_cache::get_prev_tokens()
    void get_prev_tokens(const llmcomms_ubatch & ubatch, uint32_t n, std::vector<llmcomms_token> & res) const;

private:
    llmcomms_memory_status status;

    llmcomms_kv_cache * kv;
    llmcomms_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llmcomms_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;

    // see llmcomms_kv_cache::get_n_kv_rows
    uint32_t n_kv_rows = 0;
};
