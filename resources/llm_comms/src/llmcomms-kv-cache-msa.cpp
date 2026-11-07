#include "llmcomms-kv-cache-msa.h"

#include "llmcomms-impl.h"
#include "llmcomms-batch.h"
#include "llmcomms-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>

// llmcomms_kv_cache_msa

llmcomms_kv_cache_msa::llmcomms_kv_cache_msa(
        const llmcomms_model & model,
                     bool   no_alloc,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_swa,
           llmcomms_swa_type   swa_type,
    const layer_filter_cb & filter,
    const layer_filter_cb & filter_idx,
    const  layer_reuse_cb & reuse) :
    hparams_idx(model.hparams),
    n_stream(unified ? 1 : n_seq_max), n_seq_max(n_seq_max),
    n_swa(n_swa), swa_type(swa_type) {

    LLMCOMMS_LOG_INFO("%s: creating main KV cache, size = %u cells\n", __func__, kv_size);

    kv_base = std::make_unique<llmcomms_kv_cache>(
            model, no_alloc, model.hparams, type_k, type_v,
            v_trans, offload, unified, kv_size, n_seq_max,
            n_swa, swa_type, nullptr, filter, reuse, nullptr);

    // the MSA indexer uses a single key head per layer
    std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
    hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;
    // the rope parameters are kept identical to the main cache

    LLMCOMMS_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

    kv_idx = std::make_unique<llmcomms_kv_cache>(
            model, no_alloc, hparams_idx, type_k, type_v,
            v_trans, offload, unified, kv_size, n_seq_max,
            n_swa, swa_type, nullptr, filter_idx, reuse, nullptr);
}

void llmcomms_kv_cache_msa::clear(bool data) {
    kv_base->clear(data);
    kv_idx ->clear(data);
}

bool llmcomms_kv_cache_msa::seq_rm(llmcomms_seq_id seq_id, llmcomms_pos p0, llmcomms_pos p1) {
    bool res = true;

    res = res & kv_base->seq_rm(seq_id, p0, p1);
    res = res & kv_idx ->seq_rm(seq_id, p0, p1);

    return res;
}

void llmcomms_kv_cache_msa::seq_cp(llmcomms_seq_id seq_id_src, llmcomms_seq_id seq_id_dst, llmcomms_pos p0, llmcomms_pos p1) {
    kv_base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_idx ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llmcomms_kv_cache_msa::seq_keep(llmcomms_seq_id seq_id) {
    kv_base->seq_keep(seq_id);
    kv_idx ->seq_keep(seq_id);
}

void llmcomms_kv_cache_msa::seq_add(llmcomms_seq_id seq_id, llmcomms_pos p0, llmcomms_pos p1, llmcomms_pos shift) {
    kv_base->seq_add(seq_id, p0, p1, shift);
    kv_idx ->seq_add(seq_id, p0, p1, shift);
}

void llmcomms_kv_cache_msa::seq_div(llmcomms_seq_id seq_id, llmcomms_pos p0, llmcomms_pos p1, int d) {
    kv_base->seq_div(seq_id, p0, p1, d);
    kv_idx ->seq_div(seq_id, p0, p1, d);
}

llmcomms_pos llmcomms_kv_cache_msa::seq_pos_min(llmcomms_seq_id seq_id) const {
    return kv_base->seq_pos_min(seq_id);
}

llmcomms_pos llmcomms_kv_cache_msa::seq_pos_max(llmcomms_seq_id seq_id) const {
    return kv_base->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llmcomms_kv_cache_msa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_base->memory_breakdown();
    for (const auto & buft_size : kv_idx->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

llmcomms_memory_context_ptr llmcomms_kv_cache_msa::init_batch(
            llmcomms_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llmcomms_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch));
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_idx = kv_idx->prepare(ubatches);
        if (sinfos_idx.empty()) {
            break;
        }

        assert(sinfos_base.size() == sinfos_idx.size());

        return std::make_unique<llmcomms_kv_cache_msa_context>(
                this, std::move(sinfos_base), std::move(sinfos_idx), std::move(ubatches));
    } while (false);

    return std::make_unique<llmcomms_kv_cache_msa_context>(LLMCOMMS_MEMORY_STATUS_FAILED_PREPARE);
}

llmcomms_memory_context_ptr llmcomms_kv_cache_msa::init_full() {
    return std::make_unique<llmcomms_kv_cache_msa_context>(this);
}

llmcomms_memory_context_ptr llmcomms_kv_cache_msa::init_update(llmcomms_context * lctx, bool optimize) {
    return std::make_unique<llmcomms_kv_cache_msa_context>(this, lctx, optimize);
}

bool llmcomms_kv_cache_msa::get_can_shift() const {
    return kv_base->get_can_shift() &&
           kv_idx ->get_can_shift() &&
           kv_base->get_size() == kv_idx->get_size();
}

void llmcomms_kv_cache_msa::state_write(llmcomms_io_write_i & io, llmcomms_seq_id seq_id, llmcomms_state_seq_flags flags) const {
    kv_base->state_write(io, seq_id, flags);
    kv_idx ->state_write(io, seq_id, flags);
}

void llmcomms_kv_cache_msa::state_read(llmcomms_io_read_i & io, llmcomms_seq_id seq_id, llmcomms_state_seq_flags flags) {
    kv_base->state_read(io, seq_id, flags);
    kv_idx ->state_read(io, seq_id, flags);
}

llmcomms_kv_cache * llmcomms_kv_cache_msa::get_base() const {
    return kv_base.get();
}

llmcomms_kv_cache * llmcomms_kv_cache_msa::get_idx() const {
    return kv_idx.get();
}

// llmcomms_kv_cache_msa_context

llmcomms_kv_cache_msa_context::llmcomms_kv_cache_msa_context(llmcomms_memory_status status) :
    kv(nullptr), status(status) {}

llmcomms_kv_cache_msa_context::llmcomms_kv_cache_msa_context(
        llmcomms_kv_cache_msa * kv) :
    kv(kv),
    ctx_base(kv->get_base()->init_full()),
    ctx_idx (kv->get_idx ()->init_full()),
    status(llmcomms_memory_status_combine(ctx_base->get_status(), ctx_idx->get_status())) {
}

llmcomms_kv_cache_msa_context::llmcomms_kv_cache_msa_context(
        llmcomms_kv_cache_msa * kv,
        llmcomms_context * lctx,
        bool optimize) :
    kv(kv),
    ctx_base(kv->get_base()->init_update(lctx, optimize)),
    ctx_idx (kv->get_idx ()->init_update(lctx, optimize)),
    status(llmcomms_memory_status_combine(ctx_base->get_status(), ctx_idx->get_status())) {
}

llmcomms_kv_cache_msa_context::llmcomms_kv_cache_msa_context(
        llmcomms_kv_cache_msa * kv,
        slot_info_vec_t sinfos_base,
        slot_info_vec_t sinfos_idx,
        std::vector<llmcomms_ubatch> ubatches) :
    kv(kv),
    ubatches(std::move(ubatches)),
    // here we copy the ubatches. not sure if this is ideal
    ctx_base(new llmcomms_kv_cache_context(kv->get_base(), std::move(sinfos_base), this->ubatches)),
    ctx_idx (new llmcomms_kv_cache_context(kv->get_idx (), std::move(sinfos_idx),  this->ubatches)),
    status(llmcomms_memory_status_combine(ctx_base->get_status(), ctx_idx->get_status())) {
}

llmcomms_kv_cache_msa_context::~llmcomms_kv_cache_msa_context() = default;

bool llmcomms_kv_cache_msa_context::next() {
    assert(status == LLMCOMMS_MEMORY_STATUS_SUCCESS);

    ctx_base->next();
    ctx_idx ->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llmcomms_kv_cache_msa_context::apply() {
    assert(!llmcomms_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_base->apply();
    res = res & ctx_idx ->apply();

    return res;
}

llmcomms_memory_status llmcomms_kv_cache_msa_context::get_status() const {
    return status;
}

const llmcomms_ubatch & llmcomms_kv_cache_msa_context::get_ubatch() const {
    assert(status == LLMCOMMS_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llmcomms_kv_cache_context * llmcomms_kv_cache_msa_context::get_base() const {
    assert(status == LLMCOMMS_MEMORY_STATUS_SUCCESS);

    return static_cast<const llmcomms_kv_cache_context *>(ctx_base.get());
}

const llmcomms_kv_cache_context * llmcomms_kv_cache_msa_context::get_idx() const {
    assert(status == LLMCOMMS_MEMORY_STATUS_SUCCESS);

    return static_cast<const llmcomms_kv_cache_context *>(ctx_idx.get());
}

uint32_t llmcomms_kv_cache_msa_context::get_n_pos() const {
    // pad the value so that the graph remains constant across batches and can be reused
    const uint32_t n_pad = 256;

    llmcomms_pos pos_max = -1;

    for (llmcomms_seq_id seq_id = 0; seq_id < (llmcomms_seq_id) kv->get_n_seq_max(); ++seq_id) {
        pos_max = std::max(pos_max, kv->seq_pos_max(seq_id));
    }

    return std::max(n_pad, GGML_PAD((uint32_t) (pos_max + 1), n_pad));
}

void llmcomms_kv_cache_msa_context::set_input_cell_pos(ggml_tensor * dst, const llmcomms_ubatch * ubatch, int32_t div) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(div > 0);

    const int64_t n_tokens    = ubatch->n_tokens;
    const int64_t n_kv        = dst->ne[0];
    const int64_t n_stream_ub = dst->ne[1];

    GGML_ASSERT(n_tokens % n_stream_ub == 0);
    const int64_t n_tps = n_tokens/n_stream_ub;

    int32_t * data = (int32_t *) dst->data;

    for (int64_t s = 0; s < n_stream_ub; ++s) {
        const llmcomms_seq_id seq_id = ubatch->seq_id[s*n_tps][0];

        const auto & cells = kv->get_base()->get_cells(seq_id);

        for (int64_t j = 0; j < n_kv; ++j) {
            // the value for empty or other-sequence cells is irrelevant as consumers mask them
            data[s*n_kv + j] =
                cells.is_empty(j) || !cells.seq_has(j, seq_id)
                    ? 0
                    : (int32_t) (cells.pos_get(j)/div);
        }
    }
}

void llmcomms_kv_cache_msa_context::set_input_pos_slot(ggml_tensor * dst, const llmcomms_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I32 || dst->type == GGML_TYPE_F32);

    const int64_t n_tokens    = ubatch->n_tokens;
    const int64_t n_pos       = dst->ne[0];
    const int64_t n_stream_ub = dst->ne[1];

    GGML_ASSERT(n_tokens % n_stream_ub == 0);
    const int64_t n_tps = n_tokens/n_stream_ub;

    for (int64_t s = 0; s < n_stream_ub; ++s) {
        const llmcomms_seq_id seq_id = ubatch->seq_id[s*n_tps][0];

        const auto & cells = kv->get_base()->get_cells(seq_id);

        std::vector<int32_t> map(n_pos, 0);

        for (uint32_t j = 0; j < cells.size(); ++j) {
            if (cells.is_empty(j) || !cells.seq_has(j, seq_id)) {
                continue;
            }

            const llmcomms_pos p0 = cells.pos_get(j);

            if (p0 < 0 || p0 >= n_pos) {
                continue;
            }

            map[p0] = (int32_t) j;
        }

        if (dst->type == GGML_TYPE_I32) {
            int32_t * data = (int32_t *) dst->data + s*n_pos;
            std::copy(map.begin(), map.end(), data);
        } else {
            float * data = (float *) dst->data + s*n_pos;
            for (int64_t p = 0; p < n_pos; ++p) {
                data[p] = (float) map[p];
            }
        }
    }
}

void llmcomms_kv_cache_msa_context::set_input_pos_mask(ggml_tensor * dst, const llmcomms_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_pos    = dst->ne[0];

    GGML_ASSERT(dst->ne[1] == n_tokens);

    const uint32_t       n_swa    = kv->get_n_swa();
    const llmcomms_swa_type swa_type = kv->get_swa_type();

    float * data = (float *) dst->data;

    std::fill(data, data + n_pos*n_tokens, -INFINITY);

    for (int64_t i = 0; i < n_tokens; ++i) {
        const llmcomms_seq_id seq_id = ubatch->seq_id[i][0];

        const auto & cells = kv->get_base()->get_cells(seq_id);

        const llmcomms_pos p1 = ubatch->pos[i];

        for (uint32_t j = 0; j < cells.size(); ++j) {
            if (cells.is_empty(j) || !cells.seq_has(j, seq_id)) {
                continue;
            }

            const llmcomms_pos p0 = cells.pos_get(j);

            if (p0 < 0 || p0 >= n_pos) {
                continue;
            }

            // causal mask
            if (p0 > p1) {
                continue;
            }

            // apply SWA if any
            if (llmcomms_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                continue;
            }

            data[i*n_pos + p0] = 0.0f;
        }
    }
}
