#include "llmcomms-kv-cache-iswa.h"

#include "llmcomms-impl.h"
#include "llmcomms-batch.h"
#include "llmcomms-model.h"

#include <algorithm>
#include <cassert>

//
// llmcomms_kv_cache_iswa
//

llmcomms_kv_cache_iswa::llmcomms_kv_cache_iswa(
        const llmcomms_model & model,
                     bool   no_alloc,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
           llmcomms_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share) :
    llmcomms_kv_cache_iswa(model, no_alloc, model.hparams, type_k, type_v, v_trans, offload, swa_full, unified,
            kv_size, n_seq_max, n_ubatch, mem_other, filter, reuse, share) {
}

llmcomms_kv_cache_iswa::llmcomms_kv_cache_iswa(
        const llmcomms_model & model,
                     bool   no_alloc,
      const llmcomms_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
           llmcomms_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share) : unified(unified) {

    // chain filters
    const layer_filter_cb filter_base = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return !model.hparams.is_swa(il);
    };

    const layer_filter_cb filter_swa  = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return  model.hparams.is_swa(il);
    };

    const uint32_t size_base = kv_size;

    // note: the SWA cache is always padded to 256 for performance
    //       https://github.com/ggml-org/llama.cpp/issues/17037
    uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);

    // when using full-size SWA cache, we set the SWA cache size to be equal to the base cache size
    if (swa_full) {
        LLMCOMMS_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");

        size_swa = size_base;
    }

    LLMCOMMS_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, size_base);

    llmcomms_memory_t mem_other_base = nullptr;
    if (mem_other) {
        mem_other_base = static_cast<llmcomms_kv_cache_iswa *>(mem_other)->get_base();
    }

    llmcomms_memory_t mem_other_swa = nullptr;
    if (mem_other) {
        mem_other_swa = static_cast<llmcomms_kv_cache_iswa *>(mem_other)->get_swa();
    }

    kv_base = std::make_unique<llmcomms_kv_cache>(
            model, no_alloc, hparams, type_k, type_v,
            v_trans, offload, unified, size_base, n_seq_max,
            0, LLMCOMMS_SWA_TYPE_NONE, mem_other_base, filter_base, reuse, share);

    LLMCOMMS_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, size_swa);

    kv_swa = std::make_unique<llmcomms_kv_cache>(
            model, no_alloc, hparams, type_k, type_v,
            v_trans, offload, unified, size_swa, n_seq_max,
            hparams.n_swa, hparams.swa_type, mem_other_swa, filter_swa, reuse, share);
}

void llmcomms_kv_cache_iswa::clear(bool data) {
    kv_base->clear(data);
    kv_swa ->clear(data);
}

bool llmcomms_kv_cache_iswa::seq_rm(llmcomms_seq_id seq_id, llmcomms_pos p0, llmcomms_pos p1) {
    bool res = true;

    res = res & kv_base->seq_rm(seq_id, p0, p1);
    res = res & kv_swa ->seq_rm(seq_id, p0, p1);

    return res;
}

void llmcomms_kv_cache_iswa::seq_cp(llmcomms_seq_id seq_id_src, llmcomms_seq_id seq_id_dst, llmcomms_pos p0, llmcomms_pos p1) {
    kv_base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_swa ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llmcomms_kv_cache_iswa::seq_keep(llmcomms_seq_id seq_id) {
    kv_base->seq_keep(seq_id);
    kv_swa ->seq_keep(seq_id);
}

void llmcomms_kv_cache_iswa::seq_add(llmcomms_seq_id seq_id, llmcomms_pos p0, llmcomms_pos p1, llmcomms_pos shift) {
    kv_base->seq_add(seq_id, p0, p1, shift);
    kv_swa ->seq_add(seq_id, p0, p1, shift);
}

void llmcomms_kv_cache_iswa::seq_div(llmcomms_seq_id seq_id, llmcomms_pos p0, llmcomms_pos p1, int d) {
    kv_base->seq_div(seq_id, p0, p1, d);
    kv_swa ->seq_div(seq_id, p0, p1, d);
}

llmcomms_pos llmcomms_kv_cache_iswa::seq_pos_min(llmcomms_seq_id seq_id) const {
    // the base cache is a superset of the SWA cache, so we can just check the SWA cache
    return kv_swa->seq_pos_min(seq_id);
}

llmcomms_pos llmcomms_kv_cache_iswa::seq_pos_max(llmcomms_seq_id seq_id) const {
    return kv_swa->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llmcomms_kv_cache_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_base->memory_breakdown();
    for (const auto & buft_size : kv_swa->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

llmcomms_memory_context_ptr llmcomms_kv_cache_iswa::init_batch(llmcomms_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // first try simple split
    do {
        if (!unified) {
            // requires equal splits, so we skip the simple split
            break;
        }

        balloc.split_reset();

        std::vector<llmcomms_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        assert(sinfos_base.size() == sinfos_swa.size());

        return std::make_unique<llmcomms_kv_cache_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    // if it fails, try equal split
    do {
        balloc.split_reset();

        std::vector<llmcomms_ubatch> ubatches;
        while (true) {
            auto ubatch = embd_all ? balloc.split_seq(n_ubatch) : balloc.split_equal(n_ubatch, !unified, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        assert(sinfos_base.size() == sinfos_swa.size());

        return std::make_unique<llmcomms_kv_cache_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    // TODO: if we fail again, we should attempt different splitting strategies
    //       but to do that properly, we first have to refactor the batches to be more flexible

    return std::make_unique<llmcomms_kv_cache_iswa_context>(LLMCOMMS_MEMORY_STATUS_FAILED_PREPARE);
}

llmcomms_memory_context_ptr llmcomms_kv_cache_iswa::init_full() {
    return std::make_unique<llmcomms_kv_cache_iswa_context>(this);
}

llmcomms_memory_context_ptr llmcomms_kv_cache_iswa::init_update(llmcomms_context * lctx, bool optimize) {
    return std::make_unique<llmcomms_kv_cache_iswa_context>(this, lctx, optimize);
}

bool llmcomms_kv_cache_iswa::get_can_shift() const {
    return kv_base->get_can_shift() &&
           kv_swa->get_can_shift() &&
           kv_base->get_size() == kv_swa->get_size();
}

void llmcomms_kv_cache_iswa::state_write(llmcomms_io_write_i & io, llmcomms_seq_id seq_id, llmcomms_state_seq_flags flags) const {
    kv_base->state_write(io, seq_id, flags);
    kv_swa ->state_write(io, seq_id, flags);
}

void llmcomms_kv_cache_iswa::state_read(llmcomms_io_read_i & io, llmcomms_seq_id seq_id, llmcomms_state_seq_flags flags) {
    kv_base->state_read(io, seq_id, flags);
    kv_swa ->state_read(io, seq_id, flags);
}

llmcomms_kv_cache * llmcomms_kv_cache_iswa::get_base() const {
    return kv_base.get();
}

llmcomms_kv_cache * llmcomms_kv_cache_iswa::get_swa() const {
    return kv_swa.get();
}

//
// llmcomms_kv_cache_iswa_context
//

llmcomms_kv_cache_iswa_context::llmcomms_kv_cache_iswa_context(llmcomms_memory_status status) : status(status) {}

llmcomms_kv_cache_iswa_context::llmcomms_kv_cache_iswa_context(
        llmcomms_kv_cache_iswa * kv) :
    ctx_base(kv->get_base()->init_full()),
    ctx_swa (kv->get_swa ()->init_full()),
    status(llmcomms_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llmcomms_kv_cache_iswa_context::llmcomms_kv_cache_iswa_context(
        llmcomms_kv_cache_iswa * kv,
        llmcomms_context * lctx,
        bool optimize) :
    ctx_base(kv->get_base()->init_update(lctx, optimize)),
    ctx_swa (kv->get_swa ()->init_update(lctx, optimize)),
    status(llmcomms_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llmcomms_kv_cache_iswa_context::llmcomms_kv_cache_iswa_context(
        llmcomms_kv_cache_iswa * kv,
        slot_info_vec_t sinfos_base,
        slot_info_vec_t sinfos_swa,
        std::vector<llmcomms_ubatch> ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_base(new llmcomms_kv_cache_context(kv->get_base(), std::move(sinfos_base), this->ubatches)),
    ctx_swa (new llmcomms_kv_cache_context(kv->get_swa (), std::move(sinfos_swa),  this->ubatches)),
    status(llmcomms_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llmcomms_kv_cache_iswa_context:: ~llmcomms_kv_cache_iswa_context() = default;

bool llmcomms_kv_cache_iswa_context::next() {
    assert(status == LLMCOMMS_MEMORY_STATUS_SUCCESS);

    ctx_base->next();
    ctx_swa ->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llmcomms_kv_cache_iswa_context::apply() {
    assert(!llmcomms_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_base->apply();
    res = res & ctx_swa ->apply();

    return res;
}

llmcomms_memory_status llmcomms_kv_cache_iswa_context::get_status() const {
    return status;
}

const llmcomms_ubatch & llmcomms_kv_cache_iswa_context::get_ubatch() const {
    assert(status == LLMCOMMS_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llmcomms_kv_cache_context * llmcomms_kv_cache_iswa_context::get_base() const {
    assert(status == LLMCOMMS_MEMORY_STATUS_SUCCESS);

    return static_cast<const llmcomms_kv_cache_context *>(ctx_base.get());
}

const llmcomms_kv_cache_context * llmcomms_kv_cache_iswa_context::get_swa()  const {
    assert(status == LLMCOMMS_MEMORY_STATUS_SUCCESS);

    return static_cast<const llmcomms_kv_cache_context *>(ctx_swa.get());
}
