#pragma once

#include "llmcomms-core.h"

#include <vector>

struct llmcomms_vocab;
struct llmcomms_grammar;

// sampler chain

struct llmcomms_sampler_chain {
    llmcomms_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    uint32_t n_nodes = 0;

    struct info {
        bool is_backend;

        llmcomms_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llmcomms_sampler_sample to avoid repeated allocations
    std::vector<llmcomms_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

uint32_t llmcomms_sampler_backend_n_nodes(const llmcomms_sampler * sampler);
void llmcomms_sampler_backend_begin(llmcomms_sampler * sampler);

struct llmcomms_sampler * llmcomms_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llmcomms_token>> & seq_breakers);
