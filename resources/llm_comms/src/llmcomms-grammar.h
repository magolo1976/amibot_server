#pragma once

#include "llmcomms-core.h"

#include <map>
#include <regex>
#include <string>
#include <vector>

struct llmcomms_vocab;

// grammar element type
enum llmcomms_gretype {
    // end of rule definition
    LLMCOMMS_GRETYPE_END            = 0,

    // start of alternate definition for rule
    LLMCOMMS_GRETYPE_ALT            = 1,

    // non-terminal element: reference to rule
    LLMCOMMS_GRETYPE_RULE_REF       = 2,

    // terminal element: character (code point)
    LLMCOMMS_GRETYPE_CHAR           = 3,

    // inverse char(s) ([^a], [^a-b] [^abc])
    LLMCOMMS_GRETYPE_CHAR_NOT       = 4,

    // modifies a preceding LLMCOMMS_GRETYPE_CHAR or LLMCOMMS_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    LLMCOMMS_GRETYPE_CHAR_RNG_UPPER = 5,

    // modifies a preceding LLMCOMMS_GRETYPE_CHAR or
    // LLMCOMMS_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
    LLMCOMMS_GRETYPE_CHAR_ALT       = 6,

    // any character (.)
    LLMCOMMS_GRETYPE_CHAR_ANY       = 7,

    // terminal element: token (<[token-id]>)
    LLMCOMMS_GRETYPE_TOKEN          = 8,

    // inverse token (!<[token-id]>)
    LLMCOMMS_GRETYPE_TOKEN_NOT      = 9,
};

typedef struct llmcomms_grammar_element {
    enum llmcomms_gretype type;
    uint32_t           value; // Unicode code point, rule ID, or token ID
} llmcomms_grammar_element;

struct llmcomms_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct llmcomms_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    llmcomms_partial_utf8   partial_utf8;
    llmcomms_token          id;
};

using llmcomms_grammar_rule  = std::vector<      llmcomms_grammar_element>;
using llmcomms_grammar_stack = std::vector<const llmcomms_grammar_element *>;

using llmcomms_grammar_rules      = std::vector<llmcomms_grammar_rule>;
using llmcomms_grammar_stacks     = std::vector<llmcomms_grammar_stack>;
using llmcomms_grammar_candidates = std::vector<llmcomms_grammar_candidate>;

// TODO: remove, needed for tests atm
const llmcomms_grammar_rules  & llmcomms_grammar_get_rules (const struct llmcomms_grammar * grammar);
      llmcomms_grammar_stacks & llmcomms_grammar_get_stacks(      struct llmcomms_grammar * grammar);

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `llmcomms_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
void llmcomms_grammar_accept(struct llmcomms_grammar * grammar, uint32_t chr);

std::vector<llmcomms_grammar_candidate> llmcomms_grammar_reject_candidates_for_stack(
        const llmcomms_grammar_rules      & rules,
        const llmcomms_grammar_stack      & stack,
        const llmcomms_grammar_candidates & candidates);

struct llmcomms_grammar_parser {
    const llmcomms_vocab * vocab;
    std::map<std::string, uint32_t> symbol_ids;

    llmcomms_grammar_rules rules;

    llmcomms_grammar_parser(const struct llmcomms_vocab * vocab = nullptr) : vocab(vocab) {}

    llmcomms_grammar_stack c_rules() const;

    uint32_t get_symbol_id(const char * src, size_t len);
    uint32_t generate_symbol_id(const std::string & base_name);

    void add_rule(uint32_t rule_id, const llmcomms_grammar_rule & rule);

    const char * parse_alternates(
            const char        * src,
            const std::string & rule_name,
            uint32_t            rule_id,
            bool                is_nested,
            uint32_t            depth = 0);

    const char * parse_sequence(
            const char         * src,
            const std::string  & rule_name,
            llmcomms_grammar_rule & rule,
            bool               is_nested,
            uint32_t           depth = 0);

    const char * parse_rule(const char * src);

    bool parse(const char * src);
    void print(FILE * file);
};

struct llmcomms_grammar_trigger_pattern {
    std::string pattern;
    std::regex  regex;

    size_t find(const std::string & input) const;
};

struct llmcomms_grammar {
    // maintain a list of llmcomms_tokens and their positions in the trigger_buffer
    using token_pos = std::pair<llmcomms_token, std::pair<size_t, size_t>>;

    // note: allow null vocab for testing (not great)
    const llmcomms_vocab * vocab;

    const llmcomms_grammar_rules  rules;  // TODO: shared ptr
          llmcomms_grammar_stacks stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llmcomms_partial_utf8 partial_utf8;

    // lazy grammars wait for trigger words or tokens before constraining the sampling.
    // we still have trigger_tokens for non-lazy grammars to force printing of special trigger tokens.
    // (useful e.g. for tool_choice=required)
    bool                     lazy             = false;
    bool                     awaiting_trigger = false; // Initialized to true for lazy grammars only
    std::string              trigger_buffer;           // Output buffered by lazy grammar. Will be cleared once trigger is found.
    std::vector<token_pos>   trigger_buffer_positions; // Tokens buffered by lazy grammar. Used to replay when a trigger is found.
    std::vector<llmcomms_token> trigger_tokens;           // Tokens that trigger a lazy grammar, or tokens to force printing of (even if special).
    std::vector<llmcomms_grammar_trigger_pattern>
                             trigger_patterns;         // Regular expressions that trigger a lazy grammar. Must be a full match of the entire generated
                                                       // string, and the grammar will be given the string from the first match group onwards.

};

//
// internal API
//

// note: needed for tests (not great)
struct llmcomms_grammar * llmcomms_grammar_init_impl(
        const struct llmcomms_vocab * vocab,
        const llmcomms_grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index);

struct llmcomms_grammar * llmcomms_grammar_init_impl(
        const struct llmcomms_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llmcomms_token * trigger_tokens,
                            size_t num_trigger_tokens);

void llmcomms_grammar_free_impl(struct llmcomms_grammar * grammar);

struct llmcomms_grammar * llmcomms_grammar_clone_impl(const struct llmcomms_grammar & grammar);

// TODO: move the API below as member functions of llmcomms_grammar
void llmcomms_grammar_apply_impl(
        const struct llmcomms_grammar & grammar,
            llmcomms_token_data_array * cur_p);

void llmcomms_grammar_accept_impl(
              struct llmcomms_grammar & grammar,
                       llmcomms_token   token);

void llmcomms_grammar_accept_str(
              struct llmcomms_grammar & grammar,
                 const std::string & piece);

void llmcomms_grammar_accept_token(
              struct llmcomms_grammar & grammar,
                       llmcomms_token   token,
                 const std::string & piece);
