#include "llm_comms.h"
#include "llmcomms-core.h"

#include <vector>
#include <thread>
#include <cstring>
#include <iostream>
#include <cctype>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif

namespace llm_comms {

static void ensure_utf8_console() {
#ifdef _WIN32
    static bool initialized = false;
    if (!initialized) {
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
        initialized = true;
    }
#endif
}

static bool is_allowed_codepoint(uint32_t cp) {
    // Saltos de linea y espacio normal
    if (cp == '\n' || cp == ' ') {
        return true;
    }
    if (cp == '\t') {
        return true;
    }
    // Caracteres de control ASCII o DEL
    if (cp < 0x20 || (cp >= 0x7F && cp <= 0xA0)) {
        return false;
    }

    // Rango ASCII imprimible
    if (cp < 0x80) {
        // Letras y numeros
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || (cp >= '0' && cp <= '9')) {
            return true;
        }
        // Puntuacion basica limpia permitida (sin markdown *, _, #, `, ~, ni corchetes/llaves/simbolos raros)
        switch (cp) {
            case '.': case ',': case ':': case ';':
            case '!': case '?': case '(': case ')':
            case '-': case '\'': case '\"': case '/':
            case '%': case '+': case '=':
                return true;
            default:
                return false;
        }
    }

    // Simbolos Latin-1 suplementarios (incluyendo ¡ U+00A1, ¿ U+00BF, «, », °, ±, etc.)
    if (cp >= 0x00A1 && cp <= 0x00BF) {
        return false;
    }
    // Signos matematicos × (U+00D7) y ÷ (U+00F7)
    if (cp == 0x00D7 || cp == 0x00F7) {
        return false;
    }

    // Bloques de puntuacion general especial, flechas, simbolos matematicos, cajas, dingbats y simbolos varios
    // (U+2000 a U+2BFF incluye emojis antiguos ✨, ✅, ⚠️, ❤️, flechas, etc.)
    if (cp >= 0x2000 && cp <= 0x2BFF) {
        return false;
    }

    // Simbolos CJK y puntuacion decorativa (U+3000..U+303F)
    if (cp >= 0x3000 && cp <= 0x303F) {
        return false;
    }

    // Area de uso privado, selectores de variacion (FE00..FE0F) y especiales (FFF0..FFFF)
    if ((cp >= 0xE000 && cp <= 0xF8FF) || (cp >= 0xFE00 && cp <= 0xFEFF) || (cp >= 0xFFF0 && cp <= 0xFFFF)) {
        return false;
    }

    // Todos los bloques de Emojis modernos, pictografias, transporte, banderas y simbolos suplementarios
    if (cp >= 0x1F000) {
        return false;
    }

    // El resto son letras legítimas de cualquier idioma (Español/Latin extendido áéíóúñü, Cirílico, Griego, Árabe, CJK, etc.)
    return true;
}

static uint32_t fold_latin_diacritic(uint32_t cp) {
    switch (cp) {
        case 0x00C0: case 0x00C1: case 0x00C2: case 0x00C3: case 0x00C4: case 0x00C5: return 'A';
        case 0x00C7: return 'C';
        case 0x00C8: case 0x00C9: case 0x00CA: case 0x00CB: return 'E';
        case 0x00CC: case 0x00CD: case 0x00CE: case 0x00CF: return 'I';
        case 0x00D1: return 'N';
        case 0x00D2: case 0x00D3: case 0x00D4: case 0x00D5: case 0x00D6: return 'O';
        case 0x00D9: case 0x00DA: case 0x00DB: case 0x00DC: return 'U';
        case 0x00DD: return 'Y';
        case 0x00E0: case 0x00E1: case 0x00E2: case 0x00E3: case 0x00E4: case 0x00E5: return 'a';
        case 0x00E7: return 'c';
        case 0x00E8: case 0x00E9: case 0x00EA: case 0x00EB: return 'e';
        case 0x00EC: case 0x00ED: case 0x00EE: case 0x00EF: return 'i';
        case 0x00F1: return 'n';
        case 0x00F2: case 0x00F3: case 0x00F4: case 0x00F5: case 0x00F6: return 'o';
        case 0x00F9: case 0x00FA: case 0x00FB: case 0x00FC: return 'u';
        case 0x00FD: case 0x00FF: return 'y';
        default: return cp;
    }
}

static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// Procesa un buffer UTF-8 extrayendo solo codepoints completos, dejando en 'pending' cualquier secuencia incompleta al final
static std::string clean_utf8_stream_chunk(std::string& buffer) {
    std::string out;
    out.reserve(buffer.size());

    size_t i = 0;
    const size_t n = buffer.size();

    while (i < n) {
        unsigned char c0 = static_cast<unsigned char>(buffer[i]);
        uint32_t cp = 0;
        size_t seq_len = 0;

        if ((c0 & 0x80) == 0) {
            cp = c0;
            seq_len = 1;
        } else if ((c0 & 0xE0) == 0xC0) {
            seq_len = 2;
            if (i + seq_len > n) break; // esperar al siguiente token
            unsigned char c1 = static_cast<unsigned char>(buffer[i + 1]);
            if ((c1 & 0xC0) != 0x80) { ++i; continue; }
            cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        } else if ((c0 & 0xF0) == 0xE0) {
            seq_len = 3;
            if (i + seq_len > n) break; // esperar al siguiente token
            unsigned char c1 = static_cast<unsigned char>(buffer[i + 1]);
            unsigned char c2 = static_cast<unsigned char>(buffer[i + 2]);
            if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) { ++i; continue; }
            cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
        } else if ((c0 & 0xF8) == 0xF0) {
            seq_len = 4;
            if (i + seq_len > n) break; // esperar al siguiente token
            unsigned char c1 = static_cast<unsigned char>(buffer[i + 1]);
            unsigned char c2 = static_cast<unsigned char>(buffer[i + 2]);
            unsigned char c3 = static_cast<unsigned char>(buffer[i + 3]);
            if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) { ++i; continue; }
            cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
        } else {
            // Byte invalido suelto
            ++i;
            continue;
        }

        if (cp == '\t') {
            cp = ' ';
        }
        if (cp == 0x00A0) {
            cp = ' '; // non-breaking space -> espacio normal
        }
        cp = fold_latin_diacritic(cp);

        if (is_allowed_codepoint(cp)) {
            append_utf8(out, cp);
        }

        i += seq_len;
    }

    buffer.erase(0, i);
    return out;
}

std::string clean_text(const std::string& input) {
    std::string temp = input;
    std::string raw_clean = clean_utf8_stream_chunk(temp);

    // Colapsar espacios multiples y recortar espacios al inicio/final
    std::string normalized;
    normalized.reserve(raw_clean.size());
    bool prev_space = false;
    int newline_count = 0;

    for (char c : raw_clean) {
        if (c == '\n') {
            if (newline_count < 2) {
                normalized.push_back('\n');
            }
            ++newline_count;
            prev_space = false;
        } else if (c == ' ') {
            if (!prev_space && !normalized.empty() && normalized.back() != '\n') {
                normalized.push_back(' ');
            }
            prev_space = true;
        } else {
            normalized.push_back(c);
            prev_space = false;
            newline_count = 0;
        }
    }

    // Trim inicio y fin
    size_t start = normalized.find_first_not_of(" \n\r\t");
    if (start == std::string::npos) return "";
    size_t end = normalized.find_last_not_of(" \n\r\t");
    return normalized.substr(start, end - start + 1);
}

static void batch_set_tokens(llmcomms_batch_ext* batch, const llmcomms_token* tokens, int32_t count, llmcomms_pos pos_0) {
    llmcomms_batch_ext_clear(batch);
    for (int32_t i = 0; i < count; ++i) {
        const int32_t idx = llmcomms_batch_ext_add_token(batch, 0, tokens[i]);
        const llmcomms_pos pos = pos_0 + i;
        llmcomms_batch_ext_set_pos(batch, idx, &pos);
    }
    llmcomms_batch_ext_set_output_logits(batch, count - 1, true);
}

class LLMClient::Impl {
public:
    llmcomms_model* model = nullptr;
    const llmcomms_vocab* vocab = nullptr;
    Config config;
    std::string last_error;

    Impl() {
        ensure_utf8_console();
    }
    ~Impl() {
        close();
    }

    void close() {
        if (model) {
            llmcomms_model_free(model);
            model = nullptr;
            vocab = nullptr;
        }
    }

    bool init(const Config& cfg) {
        ensure_utf8_console();
        close();
        config = cfg;

        if (!config.verbose) {
            llmcomms_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
        }

        llmcomms_backend_init();

        llmcomms_model_params mparams = llmcomms_model_default_params();
        mparams.n_gpu_layers = config.n_gpu_layers;

        model = llmcomms_model_load_from_file(config.model_path.c_str(), mparams);
        if (!model) {
            last_error = "Error al cargar el modelo desde: " + config.model_path;
            return false;
        }

        vocab = llmcomms_model_get_vocab(model);
        return true;
    }

    std::string format_chat_prompt(const std::vector<ChatMessage>& messages, bool add_assistant) const {
        if (!model) return "";

        const char* tmpl = llmcomms_model_chat_template(model, nullptr);

        std::vector<llmcomms_chat_message> raw_msgs;
        raw_msgs.reserve(messages.size());
        for (const auto& msg : messages) {
            raw_msgs.push_back({ msg.role.c_str(), msg.content.c_str() });
        }

        int32_t needed = llmcomms_chat_apply_template(tmpl, raw_msgs.data(), raw_msgs.size(), add_assistant, nullptr, 0);
        if (needed <= 0) {
            std::string fallback;
            for (const auto& m : messages) {
                fallback += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
            }
            if (add_assistant) {
                fallback += "<|im_start|>assistant\n";
            }
            return fallback;
        }

        std::vector<char> buf(needed + 1);
        llmcomms_chat_apply_template(tmpl, raw_msgs.data(), raw_msgs.size(), add_assistant, buf.data(), (int32_t)buf.size());
        return std::string(buf.data(), needed);
    }

    std::string send_text(const std::string& prompt, const GenerateOptions& options) {
        ensure_utf8_console();
        if (!model || !vocab) {
            last_error = "El modelo no esta inicializado";
            return "";
        }

        // 1. Tokenizar prompt
        const int n_prompt = -llmcomms_tokenize(vocab, prompt.c_str(), (int)prompt.size(), NULL, 0, true, true);
        if (n_prompt <= 0) {
            last_error = "Error o prompt vacio al tokenizar";
            return "";
        }

        std::vector<llmcomms_token> prompt_tokens(n_prompt);
        if (llmcomms_tokenize(vocab, prompt.c_str(), (int)prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            last_error = "Fallo la tokenizacion del prompt";
            return "";
        }

        // 2. Inicializar contexto
        llmcomms_context_params cparams = llmcomms_context_default_params();
        int req_ctx = n_prompt + options.max_tokens;
        cparams.n_ctx = (req_ctx > config.n_ctx) ? req_ctx : config.n_ctx;
        cparams.n_batch = n_prompt > 512 ? n_prompt : 512;
        int threads = config.n_threads > 0 ? config.n_threads : (int)std::thread::hardware_concurrency();
        if (threads <= 0) threads = 4;
        cparams.n_threads = threads;
        cparams.n_threads_batch = threads;
        cparams.no_perf = true;

        llmcomms_context* ctx = llmcomms_init_from_model(model, cparams);
        if (!ctx) {
            last_error = "No se pudo crear el contexto de inferencia llmcomms_context";
            return "";
        }

        // 3. Configurar Sampler (temperatura, top_p)
        auto sparams = llmcomms_sampler_chain_default_params();
        sparams.no_perf = true;
        llmcomms_sampler* smpl = llmcomms_sampler_chain_init(sparams);
        if (options.temperature <= 0.0f) {
            llmcomms_sampler_chain_add(smpl, llmcomms_sampler_init_greedy());
        } else {
            llmcomms_sampler_chain_add(smpl, llmcomms_sampler_init_temp(options.temperature));
            llmcomms_sampler_chain_add(smpl, llmcomms_sampler_init_top_p(options.top_p, 1));
            llmcomms_sampler_chain_add(smpl, llmcomms_sampler_init_dist(1234));
        }

        // 4. Preparar batch inicial con el prompt
        llmcomms_batch_ext* batch = llmcomms_batch_ext_init(ctx);
        batch_set_tokens(batch, prompt_tokens.data(), (int32_t)prompt_tokens.size(), 0);

        int n_tokens = (int)prompt_tokens.size();
        std::string raw_result;
        std::string utf8_stream_pending;
        bool started_streaming = false;

        // 5. Bucle de decodificacion y generacion
        for (int n_pos = 0; n_pos + n_tokens < (int)prompt_tokens.size() + options.max_tokens; ) {
            if (llmcomms_process(ctx, LLMCOMMS_PROCESS_TYPE_DECODE, batch)) {
                last_error = "Fallo en la evaluacion (llmcomms_process DECODE)";
                break;
            }

            n_pos += n_tokens;

            llmcomms_token new_token_id = llmcomms_sampler_sample(smpl, ctx, -1);
            if (llmcomms_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char piece_buf[256];
            int n_piece = llmcomms_token_to_piece(vocab, new_token_id, piece_buf, sizeof(piece_buf), 0, true);
            if (n_piece > 0) {
                std::string piece(piece_buf, n_piece);
                raw_result += piece;

                if (options.stream_cb) {
                    utf8_stream_pending += piece;
                    std::string clean_chunk = clean_utf8_stream_chunk(utf8_stream_pending);
                    if (!started_streaming) {
                        size_t first_non_ws = clean_chunk.find_first_not_of(" \n\r\t");
                        if (first_non_ws == std::string::npos) {
                            clean_chunk.clear();
                        } else {
                            clean_chunk = clean_chunk.substr(first_non_ws);
                            started_streaming = true;
                        }
                    }
                    if (!clean_chunk.empty()) {
                        if (!options.stream_cb(clean_chunk)) {
                            break;
                        }
                    }
                }
            }

            batch_set_tokens(batch, &new_token_id, 1, n_pos);
            n_tokens = 1;
        }

        // Limpieza de recursos por solicitud
        llmcomms_batch_ext_free(batch);
        llmcomms_sampler_free(smpl);
        llmcomms_free(ctx);

        return clean_text(raw_result);
    }
};

LLMClient::LLMClient() : pimpl(new Impl()) {}
LLMClient::~LLMClient() { delete pimpl; }

bool LLMClient::init(const Config& config) { return pimpl->init(config); }
bool LLMClient::is_ready() const { return pimpl->model != nullptr; }
void LLMClient::close() { pimpl->close(); }
std::string LLMClient::get_last_error() const { return pimpl->last_error; }

std::string LLMClient::send_text(const std::string& prompt, const GenerateOptions& options) {
    return pimpl->send_text(prompt, options);
}

std::string LLMClient::send_chat(const std::vector<ChatMessage>& messages, const GenerateOptions& options) {
    std::string prompt = pimpl->format_chat_prompt(messages, true);
    return pimpl->send_text(prompt, options);
}

std::string LLMClient::format_chat_prompt(const std::vector<ChatMessage>& messages, bool add_assistant) const {
    return pimpl->format_chat_prompt(messages, add_assistant);
}

// Interfaz C
extern "C" {
    llm_comms_handle llm_comms_create() {
        return new LLMClient();
    }

    int llm_comms_init(llm_comms_handle handle, const char* model_path, int n_ctx, int n_gpu_layers, int n_threads) {
        if (!handle || !model_path) return -1;
        LLMClient* client = static_cast<LLMClient*>(handle);
        Config cfg;
        cfg.model_path = model_path;
        if (n_ctx > 0) cfg.n_ctx = n_ctx;
        cfg.n_gpu_layers = n_gpu_layers;
        if (n_threads > 0) cfg.n_threads = n_threads;
        return client->init(cfg) ? 0 : -1;
    }

    int llm_comms_send_text(llm_comms_handle handle, const char* prompt, char* out_buf, int max_out_len, int max_tokens, float temperature) {
        if (!handle || !prompt || !out_buf || max_out_len <= 0) return -1;
        LLMClient* client = static_cast<LLMClient*>(handle);
        GenerateOptions opts;
        if (max_tokens > 0) opts.max_tokens = max_tokens;
        opts.temperature = temperature;
        std::string response = client->send_text(prompt, opts);
        int copy_len = (int)response.size();
        if (copy_len >= max_out_len) copy_len = max_out_len - 1;
        std::memcpy(out_buf, response.data(), copy_len);
        out_buf[copy_len] = '\0';
        return copy_len;
    }

    int llm_comms_send_chat_simple(llm_comms_handle handle, const char* system_prompt, const char* user_prompt, char* out_buf, int max_out_len, int max_tokens, float temperature) {
        if (!handle || !user_prompt || !out_buf || max_out_len <= 0) return -1;
        LLMClient* client = static_cast<LLMClient*>(handle);
        GenerateOptions opts;
        if (max_tokens > 0) opts.max_tokens = max_tokens;
        opts.temperature = temperature;
        std::vector<ChatMessage> msgs;
        if (system_prompt && std::strlen(system_prompt) > 0) {
            msgs.push_back({ "system", system_prompt });
        }
        msgs.push_back({ "user", user_prompt });
        std::string response = client->send_chat(msgs, opts);
        int copy_len = (int)response.size();
        if (copy_len >= max_out_len) copy_len = max_out_len - 1;
        std::memcpy(out_buf, response.data(), copy_len);
        out_buf[copy_len] = '\0';
        return copy_len;
    }

    void llm_comms_destroy(llm_comms_handle handle) {
        if (handle) {
            delete static_cast<LLMClient*>(handle);
        }
    }
}

} // namespace llm_comms
