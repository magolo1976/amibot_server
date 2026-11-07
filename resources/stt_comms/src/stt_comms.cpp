#include "stt_comms.h"
#include "sttcomms-core.h"

#include <thread>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <cmath>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace stt_comms {

namespace {
    static void silent_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
        (void)level;
        (void)text;
        (void)user_data;
    }

    static bool decode_utf8_codepoint(const std::string& s, size_t& i, uint32_t& cp) {
        const unsigned char c0 = static_cast<unsigned char>(s[i]);
        if (c0 < 0x80) {
            cp = c0;
            i += 1;
            return true;
        }
        if ((c0 & 0xE0) == 0xC0) {
            if (i + 1 >= s.size()) { i += 1; return false; }
            const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
            if ((c1 & 0xC0) != 0x80) { i += 1; return false; }
            cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
            i += 2;
            return cp >= 0x80;
        }
        if ((c0 & 0xF0) == 0xE0) {
            if (i + 2 >= s.size()) { i += 1; return false; }
            const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
            if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) { i += 1; return false; }
            cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
            i += 3;
            return cp >= 0x800;
        }
        if ((c0 & 0xF8) == 0xF0) {
            if (i + 3 >= s.size()) { i += 1; return false; }
            const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
            const unsigned char c3 = static_cast<unsigned char>(s[i + 3]);
            if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) { i += 1; return false; }
            cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
            i += 4;
            return cp >= 0x10000 && cp <= 0x10FFFF;
        }
        i += 1;
        return false;
    }

    static void append_utf8_codepoint(std::string& out, uint32_t cp) {
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

    static bool is_allowed_clean_codepoint(uint32_t cp, uint32_t& mapped_cp) {
        cp = fold_latin_diacritic(cp);
        mapped_cp = cp;
        if (cp == '\n' || cp == '\t') {
            return true;
        }
        if (cp == '\r') {
            return false;
        }
        if (cp >= 0x20 && cp <= 0x7E) {
            const char c = static_cast<char>(cp);
            if (c == '*' || c == '#' || c == '`' || c == '_' || c == '~' ||
                c == '^' || c == '|' || c == '\\' || c == '<' || c == '>' ||
                c == '[' || c == ']' || c == '{' || c == '}' || c == '@') {
                return false;
            }
            return true;
        }
        if (cp == 0x00A0) {
            mapped_cp = ' ';
            return true;
        }
        if (cp == 0x00A1 || cp == 0x00BF) {
            return false;
        }
        if (cp == 0x2018 || cp == 0x2019) {
            mapped_cp = '\'';
            return true;
        }
        if (cp == 0x201C || cp == 0x201D || cp == 0x00AB || cp == 0x00BB) {
            mapped_cp = '"';
            return true;
        }
        if (cp == 0x2013 || cp == 0x2014) {
            mapped_cp = '-';
            return true;
        }
        if (cp == 0x2026) {
            mapped_cp = '.';
            return true;
        }
        if ((cp >= 0x00C0 && cp <= 0x024F && cp != 0x00D7 && cp != 0x00F7) ||
            (cp >= 0x0370 && cp <= 0x052F) ||
            (cp >= 0x0590 && cp <= 0x06FF) ||
            (cp >= 0x0900 && cp <= 0x0DFF) ||
            (cp >= 0x0E00 && cp <= 0x0E7F) ||
            (cp >= 0x1E00 && cp <= 0x1EFF) ||
            (cp >= 0x3040 && cp <= 0x30FF) ||
            (cp >= 0x3400 && cp <= 0x9FFF) ||
            (cp >= 0xAC00 && cp <= 0xD7AF)) {
            return true;
        }
        return false;
    }

    // Elimina bloques de etiquetas de audio como [BLANK_AUDIO], [MUSIC], (music), etc.
    static std::string strip_audio_tags(const std::string& input) {
        std::string res;
        res.reserve(input.size());
        bool in_brackets = false;
        for (size_t i = 0; i < input.size(); ++i) {
            char c = input[i];
            if (c == '[') {
                in_brackets = true;
                continue;
            }
            if (c == ']' && in_brackets) {
                in_brackets = false;
                continue;
            }
            if (!in_brackets) {
                res.push_back(c);
            }
        }
        return res;
    }
}

std::string clean_text(const std::string& input) {
    std::string stripped = strip_audio_tags(input);
    std::string out;
    out.reserve(stripped.size());
    size_t i = 0;
    bool prev_space = false;

    while (i < stripped.size()) {
        uint32_t cp = 0;
        if (!decode_utf8_codepoint(stripped, i, cp)) {
            continue;
        }
        uint32_t mapped = 0;
        if (is_allowed_clean_codepoint(cp, mapped)) {
            if (mapped == ' ') {
                if (prev_space || out.empty()) {
                    continue;
                }
                prev_space = true;
            } else {
                prev_space = false;
            }
            append_utf8_codepoint(out, mapped);
        }
    }

    while (!out.empty() && (out.back() == ' ' || out.back() == '\n' || out.back() == '\t')) {
        out.pop_back();
    }
    return out;
}

bool read_wav_bytes_16k_mono(const uint8_t* wav_bytes, size_t n_bytes, std::vector<float>& out_samples, std::string& out_error) {
    out_samples.clear();
    if (!wav_bytes || n_bytes < 44) {
        out_error = "Buffer WAV vacio o demasiado corto";
        return false;
    }
    if (std::memcmp(wav_bytes, "RIFF", 4) != 0 || std::memcmp(wav_bytes + 8, "WAVE", 4) != 0) {
        out_error = "El buffer no tiene formato RIFF/WAVE valido";
        return false;
    }

    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    const uint8_t* raw_data = nullptr;
    size_t raw_size = 0;

    size_t pos = 12;
    while (pos + 8 <= n_bytes) {
        const char* chunk_id = reinterpret_cast<const char*>(wav_bytes + pos);
        uint32_t chunk_size = 0;
        std::memcpy(&chunk_size, wav_bytes + pos + 4, 4);
        pos += 8;

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            if (chunk_size < 16 || pos + 16 > n_bytes) {
                out_error = "Chunk fmt invalido en WAV";
                return false;
            }
            std::memcpy(&audio_format, wav_bytes + pos, 2);
            std::memcpy(&num_channels, wav_bytes + pos + 2, 2);
            std::memcpy(&sample_rate, wav_bytes + pos + 4, 4);
            std::memcpy(&bits_per_sample, wav_bytes + pos + 14, 2);
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            raw_data = wav_bytes + pos;
            raw_size = std::min(static_cast<size_t>(chunk_size), n_bytes - pos);
            break;
        }
        pos += chunk_size + (chunk_size & 1);
    }

    if (!raw_data || raw_size == 0 || num_channels == 0 || sample_rate == 0) {
        out_error = "El archivo WAV no contiene muestras de audio validas";
        return false;
    }

    std::vector<float> mono_samples;
    if (audio_format == 1 && bits_per_sample == 16) {
        const size_t frames = (raw_size / sizeof(int16_t)) / num_channels;
        mono_samples.resize(frames);
        const int16_t* pcm = reinterpret_cast<const int16_t*>(raw_data);
        for (size_t f = 0; f < frames; ++f) {
            float acc = 0.0f;
            for (uint16_t c = 0; c < num_channels; ++c) {
                acc += static_cast<float>(pcm[f * num_channels + c]) / 32768.0f;
            }
            mono_samples[f] = acc / static_cast<float>(num_channels);
        }
    } else if (audio_format == 3 && bits_per_sample == 32) {
        const size_t frames = (raw_size / sizeof(float)) / num_channels;
        mono_samples.resize(frames);
        const float* pcm = reinterpret_cast<const float*>(raw_data);
        for (size_t f = 0; f < frames; ++f) {
            float acc = 0.0f;
            for (uint16_t c = 0; c < num_channels; ++c) {
                acc += pcm[f * num_channels + c];
            }
            mono_samples[f] = acc / static_cast<float>(num_channels);
        }
    } else if (audio_format == 1 && bits_per_sample == 8) {
        const size_t frames = raw_size / num_channels;
        mono_samples.resize(frames);
        for (size_t f = 0; f < frames; ++f) {
            float acc = 0.0f;
            for (uint16_t c = 0; c < num_channels; ++c) {
                acc += (static_cast<float>(raw_data[f * num_channels + c]) - 128.0f) / 128.0f;
            }
            mono_samples[f] = acc / static_cast<float>(num_channels);
        }
    } else {
        out_error = "Formato WAV no soportado (se requiere PCM 8/16-bit o Float 32-bit)";
        return false;
    }

    if (sample_rate == STTCOMMS_SAMPLE_RATE) {
        out_samples = std::move(mono_samples);
        return true;
    }

    const double ratio = static_cast<double>(sample_rate) / static_cast<double>(STTCOMMS_SAMPLE_RATE);
    const size_t out_len = static_cast<size_t>(static_cast<double>(mono_samples.size()) / ratio);
    out_samples.resize(out_len);
    for (size_t i = 0; i < out_len; ++i) {
        const double src_pos = static_cast<double>(i) * ratio;
        const size_t idx0 = static_cast<size_t>(src_pos);
        const size_t idx1 = std::min(idx0 + 1, mono_samples.size() - 1);
        const float frac = static_cast<float>(src_pos - static_cast<double>(idx0));
        out_samples[i] = mono_samples[idx0] * (1.0f - frac) + mono_samples[idx1] * frac;
    }
    return true;
}

bool read_wav_16k_mono(const std::string& wav_path, std::vector<float>& out_samples, std::string& out_error) {
    out_samples.clear();
    std::ifstream file(wav_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        out_error = "No se pudo abrir el archivo WAV: " + wav_path;
        return false;
    }
    const std::streamsize sz = file.tellg();
    if (sz < 44) {
        out_error = "Archivo WAV vacio o demasiado corto";
        return false;
    }
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(sz));
    if (!file.read(reinterpret_cast<char*>(bytes.data()), sz)) {
        out_error = "Error al leer el archivo WAV: " + wav_path;
        return false;
    }
    return read_wav_bytes_16k_mono(bytes.data(), bytes.size(), out_samples, out_error);
}

class STTClient::Impl {
public:
    sttcomms_context* ctx = nullptr;
    Config cfg;
    std::string last_error;

    ~Impl() {
        close();
    }

    void close() {
        if (ctx) {
            sttcomms_free(ctx);
            ctx = nullptr;
        }
    }
};

STTClient::STTClient() : pimpl(new Impl()) {}

STTClient::~STTClient() {
    delete pimpl;
}

bool STTClient::init(const Config& config) {
    pimpl->close();
    pimpl->cfg = config;

#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    if (config.model_path.empty()) {
        pimpl->last_error = "La ruta del modelo STT (model_path) esta vacia";
        return false;
    }

    if (!config.verbose) {
        sttcomms_log_set(silent_log_callback, nullptr);
    }

    sttcomms_context_params cparams = sttcomms_context_default_params();
    cparams.use_gpu = config.use_gpu;
    cparams.flash_attn = false;

    pimpl->ctx = sttcomms_init_from_file_with_params(config.model_path.c_str(), cparams);
    if (!pimpl->ctx) {
        pimpl->last_error = "Fallo al cargar el modelo STT desde: " + config.model_path;
        return false;
    }

    return true;
}

bool STTClient::is_ready() const {
    return pimpl->ctx != nullptr;
}

void STTClient::close() {
    pimpl->close();
}

std::string STTClient::get_last_error() const {
    return pimpl->last_error;
}

std::string STTClient::transcribe_samples(const float* samples, size_t n_samples, const TranscribeOptions& options) {
    if (!is_ready()) {
        pimpl->last_error = "STTClient no esta inicializado";
        return "";
    }
    if (!samples || n_samples == 0) {
        pimpl->last_error = "El buffer de muestras de audio esta vacio";
        return "";
    }

    sttcomms_full_params wparams = sttcomms_full_default_params(STTCOMMS_SAMPLING_GREEDY);
    int32_t threads = pimpl->cfg.n_threads;
    if (threads <= 0) {
        threads = std::min(8, std::max(1, static_cast<int32_t>(std::thread::hardware_concurrency())));
    }
    wparams.n_threads = threads;
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_special = false;
    wparams.print_timestamps = false;
    wparams.translate = options.translate;
    wparams.max_tokens = options.max_tokens;
    wparams.no_timestamps = true;
    wparams.no_context = true;
    wparams.single_segment = true;
    wparams.greedy.best_of = 1;

    if (options.language.empty() || options.language == "auto") {
        wparams.language = "auto";
        wparams.detect_language = false;
    } else {
        wparams.language = options.language.c_str();
    }

    if (sttcomms_full(pimpl->ctx, wparams, samples, static_cast<int>(n_samples)) != 0) {
        pimpl->last_error = "Fallo al ejecutar la inferencia de voz a texto (sttcomms_full)";
        return "";
    }

    std::string raw_text;
    const int n_segments = sttcomms_full_n_segments(pimpl->ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char* seg = sttcomms_full_get_segment_text(pimpl->ctx, i);
        if (seg) {
            raw_text += seg;
        }
    }

    return clean_text(raw_text);
}

std::string STTClient::transcribe_wav_bytes(const uint8_t* wav_bytes, size_t n_bytes, const TranscribeOptions& options) {
    std::vector<float> samples;
    std::string err;
    if (!read_wav_bytes_16k_mono(wav_bytes, n_bytes, samples, err)) {
        pimpl->last_error = err;
        return "";
    }
    return transcribe_samples(samples.data(), samples.size(), options);
}

std::string STTClient::transcribe_wav(const std::string& wav_path, const TranscribeOptions& options) {
    std::vector<float> samples;
    std::string err;
    if (!read_wav_16k_mono(wav_path, samples, err)) {
        pimpl->last_error = err;
        return "";
    }
    return transcribe_samples(samples.data(), samples.size(), options);
}

// ========================================================================
// Implementación C ABI
// ========================================================================
extern "C" {

stt_comms_handle stt_comms_create() {
    return new STTClient();
}

int stt_comms_init(stt_comms_handle handle, const char* model_path, int use_gpu, int n_threads) {
    if (!handle || !model_path) return -1;
    STTClient* client = static_cast<STTClient*>(handle);
    Config cfg;
    cfg.model_path = model_path;
    cfg.use_gpu = (use_gpu != 0);
    cfg.n_threads = n_threads;
    return client->init(cfg) ? 0 : -2;
}

int stt_comms_transcribe_wav(stt_comms_handle handle, const char* wav_path, const char* language, char* out_buf, int max_out_len) {
    if (!handle || !wav_path || !out_buf || max_out_len <= 0) return -1;
    STTClient* client = static_cast<STTClient*>(handle);
    TranscribeOptions opts;
    opts.language = (language && language[0] != '\0') ? language : "auto";
    std::string result = client->transcribe_wav(wav_path, opts);
    if (result.empty() && !client->get_last_error().empty()) {
        return -2;
    }
    int copy_len = std::min(static_cast<int>(result.size()), max_out_len - 1);
    std::memcpy(out_buf, result.data(), copy_len);
    out_buf[copy_len] = '\0';
    return copy_len;
}

int stt_comms_transcribe_pcm(stt_comms_handle handle, const float* samples, int n_samples, const char* language, char* out_buf, int max_out_len) {
    if (!handle || !samples || n_samples <= 0 || !out_buf || max_out_len <= 0) return -1;
    STTClient* client = static_cast<STTClient*>(handle);
    TranscribeOptions opts;
    opts.language = (language && language[0] != '\0') ? language : "auto";
    std::string result = client->transcribe_samples(samples, static_cast<size_t>(n_samples), opts);
    if (result.empty() && !client->get_last_error().empty()) {
        return -2;
    }
    int copy_len = std::min(static_cast<int>(result.size()), max_out_len - 1);
    std::memcpy(out_buf, result.data(), copy_len);
    out_buf[copy_len] = '\0';
    return copy_len;
}

void stt_comms_destroy(stt_comms_handle handle) {
    if (handle) {
        delete static_cast<STTClient*>(handle);
    }
}

} // extern "C"

} // namespace stt_comms
