#include "api_core.h"

#include <filesystem>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <cstdlib>
#include <cctype>
#include <cstring>

namespace fs = std::filesystem;

namespace llm_api {

std::string json_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (unsigned char c : input) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

static size_t find_json_key_value_pos(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = 0;
    while ((pos = json.find(pattern, pos)) != std::string::npos) {
        size_t after_key = pos + pattern.size();
        while (after_key < json.size() && std::isspace(static_cast<unsigned char>(json[after_key]))) {
            ++after_key;
        }
        if (after_key < json.size() && json[after_key] == ':') {
            size_t val_pos = after_key + 1;
            while (val_pos < json.size() && std::isspace(static_cast<unsigned char>(json[val_pos]))) {
                ++val_pos;
            }
            return val_pos;
        }
        pos = after_key;
    }
    return std::string::npos;
}

std::string json_extract_string(const std::string& json, const std::string& key) {
    size_t val_pos = find_json_key_value_pos(json, key);
    if (val_pos == std::string::npos || val_pos >= json.size() || json[val_pos] != '\"') {
        return "";
    }

    std::string result;
    for (size_t i = val_pos + 1; i < json.size(); ++i) {
        char c = json[i];
        if (c == '\\' && i + 1 < json.size()) {
            char next = json[++i];
            switch (next) {
                case '\"': result += '\"'; break;
                case '\\': result += '\\'; break;
                case '/':  result += '/';  break;
                case 'b':  result += '\b'; break;
                case 'f':  result += '\f'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += next; break;
            }
        } else if (c == '\"') {
            return result;
        } else {
            result += c;
        }
    }
    return "";
}

int json_extract_int(const std::string& json, const std::string& key, int default_val) {
    size_t val_pos = find_json_key_value_pos(json, key);
    if (val_pos == std::string::npos || val_pos >= json.size()) {
        return default_val;
    }
    try {
        size_t idx = 0;
        int v = std::stoi(json.substr(val_pos, 32), &idx);
        return idx > 0 ? v : default_val;
    } catch (...) {
        return default_val;
    }
}

float json_extract_float(const std::string& json, const std::string& key, float default_val) {
    size_t val_pos = find_json_key_value_pos(json, key);
    if (val_pos == std::string::npos || val_pos >= json.size()) {
        return default_val;
    }
    try {
        size_t idx = 0;
        float v = std::stof(json.substr(val_pos, 32), &idx);
        return idx > 0 ? v : default_val;
    } catch (...) {
        return default_val;
    }
}

std::string query_extract_param(const std::string& query, const std::string& key, const std::string& default_val) {
    if (query.empty() || key.empty()) return default_val;
    std::string pat = key + "=";
    size_t pos = 0;
    while ((pos = query.find(pat, pos)) != std::string::npos) {
        if (pos == 0 || query[pos - 1] == '&' || query[pos - 1] == '?') {
            size_t start = pos + pat.size();
            size_t end = query.find('&', start);
            return (end == std::string::npos) ? query.substr(start) : query.substr(start, end - start);
        }
        pos += pat.size();
    }
    return default_val;
}

HttpRequest parse_http_request(const std::string& raw_request) {
    HttpRequest req;
    if (raw_request.empty()) {
        return req;
    }

    size_t line_end = raw_request.find("\r\n");
    if (line_end == std::string::npos) {
        line_end = raw_request.find('\n');
    }
    if (line_end == std::string::npos) {
        return req;
    }

    std::string request_line = raw_request.substr(0, line_end);
    std::istringstream iss(request_line);
    std::string version;
    if (!(iss >> req.method >> req.path >> version)) {
        return req;
    }

    size_t qpos = req.path.find('?');
    if (qpos != std::string::npos) {
        req.query = req.path.substr(qpos + 1);
        req.path = req.path.substr(0, qpos);
    }

    size_t header_end = raw_request.find("\r\n\r\n");
    if (header_end != std::string::npos) {
        req.body = raw_request.substr(header_end + 4);
    } else {
        header_end = raw_request.find("\n\n");
        if (header_end != std::string::npos) {
            req.body = raw_request.substr(header_end + 2);
        }
    }

    // Extraer credencial: X-API-Key > Authorization: Bearer > ?api_key=
    {
        size_t hdr_end = raw_request.find("\r\n\r\n");
        if (hdr_end == std::string::npos) hdr_end = raw_request.size();
        size_t pos = line_end;
        std::string bearer_key;
        while (pos < hdr_end) {
            while (pos < hdr_end && (raw_request[pos] == '\r' || raw_request[pos] == '\n')) ++pos;
            size_t eol = raw_request.find('\n', pos);
            if (eol == std::string::npos || eol > hdr_end) eol = hdr_end;
            std::string line = raw_request.substr(pos, eol - pos);
            pos = eol;
            size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string name = line.substr(0, colon);
            for (char& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            std::string value = line.substr(colon + 1);
            while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
            while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
            if (name == "x-api-key") {
                req.api_key = value;
            } else if (name == "authorization" && value.size() > 7) {
                std::string scheme = value.substr(0, 7);
                for (char& c : scheme) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                if (scheme == "bearer ") bearer_key = value.substr(7);
            }
        }
        if (req.api_key.empty()) req.api_key = bearer_key;
        if (req.api_key.empty()) req.api_key = query_extract_param(req.query, "api_key");
    }

    req.valid = !req.method.empty() && !req.path.empty();
    return req;
}

std::string configured_api_key() {
    const char* env_key = std::getenv("API_KEY");
    return env_key ? std::string(env_key) : std::string();
}

bool is_request_authorized(const HttpRequest& req, const std::string& expected_key) {
    if (expected_key.empty()) return true;
    const std::string& got = req.api_key;
    unsigned char diff = static_cast<unsigned char>(got.size() != expected_key.size());
    for (size_t i = 0; i < expected_key.size(); ++i) {
        const char g = i < got.size() ? got[i] : '\0';
        diff |= static_cast<unsigned char>(g ^ expected_key[i]);
    }
    return diff == 0;
}

std::string build_http_response(const HttpResponse& resp) {
    std::string ctype = resp.content_type.empty() ? "application/json; charset=utf-8" : resp.content_type;
    std::ostringstream oss;
    oss << "HTTP/1.1 " << resp.status_code << " " << resp.status_text << "\r\n"
        << "Content-Type: " << ctype << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << "Access-Control-Allow-Headers: Content-Type, Authorization, X-API-Key\r\n"
        << "Access-Control-Expose-Headers: *\r\n"
        << resp.extra_headers
        << "Connection: close\r\n"
        << "Content-Length: " << resp.body.size() << "\r\n\r\n";
    std::string out = oss.str();
    out.append(resp.body.data(), resp.body.size());
    return out;
}

std::string build_http_response(int status_code, const std::string& status_text, const std::string& json_body) {
    HttpResponse r;
    r.status_code = status_code;
    r.status_text = status_text;
    r.content_type = "application/json; charset=utf-8";
    r.body = json_body;
    return build_http_response(r);
}

static std::string to_lower_copy(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

static std::string sanitize_header_value(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c != '\r' && c != '\n') out.push_back(c);
        else out.push_back(' ');
    }
    return out;
}

ApiService::ApiService(const std::string& models_dir, int n_gpu_layers, int n_ctx)
    : models_dir_(find_models_directory(models_dir)),
      n_gpu_layers_(n_gpu_layers),
      n_ctx_(n_ctx) {
    ensure_model_loaded("qwen2.5-3b-instruct-q4_k_m.gguf");
    ensure_stt_loaded();
    ensure_tts_loaded();
}

ApiService::~ApiService() {
    client_.close();
    stt_client_.close();
    tts_client_.close();
}

std::string ApiService::find_models_directory(const std::string& preferred_dir) {
    if (!preferred_dir.empty() && fs::exists(preferred_dir) && fs::is_directory(preferred_dir)) {
        return fs::absolute(preferred_dir).string();
    }

    if (const char* env_dir = std::getenv("MODELS_DIR")) {
        if (fs::exists(env_dir) && fs::is_directory(env_dir)) {
            return fs::absolute(env_dir).string();
        }
    }

    const std::vector<std::string> candidates = {
        "./models",
        "./model",
        "../models",
        "../model",
        "../../models",
        "../../model",
        "../../../models",
        "../../../model",
        "../../../../models",
        "../../../../model",
        "C:/_PROYECTOS_2030/amibot/amibot_server/models",
        "C:/_PROYECTOS_2030/amibot/amibot_server/model",
        "/app/models",
        "/app/model"
    };

    for (const auto& c : candidates) {
        if (fs::exists(c) && fs::is_directory(c)) {
            return fs::absolute(c).string();
        }
    }

    return preferred_dir.empty() ? "./models" : preferred_dir;
}

bool ApiService::is_instruct_model(const std::string& model_name_or_path) {
    std::string lower = to_lower_copy(model_name_or_path);
    return lower.find("instruct") != std::string::npos ||
           lower.find("chat") != std::string::npos ||
           lower.find("qwen") != std::string::npos;
}

std::string ApiService::detect_emotion(const std::string& text) {
    std::string lower = to_lower_copy(text);
    if (lower.find("error") != std::string::npos ||
        lower.find("problema") != std::string::npos ||
        lower.find("falla") != std::string::npos) {
        return "alert";
    }
    if (lower.find("jaja") != std::string::npos ||
        lower.find("genial") != std::string::npos ||
        lower.find("excelente") != std::string::npos ||
        lower.find("gracias") != std::string::npos ||
        lower.find("feliz") != std::string::npos) {
        return "happy";
    }
    if (lower.find("segundo") != std::string::npos ||
        lower.find("revisando") != std::string::npos ||
        lower.find("consultando") != std::string::npos) {
        return "thinking";
    }
    return "neutral";
}

std::vector<std::string> ApiService::list_models() const {
    std::vector<std::string> models;
    if (!fs::exists(models_dir_) || !fs::is_directory(models_dir_)) {
        return models;
    }

    for (const auto& entry : fs::directory_iterator(models_dir_)) {
        if (entry.is_regular_file()) {
            std::string ext = to_lower_copy(entry.path().extension().string());
            if (ext == ".gguf") {
                models.push_back(entry.path().filename().string());
            }
        }
    }

    std::sort(models.begin(), models.end());
    return models;
}

std::string ApiService::resolve_model_path(const std::string& model_query) const {
    auto available = list_models();
    if (available.empty()) {
        return "";
    }

    if (model_query.empty()) {
        // Preferir Qwen por defecto si está disponible
        for (const auto& name : available) {
            if (to_lower_copy(name).find("qwen") != std::string::npos) {
                return (fs::path(models_dir_) / name).string();
            }
        }
        return (fs::path(models_dir_) / available.front()).string();
    }

    for (const auto& name : available) {
        if (name == model_query) {
            return (fs::path(models_dir_) / name).string();
        }
    }

    std::string query_lower = to_lower_copy(model_query);
    for (const auto& name : available) {
        if (to_lower_copy(name).find(query_lower) != std::string::npos) {
            return (fs::path(models_dir_) / name).string();
        }
    }

    return "";
}

std::string ApiService::resolve_stt_model_path() const {
    if (const char* env_stt = std::getenv("STT_MODEL_PATH")) {
        if (fs::exists(env_stt)) return fs::absolute(env_stt).string();
    }

    if (fs::exists(models_dir_) && fs::is_directory(models_dir_)) {
        for (const auto& entry : fs::directory_iterator(models_dir_)) {
            if (entry.is_regular_file() && to_lower_copy(entry.path().extension().string()) == ".bin") {
                return entry.path().string();
            }
        }
    }

    const std::vector<std::string> fallbacks = {
        "C:/_PROYECTOS_2030/amibot/amibot_server/model/ggml-base-q5_1.bin",
        "/app/models/ggml-base-q5_1.bin"
    };
    for (const auto& f : fallbacks) {
        if (fs::exists(f)) return f;
    }
    return "";
}

bool ApiService::ensure_model_loaded(const std::string& model_query) {
    std::string resolved = resolve_model_path(model_query);
    if (resolved.empty() || !fs::exists(resolved)) {
        return false;
    }

    if (client_.is_ready() && loaded_model_path_ == resolved) {
        return true;
    }

    llm_comms::Config cfg;
    cfg.model_path = resolved;
    cfg.n_ctx = n_ctx_;
    cfg.n_gpu_layers = n_gpu_layers_;
    cfg.n_threads = 0;
    cfg.verbose = false;

    if (!client_.init(cfg)) {
        loaded_model_path_.clear();
        loaded_model_name_.clear();
        return false;
    }

    loaded_model_path_ = resolved;
    loaded_model_name_ = fs::path(resolved).filename().string();
    return true;
}

bool ApiService::ensure_stt_loaded() {
    if (stt_client_.is_ready()) {
        return true;
    }
    std::string stt_path = resolve_stt_model_path();
    if (stt_path.empty()) {
        return false;
    }
    stt_comms::Config cfg;
    cfg.model_path = stt_path;
    cfg.use_gpu = true;
    cfg.verbose = false;
    if (!stt_client_.init(cfg)) {
        return false;
    }
    loaded_stt_path_ = stt_path;
    return true;
}

bool ApiService::ensure_tts_loaded() {
    if (tts_client_.is_ready()) {
        return true;
    }
    tts_comms::Config cfg;
    cfg.default_voice = "ef_dora";
    return tts_client_.init(cfg);
}

std::string ApiService::get_loaded_model_name() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return loaded_model_name_;
}

std::string ApiService::get_models_directory() const {
    return models_dir_;
}

GenerateResult ApiService::generate_text_unlocked(const GenerateRequest& req) {
    GenerateResult res;

    if (req.prompt.empty()) {
        res.error_message = "El campo 'prompt' es obligatorio y no puede estar vacio.";
        return res;
    }

    std::string target_model = req.model.empty() ? loaded_model_name_ : req.model;
    if (!ensure_model_loaded(target_model)) {
        res.error_message = "No se pudo encontrar o cargar el modelo solicitado: " +
                            (target_model.empty() ? "(ningun modelo disponible)" : target_model);
        return res;
    }

    llm_comms::GenerateOptions opts;
    opts.max_tokens = req.max_tokens > 0 ? req.max_tokens : 128;
    opts.temperature = req.temperature;
    opts.top_p = 0.9f;

    auto t_start = std::chrono::steady_clock::now();

    if (is_instruct_model(loaded_model_name_) || !req.system_prompt.empty()) {
        std::vector<llm_comms::ChatMessage> messages;
        std::string sys = req.system_prompt.empty()
            ? "Eres un asistente inteligente, conciso y util. Responde unicamente con texto limpio en el idioma del usuario, sin emojis, sin markdown y sin caracteres especiales."
            : req.system_prompt;
        messages.push_back({ "system", sys });
        messages.push_back({ "user", req.prompt });
        res.response_text = client_.send_chat(messages, opts);
    } else {
        res.response_text = client_.send_text(req.prompt, opts);
    }

    auto t_end = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    res.model_used = loaded_model_name_;

    if (res.response_text.empty() && !client_.get_last_error().empty()) {
        res.error_message = client_.get_last_error();
        res.success = false;
        return res;
    }

    res.success = true;
    return res;
}

GenerateResult ApiService::generate_text(const GenerateRequest& req) {
    std::lock_guard<std::mutex> lock(mutex_);
    return generate_text_unlocked(req);
}

static std::string base64_decode(const std::string& in) {
    static constexpr int T[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
    };
    std::string out;
    out.reserve((in.size() * 3) / 4);
    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (T[c] == -1) continue;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

static std::string base64_encode(const uint8_t* data, size_t len) {
    static constexpr char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1 < len) v |= static_cast<uint32_t>(data[i + 1]) << 8;
        if (i + 2 < len) v |= static_cast<uint32_t>(data[i + 2]);
        out.push_back(B64[(v >> 18) & 0x3F]);
        out.push_back(B64[(v >> 12) & 0x3F]);
        out.push_back((i + 1 < len) ? B64[(v >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < len) ? B64[v & 0x3F] : '=');
    }
    return out;
}

std::string unwrap_esp32_audio_frames(const std::string& raw_bytes) {
    if (raw_bytes.size() <= 4) return raw_bytes;
    // No desempaquetar si ya es un archivo WAV (RIFF....WAVE)
    if (raw_bytes.size() >= 12 &&
        std::memcmp(raw_bytes.data(), "RIFF", 4) == 0 &&
        std::memcmp(raw_bytes.data() + 8, "WAVE", 4) == 0) {
        return raw_bytes;
    }

    // Verificar si el buffer consiste en una o mas tramas del protocolo ESP32:
    // [tipo(1B)=0, reservado(1B)=0, longitud_BE(2B)] + payload
    size_t offset = 0;
    size_t frame_count = 0;
    std::string unpacked;
    while (offset + 4 <= raw_bytes.size()) {
        const uint8_t b0 = static_cast<uint8_t>(raw_bytes[offset]);
        const uint8_t b1 = static_cast<uint8_t>(raw_bytes[offset + 1]);
        const uint16_t payload_len = (static_cast<uint16_t>(static_cast<uint8_t>(raw_bytes[offset + 2])) << 8) |
                                      static_cast<uint16_t>(static_cast<uint8_t>(raw_bytes[offset + 3]));
        if (b0 != 0 || b1 != 0 || payload_len == 0 || offset + 4 + payload_len > raw_bytes.size()) {
            return raw_bytes;
        }
        unpacked.append(raw_bytes.data() + offset + 4, payload_len);
        offset += 4 + payload_len;
        ++frame_count;
    }

    if (frame_count > 0 && offset == raw_bytes.size() && !unpacked.empty()) {
        return unpacked;
    }
    return raw_bytes;
}

std::vector<uint8_t> pack_esp32_audio_frames(const uint8_t* data, size_t len, size_t chunk_size) {
    if (chunk_size == 0) chunk_size = 120;
    std::vector<uint8_t> out;
    out.reserve(len + (len / chunk_size + 1) * 4);
    size_t offset = 0;
    while (offset < len) {
        const size_t n = std::min(chunk_size, len - offset);
        out.push_back(0); // Tipo de trama de audio ESP32
        out.push_back(0); // Reservado
        out.push_back(static_cast<uint8_t>((n >> 8) & 0xFF));
        out.push_back(static_cast<uint8_t>(n & 0xFF));
        out.insert(out.end(), data + offset, data + offset + n);
        offset += n;
    }
    return out;
}

static std::string unwrap_multipart_if_needed(const std::string& body) {
    if (body.size() > 4 && body[0] == '-' && body[1] == '-') {
        size_t hdr_end = body.find("\r\n\r\n");
        if (hdr_end != std::string::npos) {
            size_t data_start = hdr_end + 4;
            size_t boundary_end = body.find("\r\n--", data_start);
            if (boundary_end != std::string::npos && boundary_end > data_start) {
                return body.substr(data_start, boundary_end - data_start);
            }
        }
    }
    return body;
}

static void convert_raw_pcm16_to_16k_float(const uint8_t* raw_bytes, size_t n_bytes, int in_rate, std::vector<float>& out_samples) {
    const size_t n_int16 = n_bytes / sizeof(int16_t);
    if (n_int16 == 0) {
        out_samples.clear();
        return;
    }
    std::vector<float> mono(n_int16);
    const int16_t* p16 = reinterpret_cast<const int16_t*>(raw_bytes);
    for (size_t i = 0; i < n_int16; ++i) {
        mono[i] = static_cast<float>(p16[i]) / 32768.0f;
    }
    if (in_rate <= 0 || in_rate == 16000) {
        out_samples = std::move(mono);
        return;
    }
    const double ratio = static_cast<double>(in_rate) / 16000.0;
    const size_t out_len = static_cast<size_t>(static_cast<double>(mono.size()) / ratio);
    out_samples.resize(out_len);
    for (size_t i = 0; i < out_len; ++i) {
        const double src_pos = static_cast<double>(i) * ratio;
        const size_t idx0 = static_cast<size_t>(src_pos);
        const size_t idx1 = std::min(idx0 + 1, mono.size() - 1);
        const float frac = static_cast<float>(src_pos - static_cast<double>(idx0));
        out_samples[i] = mono[idx0] * (1.0f - frac) + mono[idx1] * frac;
    }
}

VoiceResult ApiService::process_voice(const VoiceRequest& req) {
    std::lock_guard<std::mutex> lock(mutex_);
    VoiceResult res;
    auto t0 = std::chrono::steady_clock::now();

    // 1. Voz -> Texto mediante stt_comms (acepta .wav, tramas ESP32 de 4 bytes o PCM 16-bit crudo de I2S)
    if (!ensure_stt_loaded()) {
        res.error_message = "No se pudo inicializar el motor STT (stt_comms)";
        return res;
    }

    stt_comms::TranscribeOptions stt_opts;
    stt_opts.language = req.language.empty() ? "auto" : req.language;

    if (req.wav_bytes.empty()) {
        res.error_message = "No se proporciono audio de entrada en memoria";
        return res;
    }

    // 100% en RAM: desempaquetar tramas o multipart y decodificar/remuestrear sin tocar disco
    std::string payload = unwrap_esp32_audio_frames(unwrap_multipart_if_needed(req.wav_bytes));
    const bool has_riff_header = (payload.size() >= 12 &&
                                  std::memcmp(payload.data(), "RIFF", 4) == 0 &&
                                  std::memcmp(payload.data() + 8, "WAVE", 4) == 0);
    if (has_riff_header) {
        res.transcribed_text = stt_client_.transcribe_wav_bytes(
            reinterpret_cast<const uint8_t*>(payload.data()),
            payload.size(),
            stt_opts
        );
    } else {
        std::vector<float> samples_16k;
        convert_raw_pcm16_to_16k_float(
            reinterpret_cast<const uint8_t*>(payload.data()),
            payload.size(),
            req.input_sample_rate > 0 ? req.input_sample_rate : 16000,
            samples_16k
        );
        res.transcribed_text = stt_client_.transcribe_samples(samples_16k.data(), samples_16k.size(), stt_opts);
    }

    if (res.transcribed_text.empty()) {
        std::string stt_err = stt_client_.get_last_error();
        res.error_message = stt_err.empty() ? "El audio no contiene voz reconocible" : stt_err;
        return res;
    }

    // 2. Texto -> LLM en RAM mediante llm_comms
    GenerateRequest gen_req;
    gen_req.prompt = res.transcribed_text;
    gen_req.system_prompt = "Eres un asistente de voz ultrarrapido y conciso. Responde en 1 o 2 frases breves con texto limpio en el idioma del usuario, sin emojis ni caracteres especiales.";
    gen_req.model = req.model;
    gen_req.max_tokens = req.max_tokens > 0 ? req.max_tokens : 64;
    gen_req.temperature = req.temperature;

    GenerateResult gen_res = generate_text_unlocked(gen_req);
    if (!gen_res.success || gen_res.response_text.empty()) {
        res.error_message = gen_res.error_message.empty() ? "El LLM no genero respuesta" : gen_res.error_message;
        return res;
    }
    res.llm_response = gen_res.response_text;
    res.emotion = detect_emotion(res.llm_response);
    res.model_used = gen_res.model_used;

    // 3. Texto del LLM -> Audio mediante tts_comms
    if (!ensure_tts_loaded()) {
        res.error_message = "No se pudo inicializar el motor TTS (tts_comms)";
        return res;
    }

    tts_comms::SynthesizeOptions tts_opts;
    tts_opts.voice = req.voice.empty() ? "ef_dora" : req.voice;
    tts_opts.speed = req.speed > 0.0f ? req.speed : 1.0f;

    std::vector<float> pcm_samples;
    if (!tts_client_.synthesize_to_pcm(res.llm_response, pcm_samples, tts_opts)) {
        res.error_message = tts_client_.get_last_error();
        return res;
    }

    // Remuestrear salida si el ESP32 pide otra frecuencia (ej. ?out_rate=16000)
    uint32_t out_rate = (req.output_sample_rate > 0)
        ? static_cast<uint32_t>(req.output_sample_rate)
        : tts_comms::TTSCOMMS_SAMPLE_RATE;
    if (out_rate != tts_comms::TTSCOMMS_SAMPLE_RATE && !pcm_samples.empty()) {
        const double ratio = static_cast<double>(tts_comms::TTSCOMMS_SAMPLE_RATE) / static_cast<double>(out_rate);
        const size_t new_len = static_cast<size_t>(static_cast<double>(pcm_samples.size()) / ratio);
        std::vector<float> resampled(new_len);
        for (size_t i = 0; i < new_len; ++i) {
            const double src_pos = static_cast<double>(i) * ratio;
            const size_t idx0 = static_cast<size_t>(src_pos);
            const size_t idx1 = std::min(idx0 + 1, pcm_samples.size() - 1);
            const float frac = static_cast<float>(src_pos - static_cast<double>(idx0));
            resampled[i] = pcm_samples[idx0] * (1.0f - frac) + pcm_samples[idx1] * frac;
        }
        pcm_samples = std::move(resampled);
    }

    res.audio_wav = tts_comms::encode_wav_bytes(pcm_samples.data(), pcm_samples.size(), out_rate);
    if ((req.raw_pcm_output || req.esp32_framed_output) && res.audio_wav.size() > 44) {
        res.audio_wav.erase(res.audio_wav.begin(), res.audio_wav.begin() + 44);
    }
    if (req.esp32_framed_output && !res.audio_wav.empty()) {
        res.audio_wav = pack_esp32_audio_frames(res.audio_wav.data(), res.audio_wav.size(), 240);
    }
    res.voice_used = tts_opts.voice;

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    res.success = !res.audio_wav.empty();
    return res;
}

HttpResponse ApiService::handle_http_request(const HttpRequest& req) {
    if (!req.valid) {
        return { 400, "Bad Request", "application/json; charset=utf-8", "", "{\"status\":\"error\",\"error\":\"Peticion HTTP malformada\"}" };
    }

    if (req.method == "OPTIONS") {
        return { 200, "OK", "application/json; charset=utf-8", "", "{\"status\":\"ok\"}" };
    }

    if (req.method == "GET" && req.path == "/health") {
        std::string current_model = get_loaded_model_name();
        std::ostringstream oss;
        oss << "{\"status\":\"ok\",\"ready\":" << (!current_model.empty() ? "true" : "false")
            << ",\"loaded_model\":\"" << json_escape(current_model) << "\"}";
        return { 200, "OK", "application/json; charset=utf-8", "", oss.str() };
    }

    // Todo endpoint salvo OPTIONS y /health requiere API key (si API_KEY esta configurada)
    if (!is_request_authorized(req, configured_api_key())) {
        return { 401, "Unauthorized", "application/json; charset=utf-8", "WWW-Authenticate: Bearer\r\n",
                 "{\"status\":\"error\",\"error\":\"API key invalida o ausente\"}" };
    }

    if (req.method == "GET" && req.path == "/models") {
        auto models = list_models();
        std::ostringstream oss;
        oss << "{\"status\":\"ok\",\"models\":[";
        for (size_t i = 0; i < models.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "\"" << json_escape(models[i]) << "\"";
        }
        oss << "]}";
        return { 200, "OK", "application/json; charset=utf-8", "", oss.str() };
    }

    if (req.method == "POST" && (req.path == "/voice" || req.path == "/esp32" || req.path == "/generate" || req.path == "/chat" || req.path == "/")) {
        size_t first_non_ws = 0;
        while (first_non_ws < req.body.size() && std::isspace(static_cast<unsigned char>(req.body[first_non_ws]))) {
            ++first_non_ws;
        }
        const bool is_json_body = (first_non_ws < req.body.size() && req.body[first_non_ws] == '{');
        const bool is_binary_audio = (!req.body.empty() && !is_json_body);
        const std::string audio_b64 = is_json_body ? json_extract_string(req.body, "audio") : "";

        // Flujo de Voz 100% en RAM (ESP32 / Cliente): acepta .wav, tramas ESP32 de 4 bytes, PCM 16-bit crudo de I2S, multipart o Base64 en JSON
        if (is_binary_audio || !audio_b64.empty() || req.path == "/voice" || req.path == "/esp32") {
            VoiceRequest vreq;
            std::string fmt;
            if (is_binary_audio) {
                vreq.wav_bytes = req.body;
                vreq.model = query_extract_param(req.query, "model", "");
                vreq.voice = query_extract_param(req.query, "voice", "ef_dora");
                vreq.language = query_extract_param(req.query, "lang", "auto");
                vreq.max_tokens = std::atoi(query_extract_param(req.query, "max_tokens", "64").c_str());
                vreq.input_sample_rate = std::atoi(query_extract_param(req.query, "sample_rate",
                                         query_extract_param(req.query, "rate", "16000")).c_str());
                vreq.output_sample_rate = std::atoi(query_extract_param(req.query, "out_rate", "24000").c_str());
                fmt = query_extract_param(req.query, "format", query_extract_param(req.query, "output", "wav"));
            } else {
                if (!audio_b64.empty()) {
                    vreq.wav_bytes = base64_decode(audio_b64);
                }
                vreq.model = json_extract_string(req.body, "model");
                vreq.voice = json_extract_string(req.body, "voice");
                if (vreq.voice.empty()) vreq.voice = "ef_dora";
                vreq.language = json_extract_string(req.body, "lang");
                if (vreq.language.empty()) vreq.language = "auto";
                vreq.max_tokens = json_extract_int(req.body, "max_tokens", 64);
                vreq.temperature = json_extract_float(req.body, "temperature", 0.7f);
                vreq.speed = json_extract_float(req.body, "speed", 1.0f);
                vreq.input_sample_rate = json_extract_int(req.body, "sample_rate", 16000);
                vreq.output_sample_rate = json_extract_int(req.body, "out_rate", 24000);
                fmt = json_extract_string(req.body, "format");
                if (fmt.empty()) fmt = query_extract_param(req.query, "format", "wav");
            }
            vreq.raw_pcm_output = (fmt == "pcm" || fmt == "raw");
            vreq.esp32_framed_output = (fmt == "frames" || fmt == "esp32");

            VoiceResult vres = process_voice(vreq);
            if (!vres.success) {
                std::ostringstream err;
                err << "{\"status\":\"error\",\"error\":\"" << json_escape(vres.error_message) << "\"}";
                return { 400, "Bad Request", "application/json; charset=utf-8", "", err.str() };
            }

            HttpResponse resp;
            resp.status_code = 200;
            resp.status_text = "OK";
            resp.extra_headers =
                "X-Transcribed-Text: " + sanitize_header_value(vres.transcribed_text) + "\r\n" +
                "X-LLM-Response: " + sanitize_header_value(vres.llm_response) + "\r\n" +
                "X-Emotion: " + sanitize_header_value(vres.emotion) + "\r\n" +
                "X-Model-Used: " + sanitize_header_value(vres.model_used) + "\r\n" +
                "X-Voice-Used: " + sanitize_header_value(vres.voice_used) + "\r\n" +
                "X-Elapsed-Ms: " + std::to_string(vres.elapsed_ms) + "\r\n";

            if (fmt == "json") {
                resp.content_type = "application/json; charset=utf-8";
                std::string b64_audio = base64_encode(vres.audio_wav.data(), vres.audio_wav.size());
                std::ostringstream jout;
                jout << "{\"status\":\"ok\""
                     << ",\"stt\":{\"type\":\"stt\",\"text\":\"" << json_escape(vres.transcribed_text) << "\"}"
                     << ",\"llm\":{\"type\":\"llm\",\"text\":\"" << json_escape(vres.llm_response) << "\",\"emotion\":\"" << json_escape(vres.emotion) << "\"}"
                     << ",\"audio\":\"" << b64_audio << "\"}";
                resp.body = jout.str();
                return resp;
            }

            resp.content_type = vreq.raw_pcm_output
                ? "audio/pcm"
                : (vreq.esp32_framed_output ? "application/octet-stream" : "audio/wav");
            resp.body.assign(reinterpret_cast<const char*>(vres.audio_wav.data()), vres.audio_wav.size());
            return resp;
        }

        // Compatibilidad con peticiones JSON de texto en /generate
        std::string prompt = json_extract_string(req.body, "prompt");
        if (prompt.empty()) {
            prompt = json_extract_string(req.body, "text");
        }
        if (prompt.empty()) {
            return { 400, "Bad Request", "application/json; charset=utf-8", "", "{\"status\":\"error\",\"error\":\"Se requiere un audio WAV o un campo 'prompt'\"}" };
        }

        GenerateRequest gen_req;
        gen_req.prompt = prompt;
        gen_req.system_prompt = json_extract_string(req.body, "system");
        gen_req.model = json_extract_string(req.body, "model");
        gen_req.max_tokens = json_extract_int(req.body, "max_tokens", 128);
        gen_req.temperature = json_extract_float(req.body, "temperature", 0.7f);

        GenerateResult gen_res = generate_text(gen_req);
        if (!gen_res.success) {
            std::ostringstream err;
            err << "{\"status\":\"error\",\"error\":\"" << json_escape(gen_res.error_message) << "\"}";
            return { 404, "Not Found", "application/json; charset=utf-8", "", err.str() };
        }

        std::ostringstream oss;
        oss << "{\"status\":\"ok\""
            << ",\"model\":\"" << json_escape(gen_res.model_used) << "\""
            << ",\"response\":\"" << json_escape(gen_res.response_text) << "\""
            << ",\"elapsed_ms\":" << gen_res.elapsed_ms << "}";
        return { 200, "OK", "application/json; charset=utf-8", "", oss.str() };
    }

    return { 404, "Not Found", "application/json; charset=utf-8", "", "{\"status\":\"error\",\"error\":\"Endpoint no encontrado\"}" };
}

} // namespace llm_api
