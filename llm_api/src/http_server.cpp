#include "http_server.h"

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef SOCKET socket_t;
    #define INVALID_SOCK INVALID_SOCKET
    #define CLOSE_SOCK(s) closesocket(s)
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <netdb.h>
    typedef int socket_t;
    #define INVALID_SOCK (-1)
    #define CLOSE_SOCK(s) close(s)
#endif

namespace llm_api {

// ============================================================================
// SHA-1 + Base64 para Handshake WebSocket RFC 6455 (Conexion directa ESP32)
// ============================================================================
static std::string ws_sha1_base64(const std::string& input) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE, h3 = 0x10325476, h4 = 0xC3D2E1F0;
    std::vector<uint8_t> msg(input.begin(), input.end());
    uint64_t bit_len = static_cast<uint64_t>(msg.size()) * 8;
    msg.push_back(0x80);
    while ((msg.size() % 64) != 56) msg.push_back(0x00);
    for (int i = 7; i >= 0; --i) msg.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xFF));

    auto rol = [](uint32_t v, int b) -> uint32_t { return (v << b) | (v >> (32 - b)); };
    for (size_t chunk = 0; chunk < msg.size(); chunk += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(msg[chunk + i * 4]) << 24) |
                   (static_cast<uint32_t>(msg[chunk + i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(msg[chunk + i * 4 + 2]) << 8) |
                   (static_cast<uint32_t>(msg[chunk + i * 4 + 3]));
        }
        for (int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | ((~b) & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;            k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else             { f = b ^ c ^ d;            k = 0xCA62C1D6; }
            uint32_t temp = rol(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rol(b, 30); b = a; a = temp;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }

    uint8_t digest[20];
    uint32_t hs[5] = { h0, h1, h2, h3, h4 };
    for (int i = 0; i < 5; ++i) {
        digest[i * 4]     = static_cast<uint8_t>((hs[i] >> 24) & 0xFF);
        digest[i * 4 + 1] = static_cast<uint8_t>((hs[i] >> 16) & 0xFF);
        digest[i * 4 + 2] = static_cast<uint8_t>((hs[i] >> 8) & 0xFF);
        digest[i * 4 + 3] = static_cast<uint8_t>(hs[i] & 0xFF);
    }

    static constexpr char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (int i = 0; i < 20; i += 3) {
        uint32_t v = static_cast<uint32_t>(digest[i]) << 16;
        if (i + 1 < 20) v |= static_cast<uint32_t>(digest[i + 1]) << 8;
        if (i + 2 < 20) v |= static_cast<uint32_t>(digest[i + 2]);
        out.push_back(B64[(v >> 18) & 0x3F]);
        out.push_back(B64[(v >> 12) & 0x3F]);
        out.push_back((i + 1 < 20) ? B64[(v >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < 20) ? B64[v & 0x3F] : '=');
    }
    return out;
}

static bool send_all(socket_t sock, const void* data, size_t len) {
    const char* ptr = static_cast<const char*>(data);
    size_t sent = 0;
    while (sent < len) {
        int n = send(sock, ptr + sent, static_cast<int>(len - sent), 0);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

static bool recv_exact(socket_t sock, uint8_t* buf, size_t len) {
    size_t got = 0;
    while (got < len) {
        int n = recv(sock, reinterpret_cast<char*>(buf + got), static_cast<int>(len - got), 0);
        if (n <= 0) return false;
        got += static_cast<size_t>(n);
    }
    return true;
}

static bool ws_send_frame(socket_t sock, uint8_t opcode, const void* payload, size_t len) {
    uint8_t hdr[10];
    size_t hdr_len = 0;
    hdr[0] = static_cast<uint8_t>(0x80 | (opcode & 0x0F));
    if (len < 126) {
        hdr[1] = static_cast<uint8_t>(len);
        hdr_len = 2;
    } else if (len <= 65535) {
        hdr[1] = 126;
        hdr[2] = static_cast<uint8_t>((len >> 8) & 0xFF);
        hdr[3] = static_cast<uint8_t>(len & 0xFF);
        hdr_len = 4;
    } else {
        hdr[1] = 127;
        uint64_t l64 = static_cast<uint64_t>(len);
        for (int i = 0; i < 8; ++i) hdr[2 + i] = static_cast<uint8_t>((l64 >> ((7 - i) * 8)) & 0xFF);
        hdr_len = 10;
    }
    if (!send_all(sock, hdr, hdr_len)) return false;
    return len == 0 || send_all(sock, payload, len);
}

static bool ws_send_json(socket_t sock, const std::string& json) {
    return ws_send_frame(sock, 0x01, json.data(), json.size());
}

static void set_socket_recv_timeout_ms(socket_t sock, int timeout_ms) {
#ifdef _WIN32
    DWORD tv = static_cast<DWORD>(timeout_ms);
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv));
#else
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

static std::string extract_ws_key(const std::string& raw_req) {
    std::string lower = raw_req;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower.find("upgrade: websocket") == std::string::npos) return "";
    size_t pos = lower.find("sec-websocket-key:");
    if (pos == std::string::npos) return "";
    pos += 18;
    while (pos < raw_req.size() && (raw_req[pos] == ' ' || raw_req[pos] == '\t')) ++pos;
    size_t end = raw_req.find("\r\n", pos);
    if (end == std::string::npos) end = raw_req.find('\n', pos);
    if (end == std::string::npos) return "";
    std::string key = raw_req.substr(pos, end - pos);
    while (!key.empty() && std::isspace(static_cast<unsigned char>(key.back()))) key.pop_back();
    return key;
}

static void handle_esp32_websocket_session(socket_t sock, ApiService& service, const std::string& ws_key, const std::string& query) {
    std::string accept_key = ws_sha1_base64(ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    std::string handshake =
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: " + accept_key + "\r\n\r\n";
    if (!send_all(sock, handshake.data(), handshake.size())) {
        return;
    }

    std::string audio_buffer;
    bool has_pending_audio = false;

    auto finish_user_speech = [&]() {
        if (audio_buffer.empty()) return;
        VoiceRequest vreq;
        vreq.wav_bytes = std::move(audio_buffer);
        audio_buffer.clear();
        has_pending_audio = false;

        vreq.model = query_extract_param(query, "model", "qwen");
        vreq.voice = query_extract_param(query, "voice", "ef_dora");
        vreq.language = query_extract_param(query, "lang", "auto");
        vreq.input_sample_rate = std::atoi(query_extract_param(query, "rate", "16000").c_str());
        vreq.output_sample_rate = std::atoi(query_extract_param(query, "out_rate", "16000").c_str());
        vreq.raw_pcm_output = true; // Para WebSocket ESP32 enviamos tramas sobre el audio sin cabecera WAV

        VoiceResult vres = service.process_voice(vreq);
        if (!vres.success || vres.transcribed_text.empty()) {
            return;
        }

        // 1. Enviar transcripcion STT al ESP32 (para pantalla OLED)
        ws_send_json(sock, "{\"type\":\"stt\",\"text\":\"" + json_escape(vres.transcribed_text) + "\"}");

        // 2. Enviar respuesta LLM + emocion al ESP32
        ws_send_json(sock, "{\"type\":\"llm\",\"text\":\"" + json_escape(vres.llm_response) +
                           "\",\"emotion\":\"" + json_escape(vres.emotion) + "\"}");

        // 3. Notificar inicio TTS y transmitir tramas binarias con cabecera de 4 bytes [0, 0, len_HI, len_LO] + chunk
        ws_send_json(sock, "{\"type\":\"tts\",\"state\":\"start\"}");

        const size_t CHUNK_SIZE = 240;
        size_t offset = 0;
        while (offset < vres.audio_wav.size()) {
            const size_t n = std::min(CHUNK_SIZE, vres.audio_wav.size() - offset);
            std::vector<uint8_t> packet(4 + n);
            packet[0] = 0;
            packet[1] = 0;
            packet[2] = static_cast<uint8_t>((n >> 8) & 0xFF);
            packet[3] = static_cast<uint8_t>(n & 0xFF);
            std::memcpy(packet.data() + 4, vres.audio_wav.data() + offset, n);
            if (!ws_send_frame(sock, 0x02, packet.data(), packet.size())) break;
            offset += n;
        }

        ws_send_json(sock, "{\"type\":\"tts\",\"state\":\"stop\"}");
    };

    // Timeout de 1200ms para detectar silencio automatico (igual que voice_session.js en colo-server)
    set_socket_recv_timeout_ms(sock, 1200);

    while (true) {
        uint8_t hdr[2];
        int r = recv(sock, reinterpret_cast<char*>(hdr), 2, 0);
        if (r <= 0) {
            if (has_pending_audio && !audio_buffer.empty()) {
                finish_user_speech();
                continue;
            }
            break;
        }
        if (r == 1) {
            if (!recv_exact(sock, hdr + 1, 1)) break;
        }

        const uint8_t opcode = hdr[0] & 0x0F;
        const bool masked = (hdr[1] & 0x80) != 0;
        uint64_t payload_len = hdr[1] & 0x7F;

        if (payload_len == 126) {
            uint8_t ext[2];
            if (!recv_exact(sock, ext, 2)) break;
            payload_len = (static_cast<uint64_t>(ext[0]) << 8) | ext[1];
        } else if (payload_len == 127) {
            uint8_t ext[8];
            if (!recv_exact(sock, ext, 8)) break;
            payload_len = 0;
            for (int i = 0; i < 8; ++i) payload_len = (payload_len << 8) | ext[i];
        }

        uint8_t mask_key[4] = {0, 0, 0, 0};
        if (masked && !recv_exact(sock, mask_key, 4)) break;

        if (payload_len > 16 * 1024 * 1024) break;
        std::string payload(static_cast<size_t>(payload_len), '\0');
        if (payload_len > 0 && !recv_exact(sock, reinterpret_cast<uint8_t*>(&payload[0]), static_cast<size_t>(payload_len))) {
            break;
        }
        if (masked) {
            for (size_t i = 0; i < payload.size(); ++i) {
                payload[i] ^= static_cast<char>(mask_key[i & 3]);
            }
        }

        if (opcode == 0x08) { // Close
            ws_send_frame(sock, 0x08, nullptr, 0);
            break;
        } else if (opcode == 0x09) { // Ping -> Pong
            ws_send_frame(sock, 0x0A, payload.data(), payload.size());
        } else if (opcode == 0x02) { // Trama binaria de audio del ESP32
            if (payload.size() > 4) {
                const uint16_t inner_len = (static_cast<uint16_t>(static_cast<uint8_t>(payload[2])) << 8) |
                                            static_cast<uint16_t>(static_cast<uint8_t>(payload[3]));
                if (static_cast<uint8_t>(payload[0]) == 0 && static_cast<uint8_t>(payload[1]) == 0 &&
                    static_cast<size_t>(4 + inner_len) == payload.size()) {
                    audio_buffer.append(payload.data() + 4, inner_len);
                } else {
                    audio_buffer.append(payload);
                }
            } else {
                audio_buffer.append(payload);
            }
            has_pending_audio = true;
        } else if (opcode == 0x01) { // Mensaje de control JSON del ESP32
            std::string msg_type = json_extract_string(payload, "type");
            if (msg_type == "hello") {
                ws_send_json(sock, "{\"type\":\"hello\",\"transport\":\"websocket\",\"session_id\":\"sess_esp32\"}");
            } else if (msg_type == "listen") {
                std::string state = json_extract_string(payload, "state");
                if (state == "start") {
                    audio_buffer.clear();
                    has_pending_audio = false;
                } else if (state == "stop") {
                    finish_user_speech();
                }
            } else if (msg_type == "abort") {
                audio_buffer.clear();
                has_pending_audio = false;
            }
        }
    }
}

static void ensure_network_init() {
#ifdef _WIN32
    static bool initialized = false;
    if (!initialized) {
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
        initialized = true;
    }
#endif
}

static size_t extract_content_length(const std::string& headers) {
    std::string lower = headers;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    size_t pos = lower.find("content-length:");
    if (pos == std::string::npos) return 0;
    pos += 15;
    while (pos < lower.size() && std::isspace(static_cast<unsigned char>(lower[pos]))) ++pos;
    return static_cast<size_t>(std::strtoul(lower.c_str() + pos, nullptr, 10));
}

static std::string read_full_http_message(socket_t sock) {
    std::string data;
    std::vector<char> buffer(65536);
    size_t header_end = std::string::npos;
    size_t content_len = 0;

    while (true) {
        int bytes = recv(sock, buffer.data(), static_cast<int>(buffer.size()), 0);
        if (bytes <= 0) break;
        data.append(buffer.data(), bytes);

        if (header_end == std::string::npos) {
            header_end = data.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                content_len = extract_content_length(data.substr(0, header_end));
                header_end += 4;
                if (content_len > 0 && content_len <= 32 * 1024 * 1024) {
                    data.reserve(header_end + content_len);
                }
            }
        }

        if (header_end != std::string::npos) {
            if (data.size() >= header_end + content_len) {
                break;
            }
        }
    }
    return data;
}

HttpServer::HttpServer(ApiService& service, int port)
    : service_(service), port_(port), listen_sock_(static_cast<uintptr_t>(INVALID_SOCK)) {}

HttpServer::~HttpServer() {
    stop();
}

static bool bind_and_listen(socket_t& out_sock, int port) {
    ensure_network_init();

    socket_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCK) {
        return false;
    }

    int opt = 1;
#ifdef _WIN32
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        CLOSE_SOCK(s);
        return false;
    }

    if (listen(s, 16) != 0) {
        CLOSE_SOCK(s);
        return false;
    }

    out_sock = s;
    return true;
}

bool HttpServer::start_blocking() {
    socket_t s = INVALID_SOCK;
    if (!bind_and_listen(s, port_)) {
        return false;
    }
    listen_sock_ = static_cast<uintptr_t>(s);
    running_ = true;
    accept_loop();
    return true;
}

bool HttpServer::start_async() {
    socket_t s = INVALID_SOCK;
    if (!bind_and_listen(s, port_)) {
        return false;
    }
    listen_sock_ = static_cast<uintptr_t>(s);
    running_ = true;
    server_thread_ = std::thread(&HttpServer::accept_loop, this);
    return true;
}

void HttpServer::stop() {
    running_ = false;
    socket_t s = static_cast<socket_t>(listen_sock_);
    if (s != INVALID_SOCK) {
        listen_sock_ = static_cast<uintptr_t>(INVALID_SOCK);
        CLOSE_SOCK(s);
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

bool HttpServer::is_running() const {
    return running_.load();
}

int HttpServer::get_port() const {
    return port_;
}

void HttpServer::accept_loop() {
    while (running_) {
        socket_t s = static_cast<socket_t>(listen_sock_);
        if (s == INVALID_SOCK) break;

        sockaddr_in client_addr{};
#ifdef _WIN32
        int client_len = sizeof(client_addr);
#else
        socklen_t client_len = sizeof(client_addr);
#endif
        socket_t client_sock = accept(s, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_sock == INVALID_SOCK) {
            if (!running_) break;
            continue;
        }

        int nodelay = 1;
#ifdef _WIN32
        setsockopt(client_sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));
#else
        setsockopt(client_sock, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
#endif

        std::string raw_req = read_full_http_message(client_sock);
        if (!raw_req.empty()) {
            HttpRequest parsed = parse_http_request(raw_req);
            std::string ws_key = extract_ws_key(raw_req);
            if (!ws_key.empty()) {
                if (!is_request_authorized(parsed, configured_api_key())) {
                    std::string deny = build_http_response(401, "Unauthorized",
                        "{\"status\":\"error\",\"error\":\"API key invalida o ausente\"}");
                    send(client_sock, deny.data(), static_cast<int>(deny.size()), 0);
                    CLOSE_SOCK(client_sock);
                    continue;
                }
                std::string q = parsed.query;
                std::thread([this, client_sock, ws_key, q]() {
                    handle_esp32_websocket_session(client_sock, service_, ws_key, q);
                    CLOSE_SOCK(client_sock);
                }).detach();
                continue;
            }

            HttpResponse resp = service_.handle_http_request(parsed);
            std::string raw_resp = build_http_response(resp);
            size_t total_sent = 0;
            while (total_sent < raw_resp.size()) {
                int n = send(client_sock, raw_resp.data() + total_sent, static_cast<int>(raw_resp.size() - total_sent), 0);
                if (n <= 0) break;
                total_sent += static_cast<size_t>(n);
            }
        }

        CLOSE_SOCK(client_sock);
    }
}

} // namespace llm_api
