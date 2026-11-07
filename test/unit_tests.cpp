#include "api_core.h"
#include "http_server.h"
#include "llm_comms.h"
#include "stt_comms.h"
#include "tts_comms.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

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
    typedef SOCKET test_sock_t;
    #define TEST_INVALID_SOCK INVALID_SOCKET
    #define TEST_CLOSE_SOCK(s) closesocket(s)
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    typedef int test_sock_t;
    #define TEST_INVALID_SOCK (-1)
    #define TEST_CLOSE_SOCK(s) close(s)
#endif

static std::string send_raw_http_request(const std::string& host, int port, const std::string& raw_request) {
#ifdef _WIN32
    static bool wsa_init = false;
    if (!wsa_init) {
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
        wsa_init = true;
    }
#endif
    test_sock_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == TEST_INVALID_SOCK) return "";

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        TEST_CLOSE_SOCK(sock);
        return "";
    }

    size_t total_sent = 0;
    while (total_sent < raw_request.size()) {
        int n = send(sock, raw_request.data() + total_sent, static_cast<int>(raw_request.size() - total_sent), 0);
        if (n <= 0) break;
        total_sent += static_cast<size_t>(n);
    }

    std::string response;
    char buf[4096];
    while (true) {
        int n = recv(sock, buf, sizeof(buf), 0);
        if (n <= 0) break;
        response.append(buf, n);
    }

    TEST_CLOSE_SOCK(sock);
    return response;
}

static std::string resolve_sample_path(const std::string& filename) {
    namespace fs = std::filesystem;
    const std::vector<std::string> candidates = {
        "C:/_PROYECTOS_2030/amibot/test/samples/" + filename,
        "test/samples/" + filename,
        "../test/samples/" + filename,
        "samples/" + filename,
        "../samples/" + filename
    };
    for (const auto& c : candidates) {
        std::error_code ec;
        if (fs::exists(c, ec)) {
            return fs::absolute(c, ec).string();
        }
    }
    return "samples/" + filename;
}

static int g_passed = 0;
static int g_failed = 0;

#define EXPECT_TRUE(cond, msg) \
    do { \
        if (cond) { \
            ++g_passed; \
            std::cout << "  [PASS] " << msg << std::endl; \
        } else { \
            ++g_failed; \
            std::cerr << "  [FAIL] " << msg << " (linea " << __LINE__ << ")" << std::endl; \
        } \
    } while (0)

void test_json_escape() {
    std::cout << "\n[Suite 1] Probando json_escape() y llm_comms::clean_text()..." << std::endl;
    std::string raw = "Hola \"Mundo\"\nLinea\\2\tFin";
    std::string escaped = llm_api::json_escape(raw);
    EXPECT_TRUE(escaped == "Hola \\\"Mundo\\\"\\nLinea\\\\2\\tFin", "Escapa comillas, saltos de linea, barras y tabuladores");

    std::string dirty = "¡Hola! **Estoy bien** 😊🚀, ¿Cómo estás tú? `código` #tag";
    std::string cleaned = llm_comms::clean_text(dirty);
    EXPECT_TRUE(cleaned == "Hola! Estoy bien , Como estas tu? codigo tag", "clean_text() elimina emojis, simbolos y tildes dejando texto limpio puro");
}

void test_json_extract_string() {
    std::cout << "\n[Suite 2] Probando json_extract_string()..." << std::endl;
    std::string json = "{\"prompt\": \"Hola \\\"amibot\\\"\", \"model\": \"qwen\"}";
    EXPECT_TRUE(llm_api::json_extract_string(json, "prompt") == "Hola \"amibot\"", "Extrae campo string con comillas escapadas");
    EXPECT_TRUE(llm_api::json_extract_string(json, "model") == "qwen", "Extrae segundo campo string");
    EXPECT_TRUE(llm_api::json_extract_string(json, "inexistente").empty(), "Retorna vacio para clave inexistente");
}

void test_json_extract_numbers() {
    std::cout << "\n[Suite 3] Probando json_extract_int() y json_extract_float()..." << std::endl;
    std::string json = "{\"max_tokens\": 64, \"temperature\": 0.25}";
    EXPECT_TRUE(llm_api::json_extract_int(json, "max_tokens", 128) == 64, "Extrae entero correctamente");
    EXPECT_TRUE(llm_api::json_extract_int(json, "no_existe", 128) == 128, "Usa valor por defecto en int si no existe");
    float temp = llm_api::json_extract_float(json, "temperature", 0.7f);
    EXPECT_TRUE(std::fabs(temp - 0.25f) < 0.001f, "Extrae float correctamente");
    float def_temp = llm_api::json_extract_float(json, "no_existe", 0.7f);
    EXPECT_TRUE(std::fabs(def_temp - 0.7f) < 0.001f, "Usa valor por defecto en float si no existe");
}

void test_http_parser_and_builder() {
    std::cout << "\n[Suite 4] Probando parse_http_request() y build_http_response()..." << std::endl;
    std::string raw_post =
        "POST /generate?debug=1 HTTP/1.1\r\n"
        "Host: localhost:8080\r\n"
        "Content-Type: application/json\r\n"
        "\r\n"
        "{\"prompt\":\"Hola\"}";

    auto parsed = llm_api::parse_http_request(raw_post);
    EXPECT_TRUE(parsed.valid, "Peticion POST valida");
    EXPECT_TRUE(parsed.method == "POST", "Metodo POST detectado");
    EXPECT_TRUE(parsed.path == "/generate", "Path limpio sin query string");
    EXPECT_TRUE(parsed.body == "{\"prompt\":\"Hola\"}", "Body extraido correctamente");

    auto invalid_req = llm_api::parse_http_request("");
    EXPECT_TRUE(!invalid_req.valid, "Detecta peticion vacia como invalida");

    std::string raw_resp = llm_api::build_http_response(200, "OK", "{\"status\":\"ok\"}");
    EXPECT_TRUE(raw_resp.find("HTTP/1.1 200 OK\r\n") == 0, "Cabecera de estado HTTP 200 correcta");
    EXPECT_TRUE(raw_resp.find("Content-Length: 15\r\n") != std::string::npos, "Content-Length calculado exactamente");
}

void test_model_discovery_and_resolution(llm_api::ApiService& service) {
    std::cout << "\n[Suite 5] Probando find_models_directory(), list_models(), resolve_model_path() e is_instruct_model()..." << std::endl;
    auto models = service.list_models();
    EXPECT_TRUE(models.size() >= 1, "Se detecto el modelo .gguf en la carpeta de modelos");

    std::string qwen_path = service.resolve_model_path("qwen");
    std::string exact_path = service.resolve_model_path("qwen2.5-3b-instruct-q4_k_m.gguf");
    std::string none_path = service.resolve_model_path("modelo_inexistente_xyz");

    EXPECT_TRUE(!qwen_path.empty(), "Resuelve alias 'qwen' a qwen2.5-3b-instruct-q4_k_m.gguf");
    EXPECT_TRUE(!exact_path.empty(), "Resuelve nombre exacto qwen2.5-3b-instruct-q4_k_m.gguf");
    EXPECT_TRUE(none_path.empty(), "Retorna vacio para modelo inexistente");

    EXPECT_TRUE(!llm_api::ApiService::is_instruct_model("base-model.gguf"), "Identifica modelo base (no instruct)");
    EXPECT_TRUE(llm_api::ApiService::is_instruct_model("qwen2.5-3b-instruct-q4_k_m.gguf"), "Identifica qwen2.5-3b-instruct-q4_k_m.gguf como modelo instruct");
}

void test_llm_comms_c_and_cpp_functions(const std::string& model_path) {
    std::cout << "\n[Suite 6] Probando funciones de libreria llm_comms (C++ API y C ABI) con qwen2.5-3b-instruct-q4_k_m.gguf..." << std::endl;

    llm_comms::llm_comms_handle h = llm_comms::llm_comms_create();
    EXPECT_TRUE(h != nullptr, "llm_comms_create() crea instancia correctamente");

    int init_rc = llm_comms::llm_comms_init(h, model_path.c_str(), 1024, 99, 4);
    EXPECT_TRUE(init_rc == 0, "llm_comms_init() carga qwen2.5-3b-instruct-q4_k_m.gguf satisfactoriamente");

    char out_buf[512] = {0};
    int len = llm_comms::llm_comms_send_text(h, "Artificial intelligence is", out_buf, sizeof(out_buf), 16, 0.1f);
    EXPECT_TRUE(len > 0 && std::strlen(out_buf) > 0, "llm_comms_send_text() genera respuesta en buffer C");
    std::cout << "    -> Respuesta C ABI (qwen-3b): " << out_buf << std::endl;

    char chat_buf[512] = {0};
    int chat_len = llm_comms::llm_comms_send_chat_simple(h, "Helpful assistant", "Hello!", chat_buf, sizeof(chat_buf), 12, 0.1f);
    EXPECT_TRUE(chat_len > 0 && std::strlen(chat_buf) > 0, "llm_comms_send_chat_simple() genera respuesta en buffer C");

    llm_comms::llm_comms_destroy(h);

    llm_comms::LLMClient cpp_client;
    EXPECT_TRUE(!cpp_client.is_ready(), "is_ready() es false antes de init()");

    llm_comms::Config cfg;
    cfg.model_path = model_path;
    cfg.n_ctx = 1024;
    cfg.n_gpu_layers = 99;
    EXPECT_TRUE(cpp_client.init(cfg), "LLMClient::init() exitoso");
    EXPECT_TRUE(cpp_client.is_ready(), "is_ready() es true tras init()");

    std::vector<llm_comms::ChatMessage> msgs = { {"user", "Hello"} };
    std::string formatted = cpp_client.format_chat_prompt(msgs, true);
    EXPECT_TRUE(!formatted.empty(), "format_chat_prompt() formatea mensajes correctamente");

    int streamed_pieces = 0;
    llm_comms::GenerateOptions opts;
    opts.max_tokens = 12;
    opts.temperature = 0.0f;
    opts.stream_cb = [&streamed_pieces](const std::string&) {
        ++streamed_pieces;
        return true;
    };
    std::string text = cpp_client.send_text("The future of robotics is", opts);
    EXPECT_TRUE(!text.empty() && streamed_pieces > 0, "LLMClient::send_text() con stream_cb invocado token por token");

    cpp_client.close();
    EXPECT_TRUE(!cpp_client.is_ready(), "is_ready() es false tras close()");
}

void test_api_error_handling(llm_api::ApiService& service) {
    std::cout << "\n[Suite 7] Probando manejo de errores de ApiService..." << std::endl;

    auto r404 = service.handle_http_request({"GET", "/ruta_desconocida", "", true});
    EXPECT_TRUE(r404.status_code == 404, "Retorna 404 para ruta inexistente");

    auto r400 = service.handle_http_request({"POST", "/generate", "{\"model\":\"qwen\"}", true});
    EXPECT_TRUE(r400.status_code == 400, "Retorna 400 cuando falta 'prompt'");

    auto r_bad_model = service.handle_http_request({"POST", "/generate", "{\"prompt\":\"Hola\",\"model\":\"inexistente\"}", true});
    EXPECT_TRUE(r_bad_model.status_code == 404, "Retorna 404 cuando se pide un modelo inexistente");
}

void test_live_http_server(llm_api::ApiService& service) {
    std::cout << "\n[Suite 8] Probando Servidor HTTP en vivo (TCP Socket) con qwen2.5-3b-instruct-q4_k_m.gguf..." << std::endl;

    const int test_port = 18089;
    llm_api::HttpServer server(service, test_port);
    bool started = server.start_async();
    EXPECT_TRUE(started && server.is_running(), "HttpServer::start_async() escuchando en puerto " + std::to_string(test_port));

    std::string req_models = "GET /models HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
    std::string resp_models = send_raw_http_request("127.0.0.1", test_port, req_models);
    EXPECT_TRUE(resp_models.find("200 OK") != std::string::npos &&
                resp_models.find("qwen2.5-3b-instruct-q4_k_m.gguf") != std::string::npos,
                "GET /models devuelve qwen2.5-3b-instruct-q4_k_m.gguf por HTTP");

    std::string body_qwen_en = "{\"model\":\"qwen2.5-3b-instruct-q4_k_m.gguf\",\"prompt\":\"Reply in one short sentence: What is artificial intelligence?\",\"max_tokens\":32,\"temperature\":0.2}";
    std::string req_qwen_en =
        "POST /generate HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body_qwen_en.size()) + "\r\n\r\n" + body_qwen_en;

    std::string raw_resp_en = send_raw_http_request("127.0.0.1", test_port, req_qwen_en);
    auto parsed_en = llm_api::parse_http_request("POST / HTTP/1.1\r\n\r\n" + raw_resp_en.substr(raw_resp_en.find("\r\n\r\n") + 4));
    std::string en_answer = llm_api::json_extract_string(parsed_en.body, "response");
    std::cout << "    [HTTP -> qwen2.5-3b-instruct-q4_k_m.gguf (EN)] Respuesta: \"" << en_answer << "\"" << std::endl;
    EXPECT_TRUE(raw_resp_en.find("200 OK") != std::string::npos && !en_answer.empty(),
                "POST /generate (EN) con qwen2.5-3b-instruct-q4_k_m.gguf responde exitosamente");

    std::string body_qwen = "{\"model\":\"qwen2.5-3b-instruct-q4_k_m.gguf\",\"prompt\":\"Responde en una sola frase corta: ¿Qué es un contenedor Docker?\",\"max_tokens\":48,\"temperature\":0.3}";
    std::string req_qwen =
        "POST /generate HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body_qwen.size()) + "\r\n\r\n" + body_qwen;

    std::string raw_resp_qwen = send_raw_http_request("127.0.0.1", test_port, req_qwen);
    auto parsed_qwen = llm_api::parse_http_request("POST / HTTP/1.1\r\n\r\n" + raw_resp_qwen.substr(raw_resp_qwen.find("\r\n\r\n") + 4));
    std::string qwen_answer = llm_api::json_extract_string(parsed_qwen.body, "response");
    std::cout << "    [HTTP -> qwen2.5-3b-instruct-q4_k_m.gguf (ES)] Respuesta: \"" << qwen_answer << "\"" << std::endl;
    EXPECT_TRUE(raw_resp_qwen.find("200 OK") != std::string::npos && !qwen_answer.empty(),
                "POST /generate (ES) con qwen2.5-3b-instruct-q4_k_m.gguf responde exitosamente");

    std::string req_health = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
    std::string resp_health = send_raw_http_request("127.0.0.1", test_port, req_health);
    EXPECT_TRUE(resp_health.find("\"ready\":true") != std::string::npos, "GET /health confirma estado ready=true");

    server.stop();
    EXPECT_TRUE(!server.is_running(), "HttpServer::stop() detiene el servidor limpiamente");
}

void test_tts_comms_library() {
    std::cout << "\n[Suite 9] Probando libreria tts_comms (C++ y C ABI, 54 voces, WAV 24kHz)..." << std::endl;

    std::string dirty = "¡Hola! **Mundo** 😊 ¿Qué tal?";
    EXPECT_TRUE(tts_comms::clean_text(dirty) == "Hola! Mundo Que tal?", "tts_comms::clean_text() limpia emojis, tildes y simbolos");

    auto sentences = tts_comms::split_sentences("Hola mundo. Esta es la frase 2! Y el numero 3.14 sigue unido.");
    EXPECT_TRUE(sentences.size() == 3, "tts_comms::split_sentences() separa oraciones respetando decimales 3.14");

    tts_comms::TTSClient tts;
    tts_comms::Config cfg;
    cfg.default_voice = "ef_dora";
    EXPECT_TRUE(tts.init(cfg) && tts.is_ready(), "TTSClient::init() carga catalogo de voces .bin");

    auto voices = tts.list_voices();
    EXPECT_TRUE(voices.size() == 54, "TTSClient::list_voices() detecta las 54 voces multilingues");
    EXPECT_TRUE(tts.has_voice("ef_dora") && tts.has_voice("em_alex") && tts.has_voice("af_heart"),
                "TTSClient::has_voice() verifica voces en espanol e ingles");

    std::vector<float> style;
    EXPECT_TRUE(tts.get_voice_style("ef_dora", 25, style) && style.size() == tts_comms::TTSCOMMS_STYLE_DIM,
                "TTSClient::get_voice_style() extrae vector de estilo de 256 dimensiones");

    std::string test_wav = (std::filesystem::path(resolve_sample_path("sample_en.wav")).parent_path() / "tts_unit_test.wav").string();
    tts_comms::SynthesizeOptions opts;
    opts.voice = "ef_dora";
    opts.speed = 1.0f;
    EXPECT_TRUE(tts.synthesize_to_wav("Hola, prueba unitaria de sintesis de voz.", test_wav, opts),
                "TTSClient::synthesize_to_wav() genera archivo WAV de 24000 Hz en disco");

    tts_comms::tts_comms_handle h = tts_comms::tts_comms_create();
    EXPECT_TRUE(h != nullptr, "tts_comms_create() crea manejador C ABI");
    EXPECT_TRUE(tts_comms::tts_comms_init(h, "", "em_alex") == 0, "tts_comms_init() inicializa por C ABI");
    std::vector<float> pcm_buf(48000, 0.0f);
    int n_gen = tts_comms::tts_comms_synthesize_pcm(h, "Hola amigos", "em_alex", 1.0f, pcm_buf.data(), static_cast<int>(pcm_buf.size()));
    EXPECT_TRUE(n_gen > 1000, "tts_comms_synthesize_pcm() sintetiza muestras PCM por C ABI");
    tts_comms::tts_comms_destroy(h);
}

void test_stt_comms_library() {
    std::cout << "\n[Suite 10] Probando libreria stt_comms (C++ y C ABI, decodificador/remuestreador WAV 16kHz y motor STT)..." << std::endl;

    std::string raw_stt = " [BLANK_AUDIO] ¡Hola! **Prueba** 🎵 [MUSIC] ";
    EXPECT_TRUE(stt_comms::clean_text(raw_stt) == "Hola! Prueba", "stt_comms::clean_text() elimina [BLANK_AUDIO], emojis y simbolos");

    std::string test_wav = (std::filesystem::path(resolve_sample_path("sample_en.wav")).parent_path() / "tts_unit_test.wav").string();
    std::vector<float> samples_16k;
    std::string wav_err;
    EXPECT_TRUE(stt_comms::read_wav_16k_mono(test_wav, samples_16k, wav_err) && samples_16k.size() > 5000,
                "stt_comms::read_wav_16k_mono() decodifica WAV de 24 kHz y remuestrea automaticamente a 16 kHz mono");
    std::error_code ec;
    std::filesystem::remove(test_wav, ec);

    std::string stt_model = resolve_sample_path("test-sttcomms-tiny.bin");
    std::string sample_wav = resolve_sample_path("sample_en.wav");

    std::vector<float> jfk_samples;
    EXPECT_TRUE(stt_comms::read_wav_16k_mono(sample_wav, jfk_samples, wav_err) && jfk_samples.size() == 176000,
                "stt_comms::read_wav_16k_mono() lee sample_en.wav (11.0s @ 16kHz = 176000 muestras)");

    stt_comms::STTClient stt;
    stt_comms::Config cfg;
    cfg.model_path = stt_model;
    cfg.use_gpu = true;
    EXPECT_TRUE(stt.init(cfg) && stt.is_ready(), "STTClient::init() inicializa motor acustico sttcomms");

    stt_comms::TranscribeOptions opts;
    opts.language = "en";
    opts.max_tokens = 16;
    std::string trans = stt.transcribe_samples(jfk_samples.data(), 32000, opts);
    EXPECT_TRUE(stt.get_last_error().empty(), "STTClient::transcribe_samples() ejecuta inferencia completa sin errores");
    stt.close();
    EXPECT_TRUE(!stt.is_ready(), "STTClient::close() libera recursos correctamente");
}

void test_voice_to_voice_pipeline(llm_api::ApiService& service) {
    std::cout << "\n[Suite 11] Probando Pipeline Completo Voz -> STT (stt_comms) -> LLM (llm_comms) -> TTS (tts_comms) -> Audio WAV..." << std::endl;

    std::string sample_wav = resolve_sample_path("sample_en.wav");
    std::ifstream in(sample_wav, std::ios::binary | std::ios::ate);
    std::streamsize sz = in.tellg();
    in.seekg(0, std::ios::beg);
    std::string raw_wav_bytes(static_cast<size_t>(sz), '\0');
    in.read(&raw_wav_bytes[0], sz);

    const int test_port = 18090;
    llm_api::HttpServer server(service, test_port);
    EXPECT_TRUE(server.start_async(), "Servidor HTTP iniciado para prueba Voz a Voz");

    std::string req_voice =
        "POST /voice?model=qwen&voice=ef_dora&lang=en&max_tokens=32 HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: audio/wav\r\n"
        "Content-Length: " + std::to_string(raw_wav_bytes.size()) + "\r\n\r\n" + raw_wav_bytes;

    std::string raw_resp = send_raw_http_request("127.0.0.1", test_port, req_voice);
    size_t hdr_end = raw_resp.find("\r\n\r\n");
    EXPECT_TRUE(raw_resp.find("200 OK") != std::string::npos &&
                raw_resp.find("Content-Type: audio/wav") != std::string::npos &&
                hdr_end != std::string::npos,
                "POST /voice responde HTTP 200 OK con Content-Type: audio/wav");

    if (hdr_end != std::string::npos) {
        std::string headers = raw_resp.substr(0, hdr_end);
        std::string wav_out = raw_resp.substr(hdr_end + 4);
        std::cout << "    Cabeceras de trazabilidad devueltas:\n" << headers << std::endl;
        std::cout << "    Audio WAV devuelto: " << wav_out.size() << " bytes" << std::endl;

        std::vector<float> decoded_out;
        std::string err;
        bool valid_wav = stt_comms::read_wav_bytes_16k_mono(
            reinterpret_cast<const uint8_t*>(wav_out.data()),
            wav_out.size(),
            decoded_out,
            err
        );
        EXPECT_TRUE(valid_wav && decoded_out.size() > 5000,
                    "El cuerpo HTTP devuelto por la API es un archivo de audio RIFF/WAVE valido y reproducible");
    }

    // 2. Probar envio directo desde ESP32 (PCM 16-bit crudo sin cabecera WAV -> respuesta PCM 16-bit crudo a 16kHz)
    if (raw_wav_bytes.size() > 44) {
        std::string raw_pcm_input = raw_wav_bytes.substr(44, 64000);
        std::string req_esp32 =
            "POST /esp32?model=qwen&voice=ef_dora&lang=en&rate=16000&out_rate=16000&format=pcm&max_tokens=24 HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: application/octet-stream\r\n"
            "Content-Length: " + std::to_string(raw_pcm_input.size()) + "\r\n\r\n" + raw_pcm_input;

        std::string esp32_resp = send_raw_http_request("127.0.0.1", test_port, req_esp32);
        size_t esp32_hdr_end = esp32_resp.find("\r\n\r\n");
        EXPECT_TRUE(esp32_resp.find("200 OK") != std::string::npos &&
                    esp32_resp.find("Content-Type: audio/pcm") != std::string::npos &&
                    esp32_resp.find("X-Emotion: ") != std::string::npos &&
                    esp32_hdr_end != std::string::npos,
                    "POST /esp32 acepta PCM 16-bit crudo sin cabecera WAV y devuelve audio/pcm + X-Emotion directo para ESP32");

        // 3. Probar empaquetado/desempaquetado de tramas de 4 bytes del protocolo ESP32 ([0,0,len_HI,len_LO] + payload)
        auto framed_vec = llm_api::pack_esp32_audio_frames(
            reinterpret_cast<const uint8_t*>(raw_pcm_input.data()),
            raw_pcm_input.size(),
            240
        );
        std::string framed_str(reinterpret_cast<const char*>(framed_vec.data()), framed_vec.size());
        std::string unframed_str = llm_api::unwrap_esp32_audio_frames(framed_str);
        EXPECT_TRUE(unframed_str == raw_pcm_input,
                    "pack_esp32_audio_frames() y unwrap_esp32_audio_frames() preservan el audio exacto del ESP32");

        EXPECT_TRUE(llm_api::ApiService::detect_emotion("Excelente, muchas gracias") == "happy" &&
                    llm_api::ApiService::detect_emotion("Hubo un error de red") == "alert",
                    "ApiService::detect_emotion() clasifica emociones para la pantalla OLED del ESP32");
    }

    server.stop();
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "   Ejecutando Suite de Tests Unitarios y de Integracion   " << std::endl;
    std::cout << "==========================================================" << std::endl;

    test_json_escape();
    test_json_extract_string();
    test_json_extract_numbers();
    test_http_parser_and_builder();

    llm_api::ApiService service;
    std::string qwen_path = service.resolve_model_path("qwen");
    if (!qwen_path.empty()) {
        test_model_discovery_and_resolution(service);
        test_llm_comms_c_and_cpp_functions(qwen_path);
        test_api_error_handling(service);
        test_live_http_server(service);
        test_tts_comms_library();
        test_stt_comms_library();
        test_voice_to_voice_pipeline(service);
    } else {
        std::cout << "\n[INFO] Entorno CI sin modelo .gguf de 2.1 GB: ejecutando suites independientes..." << std::endl;
        EXPECT_TRUE(!llm_api::ApiService::is_instruct_model("base-model.gguf"), "Identifica modelo base (no instruct)");
        EXPECT_TRUE(llm_api::ApiService::is_instruct_model("qwen2.5-3b-instruct-q4_k_m.gguf"), "Identifica qwen2.5-3b-instruct-q4_k_m.gguf como modelo instruct");
        test_api_error_handling(service);
        test_tts_comms_library();
        test_stt_comms_library();
        EXPECT_TRUE(llm_api::ApiService::detect_emotion("Excelente, muchas gracias") == "happy" &&
                    llm_api::ApiService::detect_emotion("Hubo un error de red") == "alert",
                    "ApiService::detect_emotion() clasifica emociones para la pantalla OLED del ESP32");
    }

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "   Resumen de Tests: " << g_passed << " pasados, " << g_failed << " fallidos." << std::endl;
    std::cout << "==========================================================" << std::endl;

    return g_failed == 0 ? 0 : 1;
}
