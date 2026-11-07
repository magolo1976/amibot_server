#include "api_core.h"
#include "http_server.h"

#include <iostream>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    int port = 8080;
    std::string models_dir;
    std::string preload_model = "qwen2.5-3b-instruct-q4_k_m.gguf";

    if (const char* env_port = std::getenv("PORT")) {
        port = std::atoi(env_port);
    }
    if (const char* env_preload = std::getenv("PRELOAD_MODEL")) {
        preload_model = env_preload;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--models-dir" && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            preload_model = argv[++i];
        }
    }

    llm_api::ApiService service(models_dir);

    std::cout << "================================================" << std::endl;
    std::cout << "   llm_api - Servidor HTTP Nativo (C++17)       " << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "[*] Directorio de modelos: " << service.get_models_directory() << std::endl;

    auto models = service.list_models();
    std::cout << "[*] Modelos detectados (" << models.size() << "):" << std::endl;
    for (const auto& m : models) {
        std::cout << "    - " << m << std::endl;
    }

    if (!preload_model.empty()) {
        std::cout << "[*] Precargando modelo: " << preload_model << " ..." << std::endl;
        if (service.ensure_model_loaded(preload_model)) {
            std::cout << "[+] Modelo activo: " << service.get_loaded_model_name() << std::endl;
        } else {
            std::cerr << "[-] No se pudo precargar el modelo indicado." << std::endl;
        }
    }

    llm_api::HttpServer server(service, port);
    std::cout << "[+] Escuchando en http://0.0.0.0:" << port << " y ws://0.0.0.0:" << port << std::endl;
    std::cout << "    Endpoints disponibles (ESP32 y Clientes):" << std::endl;
    std::cout << "      WS   ws://0.0.0.0:" << port << "   (Protocolo WebSocket ESP32: hello, listen, tramas 4B -> stt, llm+emotion, tts)" << std::endl;
    std::cout << "      POST /esp32      (Acepta PCM 16-bit I2S, tramas 4B, .wav, multipart o Base64 -> devuelve WAV, PCM, tramas o JSON)" << std::endl;
    std::cout << "      POST /voice      (Entrada: voz -> stt_comms -> llm_comms -> tts_comms -> Salida: audio/wav o audio/pcm)" << std::endl;
    std::cout << "      POST /generate   (Acepta voz o JSON)" << std::endl;
    std::cout << "      GET  /health" << std::endl;
    std::cout << "      GET  /models" << std::endl;

    if (!server.start_blocking()) {
        std::cerr << "[-] Error al iniciar el servidor en el puerto " << port << std::endl;
        return 1;
    }

    return 0;
}
