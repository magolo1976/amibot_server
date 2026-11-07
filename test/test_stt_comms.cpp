#include "stt_comms.h"
#include <iostream>
#include <chrono>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

int main(int argc, char** argv) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    if (argc < 3) {
        std::cout << "Uso: test_stt_comms <ruta_modelo.bin> <ruta_audio.wav> [idioma=auto]\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string wav_path = argv[2];
    std::string language = (argc >= 4) ? argv[3] : "auto";

    stt_comms::STTClient client;
    stt_comms::Config cfg;
    cfg.model_path = model_path;
    cfg.use_gpu = true;

    std::cout << "[stt_comms] Cargando modelo: " << model_path << "...\n";
    if (!client.init(cfg)) {
        std::cerr << "[Error] " << client.get_last_error() << "\n";
        return 1;
    }

    stt_comms::TranscribeOptions opts;
    opts.language = language;

    auto t0 = std::chrono::high_resolution_clock::now();
    std::string text = client.transcribe_wav(wav_path, opts);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (text.empty() && !client.get_last_error().empty()) {
        std::cerr << "[Error] " << client.get_last_error() << "\n";
        return 1;
    }

    std::cout << "--- Texto Transcrito (" << ms << " ms) ---\n";
    std::cout << text << "\n";
    std::cout << "-----------------------------------------\n";
    return 0;
}
