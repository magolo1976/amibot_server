#ifndef LLM_API_HTTP_SERVER_H
#define LLM_API_HTTP_SERVER_H

#include "api_core.h"
#include <atomic>
#include <thread>
#include <string>

namespace llm_api {

class HttpServer {
public:
    HttpServer(ApiService& service, int port = 8080);
    ~HttpServer();

    // Inicia el servidor en el hilo actual (bloqueante)
    bool start_blocking();

    // Inicia el servidor en un hilo en segundo plano
    bool start_async();

    // Detiene el servidor limpiamente
    void stop();

    bool is_running() const;
    int get_port() const;

private:
    void accept_loop();

    ApiService& service_;
    int port_;
    std::atomic<bool> running_{false};
    uintptr_t listen_sock_;
    std::thread server_thread_;
};

} // namespace llm_api

#endif // LLM_API_HTTP_SERVER_H
