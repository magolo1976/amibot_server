# `llm_api` — Microservicio HTTP Nativo en C++17 para Inferencia LLM

API ultraligera y de máxima velocidad escrita en **C++17 nativo** que enlaza directamente en memoria con la librería compilada en **`server/libraries/llm_comms`**.

## ⚡ ¿Por qué C++17?
* **Cero sobrecarga FFI / serialización**: Se comunica directamente con `llm_comms::LLMClient` en el mismo espacio de memoria sin puentes intermedios (ni Python `ctypes` ni Node.js FFI).
* **Arranque instantáneo y huella mínima**: El ejecutable pesa ~70 KB y consume menos de 5 MB de RAM base (más el tamaño del modelo `.gguf` cargado).
* **Multiplataforma real**: Compila sin dependencias externas tanto en Windows (Winsock2 + MSVC) como en contenedores Docker Linux (`debian:bookworm-slim` + GCC + POSIX sockets).

---

## 🌐 Endpoints HTTP

### 1. `GET /health`
Verifica el estado del servicio y qué modelo se encuentra actualmente cargado en memoria.
```json
{
  "status": "ok",
  "ready": true,
  "loaded_model": "qwen2.5-3b-instruct-q4_k_m.gguf"
}
```

### 2. `GET /models`
Lista todos los modelos `.gguf` disponibles en la carpeta de modelos.
```json
{
  "status": "ok",
  "models": [
    "qwen2.5-3b-instruct-q4_k_m.gguf"
  ]
}
```

### 3. `POST /generate`
Envía un texto al LLM indicado (soporta nombre completo o alias parcial como `"qwen"`) y retorna la respuesta generada.

**Request:**
```json
{
  "prompt": "Hola, ¿qué puedes hacer?",
  "model": "qwen2.5-3b-instruct-q4_k_m.gguf",
  "max_tokens": 64,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "status": "ok",
  "model": "qwen2.5-3b-instruct-q4_k_m.gguf",
  "response": "Hola! Puedo responder preguntas, redactar textos y ayudarte a resolver problemas.",
  "elapsed_ms": 320
}
```

---

## 🐳 Despliegue en VPS Linux con Docker

Desde la carpeta `server/llm_api`:

```bash
docker compose up -d --build
```

El `Dockerfile` multi-etapa compila automáticamente `resources/llm_comms` generando `libllm_comms.so` en Linux, compila `llm_api_server` y monta la carpeta de modelos en `/app/models`.

---

## 🧪 Tests Unitarios y de Integración

Para compilar y ejecutar la suite completa de tests unitarios (que valida cada función de JSON, HTTP, descubrimiento de modelos, ABI C, API C++ y peticiones reales contra ambos modelos):

```powershell
cmake -B build -A x64
cmake --build build --config Release
.\build\bin\Release\llm_api_tests.exe
```
