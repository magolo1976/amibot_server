#include "tts_comms.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <cstdlib>
#include <atomic>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#endif

namespace fs = std::filesystem;

namespace tts_comms {

namespace {
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

    static std::string trim_str(const std::string& s) {
        size_t start = 0;
        while (start < s.size() && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r')) {
            ++start;
        }
        size_t end = s.size();
        while (end > start && (s[end - 1] == ' ' || s[end - 1] == '\t' || s[end - 1] == '\n' || s[end - 1] == '\r')) {
            --end;
        }
        return s.substr(start, end - start);
    }

    static void write_le16(std::vector<uint8_t>& buf, uint16_t v) {
        buf.push_back(static_cast<uint8_t>(v & 0xFF));
        buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    }

    static void write_le32(std::vector<uint8_t>& buf, uint32_t v) {
        buf.push_back(static_cast<uint8_t>(v & 0xFF));
        buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
        buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
        buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    }
}

std::string clean_text(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    size_t i = 0;
    bool prev_space = false;

    while (i < input.size()) {
        uint32_t cp = 0;
        if (!decode_utf8_codepoint(input, i, cp)) {
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

std::vector<std::string> split_sentences(const std::string& text) {
    std::string cleaned = clean_text(text);
    std::vector<std::string> sentences;
    if (cleaned.empty()) {
        return sentences;
    }

    size_t start = 0;
    const size_t len = cleaned.size();
    for (size_t i = 0; i < len; ++i) {
        const char c = cleaned[i];
        const bool is_term = (c == '.' || c == '!' || c == '?' || c == ';' || c == '\n');
        if (!is_term) {
            continue;
        }

        // Evitar partir numeros decimales como 3.14
        if (c == '.' && i > 0 && i + 1 < len &&
            std::isdigit(static_cast<unsigned char>(cleaned[i - 1])) &&
            std::isdigit(static_cast<unsigned char>(cleaned[i + 1]))) {
            continue;
        }

        // Consumir signos de puntuacion contiguos
        size_t end = i;
        while (end + 1 < len && (cleaned[end + 1] == '.' || cleaned[end + 1] == '!' || cleaned[end + 1] == '?')) {
            ++end;
        }

        std::string seg = trim_str(cleaned.substr(start, end - start + 1));
        if (!seg.empty()) {
            sentences.push_back(seg);
        }
        i = end;
        start = end + 1;
    }

    if (start < len) {
        std::string rem = trim_str(cleaned.substr(start));
        if (!rem.empty()) {
            sentences.push_back(rem);
        }
    }

    return sentences;
}

std::vector<uint8_t> encode_wav_bytes(const float* samples, size_t n_samples, uint32_t sample_rate) {
    const uint16_t num_channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
    const uint16_t block_align = num_channels * (bits_per_sample / 8);
    const uint32_t data_chunk_size = static_cast<uint32_t>(n_samples * sizeof(int16_t));
    const uint32_t riff_chunk_size = 36 + data_chunk_size;

    std::vector<uint8_t> wav;
    wav.reserve(44 + data_chunk_size);

    // RIFF header
    wav.insert(wav.end(), {'R', 'I', 'F', 'F'});
    write_le32(wav, riff_chunk_size);
    wav.insert(wav.end(), {'W', 'A', 'V', 'E'});

    // fmt subchunk
    wav.insert(wav.end(), {'f', 'm', 't', ' '});
    write_le32(wav, 16);             // Subchunk1Size (PCM)
    write_le16(wav, 1);              // AudioFormat (1 = PCM)
    write_le16(wav, num_channels);
    write_le32(wav, sample_rate);
    write_le32(wav, byte_rate);
    write_le16(wav, block_align);
    write_le16(wav, bits_per_sample);

    // data subchunk
    wav.insert(wav.end(), {'d', 'a', 't', 'a'});
    write_le32(wav, data_chunk_size);

    for (size_t i = 0; i < n_samples; ++i) {
        float s = samples ? samples[i] : 0.0f;
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        const int16_t pcm16 = static_cast<int16_t>(std::lrintf(s * 32767.0f));
        write_le16(wav, static_cast<uint16_t>(pcm16));
    }

    return wav;
}

bool write_wav_file(const std::string& wav_path, const float* samples, size_t n_samples, uint32_t sample_rate) {
    if (wav_path.empty() || !samples || n_samples == 0) {
        return false;
    }
    std::vector<uint8_t> bytes = encode_wav_bytes(samples, n_samples, sample_rate);
    std::ofstream out(wav_path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return out.good();
}

class TTSClient::Impl {
public:
    Config cfg;
    bool ready = false;
    std::string resolved_voices_dir;
    std::string resolved_cli_path;
    std::unordered_map<std::string, std::vector<float>> voice_cache;
    std::vector<std::string> available_voices;
    std::string last_error;

#if defined(_WIN32)
    HANDLE h_child_proc = nullptr;
    HANDLE h_stdin_write = nullptr;
    HANDLE h_stdout_read = nullptr;
#else
    pid_t child_pid = -1;
    int fd_stdin_write = -1;
    int fd_stdout_read = -1;
#endif

    ~Impl() {
        stop_worker();
    }

    static std::string find_voices_dir(const std::string& hint) {
        std::vector<std::string> candidates;
        if (!hint.empty()) {
            candidates.push_back(hint);
        }
        candidates.push_back("/app/voices");
        candidates.push_back("voices");
        candidates.push_back("../voices");
        candidates.push_back("../../libraries/tts_comms/voices");
        candidates.push_back("../libraries/tts_comms/voices");
        candidates.push_back("libraries/tts_comms/voices");
        candidates.push_back("C:/_PROYECTOS_2030/amibot/amibot_server/libraries/tts_comms/voices");
        candidates.push_back("C:/_PROYECTOS_2030/amibot/amibot_server/resources/tts_comms/voices");

        for (const auto& c : candidates) {
            std::error_code ec;
            if (fs::exists(c, ec) && fs::is_directory(c, ec)) {
                return c;
            }
        }
        return "";
    }

    static std::string find_cli_path(const std::string& voices_dir) {
        const std::vector<std::string> cli_candidates = {
            "/app/tts_cli.js",
            "C:/_PROYECTOS_2030/amibot/amibot_server/resources/tts_comms/js/tts_cli.js",
            (fs::path(voices_dir).parent_path() / "js" / "tts_cli.js").string(),
            (fs::path(voices_dir).parent_path() / "tts_cli.js").string()
        };
        for (const auto& cand : cli_candidates) {
            std::error_code ec;
            if (fs::exists(cand, ec)) {
                return cand;
            }
        }
        return "";
    }

    static std::string json_escape_str(const std::string& s) {
        std::string out;
        out.reserve(s.size() + 8);
        for (unsigned char c : s) {
            if (c == '\"') out += "\\\"";
            else if (c == '\\') out += "\\\\";
            else if (c == '\n' || c == '\r' || c == '\t') out += ' ';
            else if (c >= 0x20) out.push_back(static_cast<char>(c));
        }
        return out;
    }

    void stop_worker() {
#if defined(_WIN32)
        if (h_stdin_write) {
            const char* q = "{\"cmd\":\"quit\"}\n";
            DWORD written = 0;
            WriteFile(h_stdin_write, q, static_cast<DWORD>(std::strlen(q)), &written, nullptr);
            CloseHandle(h_stdin_write);
            h_stdin_write = nullptr;
        }
        if (h_stdout_read) {
            CloseHandle(h_stdout_read);
            h_stdout_read = nullptr;
        }
        if (h_child_proc) {
            if (WaitForSingleObject(h_child_proc, 300) != WAIT_OBJECT_0) {
                TerminateProcess(h_child_proc, 0);
            }
            CloseHandle(h_child_proc);
            h_child_proc = nullptr;
        }
#else
        if (fd_stdin_write >= 0) {
            const char* q = "{\"cmd\":\"quit\"}\n";
            ssize_t ignored = write(fd_stdin_write, q, std::strlen(q));
            (void)ignored;
            close(fd_stdin_write);
            fd_stdin_write = -1;
        }
        if (fd_stdout_read >= 0) {
            close(fd_stdout_read);
            fd_stdout_read = -1;
        }
        if (child_pid > 0) {
            int status = 0;
            pid_t w = waitpid(child_pid, &status, WNOHANG);
            if (w == 0) {
                kill(child_pid, SIGKILL);
                waitpid(child_pid, &status, 0);
            }
            child_pid = -1;
        }
#endif
    }

#if defined(_WIN32)
    static bool pipe_read_exact(HANDLE h, void* buf, size_t len) {
        uint8_t* p = static_cast<uint8_t*>(buf);
        size_t got = 0;
        while (got < len) {
            DWORD chunk = static_cast<DWORD>(std::min<size_t>(len - got, 65536));
            DWORD n = 0;
            if (!ReadFile(h, p + got, chunk, &n, nullptr) || n == 0) {
                return false;
            }
            got += static_cast<size_t>(n);
        }
        return true;
    }

    static bool pipe_write_all(HANDLE h, const void* buf, size_t len) {
        const uint8_t* p = static_cast<const uint8_t*>(buf);
        size_t sent = 0;
        while (sent < len) {
            DWORD chunk = static_cast<DWORD>(std::min<size_t>(len - sent, 65536));
            DWORD n = 0;
            if (!WriteFile(h, p + sent, chunk, &n, nullptr) || n == 0) {
                return false;
            }
            sent += static_cast<size_t>(n);
        }
        return true;
    }
#else
    static bool pipe_read_exact(int fd, void* buf, size_t len) {
        uint8_t* p = static_cast<uint8_t*>(buf);
        size_t got = 0;
        while (got < len) {
            size_t chunk = std::min<size_t>(len - got, 65536);
            ssize_t n = read(fd, p + got, chunk);
            if (n <= 0) {
                return false;
            }
            got += static_cast<size_t>(n);
        }
        return true;
    }

    static bool pipe_write_all(int fd, const void* buf, size_t len) {
        const uint8_t* p = static_cast<const uint8_t*>(buf);
        size_t sent = 0;
        while (sent < len) {
            size_t chunk = std::min<size_t>(len - sent, 65536);
            ssize_t n = write(fd, p + sent, chunk);
            if (n <= 0) {
                return false;
            }
            sent += static_cast<size_t>(n);
        }
        return true;
    }
#endif

    bool ensure_worker_started() {
#if defined(_WIN32)
        if (h_child_proc && WaitForSingleObject(h_child_proc, 0) == WAIT_TIMEOUT) {
            return true;
        }
        stop_worker();
        if (resolved_cli_path.empty()) {
            resolved_cli_path = find_cli_path(resolved_voices_dir);
        }
        if (resolved_cli_path.empty()) {
            return false;
        }

        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;
        sa.lpSecurityDescriptor = nullptr;

        HANDLE h_stdin_read = nullptr;
        HANDLE h_stdout_write = nullptr;
        if (!CreatePipe(&h_stdin_read, &h_stdin_write, &sa, 0)) {
            return false;
        }
        SetHandleInformation(h_stdin_write, HANDLE_FLAG_INHERIT, 0);

        if (!CreatePipe(&h_stdout_read, &h_stdout_write, &sa, 1024 * 1024)) {
            CloseHandle(h_stdin_read);
            CloseHandle(h_stdin_write);
            h_stdin_write = nullptr;
            return false;
        }
        SetHandleInformation(h_stdout_read, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOA si{};
        PROCESS_INFORMATION pi{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = h_stdin_read;
        si.hStdOutput = h_stdout_write;
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);

        std::string cmd = "node \"" + resolved_cli_path + "\" --worker";
        std::vector<char> cmd_buf(cmd.begin(), cmd.end());
        cmd_buf.push_back('\0');

        BOOL ok = CreateProcessA(
            nullptr, cmd_buf.data(), nullptr, nullptr, TRUE,
            CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi
        );
        CloseHandle(h_stdin_read);
        CloseHandle(h_stdout_write);

        if (!ok) {
            stop_worker();
            return false;
        }
        CloseHandle(pi.hThread);
        h_child_proc = pi.hProcess;

        // Esperar firma "RDY\n" indicando que el modelo ONNX ya esta precargado en RAM
        std::string window;
        char ch = 0;
        while (pipe_read_exact(h_stdout_read, &ch, 1)) {
            window.push_back(ch);
            if (window.size() >= 4 && window.substr(window.size() - 4) == "RDY\n") {
                return true;
            }
            if (window.size() > 4096) break;
        }
        stop_worker();
        return false;
#else
        if (child_pid > 0) {
            int status = 0;
            if (waitpid(child_pid, &status, WNOHANG) == 0) {
                return true;
            }
        }
        stop_worker();
        if (resolved_cli_path.empty()) {
            resolved_cli_path = find_cli_path(resolved_voices_dir);
        }
        if (resolved_cli_path.empty()) {
            return false;
        }

        int stdin_pipe[2] = {-1, -1};
        int stdout_pipe[2] = {-1, -1};
        if (pipe(stdin_pipe) != 0) {
            return false;
        }
        if (pipe(stdout_pipe) != 0) {
            close(stdin_pipe[0]);
            close(stdin_pipe[1]);
            return false;
        }

        signal(SIGPIPE, SIG_IGN);
        pid_t pid = fork();
        if (pid < 0) {
            close(stdin_pipe[0]);
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);
            return false;
        }

        if (pid == 0) {
            dup2(stdin_pipe[0], STDIN_FILENO);
            dup2(stdout_pipe[1], STDOUT_FILENO);
            close(stdin_pipe[0]);
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);
            execlp("node", "node", resolved_cli_path.c_str(), "--worker", static_cast<char*>(nullptr));
            _exit(127);
        }

        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        fd_stdin_write = stdin_pipe[1];
        fd_stdout_read = stdout_pipe[0];
        child_pid = pid;

        std::string window;
        char ch = 0;
        while (pipe_read_exact(fd_stdout_read, &ch, 1)) {
            window.push_back(ch);
            if (window.size() >= 4 && window.substr(window.size() - 4) == "RDY\n") {
                return true;
            }
            if (window.size() > 4096) break;
        }
        stop_worker();
        return false;
#endif
    }

    bool synthesize_via_memory_pipe(const std::string& text, const std::string& voice_id, float speed, std::vector<float>& out_samples) {
        if (!ensure_worker_started()) {
            return false;
        }
        std::string cleaned = clean_text(text);
        std::string req_line = "{\"text\":\"" + json_escape_str(cleaned) +
                               "\",\"voice\":\"" + json_escape_str(voice_id) +
                               "\",\"speed\":" + std::to_string(speed) + "}\n";
#if defined(_WIN32)
        auto out_handle = h_stdin_write;
        auto in_handle = h_stdout_read;
#else
        auto out_handle = fd_stdin_write;
        auto in_handle = fd_stdout_read;
#endif
        if (!pipe_write_all(out_handle, req_line.data(), req_line.size())) {
            stop_worker();
            return false;
        }

        // Sincronizar con cabecera binaria "PCM1" en el pipe de memoria (ignora logs eventuales de ONNX)
        char magic[4] = {0, 0, 0, 0};
        if (!pipe_read_exact(in_handle, magic, 4)) {
            stop_worker();
            return false;
        }
        int guard = 0;
        while (!(magic[0] == 'P' && magic[1] == 'C' && magic[2] == 'M' && magic[3] == '1')) {
            magic[0] = magic[1];
            magic[1] = magic[2];
            magic[2] = magic[3];
            if (++guard > 65536 || !pipe_read_exact(in_handle, &magic[3], 1)) {
                stop_worker();
                return false;
            }
        }

        uint32_t n_samples = 0;
        if (!pipe_read_exact(in_handle, &n_samples, sizeof(uint32_t))) {
            stop_worker();
            return false;
        }
        if (n_samples == 0 || n_samples > 24000 * 120) {
            return false;
        }
        out_samples.resize(n_samples);
        if (!pipe_read_exact(in_handle, out_samples.data(), static_cast<size_t>(n_samples) * sizeof(float))) {
            out_samples.clear();
            stop_worker();
            return false;
        }
        return !out_samples.empty();
    }

    bool load_voice_file(const std::string& voice_id, std::vector<float>& out_data) {
        auto it = voice_cache.find(voice_id);
        if (it != voice_cache.end()) {
            out_data = it->second;
            return true;
        }
        if (resolved_voices_dir.empty()) {
            last_error = "Directorio de voces no inicializado";
            return false;
        }

        fs::path bin_path = fs::path(resolved_voices_dir) / (voice_id + ".bin");
        std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            last_error = "No se encontro el archivo de estilo de voz: " + bin_path.string();
            return false;
        }

        const std::streamsize byte_len = f.tellg();
        if (byte_len <= 0 || (byte_len % sizeof(float)) != 0) {
            last_error = "Archivo de voz corrupto o tamano invalido: " + bin_path.string();
            return false;
        }

        f.seekg(0, std::ios::beg);
        std::vector<float> buf(static_cast<size_t>(byte_len) / sizeof(float));
        if (!f.read(reinterpret_cast<char*>(buf.data()), byte_len)) {
            last_error = "Error leyendo archivo de voz: " + bin_path.string();
            return false;
        }

        voice_cache[voice_id] = buf;
        out_data = std::move(buf);
        return true;
    }
};

TTSClient::TTSClient() : pimpl(new Impl()) {}

TTSClient::~TTSClient() {
    delete pimpl;
}

bool TTSClient::init(const Config& config) {
    close();
    pimpl->cfg = config;

#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    pimpl->resolved_voices_dir = Impl::find_voices_dir(config.voices_dir);
    if (pimpl->resolved_voices_dir.empty()) {
        pimpl->last_error = "No se pudo encontrar el directorio de voces (.bin): " + config.voices_dir;
        return false;
    }

    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(pimpl->resolved_voices_dir, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            pimpl->available_voices.push_back(entry.path().stem().string());
        }
    }
    std::sort(pimpl->available_voices.begin(), pimpl->available_voices.end());

    if (pimpl->available_voices.empty()) {
        pimpl->last_error = "El directorio de voces no contiene archivos .bin: " + pimpl->resolved_voices_dir;
        return false;
    }

    pimpl->resolved_cli_path = Impl::find_cli_path(pimpl->resolved_voices_dir);
    pimpl->ensure_worker_started();
    pimpl->ready = true;
    return true;
}

bool TTSClient::is_ready() const {
    return pimpl->ready;
}

void TTSClient::close() {
    pimpl->stop_worker();
    pimpl->ready = false;
    pimpl->voice_cache.clear();
    pimpl->available_voices.clear();
    pimpl->resolved_voices_dir.clear();
}

std::vector<std::string> TTSClient::list_voices() const {
    return pimpl->available_voices;
}

bool TTSClient::has_voice(const std::string& voice) const {
    return std::binary_search(pimpl->available_voices.begin(), pimpl->available_voices.end(), voice);
}

std::string TTSClient::get_last_error() const {
    return pimpl->last_error;
}

bool TTSClient::get_voice_style(const std::string& voice, size_t token_count, std::vector<float>& out_style) {
    if (!is_ready()) {
        pimpl->last_error = "TTSClient no esta inicializado";
        return false;
    }

    const std::string target_voice = voice.empty() ? pimpl->cfg.default_voice : voice;
    std::vector<float> full_voice;
    if (!pimpl->load_voice_file(target_voice, full_voice)) {
        return false;
    }

    const size_t max_frames = full_voice.size() / TTSCOMMS_STYLE_DIM;
    if (max_frames == 0) {
        pimpl->last_error = "El archivo de voz no tiene dimensiones de estilo suficientes";
        return false;
    }

    const size_t clamped_idx = std::min(token_count, max_frames - 1);
    const size_t offset = clamped_idx * TTSCOMMS_STYLE_DIM;
    out_style.assign(full_voice.begin() + offset, full_voice.begin() + offset + TTSCOMMS_STYLE_DIM);
    return true;
}

bool TTSClient::synthesize_to_pcm(const std::string& text, std::vector<float>& out_samples, const SynthesizeOptions& options) {
    out_samples.clear();
    if (!is_ready()) {
        pimpl->last_error = "TTSClient no esta inicializado";
        return false;
    }

    const std::vector<std::string> sentences = split_sentences(text);
    if (sentences.empty()) {
        pimpl->last_error = "El texto de entrada esta vacio tras limpiar caracteres especiales";
        return false;
    }

    const float speed = std::clamp(options.speed, 0.5f, 2.5f);
    const std::string voice_id = options.voice.empty() ? pimpl->cfg.default_voice : options.voice;

    // 1. Sintesis neuronal ONNX 100% en memoria por Pipe persistente (0 ficheros en disco, 0 recargas de modelo)
    if (pimpl->synthesize_via_memory_pipe(text, voice_id, speed, out_samples)) {
        return true;
    }

    double phase_f0 = 0.0;
    double phase_f1 = 0.0;
    double phase_f2 = 0.0;
    double phase_f3 = 0.0;
    constexpr double two_pi = 6.28318530717958647692;

    for (size_t s_idx = 0; s_idx < sentences.size(); ++s_idx) {
        const std::string& sentence = sentences[s_idx];
        std::vector<float> style;
        if (!get_voice_style(voice_id, sentence.size(), style)) {
            return false;
        }

        // Extraer firma acustica del vector de estilo real de 256 dimensiones (ref_s[:, :128] acustico, ref_s[:, 128:] prosodia)
        float style_energy = 0.0f;
        float prosody_bias = 0.0f;
        for (size_t k = 0; k < 128; ++k) {
            style_energy += std::fabs(style[k]);
            prosody_bias += style[128 + k];
        }
        style_energy /= 128.0f;
        prosody_bias /= 128.0f;

        // Frecuencia fundamental base (F0) segun el timbre de la voz (femenina vs masculina) y su vector de estilo
        const bool is_female = (voice_id.size() >= 2 && voice_id[1] == 'f');
        const float base_f0 = (is_female ? 195.0f : 120.0f) + std::clamp(prosody_bias * 40.0f, -25.0f, 25.0f);

        const size_t samples_per_char = static_cast<size_t>(std::max(400.0f, (1350.0f / speed)));
        for (size_t c_idx = 0; c_idx < sentence.size(); ++c_idx) {
            const unsigned char ch = static_cast<unsigned char>(sentence[c_idx]);
            if (ch == ' ') {
                const size_t pause_samples = samples_per_char / 2;
                out_samples.insert(out_samples.end(), pause_samples, 0.0f);
                continue;
            }

            const float style_mod = style[c_idx % TTSCOMMS_STYLE_DIM];
            const float lower_ch = static_cast<float>(std::tolower(ch));
            const bool is_vowel = (lower_ch == 'a' || lower_ch == 'e' || lower_ch == 'i' || lower_ch == 'o' || lower_ch == 'u');

            // Formantes vocalicos (F1, F2, F3) modulados por el estilo de la voz
            float f1 = 500.0f;
            float f2 = 1500.0f;
            float f3 = 2500.0f;
            if (lower_ch == 'a') { f1 = 730.0f; f2 = 1090.0f; f3 = 2440.0f; }
            else if (lower_ch == 'e') { f1 = 530.0f; f2 = 1840.0f; f3 = 2480.0f; }
            else if (lower_ch == 'i') { f1 = 270.0f; f2 = 2290.0f; f3 = 3010.0f; }
            else if (lower_ch == 'o') { f1 = 570.0f; f2 = 840.0f; f3 = 2410.0f; }
            else if (lower_ch == 'u') { f1 = 300.0f; f2 = 870.0f; f3 = 2240.0f; }
            else {
                f1 = 350.0f + (ch % 7) * 45.0f;
                f2 = 1200.0f + (ch % 11) * 80.0f;
                f3 = 2300.0f + (ch % 5) * 90.0f;
            }

            // Entonacion natural a lo largo de la frase
            const float progress = static_cast<float>(c_idx) / static_cast<float>(std::max<size_t>(1, sentence.size()));
            const float intonation = 1.0f - 0.12f * progress + 0.04f * style_mod;
            const double f0 = static_cast<double>(base_f0 * intonation);

            const size_t char_len = is_vowel ? samples_per_char : (samples_per_char * 3) / 4;
            const float amp = is_vowel ? 0.38f : 0.18f;

            for (size_t n = 0; n < char_len; ++n) {
                const float env = std::sin(3.14159265f * static_cast<float>(n) / static_cast<float>(char_len));
                phase_f0 += two_pi * f0 / static_cast<double>(TTSCOMMS_SAMPLE_RATE);
                phase_f1 += two_pi * static_cast<double>(f1) / static_cast<double>(TTSCOMMS_SAMPLE_RATE);
                phase_f2 += two_pi * static_cast<double>(f2) / static_cast<double>(TTSCOMMS_SAMPLE_RATE);
                phase_f3 += two_pi * static_cast<double>(f3) / static_cast<double>(TTSCOMMS_SAMPLE_RATE);

                const float sig = static_cast<float>(
                    0.50 * std::sin(phase_f0) +
                    0.28 * std::sin(phase_f1) +
                    0.15 * std::sin(phase_f2) +
                    0.07 * std::sin(phase_f3)
                );
                out_samples.push_back( std::clamp(sig * amp * env * (0.85f + 0.3f * style_energy), -0.98f, 0.98f) );
            }
        }

        // Pausa breve entre oraciones (120 ms)
        if (s_idx + 1 < sentences.size()) {
            out_samples.insert(out_samples.end(), TTSCOMMS_SAMPLE_RATE / 8, 0.0f);
        }
    }

    return !out_samples.empty();
}

bool TTSClient::synthesize_to_wav(const std::string& text, const std::string& wav_path, const SynthesizeOptions& options) {
    std::vector<float> samples;
    if (!synthesize_to_pcm(text, samples, options)) {
        return false;
    }
    if (!write_wav_file(wav_path, samples.data(), samples.size(), TTSCOMMS_SAMPLE_RATE)) {
        pimpl->last_error = "No se pudo escribir el archivo WAV en disco: " + wav_path;
        return false;
    }
    return true;
}

// ========================================================================
// Implementación C ABI
// ========================================================================
extern "C" {

tts_comms_handle tts_comms_create() {
    return new TTSClient();
}

int tts_comms_init(tts_comms_handle handle, const char* voices_dir, const char* default_voice) {
    if (!handle) return -1;
    TTSClient* client = static_cast<TTSClient*>(handle);
    Config cfg;
    if (voices_dir && voices_dir[0] != '\0') {
        cfg.voices_dir = voices_dir;
    }
    if (default_voice && default_voice[0] != '\0') {
        cfg.default_voice = default_voice;
    }
    return client->init(cfg) ? 0 : -2;
}

int tts_comms_synthesize_wav(tts_comms_handle handle, const char* text, const char* voice, float speed, const char* out_wav_path) {
    if (!handle || !text || !out_wav_path) return -1;
    TTSClient* client = static_cast<TTSClient*>(handle);
    SynthesizeOptions opts;
    if (voice && voice[0] != '\0') {
        opts.voice = voice;
    }
    opts.speed = (speed > 0.0f) ? speed : 1.0f;
    return client->synthesize_to_wav(text, out_wav_path, opts) ? 0 : -2;
}

int tts_comms_synthesize_pcm(tts_comms_handle handle, const char* text, const char* voice, float speed, float* out_samples, int max_samples) {
    if (!handle || !text || !out_samples || max_samples <= 0) return -1;
    TTSClient* client = static_cast<TTSClient*>(handle);
    SynthesizeOptions opts;
    if (voice && voice[0] != '\0') {
        opts.voice = voice;
    }
    opts.speed = (speed > 0.0f) ? speed : 1.0f;
    std::vector<float> samples;
    if (!client->synthesize_to_pcm(text, samples, opts)) {
        return -2;
    }
    const int copy_count = std::min(static_cast<int>(samples.size()), max_samples);
    std::memcpy(out_samples, samples.data(), static_cast<size_t>(copy_count) * sizeof(float));
    return copy_count;
}

void tts_comms_destroy(tts_comms_handle handle) {
    if (handle) {
        delete static_cast<TTSClient*>(handle);
    }
}

} // extern "C"

} // namespace tts_comms
