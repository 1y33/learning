# HTTP Server with Thread Pool and WebSocket - Implementation Guide

## What is This Project?

Build a production-ready HTTP/1.1 web server from scratch with WebSocket support, similar to nginx or Node.js's HTTP server. This project teaches network programming, protocol implementation, concurrent request handling, and real-time communication.

## Why Build This?

- Master TCP/IP socket programming
- Understand HTTP protocol internals
- Learn thread pool patterns for concurrency
- Implement WebSocket for bidirectional communication
- Build something that can serve real web applications

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│                   HTTP Clients                         │
└─────────────────────┬──────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────┐
│              TCP Listener (epoll/kqueue)               │
└─────────────────────┬──────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────┐
│              Thread Pool (Worker Threads)              │
└─────────────────────┬──────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────┐
│         HTTP Parser & Request Router                   │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  Middleware  │→ │   Handler    │→ │  Response   │  │
│  │   Pipeline   │  │   Functions  │  │  Generator  │  │
│  └──────────────┘  └──────────────┘  └─────────────┘  │
└────────────────────────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
┌─────────▼─────────┐  ┌─────────▼──────────┐
│  Static File      │  │  WebSocket         │
│  Handler          │  │  Connection Pool   │
└───────────────────┘  └────────────────────┘
```

---

## Implementation Hints

### 1. TCP Server with epoll

**What you need:**
Non-blocking I/O multiplexing to handle thousands of concurrent connections.

**Hint:**
```cpp
class TCPServer {
private:
    int listen_fd;
    int epoll_fd;
    std::vector<struct epoll_event> events;

public:
    void start(int port, int max_connections = 10000) {
        listen_fd = socket(AF_INET, SOCK_STREAM, 0);

        // Make socket non-blocking
        int flags = fcntl(listen_fd, F_GETFL, 0);
        fcntl(listen_fd, F_SETFL, flags | O_NONBLOCK);

        // Bind and listen
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = INADDR_ANY;

        bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr));
        listen(listen_fd, 128);

        // Create epoll instance
        epoll_fd = epoll_create1(0);

        struct epoll_event ev;
        ev.events = EPOLLIN | EPOLLET; // Edge-triggered
        ev.data.fd = listen_fd;
        epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listen_fd, &ev);

        // Event loop
        events.resize(max_connections);
        while (true) {
            int nfds = epoll_wait(epoll_fd, events.data(), max_connections, -1);
            for (int i = 0; i < nfds; i++) {
                handleEvent(events[i]);
            }
        }
    }

    void handleEvent(const struct epoll_event& event) {
        if (event.data.fd == listen_fd) {
            acceptConnection();
        } else {
            readRequest(event.data.fd);
        }
    }
};
```

**Tips:**
- Use `epoll` on Linux, `kqueue` on BSD/macOS
- Edge-triggered mode for better performance
- Set `SO_REUSEADDR` and `SO_REUSEPORT` socket options
- Handle `EAGAIN`/`EWOULDBLOCK` for non-blocking I/O

### 2. HTTP/1.1 Parser

**What you need:**
Parse HTTP requests according to RFC 7230.

**Hint:**
```cpp
class HTTPRequest {
public:
    std::string method;
    std::string path;
    std::string version;
    std::map<std::string, std::string> headers;
    std::string body;

    struct QueryParams {
        std::map<std::string, std::string> params;
    };
    QueryParams query;
};

class HTTPParser {
private:
    enum State {
        REQUEST_LINE,
        HEADERS,
        BODY,
        COMPLETE
    };
    State state = REQUEST_LINE;
    std::string buffer;

public:
    std::optional<HTTPRequest> parse(const std::string& data) {
        buffer += data;

        HTTPRequest req;

        // Parse request line: GET /path HTTP/1.1
        if (state == REQUEST_LINE) {
            auto line_end = buffer.find("\r\n");
            if (line_end == std::string::npos) return std::nullopt;

            std::string request_line = buffer.substr(0, line_end);
            parseRequestLine(request_line, req);

            buffer = buffer.substr(line_end + 2);
            state = HEADERS;
        }

        // Parse headers
        if (state == HEADERS) {
            while (true) {
                auto line_end = buffer.find("\r\n");
                if (line_end == std::string::npos) return std::nullopt;

                std::string line = buffer.substr(0, line_end);
                if (line.empty()) {
                    // Empty line marks end of headers
                    buffer = buffer.substr(line_end + 2);
                    state = BODY;
                    break;
                }

                parseHeader(line, req);
                buffer = buffer.substr(line_end + 2);
            }
        }

        // Parse body (if Content-Length present)
        if (state == BODY) {
            auto it = req.headers.find("Content-Length");
            if (it != req.headers.end()) {
                size_t content_length = std::stoul(it->second);
                if (buffer.size() < content_length) return std::nullopt;

                req.body = buffer.substr(0, content_length);
                buffer = buffer.substr(content_length);
            }
            state = COMPLETE;
        }

        return req;
    }

private:
    void parseRequestLine(const std::string& line, HTTPRequest& req) {
        std::istringstream ss(line);
        ss >> req.method >> req.path >> req.version;

        // Parse query string
        auto query_pos = req.path.find('?');
        if (query_pos != std::string::npos) {
            std::string query_string = req.path.substr(query_pos + 1);
            req.path = req.path.substr(0, query_pos);
            parseQueryString(query_string, req.query);
        }
    }

    void parseHeader(const std::string& line, HTTPRequest& req) {
        auto colon_pos = line.find(':');
        std::string key = line.substr(0, colon_pos);
        std::string value = line.substr(colon_pos + 2); // Skip ": "
        req.headers[key] = value;
    }
};
```

**Tips:**
- Handle chunked transfer encoding (`Transfer-Encoding: chunked`)
- Support pipelining (multiple requests on same connection)
- Validate request line and headers
- Set reasonable limits (max header size, body size)
- Handle URL decoding (`%20` → space)

### 3. Thread Pool for Request Handling

**What you need:**
Pool of worker threads to process requests concurrently.

**Hint:**
```cpp
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;

public:
    ThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; i++) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });

                        if (stop && tasks.empty()) return;

                        task = std::move(tasks.front());
                        tasks.pop();
                    }

                    task();
                }
            });
        }
    }

    void enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.push(std::move(task));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }
};
```

**Tips:**
- Size thread pool based on CPU cores (e.g., 2x cores)
- Use work-stealing queues for better load balancing
- Monitor queue depth to detect overload
- Add thread-local caches for frequently used data

### 4. Routing System

**What you need:**
Map URL paths to handler functions, support path parameters.

**Hint:**
```cpp
class Router {
private:
    struct Route {
        std::string method;
        std::regex path_pattern;
        std::function<HTTPResponse(const HTTPRequest&)> handler;
    };
    std::vector<Route> routes;

public:
    void addRoute(const std::string& method, const std::string& path,
                  std::function<HTTPResponse(const HTTPRequest&)> handler) {
        // Convert path like "/users/:id" to regex
        std::string pattern = convertToRegex(path);
        routes.push_back({method, std::regex(pattern), handler});
    }

    HTTPResponse route(const HTTPRequest& req) {
        for (const auto& route : routes) {
            if (route.method == req.method) {
                std::smatch match;
                if (std::regex_match(req.path, match, route.path_pattern)) {
                    // Extract path parameters
                    // req.params["id"] = match[1];
                    return route.handler(req);
                }
            }
        }

        return HTTPResponse{404, "Not Found"};
    }

private:
    std::string convertToRegex(const std::string& path) {
        // Convert "/users/:id" to "/users/([^/]+)"
        std::string pattern = path;
        std::regex param_regex(":([a-zA-Z_][a-zA-Z0-9_]*)");
        pattern = std::regex_replace(pattern, param_regex, "([^/]+)");
        return "^" + pattern + "$";
    }
};

// Usage:
// router.addRoute("GET", "/users/:id", [](const HTTPRequest& req) {
//     return HTTPResponse{200, "User ID: " + req.params["id"]};
// });
```

**Tips:**
- Use trie-based routing for better performance
- Support wildcards: `/files/*`
- Method-specific routes: GET, POST, PUT, DELETE
- Add middleware support (before/after handlers)

### 5. WebSocket Protocol Implementation

**What you need:**
Upgrade HTTP connections to WebSocket for bidirectional communication.

**Hint:**
```cpp
class WebSocket {
private:
    int socket_fd;
    bool is_closed = false;

    struct Frame {
        bool fin;
        uint8_t opcode;
        bool masked;
        uint64_t payload_length;
        std::vector<uint8_t> payload;
    };

public:
    static bool isWebSocketUpgrade(const HTTPRequest& req) {
        auto upgrade = req.headers.find("Upgrade");
        auto connection = req.headers.find("Connection");

        return upgrade != req.headers.end() &&
               upgrade->second == "websocket" &&
               connection != req.headers.end() &&
               connection->second.find("Upgrade") != std::string::npos;
    }

    static HTTPResponse createHandshake(const HTTPRequest& req) {
        // Get Sec-WebSocket-Key header
        auto key_it = req.headers.find("Sec-WebSocket-Key");
        std::string key = key_it->second;

        // Compute accept key: SHA1(key + magic string)
        std::string magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
        std::string accept_key = base64(sha1(key + magic));

        HTTPResponse response;
        response.status_code = 101;
        response.status_text = "Switching Protocols";
        response.headers["Upgrade"] = "websocket";
        response.headers["Connection"] = "Upgrade";
        response.headers["Sec-WebSocket-Accept"] = accept_key;

        return response;
    }

    Frame readFrame() {
        Frame frame;
        uint8_t byte1, byte2;

        read(socket_fd, &byte1, 1);
        frame.fin = byte1 & 0x80;
        frame.opcode = byte1 & 0x0F;

        read(socket_fd, &byte2, 1);
        frame.masked = byte2 & 0x80;
        frame.payload_length = byte2 & 0x7F;

        // Handle extended payload length
        if (frame.payload_length == 126) {
            uint16_t len;
            read(socket_fd, &len, 2);
            frame.payload_length = ntohs(len);
        } else if (frame.payload_length == 127) {
            uint64_t len;
            read(socket_fd, &len, 8);
            frame.payload_length = be64toh(len);
        }

        // Read masking key if present
        uint8_t mask[4];
        if (frame.masked) {
            read(socket_fd, mask, 4);
        }

        // Read payload
        frame.payload.resize(frame.payload_length);
        read(socket_fd, frame.payload.data(), frame.payload_length);

        // Unmask payload
        if (frame.masked) {
            for (size_t i = 0; i < frame.payload.size(); i++) {
                frame.payload[i] ^= mask[i % 4];
            }
        }

        return frame;
    }

    void sendFrame(const std::vector<uint8_t>& data, uint8_t opcode = 0x01) {
        std::vector<uint8_t> frame;

        // Byte 1: FIN + opcode
        frame.push_back(0x80 | opcode);

        // Byte 2: Mask bit + payload length
        if (data.size() < 126) {
            frame.push_back(data.size());
        } else if (data.size() < 65536) {
            frame.push_back(126);
            uint16_t len = htons(data.size());
            frame.push_back(len >> 8);
            frame.push_back(len & 0xFF);
        } else {
            frame.push_back(127);
            uint64_t len = htobe64(data.size());
            for (int i = 7; i >= 0; i--) {
                frame.push_back((len >> (i * 8)) & 0xFF);
            }
        }

        // Payload
        frame.insert(frame.end(), data.begin(), data.end());

        write(socket_fd, frame.data(), frame.size());
    }
};
```

**Tips:**
- Handle ping/pong frames for keep-alive
- Support text and binary frames
- Implement fragmentation for large messages
- Add per-message compression (permessage-deflate extension)
- Use a separate thread pool for WebSocket messages

### 6. Static File Serving with Caching

**What you need:**
Efficiently serve files from disk with proper caching headers.

**Hint:**
```cpp
class StaticFileHandler {
private:
    std::string root_directory;
    std::map<std::string, std::string> mime_types = {
        {".html", "text/html"},
        {".css", "text/css"},
        {".js", "application/javascript"},
        {".json", "application/json"},
        {".png", "image/png"},
        {".jpg", "image/jpeg"}
    };

public:
    HTTPResponse serve(const std::string& path) {
        std::string file_path = root_directory + path;

        // Security: prevent directory traversal
        if (path.find("..") != std::string::npos) {
            return HTTPResponse{403, "Forbidden"};
        }

        // Check if file exists
        struct stat file_stat;
        if (stat(file_path.c_str(), &file_stat) != 0) {
            return HTTPResponse{404, "Not Found"};
        }

        // Read file
        std::ifstream file(file_path, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        // Get MIME type
        std::string extension = getExtension(path);
        std::string content_type = mime_types[extension];

        // Compute ETag (MD5 of content)
        std::string etag = computeMD5(content);

        HTTPResponse response;
        response.status_code = 200;
        response.headers["Content-Type"] = content_type;
        response.headers["ETag"] = etag;
        response.headers["Cache-Control"] = "public, max-age=3600";
        response.body = content;

        return response;
    }
};
```

**Tips:**
- Use sendfile() for zero-copy file transmission
- Implement conditional requests (If-None-Match, If-Modified-Since)
- Add gzip compression for text files
- Cache frequently accessed files in memory
- Support byte-range requests for video streaming

### 7. SSL/TLS Support with OpenSSL

**What you need:**
HTTPS support for secure connections.

**Hint:**
```cpp
class SSLServer {
private:
    SSL_CTX* ssl_ctx;

public:
    void initSSL(const std::string& cert_file, const std::string& key_file) {
        SSL_library_init();
        OpenSSL_add_all_algorithms();
        SSL_load_error_strings();

        ssl_ctx = SSL_CTX_new(TLS_server_method());

        SSL_CTX_use_certificate_file(ssl_ctx, cert_file.c_str(), SSL_FILETYPE_PEM);
        SSL_CTX_use_PrivateKey_file(ssl_ctx, key_file.c_str(), SSL_FILETYPE_PEM);

        if (!SSL_CTX_check_private_key(ssl_ctx)) {
            throw std::runtime_error("Private key does not match certificate");
        }
    }

    SSL* acceptSSL(int client_fd) {
        SSL* ssl = SSL_new(ssl_ctx);
        SSL_set_fd(ssl, client_fd);

        if (SSL_accept(ssl) <= 0) {
            ERR_print_errors_fp(stderr);
            return nullptr;
        }

        return ssl;
    }

    ssize_t readSSL(SSL* ssl, void* buf, size_t count) {
        return SSL_read(ssl, buf, count);
    }

    ssize_t writeSSL(SSL* ssl, const void* buf, size_t count) {
        return SSL_write(ssl, buf, count);
    }
};
```

**Tips:**
- Generate self-signed cert for testing: `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes`
- Use Let's Encrypt for production certificates
- Enable HTTP/2 with ALPN negotiation
- Configure cipher suites for security

---

## Project Structure

```
03_http_server/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── server/
│   │   ├── tcp_server.cpp
│   │   ├── http_parser.cpp
│   │   ├── http_response.cpp
│   │   └── thread_pool.cpp
│   ├── router/
│   │   ├── router.cpp
│   │   └── middleware.cpp
│   ├── websocket/
│   │   ├── websocket.cpp
│   │   └── frame_parser.cpp
│   ├── handlers/
│   │   ├── static_file_handler.cpp
│   │   └── api_handlers.cpp
│   └── ssl/
│       └── ssl_server.cpp
├── include/
│   └── http_server/
│       ├── server.h
│       ├── router.h
│       └── websocket.h
├── tests/
│   ├── test_parser.cpp
│   ├── test_router.cpp
│   └── test_websocket.cpp
├── benchmarks/
│   └── load_test.cpp
└── public/
    ├── index.html
    └── static/
        ├── css/
        └── js/
```

---

## Testing Strategy

1. **Unit Tests**: HTTP parser, routing, WebSocket frames
2. **Integration Tests**: Full request/response cycle
3. **Load Testing**: Use `wrk` or `ab` for benchmarks
4. **WebSocket Testing**: Multiple concurrent connections

```bash
# Load test with wrk
wrk -t12 -c400 -d30s http://localhost:8080/

# Apache Bench
ab -n 10000 -c 100 http://localhost:8080/
```

---

## Performance Goals

- **Throughput**: 50K+ requests/sec for static files
- **Latency**: < 1ms for cached responses
- **Concurrent Connections**: 10K+ simultaneous connections
- **WebSocket**: 1K+ concurrent WebSocket connections

---

## Resources

- [RFC 7230: HTTP/1.1 Message Syntax](https://tools.ietf.org/html/rfc7230)
- [RFC 6455: WebSocket Protocol](https://tools.ietf.org/html/rfc6455)
- [epoll tutorial](https://man7.org/linux/man-pages/man7/epoll.7.html)
- Book: "Unix Network Programming" by W. Richard Stevens

Good luck building your HTTP server!
