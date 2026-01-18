# Redis-like In-Memory Cache - Implementation Guide

## What is This Project?

This project involves building a high-performance, in-memory key-value store similar to Redis. Redis is one of the most popular data structure servers used for caching, session storage, pub/sub messaging, and real-time analytics. You'll learn about data structures, concurrency, networking, and persistence mechanisms.

## Why Build This?

- Understand how modern caching systems work
- Master concurrent programming with threads and locks
- Learn network programming and protocol design
- Practice memory management and optimization
- Build something that can be benchmarked against production systems

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│         Python Client (pybind11)        │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│         TCP Server (RESP Protocol)      │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│      Command Parser & Dispatcher        │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│     Data Store (Hash Tables + Types)    │
│  - Strings  - Lists  - Sets  - Hashes   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│    Persistence Layer (RDB + AOF)        │
└─────────────────────────────────────────┘
```

---

## Implementation Hints

### 1. Core Data Structure: Hash Table

**What you need:**
A fast hash table that can store different value types.

**Hint:**
```cpp
// Use a variant or union for different types
struct Value {
    enum Type { STRING, LIST, SET, HASH, ZSET };
    Type type;
    void* data;
    time_t expire_time; // For TTL support
};

class DataStore {
private:
    std::unordered_map<std::string, Value> store;
    std::shared_mutex mutex; // Read-write lock

public:
    bool set(const std::string& key, const std::string& value);
    std::optional<std::string> get(const std::string& key);
};
```

**Tips:**
- Use `std::shared_mutex` for multiple readers, single writer
- For TTL, maintain a separate min-heap or time-wheel of expiring keys
- Consider using a custom hash function for better distribution

### 2. TCP Server with RESP Protocol

**What you need:**
A server that accepts TCP connections and parses Redis RESP (REdis Serialization Protocol).

**RESP Protocol Example:**
```
Client sends: *3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n
Means: ["SET", "key", "value"]

Server responds: +OK\r\n
```

**Hint:**
```cpp
class RESPParser {
public:
    enum Type { SIMPLE_STRING, ERROR, INTEGER, BULK_STRING, ARRAY };

    struct Message {
        Type type;
        std::vector<std::string> elements;
    };

    std::optional<Message> parse(const std::string& buffer);
};

class TCPServer {
private:
    int server_fd;
    std::vector<std::thread> worker_threads;

    void handleClient(int client_fd);

public:
    void start(int port, int num_threads);
};
```

**Tips:**
- Use `epoll` (Linux) or `kqueue` (BSD/Mac) for efficient I/O multiplexing
- Implement a thread pool to handle multiple clients
- Parse RESP incrementally to handle partial reads
- Buffer incomplete messages until full command arrives

### 3. LRU Cache Eviction Policy

**What you need:**
When memory limit is reached, evict least recently used keys.

**Hint:**
```cpp
class LRUCache {
private:
    size_t max_memory;
    size_t current_memory;

    struct Node {
        std::string key;
        Node* prev;
        Node* next;
    };

    Node* head; // Most recently used
    Node* tail; // Least recently used
    std::unordered_map<std::string, Node*> cache_map;

public:
    void touch(const std::string& key); // Move to front
    std::string evict(); // Remove from tail
};
```

**Tips:**
- Use a doubly-linked list + hash map for O(1) operations
- Track approximate memory usage by summing key and value sizes
- Consider implementing multiple eviction policies (LRU, LFU, Random)

### 4. Persistence: RDB Snapshots

**What you need:**
Periodically save the entire dataset to disk in a compact binary format.

**Hint:**
```cpp
class RDBPersistence {
public:
    void save(const std::string& filename, const DataStore& store) {
        // Binary format:
        // [MAGIC][VERSION][DATA_SIZE][KEY_COUNT]
        // For each key: [TYPE][KEY_LEN][KEY][VALUE_LEN][VALUE]

        std::ofstream file(filename, std::ios::binary);
        // Write header
        file.write("REDIS", 5);
        // Iterate and serialize each key-value pair
    }

    void load(const std::string& filename, DataStore& store);
};
```

**Tips:**
- Use fork() to create a child process for saving (copy-on-write)
- Compress data with LZ4 or Snappy
- Add checksums (CRC64) for corruption detection
- Schedule saves based on time or number of writes

### 5. Persistence: Append-Only File (AOF)

**What you need:**
Log every write operation to a file for durability.

**Hint:**
```cpp
class AOFLogger {
private:
    std::ofstream aof_file;
    std::mutex write_mutex;

public:
    void log(const std::string& command) {
        std::lock_guard<std::mutex> lock(write_mutex);
        aof_file << command << "\n";
        aof_file.flush(); // Or fsync() for durability
    }

    void rewrite() {
        // Create new AOF from current state
        // Replace old AOF atomically
    }
};
```

**Tips:**
- Use buffered writes and fsync() periodically
- Implement AOF rewriting to compact the log
- On startup, replay AOF to restore state
- Consider different fsync policies (always, every second, never)

### 6. Python Bindings with pybind11

**What you need:**
Expose your C++ cache to Python users.

**Hint:**
```cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(rediscache, m) {
    py::class_<DataStore>(m, "RedisCache")
        .def(py::init<>())
        .def("set", &DataStore::set)
        .def("get", &DataStore::get)
        .def("delete", &DataStore::del)
        .def("exists", &DataStore::exists);
}
```

**Python usage:**
```python
import rediscache

cache = rediscache.RedisCache()
cache.set("user:1", "John Doe")
value = cache.get("user:1")
```

**Tips:**
- Install pybind11: `pip install pybind11`
- Compile with: `c++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) bindings.cpp -o rediscache$(python3-config --extension-suffix)`
- Handle Python GIL for thread safety
- Return `py::none()` for null values

### 7. Master-Slave Replication

**What you need:**
Replicas that sync with a master for high availability.

**Hint:**
```cpp
class ReplicationMaster {
private:
    std::vector<int> replica_fds;

public:
    void propagateWrite(const std::string& command) {
        for (int fd : replica_fds) {
            send(fd, command.c_str(), command.size(), 0);
        }
    }
};

class ReplicationSlave {
public:
    void syncWithMaster(const std::string& master_host, int port) {
        // 1. Send PSYNC command
        // 2. Receive RDB snapshot
        // 3. Apply incremental updates
    }
};
```

**Tips:**
- Use a replication buffer for partial resync
- Track replication offset on both master and slave
- Handle network failures gracefully
- Support chained replication (slave of slave)

---

## Project Structure

```
01_redis_cache/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── server/
│   │   ├── tcp_server.cpp
│   │   └── resp_parser.cpp
│   ├── storage/
│   │   ├── data_store.cpp
│   │   ├── value_types.cpp
│   │   └── lru_cache.cpp
│   ├── persistence/
│   │   ├── rdb.cpp
│   │   └── aof.cpp
│   ├── replication/
│   │   ├── master.cpp
│   │   └── slave.cpp
│   └── bindings/
│       └── python_bindings.cpp
├── include/
│   └── redis_cache/
│       ├── server.h
│       ├── storage.h
│       └── persistence.h
├── tests/
│   ├── test_storage.cpp
│   ├── test_server.cpp
│   └── test_persistence.cpp
└── benchmarks/
    └── benchmark_throughput.cpp
```

---

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
   - Hash table operations
   - RESP parser correctness
   - LRU eviction policy
   - Persistence load/save

2. **Integration Tests**: Test end-to-end workflows
   - Client connects, sends commands, receives responses
   - Persistence: Save, restart, load
   - Replication: Master writes, slave reads

3. **Performance Tests**: Compare with Redis
   - Throughput (ops/sec)
   - Latency (p50, p99, p999)
   - Memory efficiency
   - Use `redis-benchmark` tool

---

## Benchmarking

```bash
# Benchmark SET operations
redis-benchmark -t set -n 100000 -q

# Your cache:
./benchmark --operation set --count 100000

# Compare results:
# Redis: ~80,000 ops/sec
# Your implementation: Target ~50,000+ ops/sec
```

---

## Advanced Features (Optional)

1. **Pub/Sub**: Implement publish/subscribe messaging
2. **Lua Scripting**: Embed Lua for atomic operations
3. **Cluster Mode**: Sharding across multiple nodes
4. **Streams**: Append-only log data structure
5. **Modules API**: Plugin system for extensions

---

## Resources

- [Redis Protocol Specification](https://redis.io/docs/reference/protocol-spec/)
- [Redis Internals](https://redis.io/topics/internals)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- Book: "Redis 4.x Cookbook"
- [Antirez's Blog](http://antirez.com/) (Redis creator)

---

## Performance Goals

- **Throughput**: 50K+ operations/sec (single-threaded)
- **Latency**: < 1ms for simple GET/SET
- **Memory**: Efficient as std::unordered_map + overhead < 20%
- **Persistence**: RDB save < 100ms for 10K keys

Good luck building your cache!
