# Kafka-like Message Queue - Implementation Guide

## What is This Project?

Build a distributed streaming platform similar to Apache Kafka. This project teaches you about distributed systems, consensus algorithms, high-throughput I/O, and fault-tolerant architectures. Kafka is used by thousands of companies for event streaming, log aggregation, and real-time data pipelines.

## Why Build This?

- Learn distributed systems concepts (replication, partitioning, consensus)
- Understand log-based architectures
- Master high-performance I/O and zero-copy techniques
- Implement leader election algorithms (Raft)
- Build something that handles millions of messages per second

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      Producers                          │
└───────────────┬─────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────┐
│                      Broker Cluster                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Broker 1   │  │  Broker 2   │  │  Broker 3   │    │
│  │  (Leader)   │  │  (Follower) │  │  (Follower) │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                 │                 │           │
│  ┌──────▼─────────────────▼─────────────────▼──────┐   │
│  │         Topic: user-events (3 partitions)       │   │
│  │  [Partition 0] [Partition 1] [Partition 2]      │   │
│  └──────────────────────────────────────────────────┘   │
└───────────────┬─────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────┐
│              Consumer Groups (Auto-Rebalancing)         │
│  Group A: [Consumer1] [Consumer2]                       │
│  Group B: [Consumer3]                                   │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Hints

### 1. Topic and Partition Structure

**What you need:**
Topics are divided into partitions for parallelism. Each partition is an ordered, immutable log.

**Hint:**
```cpp
struct Message {
    uint64_t offset;        // Position in partition
    uint64_t timestamp;     // Message timestamp
    std::string key;        // For partitioning
    std::vector<uint8_t> value; // Actual data
    std::vector<uint8_t> headers; // Metadata
};

class Partition {
private:
    std::string topic_name;
    int partition_id;
    std::string log_directory;
    std::vector<Segment> segments; // Multiple log files
    uint64_t next_offset;

public:
    uint64_t append(const Message& msg);
    std::vector<Message> read(uint64_t start_offset, size_t max_bytes);
    void flush();
};

class Topic {
private:
    std::string name;
    std::vector<std::unique_ptr<Partition>> partitions;

public:
    int getPartition(const std::string& key) {
        // Hash-based partitioning
        return std::hash<std::string>{}(key) % partitions.size();
    }
};
```

**Tips:**
- Each partition is a separate directory with segment files
- Segment files named like: `00000000000000000000.log`
- Keep an index file for fast offset lookups
- Rotate segments when they reach size limit (e.g., 1GB)

### 2. Log Segment Storage

**What you need:**
Efficient append-only log files with index for random access.

**Hint:**
```cpp
class Segment {
private:
    std::string log_path;      // e.g., "00000000000000123456.log"
    std::string index_path;    // e.g., "00000000000000123456.index"
    int log_fd;
    uint64_t base_offset;      // First offset in this segment
    uint64_t next_offset;      // Next offset to write

    struct IndexEntry {
        uint32_t relative_offset; // Offset relative to base
        uint32_t position;        // Byte position in log file
    };
    std::vector<IndexEntry> index;

public:
    uint64_t append(const Message& msg) {
        // Serialize message
        // Write to log file
        // Update index every N messages
        // Return offset
    }

    std::vector<Message> read(uint64_t offset, size_t max_bytes);

    void flush() {
        fsync(log_fd); // Ensure durability
    }
};
```

**Tips:**
- Use memory-mapped files (`mmap`) for better performance
- Batch index updates to reduce overhead
- Use direct I/O to bypass OS cache for sequential writes
- Pre-allocate file space with `fallocate()` to reduce fragmentation

### 3. Producer API with Batching

**What you need:**
Clients that send messages efficiently with batching and compression.

**Hint:**
```cpp
class Producer {
private:
    std::string broker_address;
    std::map<std::string, std::vector<Message>> pending_batches;
    std::thread batch_sender_thread;

    struct Config {
        size_t batch_size = 16384;      // 16KB batches
        int linger_ms = 10;             // Wait up to 10ms
        CompressionType compression = SNAPPY;
        int acks = 1;                   // 0=none, 1=leader, -1=all
    } config;

public:
    void send(const std::string& topic, const Message& msg) {
        pending_batches[topic].push_back(msg);

        if (shouldFlush(topic)) {
            flush(topic);
        }
    }

private:
    bool shouldFlush(const std::string& topic) {
        auto& batch = pending_batches[topic];
        size_t batch_bytes = calculateSize(batch);
        return batch_bytes >= config.batch_size;
    }

    void flush(const std::string& topic) {
        auto& batch = pending_batches[topic];
        auto compressed = compress(batch, config.compression);
        sendToBroker(topic, compressed);
        batch.clear();
    }
};
```

**Tips:**
- Use a background thread that flushes based on time
- Implement retries with exponential backoff
- Support idempotent producers (deduplication via sequence numbers)
- Add metrics: messages sent, batch sizes, latency

### 4. Consumer Groups with Auto-Rebalancing

**What you need:**
Multiple consumers that coordinate to split partition assignments. When consumers join/leave, rebalance partitions.

**Hint:**
```cpp
class ConsumerGroup {
private:
    std::string group_id;
    std::vector<std::string> members; // Consumer IDs
    std::map<std::string, std::vector<int>> assignments; // consumer -> partitions

public:
    void rebalance(const std::vector<std::string>& topics) {
        // Rebalancing strategies:
        // 1. Range: Assign contiguous partition ranges
        // 2. RoundRobin: Distribute evenly across consumers
        // 3. Sticky: Minimize partition movement

        assignments.clear();
        int consumer_idx = 0;

        for (const auto& topic : topics) {
            int num_partitions = getPartitionCount(topic);
            for (int p = 0; p < num_partitions; p++) {
                std::string consumer = members[consumer_idx];
                assignments[consumer].push_back(p);
                consumer_idx = (consumer_idx + 1) % members.size();
            }
        }
    }
};

class Consumer {
private:
    std::string group_id;
    std::string consumer_id;
    std::map<int, uint64_t> partition_offsets; // Track progress

public:
    void subscribe(const std::vector<std::string>& topics);

    std::vector<Message> poll(int timeout_ms) {
        // 1. Send heartbeat to coordinator
        // 2. Check for rebalance
        // 3. Fetch from assigned partitions
        // 4. Return messages
    }

    void commit() {
        // Save offsets to broker
        // Stored in special __consumer_offsets topic
    }
};
```

**Tips:**
- Use Raft consensus to elect a group coordinator
- Implement heartbeat protocol (consumers send periodic heartbeats)
- On timeout, remove consumer and trigger rebalance
- Store committed offsets in an internal topic

### 5. Raft Consensus for Leader Election

**What you need:**
When a broker goes down, elect a new leader for its partitions.

**Hint:**
```cpp
class RaftNode {
private:
    enum State { FOLLOWER, CANDIDATE, LEADER };
    State state = FOLLOWER;
    int current_term = 0;
    std::string voted_for;
    std::vector<LogEntry> log;
    std::chrono::time_point<std::chrono::steady_clock> last_heartbeat;

public:
    void startElection() {
        state = CANDIDATE;
        current_term++;
        voted_for = node_id;
        int votes = 1;

        // Request votes from all peers
        for (auto& peer : peers) {
            if (peer.requestVote(current_term, node_id)) {
                votes++;
            }
        }

        if (votes > peers.size() / 2) {
            becomeLeader();
        }
    }

    void becomeLeader() {
        state = LEADER;
        // Send heartbeats to all followers
        startHeartbeatTimer();
    }

    void appendEntries(const std::vector<LogEntry>& entries) {
        if (state == LEADER) {
            log.insert(log.end(), entries.begin(), entries.end());
            // Replicate to followers
            replicateToFollowers(entries);
        }
    }
};
```

**Tips:**
- Use randomized election timeouts to avoid split votes
- Leader sends periodic heartbeats (empty AppendEntries RPCs)
- Implement log replication with consistency checks
- Test with network partitions and failures

### 6. Zero-Copy Message Delivery

**What you need:**
Transfer data from disk to network without copying to userspace.

**Hint:**
```cpp
class ZeroCopyReader {
public:
    ssize_t sendfile(int out_socket, int in_fd, off_t* offset, size_t count) {
        // Linux: Use sendfile() system call
        return ::sendfile(out_socket, in_fd, offset, count);

        // Alternative: splice() for pipe-based transfer
        // FreeBSD: Use sendfile() with different API
    }

    void sendMessages(int socket_fd, const std::string& log_path,
                      uint64_t offset, size_t max_bytes) {
        int log_fd = open(log_path.c_str(), O_RDONLY);
        off_t file_offset = calculateFileOffset(offset);

        ssize_t sent = sendfile(socket_fd, log_fd, &file_offset, max_bytes);

        close(log_fd);
    }
};
```

**Tips:**
- Use `sendfile()` on Linux/BSD for zero-copy
- On macOS, use `sendfile()` with different signature
- Fallback to regular `read()`/`write()` if unavailable
- Benchmark difference: zero-copy can be 2-3x faster

### 7. Replication and ISR (In-Sync Replicas)

**What you need:**
Each partition has a leader and N followers. Track which replicas are in sync.

**Hint:**
```cpp
class PartitionReplicaManager {
private:
    int leader_id;
    std::set<int> in_sync_replicas; // ISR
    std::map<int, uint64_t> follower_offsets; // Track replication lag

public:
    bool isInSync(int replica_id) {
        uint64_t leader_offset = getLeaderOffset();
        uint64_t follower_offset = follower_offsets[replica_id];

        // In-sync if within threshold
        return (leader_offset - follower_offset) < 100;
    }

    void replicateToFollowers(const Message& msg) {
        for (int follower : in_sync_replicas) {
            sendReplicationRequest(follower, msg);
        }

        // Wait for acks based on min.insync.replicas config
        if (config.min_insync_replicas == 2) {
            waitForAcks(1); // Leader + 1 follower
        }
    }

    void updateFollowerOffset(int replica_id, uint64_t offset) {
        follower_offsets[replica_id] = offset;

        if (!isInSync(replica_id)) {
            in_sync_replicas.erase(replica_id);
        } else {
            in_sync_replicas.insert(replica_id);
        }
    }
};
```

**Tips:**
- Remove replicas from ISR if they fall behind
- Clients can specify acks: 0 (no ack), 1 (leader), -1 (all ISR)
- Handle follower failure by shrinking ISR
- On leader failure, elect new leader from ISR

---

## Project Structure

```
02_message_queue/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── broker/
│   │   ├── broker_server.cpp
│   │   ├── topic_manager.cpp
│   │   └── partition_manager.cpp
│   ├── storage/
│   │   ├── segment.cpp
│   │   ├── log_manager.cpp
│   │   └── index.cpp
│   ├── consensus/
│   │   ├── raft_node.cpp
│   │   └── leader_election.cpp
│   ├── client/
│   │   ├── producer.cpp
│   │   ├── consumer.cpp
│   │   └── consumer_group.cpp
│   ├── protocol/
│   │   ├── message_format.cpp
│   │   └── wire_protocol.cpp
│   └── network/
│       ├── tcp_server.cpp
│       └── zero_copy.cpp
├── include/
│   └── message_queue/
│       ├── broker.h
│       ├── storage.h
│       ├── consensus.h
│       └── client.h
├── tests/
│   ├── test_storage.cpp
│   ├── test_replication.cpp
│   ├── test_consensus.cpp
│   └── test_consumer_group.cpp
└── benchmarks/
    └── throughput_test.cpp
```

---

## Testing Strategy

1. **Unit Tests**:
   - Segment append and read
   - Index lookup accuracy
   - Partition assignment algorithms
   - Raft state transitions

2. **Integration Tests**:
   - Producer sends, consumer receives
   - Rebalancing on consumer join/leave
   - Leader election on broker failure
   - Message ordering within partitions

3. **Chaos Tests**:
   - Kill random brokers during operation
   - Network partitions between brokers
   - Slow/delayed message delivery
   - Disk I/O failures

4. **Performance Tests**:
   - Throughput: messages/sec
   - Latency: end-to-end publish to consume
   - Compare with Kafka benchmarks

---

## Performance Goals

- **Throughput**: 100K+ messages/sec per broker
- **Latency**: < 10ms end-to-end (p99)
- **Durability**: fsync every batch, no data loss
- **Scalability**: Linear scaling with partitions

---

## Advanced Features (Optional)

1. **Exactly-Once Semantics**: Idempotent producers + transactional writes
2. **Compacted Topics**: Keep only latest value per key
3. **Stream Processing**: Built-in windowing and aggregations
4. **Schema Registry**: Store and version message schemas
5. **Multi-datacenter Replication**: Cross-region mirroring

---

## Resources

- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [Raft Paper](https://raft.github.io/raft.pdf)
- [Log-Structured Storage](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying)
- Book: "Kafka: The Definitive Guide"
- [Kafka Protocol Specification](https://kafka.apache.org/protocol)

---

## Debugging Tips

- Use Wireshark to inspect protocol messages
- Log all Raft state transitions
- Monitor disk I/O with `iostat`
- Profile with `perf` to find bottlenecks
- Test with different replication factors and ISR configs

Good luck building your message queue system!
