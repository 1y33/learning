# Distributed File System (HDFS-like) - Implementation Guide

## What is This Project?

Build a distributed file system similar to HDFS (Hadoop Distributed File System) with fault tolerance, replication, and scalability. This project teaches distributed systems concepts, consensus algorithms, and large-scale storage management.

## Why Build This?

- Understand distributed storage systems
- Learn fault tolerance through replication
- Master network programming for distributed systems
- Implement heartbeat and failure detection
- Build the foundation of big data systems

---

## Architecture Overview

```
┌──────────────────────────────────────────────┐
│              Client Applications             │
└────────────────────┬─────────────────────────┘
                     │
┌────────────────────▼─────────────────────────┐
│         NameNode (Metadata Server)           │
│  ┌────────────┐  ┌──────────────────────┐   │
│  │ Namespace  │  │  Block Location Map  │   │
│  └────────────┘  └──────────────────────┘   │
└────────────────────┬─────────────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
┌────────▼──┐  ┌─────▼────┐  ┌──▼────────┐
│ DataNode1 │  │DataNode2 │  │ DataNode3 │
│ [Blk1,2]  │  │[Blk1,3]  │  │ [Blk2,3]  │
└───────────┘  └──────────┘  └───────────┘
     Rack 1       Rack 1         Rack 2
```

---

## Implementation Hints

### 1. NameNode: Metadata Management

**What you need:**
Central server that maintains filesystem namespace and block locations.

**Hint:**
```cpp
struct FileMetadata {
    std::string path;
    uint64_t file_size;
    time_t created_time;
    time_t modified_time;
    uint16_t replication_factor;
    uint64_t block_size;
    std::vector<uint64_t> block_ids;
};

struct BlockInfo {
    uint64_t block_id;
    uint64_t size;
    std::vector<std::string> datanode_locations; // Replicas
    int replication_count;
};

class NameNode {
private:
    // Filesystem namespace (directory tree)
    std::map<std::string, FileMetadata> files;
    std::map<std::string, std::vector<std::string>> directories;

    // Block location mapping
    std::map<uint64_t, BlockInfo> blocks;

    // DataNode status
    std::map<std::string, DataNodeInfo> datanodes;

    std::atomic<uint64_t> next_block_id{0};
    std::mutex namespace_lock;

public:
    CreateFileResponse createFile(const std::string& path, uint16_t replication) {
        std::lock_guard<std::mutex> lock(namespace_lock);

        if (files.count(path)) {
            return {false, "File already exists"};
        }

        FileMetadata metadata;
        metadata.path = path;
        metadata.replication_factor = replication;
        metadata.block_size = 64 * 1024 * 1024; // 64MB blocks

        files[path] = metadata;

        return {true, "File created"};
    }

    AllocateBlocksResponse allocateBlocks(const std::string& path, int num_blocks) {
        std::lock_guard<std::mutex> lock(namespace_lock);

        auto it = files.find(path);
        if (it == files.end()) {
            return {false, {}};
        }

        std::vector<BlockInfo> allocated_blocks;

        for (int i = 0; i < num_blocks; i++) {
            uint64_t block_id = next_block_id++;

            // Select DataNodes for replicas (rack-aware)
            auto datanodes = selectDataNodes(it->second.replication_factor);

            BlockInfo block;
            block.block_id = block_id;
            block.datanode_locations = datanodes;
            block.replication_count = datanodes.size();

            blocks[block_id] = block;
            it->second.block_ids.push_back(block_id);

            allocated_blocks.push_back(block);
        }

        return {true, allocated_blocks};
    }

    std::vector<BlockInfo> getBlockLocations(const std::string& path) {
        std::lock_guard<std::mutex> lock(namespace_lock);

        auto it = files.find(path);
        if (it == files.end()) return {};

        std::vector<BlockInfo> result;
        for (uint64_t block_id : it->second.block_ids) {
            result.push_back(blocks[block_id]);
        }

        return result;
    }

private:
    std::vector<std::string> selectDataNodes(int replication_factor) {
        // Rack-aware placement:
        // 1. First replica: random DataNode
        // 2. Second replica: different rack
        // 3. Third replica: same rack as second

        std::vector<std::string> selected;
        std::set<std::string> used_racks;

        // First replica (random)
        auto candidates = getHealthyDataNodes();
        if (candidates.empty()) return {};

        std::string first = candidates[rand() % candidates.size()];
        selected.push_back(first);
        used_racks.insert(getRack(first));

        // Second replica (different rack if possible)
        if (replication_factor > 1) {
            for (const auto& dn : candidates) {
                if (getRack(dn) != getRack(first)) {
                    selected.push_back(dn);
                    used_racks.insert(getRack(dn));
                    break;
                }
            }
        }

        // Third replica (same rack as second)
        if (replication_factor > 2 && selected.size() > 1) {
            std::string second_rack = getRack(selected[1]);
            for (const auto& dn : candidates) {
                if (getRack(dn) == second_rack &&
                    std::find(selected.begin(), selected.end(), dn) == selected.end()) {
                    selected.push_back(dn);
                    break;
                }
            }
        }

        return selected;
    }

    std::string getRack(const std::string& datanode_id) {
        return datanodes[datanode_id].rack_id;
    }
};
```

**Tips:**
- Persist namespace to disk (fsimage + edit log)
- Implement namespace checkpointing
- Use protobuf for RPC serialization
- Add support for directories, permissions

### 2. DataNode: Block Storage

**What you need:**
Storage server that stores actual file blocks on local disk.

**Hint:**
```cpp
class DataNode {
private:
    std::string node_id;
    std::string storage_dir;
    std::map<uint64_t, std::string> block_paths; // block_id -> file path
    std::string namenode_address;

    std::thread heartbeat_thread;
    std::thread block_report_thread;

public:
    void start() {
        // Send initial block report
        sendBlockReport();

        // Start heartbeat thread
        heartbeat_thread = std::thread([this] {
            while (true) {
                sendHeartbeat();
                std::this_thread::sleep_for(std::chrono::seconds(3));
            }
        });

        // Start block report thread
        block_report_thread = std::thread([this] {
            while (true) {
                std::this_thread::sleep_for(std::chrono::hours(1));
                sendBlockReport();
            }
        });
    }

    bool writeBlock(uint64_t block_id, const std::vector<uint8_t>& data,
                    const std::vector<std::string>& replica_targets) {
        // Write block to local disk
        std::string block_path = storage_dir + "/blk_" + std::to_string(block_id);
        std::ofstream file(block_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();

        block_paths[block_id] = block_path;

        // Pipeline to next replica
        if (!replica_targets.empty()) {
            std::string next_target = replica_targets[0];
            std::vector<std::string> remaining(replica_targets.begin() + 1,
                                               replica_targets.end());

            sendToReplica(next_target, block_id, data, remaining);
        }

        return true;
    }

    std::vector<uint8_t> readBlock(uint64_t block_id) {
        auto it = block_paths.find(block_id);
        if (it == block_paths.end()) {
            return {};
        }

        std::ifstream file(it->second, std::ios::binary);
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
        return data;
    }

private:
    void sendHeartbeat() {
        HeartbeatRequest req;
        req.node_id = node_id;
        req.num_blocks = block_paths.size();
        req.available_space = getAvailableSpace();
        req.num_failed_volumes = 0;

        // Send to NameNode via RPC
        sendRPC(namenode_address, req);
    }

    void sendBlockReport() {
        BlockReportRequest req;
        req.node_id = node_id;

        for (const auto& [block_id, path] : block_paths) {
            struct stat st;
            stat(path.c_str(), &st);

            BlockReportEntry entry;
            entry.block_id = block_id;
            entry.size = st.st_size;
            entry.generation_stamp = st.st_mtime;

            req.blocks.push_back(entry);
        }

        sendRPC(namenode_address, req);
    }

    void sendToReplica(const std::string& target, uint64_t block_id,
                      const std::vector<uint8_t>& data,
                      const std::vector<std::string>& remaining) {
        // Connect to next DataNode in pipeline
        // Forward block data
    }
};
```

**Tips:**
- Use directory-based storage (e.g., `/storage/current/`)
- Implement block verification with checksums
- Handle disk failures gracefully
- Add block scanner to detect corruption

### 3. Client Library

**What you need:**
Client API for reading and writing files.

**Hint:**
```cpp
class DFSClient {
private:
    std::string namenode_address;
    static constexpr uint64_t BLOCK_SIZE = 64 * 1024 * 1024;

public:
    void writeFile(const std::string& path, const std::vector<uint8_t>& data) {
        // 1. Create file on NameNode
        createFile(path);

        // 2. Split data into blocks
        int num_blocks = (data.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // 3. Request block allocations from NameNode
        auto block_info = allocateBlocks(path, num_blocks);

        // 4. Write each block to DataNodes
        for (int i = 0; i < num_blocks; i++) {
            size_t offset = i * BLOCK_SIZE;
            size_t length = std::min(BLOCK_SIZE, data.size() - offset);

            std::vector<uint8_t> block_data(
                data.begin() + offset,
                data.begin() + offset + length
            );

            // Write to first DataNode (which pipelines to replicas)
            writeBlock(block_info[i], block_data);
        }

        // 5. Close file on NameNode
        closeFile(path);
    }

    std::vector<uint8_t> readFile(const std::string& path) {
        // 1. Get block locations from NameNode
        auto blocks = getBlockLocations(path);

        std::vector<uint8_t> result;

        // 2. Read each block from DataNodes
        for (const auto& block : blocks) {
            // Choose closest DataNode (rack-aware)
            std::string datanode = selectDataNode(block.datanode_locations);

            // Read block
            auto block_data = readBlock(datanode, block.block_id);

            // Verify checksum
            if (!verifyChecksum(block_data, block.checksum)) {
                // Try another replica
                datanode = selectDataNode(block.datanode_locations, datanode);
                block_data = readBlock(datanode, block.block_id);
            }

            result.insert(result.end(), block_data.begin(), block_data.end());
        }

        return result;
    }

private:
    void writeBlock(const BlockInfo& block, const std::vector<uint8_t>& data) {
        // Connect to first DataNode
        int sock = connectToDataNode(block.datanode_locations[0]);

        // Send write request with replica pipeline
        WriteBlockRequest req;
        req.block_id = block.block_id;
        req.replicas = std::vector<std::string>(
            block.datanode_locations.begin() + 1,
            block.datanode_locations.end()
        );

        send(sock, &req, sizeof(req), 0);
        send(sock, data.data(), data.size(), 0);

        // Wait for ack
        WriteBlockResponse resp;
        recv(sock, &resp, sizeof(resp), 0);

        close(sock);
    }
};
```

**Tips:**
- Implement client-side caching
- Add retry logic for failures
- Support append operations
- Implement read/write streaming

### 4. Heartbeat and Failure Detection

**What you need:**
NameNode monitors DataNode health and handles failures.

**Hint:**
```cpp
struct DataNodeInfo {
    std::string node_id;
    std::string address;
    std::string rack_id;
    time_t last_heartbeat;
    uint64_t available_space;
    int num_blocks;
    bool is_alive;
};

class HeartbeatManager {
private:
    std::map<std::string, DataNodeInfo> datanodes;
    std::mutex datanodes_mutex;
    std::thread monitor_thread;

public:
    void start() {
        monitor_thread = std::thread([this] {
            while (true) {
                checkDeadNodes();
                std::this_thread::sleep_for(std::chrono::seconds(10));
            }
        });
    }

    void processHeartbeat(const HeartbeatRequest& req) {
        std::lock_guard<std::mutex> lock(datanodes_mutex);

        auto& node = datanodes[req.node_id];
        node.last_heartbeat = time(nullptr);
        node.available_space = req.available_space;
        node.num_blocks = req.num_blocks;
        node.is_alive = true;
    }

private:
    void checkDeadNodes() {
        std::lock_guard<std::mutex> lock(datanodes_mutex);

        time_t now = time(nullptr);
        const int TIMEOUT = 30; // 30 seconds

        for (auto& [node_id, info] : datanodes) {
            if (now - info.last_heartbeat > TIMEOUT && info.is_alive) {
                info.is_alive = false;
                handleNodeFailure(node_id);
            }
        }
    }

    void handleNodeFailure(const std::string& node_id) {
        // 1. Find all blocks on failed node
        // 2. Check if blocks are under-replicated
        // 3. Schedule re-replication on other nodes
        scheduleReplication(node_id);
    }

    void scheduleReplication(const std::string& failed_node) {
        // Find blocks that need re-replication
        // Add to replication queue
        // Background thread picks up and replicates
    }
};
```

**Tips:**
- Use exponential backoff for timeouts
- Implement decommissioning for graceful node removal
- Add node registration protocol
- Monitor network topology changes

### 5. Block Replication

**What you need:**
Maintain replication factor when nodes fail.

**Hint:**
```cpp
class ReplicationManager {
private:
    std::priority_queue<ReplicationTask> replication_queue;
    std::thread replication_thread;

public:
    void start() {
        replication_thread = std::thread([this] {
            while (true) {
                processReplicationQueue();
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
    }

    void scheduleReplication(uint64_t block_id, int current_replicas,
                           int target_replicas) {
        ReplicationTask task;
        task.block_id = block_id;
        task.current_count = current_replicas;
        task.target_count = target_replicas;
        task.priority = calculatePriority(current_replicas, target_replicas);

        replication_queue.push(task);
    }

private:
    void processReplicationQueue() {
        if (replication_queue.empty()) return;

        ReplicationTask task = replication_queue.top();
        replication_queue.pop();

        // Find source DataNode (has the block)
        auto source = findDataNodeWithBlock(task.block_id);

        // Find target DataNode (for new replica)
        auto target = selectDataNodeForReplication(task.block_id);

        // Send replication command to source
        sendReplicationCommand(source, target, task.block_id);
    }

    int calculatePriority(int current, int target) {
        // Higher priority for blocks with fewer replicas
        return target - current;
    }
};
```

---

## Project Structure

```
05_distributed_fs/
├── CMakeLists.txt
├── src/
│   ├── namenode/
│   │   ├── namenode.cpp
│   │   ├── namespace.cpp
│   │   ├── block_manager.cpp
│   │   └── heartbeat_manager.cpp
│   ├── datanode/
│   │   ├── datanode.cpp
│   │   ├── block_storage.cpp
│   │   └── block_scanner.cpp
│   ├── client/
│   │   ├── dfs_client.cpp
│   │   └── input_stream.cpp
│   ├── protocol/
│   │   ├── rpc.cpp
│   │   └── messages.proto
│   └── common/
│       ├── block.cpp
│       └── checksum.cpp
├── tests/
│   ├── test_namenode.cpp
│   ├── test_datanode.cpp
│   └── test_replication.cpp
└── benchmarks/
    └── write_throughput.cpp
```

---

## Resources

- [HDFS Architecture Guide](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html)
- [Google File System Paper](https://static.googleusercontent.com/media/research.google.com/en//archive/gfs-sosp2003.pdf)
- Book: "Designing Data-Intensive Applications" by Martin Kleppmann

Good luck building your distributed file system!
