# Database Engine with B+ Tree Indexing - Implementation Guide

## What is This Project?

Build a relational database management system (RDBMS) from scratch with ACID transactions, SQL query support, and disk-based storage. This is one of the most comprehensive systems programming projects, teaching you about data structures, concurrency control, query optimization, and recovery mechanisms.

## Why Build This?

- Understand how databases work internally
- Master complex data structures (B+ trees)
- Learn transaction processing and ACID guarantees
- Implement query parsing and optimization
- Build the foundation of modern data systems

---

## Architecture Overview

```
┌───────────────────────────────────────────┐
│         SQL Client Interface              │
└─────────────────┬─────────────────────────┘
                  │
┌─────────────────▼─────────────────────────┐
│        SQL Parser & Analyzer              │
│  (Lexer → Parser → AST → Semantic Check)  │
└─────────────────┬─────────────────────────┘
                  │
┌─────────────────▼─────────────────────────┐
│         Query Optimizer                   │
│  (Cost Model → Plan Selection)            │
└─────────────────┬─────────────────────────┘
                  │
┌─────────────────▼─────────────────────────┐
│         Query Executor                    │
│  (Scan → Join → Aggregate → Project)      │
└─────────────────┬─────────────────────────┘
                  │
┌─────────────────▼─────────────────────────┐
│      Transaction Manager (MVCC)           │
│  ┌────────────┐  ┌──────────────────┐    │
│  │ Lock Mgr   │  │ Timestamp Oracle │    │
│  └────────────┘  └──────────────────┘    │
└─────────────────┬─────────────────────────┘
                  │
┌─────────────────▼─────────────────────────┐
│         Buffer Pool Manager               │
│  (LRU Eviction + Page Pinning)            │
└─────────────────┬─────────────────────────┘
                  │
┌─────────────────▼─────────────────────────┐
│       Disk Manager & Storage              │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │  Pages   │  │ B+ Trees │  │  WAL   │  │
│  └──────────┘  └──────────┘  └────────┘  │
└───────────────────────────────────────────┘
```

---

## Implementation Hints

### 1. Page-Based Storage Manager

**What you need:**
Store data in fixed-size pages (typically 4KB or 8KB), the fundamental unit of I/O.

**Hint:**
```cpp
constexpr size_t PAGE_SIZE = 4096;
using page_id_t = uint32_t;

struct Page {
    page_id_t page_id;
    int pin_count;
    bool is_dirty;
    uint8_t data[PAGE_SIZE];

    // Page header (first bytes)
    struct Header {
        uint16_t num_slots;      // Number of tuples
        uint16_t free_space_offset;
        uint32_t next_page;      // For linked pages
    };

    Header* getHeader() {
        return reinterpret_cast<Header*>(data);
    }
};

class DiskManager {
private:
    int db_fd;
    std::string db_file;

public:
    void readPage(page_id_t page_id, Page* page) {
        off_t offset = page_id * PAGE_SIZE;
        lseek(db_fd, offset, SEEK_SET);
        read(db_fd, page->data, PAGE_SIZE);
    }

    void writePage(page_id_t page_id, const Page* page) {
        off_t offset = page_id * PAGE_SIZE;
        lseek(db_fd, offset, SEEK_SET);
        write(db_fd, page->data, PAGE_SIZE);
        fsync(db_fd); // Flush to disk
    }

    page_id_t allocatePage() {
        // Extend file and return new page ID
        off_t file_size = lseek(db_fd, 0, SEEK_END);
        return file_size / PAGE_SIZE;
    }
};
```

**Tips:**
- Use slotted page format for variable-length records
- Implement page header with metadata
- Track free space within pages
- Use bitmap for page allocation

### 2. Buffer Pool with LRU Eviction

**What you need:**
In-memory cache of disk pages with eviction policy.

**Hint:**
```cpp
class BufferPoolManager {
private:
    Page* pages;                  // Array of pages
    std::unordered_map<page_id_t, size_t> page_table; // page_id → frame_id
    std::list<size_t> lru_list;   // LRU order (frame IDs)
    std::unordered_map<size_t, std::list<size_t>::iterator> lru_map;
    std::vector<bool> free_list;  // Available frames
    DiskManager* disk_manager;
    std::mutex latch;

public:
    Page* fetchPage(page_id_t page_id) {
        std::lock_guard<std::mutex> lock(latch);

        // Check if page in buffer pool
        auto it = page_table.find(page_id);
        if (it != page_table.end()) {
            size_t frame_id = it->second;
            pages[frame_id].pin_count++;

            // Update LRU
            lru_list.erase(lru_map[frame_id]);
            lru_list.push_front(frame_id);
            lru_map[frame_id] = lru_list.begin();

            return &pages[frame_id];
        }

        // Find victim page to evict
        size_t frame_id = findVictim();
        if (frame_id == INVALID_FRAME_ID) {
            return nullptr; // All pages pinned
        }

        // Evict page if dirty
        if (pages[frame_id].is_dirty) {
            disk_manager->writePage(pages[frame_id].page_id, &pages[frame_id]);
        }

        // Load new page
        page_table.erase(pages[frame_id].page_id);
        disk_manager->readPage(page_id, &pages[frame_id]);
        pages[frame_id].page_id = page_id;
        pages[frame_id].pin_count = 1;
        pages[frame_id].is_dirty = false;

        page_table[page_id] = frame_id;

        return &pages[frame_id];
    }

    void unpinPage(page_id_t page_id, bool is_dirty) {
        std::lock_guard<std::mutex> lock(latch);
        auto it = page_table.find(page_id);
        if (it == page_table.end()) return;

        size_t frame_id = it->second;
        pages[frame_id].pin_count--;
        pages[frame_id].is_dirty |= is_dirty;
    }

private:
    size_t findVictim() {
        // Find unpinned page (pin_count == 0) from LRU
        for (auto it = lru_list.rbegin(); it != lru_list.rend(); ++it) {
            if (pages[*it].pin_count == 0) {
                return *it;
            }
        }
        return INVALID_FRAME_ID;
    }
};
```

**Tips:**
- Pages with `pin_count > 0` cannot be evicted
- Implement clock replacement algorithm as alternative
- Add prefetching for sequential scans
- Use multiple pools for different page types

### 3. B+ Tree Index

**What you need:**
Balanced tree for efficient key lookups, range scans, and sorted iteration.

**Hint:**
```cpp
template<typename KeyType, typename ValueType, typename KeyComparator>
class BPlusTree {
private:
    struct InternalPage {
        page_id_t page_id;
        int num_keys;
        KeyType keys[MAX_INTERNAL_SIZE];
        page_id_t children[MAX_INTERNAL_SIZE + 1];

        int findChild(const KeyType& key) {
            // Binary search
            int idx = std::lower_bound(keys, keys + num_keys, key) - keys;
            return idx;
        }
    };

    struct LeafPage {
        page_id_t page_id;
        int num_keys;
        KeyType keys[MAX_LEAF_SIZE];
        ValueType values[MAX_LEAF_SIZE];
        page_id_t next_leaf; // For range scans

        int findKey(const KeyType& key) {
            return std::lower_bound(keys, keys + num_keys, key) - keys;
        }
    };

    page_id_t root_page_id;
    BufferPoolManager* buffer_pool;

public:
    bool insert(const KeyType& key, const ValueType& value) {
        // 1. Start from root
        // 2. Navigate to leaf page
        // 3. Insert key-value pair
        // 4. If leaf overflows, split and propagate up
        // 5. If root splits, create new root

        LeafPage* leaf = findLeaf(key);
        int idx = leaf->findKey(key);

        // Check for duplicate key
        if (idx < leaf->num_keys && leaf->keys[idx] == key) {
            return false;
        }

        // Insert into leaf
        if (leaf->num_keys < MAX_LEAF_SIZE) {
            insertIntoLeaf(leaf, idx, key, value);
            return true;
        }

        // Leaf is full, need to split
        splitLeaf(leaf, key, value);
        return true;
    }

    bool search(const KeyType& key, ValueType& result) {
        LeafPage* leaf = findLeaf(key);
        int idx = leaf->findKey(key);

        if (idx < leaf->num_keys && leaf->keys[idx] == key) {
            result = leaf->values[idx];
            return true;
        }
        return false;
    }

    std::vector<ValueType> rangeScan(const KeyType& start, const KeyType& end) {
        std::vector<ValueType> results;
        LeafPage* leaf = findLeaf(start);

        // Scan through leaves
        while (leaf != nullptr) {
            for (int i = 0; i < leaf->num_keys; i++) {
                if (leaf->keys[i] >= start && leaf->keys[i] <= end) {
                    results.push_back(leaf->values[i]);
                } else if (leaf->keys[i] > end) {
                    return results;
                }
            }

            // Move to next leaf
            if (leaf->next_leaf != INVALID_PAGE_ID) {
                Page* next_page = buffer_pool->fetchPage(leaf->next_leaf);
                leaf = reinterpret_cast<LeafPage*>(next_page->data);
            } else {
                break;
            }
        }

        return results;
    }

private:
    void splitLeaf(LeafPage* leaf, const KeyType& key, const ValueType& value) {
        // 1. Allocate new leaf page
        // 2. Move half the keys to new leaf
        // 3. Update next pointers
        // 4. Insert key into parent
        // 5. Recursively split parent if needed
    }
};
```

**Tips:**
- Use bulk loading for initial index creation
- Implement concurrent B+ tree (latch crabbing)
- Add deletion with merging and redistribution
- Support composite keys for multi-column indexes

### 4. SQL Parser

**What you need:**
Parse SQL queries into an Abstract Syntax Tree (AST).

**Hint:**
```cpp
enum class TokenType {
    SELECT, FROM, WHERE, INSERT, UPDATE, DELETE,
    AND, OR, NOT, EQUAL, LESS_THAN, GREATER_THAN,
    IDENTIFIER, NUMBER, STRING, SEMICOLON, COMMA
};

struct Token {
    TokenType type;
    std::string value;
    int line, column;
};

class Lexer {
public:
    std::vector<Token> tokenize(const std::string& sql) {
        std::vector<Token> tokens;
        // Scan through SQL string, identify keywords, operators, literals
        // Return list of tokens
    }
};

struct Expression {
    virtual ~Expression() = default;
};

struct SelectStatement {
    std::vector<std::string> columns;
    std::string table_name;
    std::unique_ptr<Expression> where_clause;
};

class Parser {
private:
    std::vector<Token> tokens;
    size_t current = 0;

public:
    std::unique_ptr<SelectStatement> parseSelect() {
        auto stmt = std::make_unique<SelectStatement>();

        expect(TokenType::SELECT);

        // Parse column list
        do {
            stmt->columns.push_back(current_token().value);
            advance();
        } while (match(TokenType::COMMA));

        expect(TokenType::FROM);
        stmt->table_name = current_token().value;
        advance();

        // Parse WHERE clause
        if (match(TokenType::WHERE)) {
            stmt->where_clause = parseExpression();
        }

        return stmt;
    }

private:
    bool match(TokenType type) {
        if (current_token().type == type) {
            advance();
            return true;
        }
        return false;
    }

    Token current_token() { return tokens[current]; }
    void advance() { current++; }
};
```

**Tips:**
- Use recursive descent parsing
- Support JOIN operations (INNER, LEFT, RIGHT)
- Implement operator precedence for WHERE clauses
- Add support for subqueries
- Generate meaningful error messages with line numbers

### 5. MVCC Transaction Manager

**What you need:**
Multi-Version Concurrency Control for snapshot isolation.

**Hint:**
```cpp
using txn_id_t = uint64_t;
using timestamp_t = uint64_t;

struct TransactionContext {
    txn_id_t txn_id;
    timestamp_t start_ts;
    timestamp_t commit_ts;
    bool is_committed;
    std::set<page_id_t> write_set;
    std::set<page_id_t> read_set;
};

class TransactionManager {
private:
    std::atomic<txn_id_t> next_txn_id{0};
    std::atomic<timestamp_t> current_timestamp{0};
    std::unordered_map<txn_id_t, TransactionContext> active_txns;
    std::mutex latch;

public:
    TransactionContext* begin() {
        std::lock_guard<std::mutex> lock(latch);

        txn_id_t txn_id = next_txn_id++;
        timestamp_t start_ts = current_timestamp.load();

        active_txns[txn_id] = {txn_id, start_ts, 0, false, {}, {}};
        return &active_txns[txn_id];
    }

    bool commit(txn_id_t txn_id) {
        std::lock_guard<std::mutex> lock(latch);

        auto it = active_txns.find(txn_id);
        if (it == active_txns.end()) return false;

        auto& txn = it->second;

        // Check for write-write conflicts
        for (page_id_t page : txn.write_set) {
            if (hasConflict(page, txn.start_ts)) {
                abort(txn_id);
                return false;
            }
        }

        // Assign commit timestamp
        txn.commit_ts = ++current_timestamp;
        txn.is_committed = true;

        // Write WAL record
        writeCommitLog(txn_id, txn.commit_ts);

        return true;
    }

    void abort(txn_id_t txn_id) {
        // Rollback changes
        // Release locks
        active_txns.erase(txn_id);
    }

private:
    bool hasConflict(page_id_t page, timestamp_t start_ts) {
        // Check if any transaction modified this page
        // after our start timestamp
        return false; // Simplified
    }
};

// Each tuple has version information
struct TupleHeader {
    txn_id_t created_by;
    txn_id_t deleted_by;
    timestamp_t created_ts;
    timestamp_t deleted_ts;
};
```

**Tips:**
- Store multiple versions of each tuple
- Garbage collect old versions periodically
- Implement optimistic concurrency control
- Support different isolation levels (Read Committed, Repeatable Read, Serializable)

### 6. Write-Ahead Logging (WAL)

**What you need:**
Durability through logging changes before writing to data pages.

**Hint:**
```cpp
enum class LogRecordType {
    BEGIN, COMMIT, ABORT, INSERT, UPDATE, DELETE
};

struct LogRecord {
    LogRecordType type;
    txn_id_t txn_id;
    uint64_t lsn;           // Log Sequence Number
    uint64_t prev_lsn;      // For transaction undo
    page_id_t page_id;
    size_t offset;
    std::vector<uint8_t> before_image; // Old data
    std::vector<uint8_t> after_image;  // New data
};

class WALManager {
private:
    int log_fd;
    std::atomic<uint64_t> next_lsn{0};
    std::mutex latch;

public:
    uint64_t appendLog(const LogRecord& record) {
        std::lock_guard<std::mutex> lock(latch);

        uint64_t lsn = next_lsn++;

        // Serialize log record
        std::vector<uint8_t> serialized = serialize(record);

        // Write to log file
        write(log_fd, serialized.data(), serialized.size());

        // Optionally fsync for durability
        fsync(log_fd);

        return lsn;
    }

    void recover() {
        // ARIES recovery algorithm:
        // 1. Analysis: Determine which transactions were active
        // 2. Redo: Replay all logged operations
        // 3. Undo: Rollback uncommitted transactions

        lseek(log_fd, 0, SEEK_SET);

        std::map<txn_id_t, std::vector<LogRecord>> active_txns;
        std::map<txn_id_t, bool> committed;

        // Phase 1: Analysis
        while (true) {
            LogRecord record;
            if (!readLogRecord(record)) break;

            if (record.type == LogRecordType::COMMIT) {
                committed[record.txn_id] = true;
            } else {
                active_txns[record.txn_id].push_back(record);
            }
        }

        // Phase 2: Redo
        for (const auto& [txn_id, records] : active_txns) {
            for (const auto& record : records) {
                redoOperation(record);
            }
        }

        // Phase 3: Undo uncommitted transactions
        for (const auto& [txn_id, records] : active_txns) {
            if (!committed[txn_id]) {
                for (auto it = records.rbegin(); it != records.rend(); ++it) {
                    undoOperation(*it);
                }
            }
        }
    }
};
```

**Tips:**
- Implement log checkpointing to reduce recovery time
- Use group commit to batch log writes
- Add logical undo for higher-level operations
- Compress log records to save space

### 7. Query Execution: Join Algorithms

**What you need:**
Efficiently join two tables.

**Hint:**
```cpp
class Executor {
public:
    // Nested Loop Join - O(n * m)
    std::vector<Tuple> nestedLoopJoin(
        const std::vector<Tuple>& left,
        const std::vector<Tuple>& right,
        const JoinPredicate& predicate)
    {
        std::vector<Tuple> result;

        for (const auto& l_tuple : left) {
            for (const auto& r_tuple : right) {
                if (predicate.evaluate(l_tuple, r_tuple)) {
                    result.push_back(merge(l_tuple, r_tuple));
                }
            }
        }

        return result;
    }

    // Hash Join - O(n + m)
    std::vector<Tuple> hashJoin(
        const std::vector<Tuple>& left,
        const std::vector<Tuple>& right,
        int left_key_idx,
        int right_key_idx)
    {
        // Build phase: Create hash table from left table
        std::unordered_multimap<Value, Tuple> hash_table;
        for (const auto& tuple : left) {
            Value key = tuple.getValue(left_key_idx);
            hash_table.insert({key, tuple});
        }

        // Probe phase: Scan right table and match
        std::vector<Tuple> result;
        for (const auto& r_tuple : right) {
            Value key = r_tuple.getValue(right_key_idx);
            auto range = hash_table.equal_range(key);

            for (auto it = range.first; it != range.second; ++it) {
                result.push_back(merge(it->second, r_tuple));
            }
        }

        return result;
    }

    // Merge Join - O(n + m) if inputs sorted
    std::vector<Tuple> mergeJoin(
        const std::vector<Tuple>& left,   // Sorted by join key
        const std::vector<Tuple>& right,  // Sorted by join key
        int key_idx)
    {
        std::vector<Tuple> result;
        size_t l_idx = 0, r_idx = 0;

        while (l_idx < left.size() && r_idx < right.size()) {
            Value l_key = left[l_idx].getValue(key_idx);
            Value r_key = right[r_idx].getValue(key_idx);

            if (l_key < r_key) {
                l_idx++;
            } else if (l_key > r_key) {
                r_idx++;
            } else {
                // Match found
                result.push_back(merge(left[l_idx], right[r_idx]));
                r_idx++;
            }
        }

        return result;
    }
};
```

**Tips:**
- Use hash join for equality predicates
- Use merge join when inputs are already sorted
- Implement block nested loop join for large tables
- Add statistics to choose best join algorithm

---

## Project Structure

```
04_database_engine/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── storage/
│   │   ├── disk_manager.cpp
│   │   ├── buffer_pool.cpp
│   │   └── page.cpp
│   ├── index/
│   │   ├── b_plus_tree.cpp
│   │   └── index_iterator.cpp
│   ├── parser/
│   │   ├── lexer.cpp
│   │   ├── parser.cpp
│   │   └── ast.cpp
│   ├── optimizer/
│   │   ├── optimizer.cpp
│   │   └── cost_model.cpp
│   ├── executor/
│   │   ├── executor.cpp
│   │   ├── seq_scan.cpp
│   │   ├── index_scan.cpp
│   │   ├── join.cpp
│   │   └── aggregation.cpp
│   ├── concurrency/
│   │   ├── transaction_manager.cpp
│   │   └── lock_manager.cpp
│   └── recovery/
│       └── wal_manager.cpp
├── include/
│   └── database/
│       ├── storage.h
│       ├── index.h
│       ├── executor.h
│       └── transaction.h
├── tests/
│   ├── test_storage.cpp
│   ├── test_index.cpp
│   ├── test_parser.cpp
│   ├── test_executor.cpp
│   └── test_transaction.cpp
└── benchmarks/
    ├── tpcc_benchmark.cpp
    └── scan_benchmark.cpp
```

---

## Testing Strategy

1. **Unit Tests**: Each component in isolation
2. **Integration Tests**: End-to-end SQL queries
3. **Concurrency Tests**: Multiple transactions
4. **Recovery Tests**: Crash and restart scenarios
5. **Performance Tests**: TPC-C, TPC-H benchmarks

---

## Performance Goals

- **Point Query**: < 1ms with index
- **Range Scan**: 1M rows/sec
- **Insert Throughput**: 10K inserts/sec
- **Transaction Rate**: 1K TPS

---

## Resources

- [CMU Database Systems Course](https://15445.courses.cs.cmu.edu/)
- Book: "Database System Concepts" by Silberschatz
- Book: "Transaction Processing" by Gray & Reuter
- [SQLite Architecture](https://www.sqlite.org/arch.html)
- [PostgreSQL Internals](https://www.postgresql.org/docs/current/internals.html)

Good luck building your database!
