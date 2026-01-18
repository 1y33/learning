# Memory Allocator with Garbage Collection - Implementation Guide

## What is This Project?

Build a custom memory allocator with automatic garbage collection, similar to allocators used in high-performance systems and managed languages. This project teaches memory management, heap data structures, and GC algorithms.

## Why Build This?

- Understand how malloc/free work internally
- Learn memory allocation strategies
- Master garbage collection algorithms
- Optimize for performance and fragmentation
- Build custom allocators for specific workloads

---

## Architecture Overview

```
┌──────────────────────────────────────────┐
│        Application Code (new/delete)     │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│       Allocation Interface               │
│  ┌────────────┐  ┌──────────────────┐   │
│  │  malloc()  │  │    gcNew()       │   │
│  └────────────┘  └──────────────────┘   │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│     Free List / Segregated Fits          │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│        Garbage Collector                 │
│  ┌────────────┐  ┌──────────────────┐   │
│  │  Mark &    │  │   Compaction     │   │
│  │  Sweep     │  │                  │   │
│  └────────────┘  └──────────────────┘   │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│         Heap Memory (mmap/brk)           │
└──────────────────────────────────────────┘
```

---

## Implementation Hints

### 1. Basic Heap Allocator (First-Fit)

**What you need:**
Manage a heap with free blocks linked in a list.

**Hint:**
```cpp
struct BlockHeader {
    size_t size;
    bool is_free;
    BlockHeader* next;
};

class HeapAllocator {
private:
    void* heap_start;
    size_t heap_size;
    BlockHeader* free_list;

public:
    HeapAllocator(size_t size) {
        heap_size = size;
        heap_start = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        // Initialize with one large free block
        free_list = static_cast<BlockHeader*>(heap_start);
        free_list->size = size - sizeof(BlockHeader);
        free_list->is_free = true;
        free_list->next = nullptr;
    }

    void* allocate(size_t size) {
        // Align size to 8 bytes
        size = (size + 7) & ~7;

        // First-fit: Find first free block that's large enough
        BlockHeader* current = free_list;
        BlockHeader* prev = nullptr;

        while (current != nullptr) {
            if (current->is_free && current->size >= size) {
                // Found suitable block

                // Split block if remainder is large enough
                if (current->size >= size + sizeof(BlockHeader) + 8) {
                    BlockHeader* new_block =
                        reinterpret_cast<BlockHeader*>(
                            reinterpret_cast<char*>(current) +
                            sizeof(BlockHeader) + size
                        );

                    new_block->size = current->size - size - sizeof(BlockHeader);
                    new_block->is_free = true;
                    new_block->next = current->next;

                    current->size = size;
                    current->next = new_block;
                }

                current->is_free = false;

                // Return pointer after header
                return reinterpret_cast<void*>(
                    reinterpret_cast<char*>(current) + sizeof(BlockHeader)
                );
            }

            prev = current;
            current = current->next;
        }

        // No suitable block found
        return nullptr;
    }

    void deallocate(void* ptr) {
        if (ptr == nullptr) return;

        BlockHeader* header = reinterpret_cast<BlockHeader*>(
            reinterpret_cast<char*>(ptr) - sizeof(BlockHeader)
        );

        header->is_free = true;

        // Coalesce with next block if free
        if (header->next && header->next->is_free) {
            header->size += sizeof(BlockHeader) + header->next->size;
            header->next = header->next->next;
        }

        // Coalesce with previous block (requires searching)
        // ... (implementation omitted for brevity)
    }

    ~HeapAllocator() {
        munmap(heap_start, heap_size);
    }
};
```

**Tips:**
- Use best-fit or worst-fit as alternatives
- Implement boundary tags for efficient coalescing
- Add minimum block size to reduce fragmentation
- Track heap statistics (allocated, free, fragmentation)

### 2. Segregated Free Lists

**What you need:**
Multiple free lists for different size classes (like tcmalloc).

**Hint:**
```cpp
class SegregatedAllocator {
private:
    static constexpr int NUM_SIZE_CLASSES = 32;
    static constexpr size_t MIN_BLOCK_SIZE = 16;
    static constexpr size_t MAX_SMALL_SIZE = 2048;

    struct FreeBlock {
        FreeBlock* next;
    };

    FreeBlock* free_lists[NUM_SIZE_CLASSES];
    std::mutex mutexes[NUM_SIZE_CLASSES];

    int getSizeClass(size_t size) {
        // Map size to size class index
        // Example: 16 → 0, 32 → 1, 64 → 2, ...
        if (size <= MIN_BLOCK_SIZE) return 0;
        if (size > MAX_SMALL_SIZE) return NUM_SIZE_CLASSES - 1;

        return (size - 1) / MIN_BLOCK_SIZE;
    }

    size_t classSize(int size_class) {
        if (size_class == NUM_SIZE_CLASSES - 1) {
            return MAX_SMALL_SIZE * 2; // Large objects
        }
        return (size_class + 1) * MIN_BLOCK_SIZE;
    }

public:
    void* allocate(size_t size) {
        int size_class = getSizeClass(size);
        std::lock_guard<std::mutex> lock(mutexes[size_class]);

        // Check free list for this size class
        if (free_lists[size_class] != nullptr) {
            FreeBlock* block = free_lists[size_class];
            free_lists[size_class] = block->next;
            return block;
        }

        // Allocate new block from system
        size_t block_size = classSize(size_class);
        void* ptr = mmap(nullptr, block_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        return ptr;
    }

    void deallocate(void* ptr, size_t size) {
        if (ptr == nullptr) return;

        int size_class = getSizeClass(size);
        std::lock_guard<std::mutex> lock(mutexes[size_class]);

        // Return to free list
        FreeBlock* block = static_cast<FreeBlock*>(ptr);
        block->next = free_lists[size_class];
        free_lists[size_class] = block;
    }
};
```

**Tips:**
- Use power-of-2 size classes
- Implement thread-local caches for lock-free allocation
- Add page-level allocator for large objects
- Use tcmalloc or jemalloc as reference

### 3. Mark-and-Sweep Garbage Collector

**What you need:**
Automatically reclaim unreachable objects.

**Hint:**
```cpp
struct GCObject {
    bool marked;
    size_t size;
    GCObject* next;
    uint8_t data[];
};

class MarkSweepGC {
private:
    GCObject* object_list = nullptr;
    std::vector<void**> roots; // Root pointers (stack, globals)
    size_t total_allocated = 0;
    size_t gc_threshold = 1024 * 1024; // 1MB

public:
    void* gcAlloc(size_t size) {
        // Trigger GC if threshold exceeded
        if (total_allocated > gc_threshold) {
            collect();
        }

        // Allocate object
        GCObject* obj = static_cast<GCObject*>(
            malloc(sizeof(GCObject) + size)
        );

        obj->marked = false;
        obj->size = size;
        obj->next = object_list;
        object_list = obj;

        total_allocated += size;

        return obj->data;
    }

    void registerRoot(void** root) {
        roots.push_back(root);
    }

    void collect() {
        // Phase 1: Mark reachable objects
        markAll();

        // Phase 2: Sweep unreachable objects
        sweep();
    }

private:
    void markAll() {
        // Mark from roots
        for (void** root : roots) {
            if (*root != nullptr) {
                GCObject* obj = getObject(*root);
                mark(obj);
            }
        }
    }

    void mark(GCObject* obj) {
        if (obj == nullptr || obj->marked) return;

        obj->marked = true;

        // Scan object data for pointers (conservative GC)
        size_t num_words = obj->size / sizeof(void*);
        void** words = reinterpret_cast<void**>(obj->data);

        for (size_t i = 0; i < num_words; i++) {
            if (isPointer(words[i])) {
                GCObject* referenced = getObject(words[i]);
                mark(referenced);
            }
        }
    }

    void sweep() {
        GCObject** current = &object_list;

        while (*current != nullptr) {
            if (!(*current)->marked) {
                // Unreachable, free it
                GCObject* unreachable = *current;
                *current = unreachable->next;

                total_allocated -= unreachable->size;
                free(unreachable);
            } else {
                // Reachable, unmark for next GC
                (*current)->marked = false;
                current = &(*current)->next;
            }
        }
    }

    GCObject* getObject(void* ptr) {
        // Find object that contains this pointer
        return reinterpret_cast<GCObject*>(
            reinterpret_cast<char*>(ptr) - offsetof(GCObject, data)
        );
    }

    bool isPointer(void* ptr) {
        // Conservative: Check if value looks like a pointer
        // within heap range
        return ptr >= heap_start && ptr < heap_end;
    }
};
```

**Tips:**
- Use tri-color marking for incremental GC
- Implement write barriers for concurrent GC
- Add finalizers for cleanup
- Track GC pause times

### 4. Generational Garbage Collector

**What you need:**
Optimize GC by focusing on young objects (most die young).

**Hint:**
```cpp
class GenerationalGC {
private:
    struct Generation {
        std::vector<GCObject*> objects;
        size_t size = 0;
        size_t threshold;
    };

    Generation young_gen;  // Frequent, fast collections
    Generation old_gen;    // Infrequent, slow collections

    int gc_count = 0;
    static constexpr int PROMOTION_AGE = 2;

public:
    void* gcAlloc(size_t size) {
        // Allocate in young generation
        GCObject* obj = allocateObject(size);
        obj->age = 0;

        young_gen.objects.push_back(obj);
        young_gen.size += size;

        // Minor GC if young gen full
        if (young_gen.size > young_gen.threshold) {
            minorGC();
        }

        return obj->data;
    }

private:
    void minorGC() {
        // Collect young generation only

        markFromRoots(young_gen);
        markFromOldGen(young_gen); // Old → Young references

        std::vector<GCObject*> survivors;

        for (GCObject* obj : young_gen.objects) {
            if (obj->marked) {
                obj->age++;

                // Promote to old generation if survived enough
                if (obj->age >= PROMOTION_AGE) {
                    old_gen.objects.push_back(obj);
                    old_gen.size += obj->size;
                } else {
                    survivors.push_back(obj);
                }

                obj->marked = false;
            } else {
                // Dead object
                young_gen.size -= obj->size;
                free(obj);
            }
        }

        young_gen.objects = std::move(survivors);

        // Trigger major GC if old gen too large
        if (old_gen.size > old_gen.threshold) {
            majorGC();
        }
    }

    void majorGC() {
        // Full heap collection (both generations)
        markFromRoots(young_gen);
        markFromRoots(old_gen);

        sweepGeneration(young_gen);
        sweepGeneration(old_gen);
    }

    void markFromOldGen(Generation& young) {
        // Use write barrier to track Old → Young pointers
        // (remembered set)
    }
};
```

**Tips:**
- Young gen: ~10% of heap, collected frequently
- Old gen: ~90% of heap, collected rarely
- Use card marking or remembered sets
- Tune generation sizes based on workload

### 5. Compacting Garbage Collector

**What you need:**
Move objects to eliminate fragmentation.

**Hint:**
```cpp
class CompactingGC {
private:
    void* heap_start;
    void* heap_end;
    void* free_ptr; // Next free position after compaction

public:
    void compact() {
        // Phase 1: Mark reachable objects
        markAll();

        // Phase 2: Compute new addresses
        void* new_addr = heap_start;
        std::map<GCObject*, void*> forwarding_table;

        GCObject* obj = object_list;
        while (obj != nullptr) {
            if (obj->marked) {
                forwarding_table[obj] = new_addr;
                new_addr = reinterpret_cast<void*>(
                    reinterpret_cast<char*>(new_addr) +
                    sizeof(GCObject) + obj->size
                );
            }
            obj = obj->next;
        }

        // Phase 3: Update all pointers
        updatePointers(forwarding_table);

        // Phase 4: Move objects
        obj = object_list;
        while (obj != nullptr) {
            if (obj->marked) {
                void* new_loc = forwarding_table[obj];
                memmove(new_loc, obj, sizeof(GCObject) + obj->size);
            }
            obj = obj->next;
        }

        // Phase 5: Update free pointer
        free_ptr = new_addr;
    }

private:
    void updatePointers(const std::map<GCObject*, void*>& table) {
        // Update roots
        for (void** root : roots) {
            auto it = table.find(getObject(*root));
            if (it != table.end()) {
                *root = reinterpret_cast<GCObject*>(it->second)->data;
            }
        }

        // Update internal pointers in objects
        for (const auto& [old_obj, new_addr] : table) {
            size_t num_words = old_obj->size / sizeof(void*);
            void** words = reinterpret_cast<void**>(old_obj->data);

            for (size_t i = 0; i < num_words; i++) {
                if (isPointer(words[i])) {
                    GCObject* ref = getObject(words[i]);
                    auto it = table.find(ref);
                    if (it != table.end()) {
                        words[i] = reinterpret_cast<GCObject*>(it->second)->data;
                    }
                }
            }
        }
    }
};
```

**Tips:**
- Use Cheney's algorithm (copying collector)
- Implement semi-space collector (from-space, to-space)
- Add compaction threshold (only compact if fragmented)
- Measure compaction overhead

### 6. Reference Counting (Alternative to GC)

**What you need:**
Track reference counts, free when count reaches zero.

**Hint:**
```cpp
template<typename T>
class RefCounted {
private:
    struct ControlBlock {
        T* ptr;
        std::atomic<int> ref_count;
        std::atomic<int> weak_count;
    };

    ControlBlock* control;

public:
    RefCounted(T* ptr = nullptr) {
        if (ptr) {
            control = new ControlBlock{ptr, 1, 0};
        } else {
            control = nullptr;
        }
    }

    RefCounted(const RefCounted& other) : control(other.control) {
        if (control) {
            control->ref_count++;
        }
    }

    RefCounted& operator=(const RefCounted& other) {
        if (this != &other) {
            release();
            control = other.control;
            if (control) {
                control->ref_count++;
            }
        }
        return *this;
    }

    ~RefCounted() {
        release();
    }

    T* get() const {
        return control ? control->ptr : nullptr;
    }

    T* operator->() const {
        return get();
    }

    T& operator*() const {
        return *get();
    }

private:
    void release() {
        if (control && --control->ref_count == 0) {
            delete control->ptr;

            if (control->weak_count == 0) {
                delete control;
            }
        }
    }
};
```

**Tips:**
- Handle cyclic references with weak pointers
- Use deferred reference counting to reduce overhead
- Implement cycle detection algorithm
- Compare with tracing GC performance

---

## Project Structure

```
07_memory_allocator/
├── CMakeLists.txt
├── src/
│   ├── allocator/
│   │   ├── heap_allocator.cpp
│   │   ├── segregated_allocator.cpp
│   │   └── buddy_allocator.cpp
│   ├── gc/
│   │   ├── mark_sweep.cpp
│   │   ├── generational.cpp
│   │   ├── compacting.cpp
│   │   └── ref_counting.cpp
│   └── benchmarks/
│       ├── allocation_bench.cpp
│       └── gc_bench.cpp
├── tests/
│   ├── test_allocator.cpp
│   └── test_gc.cpp
└── include/
    └── allocator.h
```

---

## Performance Goals

- **Allocation**: < 100ns for small objects
- **Deallocation**: < 50ns
- **GC Pause**: < 10ms for 100MB heap
- **Fragmentation**: < 20% waste
- **Throughput**: Match tcmalloc within 20%

---

## Resources

- [TCMalloc Design](https://google.github.io/tcmalloc/design.html)
- [Garbage Collection Handbook](http://gchandbook.org/)
- [JVM Garbage Collectors](https://docs.oracle.com/en/java/javase/17/gctuning/)
- [Memory Allocators 101](https://arjunsreedharan.org/post/148675821737/memory-allocators-101-write-a-simple-memory)

Good luck building your memory allocator!
