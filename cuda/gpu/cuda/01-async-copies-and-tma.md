# Asynchronous Data Copies and Tensor Memory Accelerator (TMA)

## Overview

Asynchronous data movement is critical for achieving high performance on modern NVIDIA GPUs. By decoupling data transfers from computation, kernels can hide memory latency and maximize throughput.

**Key Resources:**
- [CUDA Programming Guide - Async Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [CUDA Core Compute Libraries - memcpy_async](https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html)

---

## Three Async Copy Mechanisms

### 1. LDGSTS (Compute Capability 8.0+ / Ampere)

**Purpose:** Element-wise asynchronous transfers from global to shared memory.

**Capabilities:**
- Transfer sizes: 4, 8, or 16 bytes per operation
- Direction: **Global → Shared memory only**
- 16-byte transfers bypass L1 cache to avoid pollution
- Optimal performance with 128-byte alignment

**Alignment Requirements:**
- Source (global memory): Must align to transfer size (4, 8, or 16 bytes)
- Destination (shared memory): Must align to transfer size
- Best performance: 128-byte alignment for both

**Key Limitations:**
- Only supports global-to-shared direction
- Requires synchronization via barriers or pipelines

---

### 2. Tensor Memory Accelerator - TMA (Compute Capability 9.0+ / Hopper)

**Purpose:** Bulk asynchronous transfers of multi-dimensional tensors with hardware acceleration.

**Revolutionary Capabilities:**
- Transfer 1D to 5D tensors
- **Bidirectional:** Global ↔ Shared memory
- **Multicast:** Copy to multiple blocks in a cluster
- **Zero register usage** for data movement
- Hardware-accelerated reduction operations (add, min, max, bitwise)
- Swizzle patterns to eliminate bank conflicts

**Alignment Requirements:**
- Global memory: 16-byte minimum
- Shared memory: 16-byte minimum (128-byte for multi-dimensional)
- Transfer sizes: Multiple of 16 bytes

**Key Advantages:**
- Single thread can issue large transfers while block continues processing
- Enables warp-specialized code (data movement warps separate from compute warps)
- Eliminates address calculation overhead for multi-dimensional arrays
- Combined bandwidth from distributed shared memory + L2 cache

**Performance Impact:**
TMA on H100 provides access to:
- **3 TB/s HBM3 bandwidth** (93% increase over A100)
- **50 MB L2 cache** with enhanced bandwidth
- **228 KB shared memory per SM** (39% increase over A100)

---

### 3. STAS (Compute Capability 9.0+ / Hopper)

**Purpose:** Direct asynchronous copies from registers to distributed shared memory within clusters.

**Specifications:**
- Transfer sizes: 4, 8, or 16 bytes
- Direction: **Registers → Distributed Shared Memory**
- Requires alignment matching transfer size
- Enables efficient inter-block communication within clusters

---

## API Overview

### High-Level API: `cuda::memcpy_async`

```cpp
#include <cuda/barrier>
#include <cuda/std/cstdlib>

__global__ void kernel() {
    __shared__ int shared_data[128];
    int* global_data = /* ... */;

    // Create barrier for synchronization
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    __syncthreads();

    // Async copy with barrier tracking
    cuda::memcpy_async(shared_data, global_data,
                       sizeof(int) * 128, barrier);

    // Wait for completion
    barrier.arrive_and_wait();

    // Now safe to use shared_data
}
```

### Cooperative Groups API

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void kernel() {
    __shared__ float shared[256];
    float* global = /* ... */;

    cg::thread_block block = cg::this_thread_block();

    // Collective async copy
    cg::memcpy_async(block, shared, global, sizeof(float) * 256);

    // Wait for completion
    cg::wait(block);
}
```

### Low-Level Pipeline API

```cpp
__global__ void pipelined_kernel() {
    __shared__ float buffer[2][256];
    float* global_data = /* ... */;

    // Initialize pipeline
    __pipeline_empty();

    // Stage 0: Load first batch
    __pipeline_memcpy_async(&buffer[0][0], &global_data[0],
                           sizeof(float) * 256);
    __pipeline_commit();

    for (int i = 1; i < num_batches; i++) {
        // Stage i: Load next batch while processing current
        int write_idx = i % 2;
        int read_idx = (i - 1) % 2;

        __pipeline_memcpy_async(&buffer[write_idx][0],
                               &global_data[i * 256],
                               sizeof(float) * 256);
        __pipeline_commit();

        // Wait for previous batch
        __pipeline_wait_prior(1);

        // Process data from read buffer
        compute(buffer[read_idx]);
    }

    // Process final batch
    __pipeline_wait_prior(0);
    compute(buffer[(num_batches - 1) % 2]);
}
```

---

## TMA Programming Model

### Creating Tensor Maps

```cpp
#include <cuda.h>

// Host-side tensor map creation
CUtensorMap tensorMap;
CUdeviceptr global_mem;
cudaMalloc(&global_mem, tensor_size);

uint64_t tensor_dims[] = {height, width};
uint64_t tensor_strides[] = {width * sizeof(float), sizeof(float)};
uint32_t box_dims[] = {tile_height, tile_width};
uint32_t elem_strides[] = {1, 1};

cuTensorMapEncodeTiled(
    &tensorMap,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    2,                          // rank (2D tensor)
    global_mem,                 // global memory address
    tensor_dims,                // tensor dimensions
    tensor_strides,             // tensor strides
    box_dims,                   // tile dimensions
    elem_strides,               // element strides
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B, // bank conflict elimination
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

### Using TMA in Kernels

```cpp
__global__ void tma_kernel(const __grid_constant__ CUtensorMap map) {
    __shared__ float smem[TILE_H][TILE_W];

    // Create barrier
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    if (threadIdx.x == 0) {
        init(&barrier, 1); // TMA uses single thread
    }
    __syncthreads();

    // Only one thread issues TMA
    if (threadIdx.x == 0) {
        // Async bulk copy using TMA
        cuda::memcpy_async(smem, map,
                          cuda::aligned_size_t<128>(sizeof(smem)),
                          barrier);
    }

    // All threads wait
    barrier.arrive_and_wait();

    // Process tile
    process_tile(smem);
}
```

### TMA Multicast (Cluster-Level)

```cpp
__global__ void __cluster_dims__(2, 2, 1) // 2x2 cluster
tma_multicast_kernel(const __grid_constant__ CUtensorMap map) {
    __shared__ float smem[TILE_SIZE];

    auto cluster = cooperative_groups::this_cluster();
    auto block = cooperative_groups::this_thread_block();

    // Create barrier
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    if (threadIdx.x == 0) {
        init(&barrier, 1);
    }
    __syncthreads();

    // Block 0 issues multicast TMA to all blocks in cluster
    if (block.block_rank() == 0 && threadIdx.x == 0) {
        uint16_t mcast_mask = 0xFFFF; // All blocks in cluster
        cuda::memcpy_async(smem, map,
                          cuda::aligned_size_t<128>(sizeof(smem)),
                          barrier, mcast_mask);
    }

    // All blocks synchronize
    cluster.sync();
    barrier.arrive_and_wait();

    // All blocks now have the same data
    process_tile(smem);
}
```

---

## Common Usage Patterns

### Pattern 1: Double Buffering

```cpp
__global__ void double_buffer_kernel(float* input, float* output, int n) {
    __shared__ float buffer[2][256];

    // Load first tile
    cuda::memcpy_async(buffer[0], input, sizeof(float) * 256, barrier_0);

    for (int i = 0; i < n / 256; i++) {
        int curr = i % 2;
        int next = (i + 1) % 2;

        // Load next tile while processing current
        if (i < n / 256 - 1) {
            cuda::memcpy_async(buffer[next],
                             input + (i + 1) * 256,
                             sizeof(float) * 256,
                             barrier_next);
        }

        barrier_curr.arrive_and_wait();

        // Process current tile
        compute(buffer[curr], output + i * 256);

        // Swap barriers
        auto temp = barrier_curr;
        barrier_curr = barrier_next;
        barrier_next = temp;
    }
}
```

### Pattern 2: Warp-Specialized Roles

```cpp
__global__ void warp_specialized_kernel() {
    __shared__ float buffer[4][256];

    int warp_id = threadIdx.x / 32;

    if (warp_id == 0) {
        // Data movement warp
        for (int stage = 0; stage < 4; stage++) {
            cuda::memcpy_async(buffer[stage],
                             global_data + stage * 256,
                             sizeof(float) * 256,
                             barriers[stage]);
        }
    } else {
        // Compute warps
        for (int stage = 0; stage < 4; stage++) {
            barriers[stage].arrive_and_wait();
            compute(buffer[stage]);
        }
    }
}
```

---

## Synchronization

### Shared Memory Barriers

```cpp
#include <cuda/barrier>

__shared__ cuda::barrier<cuda::thread_scope_block> barrier;

// Initialize (single thread)
if (threadIdx.x == 0) {
    init(&barrier, blockDim.x); // Expected arrival count
}
__syncthreads();

// Issue async copy
cuda::memcpy_async(dest, src, size, barrier);

// Wait for completion
barrier.arrive_and_wait();
```

### Pipeline with Multiple Stages

```cpp
constexpr size_t stages = 4;
__shared__ cuda::pipeline_shared_state<
    cuda::thread_scope_block, stages> pipe_state;

cuda::pipeline<cuda::thread_scope_block> pipe =
    cuda::make_pipeline(pipe_state);

// Producer
for (int i = 0; i < N; i++) {
    pipe.producer_acquire();
    cuda::memcpy_async(dest[i % stages], src[i], size, pipe);
    pipe.producer_commit();
}

// Consumer
for (int i = 0; i < N; i++) {
    pipe.consumer_wait();
    compute(dest[i % stages]);
    pipe.consumer_release();
}
```

---

## Performance Tips

### 1. Maximize Alignment
- Ensure 128-byte alignment for both source and destination
- Use `__align__(128)` for shared memory declarations

### 2. Avoid Small Transfers
- Batch multiple small transfers into larger ones
- TMA overhead is amortized over transfer size

### 3. Hide Latency with Pipelining
- Keep multiple transfers in flight
- Process previous data while loading next batch

### 4. Use TMA for Multi-Dimensional Data
- Hardware handles address calculations
- Eliminates register pressure from index arithmetic

### 5. Leverage Swizzle Patterns
- Use 128B swizzle for matrix tiles to eliminate bank conflicts
- Particularly important for transpose operations

### 6. Cluster-Aware Data Sharing
- Use TMA multicast to share data across cluster blocks
- Reduces redundant global memory accesses

---

## Comparison Table

| Feature | LDGSTS (Ampere) | TMA (Hopper) | STAS (Hopper) |
|---------|----------------|--------------|---------------|
| **Compute Capability** | 8.0+ | 9.0+ | 9.0+ |
| **Dimensions** | Element-wise | 1D-5D tensors | Element-wise |
| **Direction** | Global→Shared | Global↔Shared | Register→Dist.Shared |
| **Register Usage** | Minimal | Zero | Minimal |
| **Multicast** | No | Yes (clusters) | No |
| **Bank Conflict Handling** | Manual | Hardware swizzle | Manual |
| **Reductions** | No | Yes (add/min/max) | No |

---

## Common Pitfalls

### 1. Forgetting Synchronization
```cpp
// WRONG: No synchronization
cuda::memcpy_async(smem, gmem, size, barrier);
use_data(smem); // Race condition!

// CORRECT: Wait for completion
cuda::memcpy_async(smem, gmem, size, barrier);
barrier.arrive_and_wait();
use_data(smem); // Safe
```

### 2. Incorrect Alignment
```cpp
// WRONG: Unaligned access
__shared__ float data[100]; // Not aligned
cuda::memcpy_async(data, gmem, sizeof(data), barrier);

// CORRECT: Proper alignment
__shared__ __align__(128) float data[128];
cuda::memcpy_async(data, gmem, sizeof(data), barrier);
```

### 3. Wrong Barrier Count
```cpp
// WRONG: Mismatched arrival count
init(&barrier, blockDim.x);
if (threadIdx.x == 0) {
    cuda::memcpy_async(..., barrier); // Only 1 arrival, expects 256!
}

// CORRECT: Match arrivals to expectation
init(&barrier, 1); // TMA issued by single thread
if (threadIdx.x == 0) {
    cuda::memcpy_async(..., barrier);
}
barrier.arrive_and_wait(); // All threads wait
```

---

## Further Reading

- [CUTLASS TMA Tutorial](https://research.colfax-intl.com/tutorial-hopper-tma/)
- [Hopper Architecture Deep Dive](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [CUDA C++ Core Libraries Documentation](https://nvidia.github.io/cccl/)
