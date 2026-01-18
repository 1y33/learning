# Thread Block Clusters (Hopper H100)

## Overview

Thread Block Clusters are a new hierarchical level introduced in NVIDIA Compute Capability 9.0 (Hopper architecture). They enable efficient collaboration between thread blocks through **Distributed Shared Memory** and hardware-accelerated synchronization.

**Key Resources:**
- [CUDA Programming Guide - Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)

---

## Thread Hierarchy

Traditional CUDA hierarchy has been extended:

```
Grid
 └─ Thread Block
     └─ Warp (32 threads)
         └─ Thread
```

**New Hopper hierarchy:**

```
Grid
 └─ Thread Block Cluster (NEW!)
     └─ Thread Block
         └─ Warp (32 threads)
             └─ Thread
```

**Guarantees:**
- Threads in a thread block: Co-scheduled on a **Streaming Multiprocessor (SM)**
- Thread blocks in a cluster: Co-scheduled on a **GPU Processing Cluster (GPC)**

---

## Key Capabilities

### 1. Distributed Shared Memory

Blocks within a cluster can access each other's shared memory, creating an intermediate memory tier:

```
Latency & Size Hierarchy:
Registers (fastest, smallest)
    ↓
Shared Memory (per-block)
    ↓
Distributed Shared Memory (per-cluster) ← NEW!
    ↓
L2 Cache
    ↓
Global Memory (slowest, largest)
```

**Benefits:**
- Faster than global memory
- Larger capacity than single-block shared memory
- Combined bandwidth from multiple SMs
- Hardware-managed coherence

### 2. Cluster-Wide Synchronization

```cpp
#include <cooperative_groups.h>

__global__ void cluster_kernel() {
    auto cluster = cooperative_groups::this_cluster();

    // Synchronize all threads in all blocks in the cluster
    cluster.sync();
}
```

### 3. Cluster Dimensions

Clusters can be 1D, 2D, or 3D:

```cpp
// 1D cluster: 4 blocks
__global__ void __cluster_dims__(4, 1, 1) kernel_1d() { }

// 2D cluster: 2x3 = 6 blocks
__global__ void __cluster_dims__(2, 3, 1) kernel_2d() { }

// 3D cluster: 2x2x2 = 8 blocks
__global__ void __cluster_dims__(2, 2, 2) kernel_3d() { }
```

**Portable Cluster Size:** Maximum 8 blocks (any dimension combination where X×Y×Z ≤ 8)

**H100 Non-Portable Mode:** Up to 16 blocks with opt-in:
```cpp
cudaFuncSetAttribute(kernel,
    cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
```

---

## Programming Model

### Launch Configuration

#### Method 1: Kernel Attribute (Compile-Time)

```cpp
__global__ void __cluster_dims__(2, 2, 1)
my_cluster_kernel(float* data) {
    // 2x2 cluster = 4 blocks per cluster
}

int main() {
    dim3 blocks(16, 16);      // 256 total blocks
    dim3 threads(16, 16);     // 256 threads per block

    // Clusters automatically configured based on attribute
    my_cluster_kernel<<<blocks, threads>>>(data);
}
```

#### Method 2: Extended Launch API (Runtime)

```cpp
int main() {
    dim3 threads_per_block(256);
    dim3 blocks_per_cluster(2, 2, 1);  // 4 blocks per cluster
    dim3 blocks_in_grid(16, 16);       // 256 total blocks

    cudaLaunchConfig_t config = {0};
    config.gridDim = blocks_in_grid;
    config.blockDim = threads_per_block;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = blocks_per_cluster.x;
    attrs[0].val.clusterDim.y = blocks_per_cluster.y;
    attrs[0].val.clusterDim.z = blocks_per_cluster.z;

    config.attrs = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, my_kernel, data);
}
```

### Query Maximum Cluster Size

```cpp
cudaOccupancyMaxPotentialClusterSize(
    int* numClusters,
    int* maxClusterSize,
    void* kernel,
    int blockSize,
    size_t dynamicSMemSize
);
```

---

## Distributed Shared Memory Access

### Basic Access Pattern

```cpp
__global__ void __cluster_dims__(2, 2, 1)
distributed_smem_kernel(float* global_data) {
    // Regular shared memory for this block
    __shared__ float local_smem[256];

    auto cluster = cooperative_groups::this_cluster();
    auto block = cooperative_groups::this_thread_block();

    // Load data to local shared memory
    int tid = threadIdx.x;
    local_smem[tid] = global_data[blockIdx.x * 256 + tid];

    // Synchronize cluster so all blocks have loaded their data
    cluster.sync();

    // Access another block's shared memory
    unsigned int other_block_rank = 1; // Access block 1's smem

    float* other_smem = (float*)cluster.map_shared_rank(
        local_smem, other_block_rank);

    // Read from neighbor block
    float neighbor_value = other_smem[tid];

    // Process combined data
    float result = local_smem[tid] + neighbor_value;

    cluster.sync();

    global_data[blockIdx.x * 256 + tid] = result;
}
```

### Halo Exchange Pattern

```cpp
__global__ void __cluster_dims__(4, 1, 1)
halo_exchange_kernel(float* input, float* output, int width) {
    __shared__ float tile[258]; // 256 + 2 halo cells

    auto cluster = cooperative_groups::this_cluster();
    int block_rank = cluster.block_rank();
    int tid = threadIdx.x;

    // Load main data
    tile[tid + 1] = input[blockIdx.x * 256 + tid];

    cluster.sync();

    // Load left halo from previous block
    if (tid == 0 && block_rank > 0) {
        float* prev_smem = (float*)cluster.map_shared_rank(
            tile, block_rank - 1);
        tile[0] = prev_smem[256]; // Last element of previous block
    }

    // Load right halo from next block
    if (tid == 255 && block_rank < cluster.num_blocks() - 1) {
        float* next_smem = (float*)cluster.map_shared_rank(
            tile, block_rank + 1);
        tile[257] = next_smem[1]; // First element of next block
    }

    cluster.sync();

    // Stencil computation with halos
    float result = 0.25f * (tile[tid] + tile[tid + 1] +
                           tile[tid + 2] + tile[tid + 1]);

    output[blockIdx.x * 256 + tid] = result;
}
```

---

## Cooperative Groups API

### Cluster Information

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2, 2, 1) cluster_info_kernel() {
    auto cluster = cg::this_cluster();
    auto block = cg::this_thread_block();

    // Cluster dimensions
    dim3 cluster_dim = cluster.dim_blocks();    // (2, 2, 1)
    unsigned int cluster_size = cluster.num_blocks(); // 4

    // This block's position in cluster
    dim3 block_idx_in_cluster = cluster.block_index(); // (0-1, 0-1, 0)
    unsigned int block_rank = cluster.block_rank();    // 0-3

    // Thread position
    unsigned int thread_rank = cluster.thread_rank();
    unsigned int num_threads = cluster.num_threads(); // 4 * blockDim
}
```

### Cluster Synchronization

```cpp
__global__ void __cluster_dims__(2, 2, 1) sync_patterns() {
    auto cluster = cg::this_cluster();
    auto block = cg::this_thread_block();

    // Block-level sync (traditional)
    __syncthreads();
    // or
    block.sync();

    // Cluster-level sync (all blocks in cluster)
    cluster.sync();

    // Thread group sync
    auto tile = cg::tiled_partition<32>(block); // Warp
    tile.sync();
}
```

---

## Advanced Patterns

### Pattern 1: Cluster-Wide Reduction

```cpp
__global__ void __cluster_dims__(4, 1, 1)
cluster_reduce(float* input, float* output, int n) {
    __shared__ float block_sum;

    auto cluster = cooperative_groups::this_cluster();
    auto block = cooperative_groups::this_thread_block();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Thread-level reduction
    float thread_sum = 0.0f;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        thread_sum += input[i];
    }

    // 2. Block-level reduction
    thread_sum = block_reduce_sum(thread_sum);

    if (threadIdx.x == 0) {
        block_sum = thread_sum;
    }
    __syncthreads();

    // 3. Cluster-level reduction via distributed smem
    cluster.sync();

    float cluster_sum = 0.0f;
    if (threadIdx.x == 0) {
        for (int rank = 0; rank < cluster.num_blocks(); rank++) {
            float* other_block_sum = (float*)cluster.map_shared_rank(
                &block_sum, rank);
            cluster_sum += *other_block_sum;
        }
    }

    // Write result
    if (block.thread_rank() == 0 && cluster.block_rank() == 0) {
        output[blockIdx.x / cluster.num_blocks()] = cluster_sum;
    }
}
```

### Pattern 2: TMA Multicast with Clusters

```cpp
__global__ void __cluster_dims__(2, 2, 1)
tma_multicast_matmul(const __grid_constant__ CUtensorMap map_a,
                     float* c, int m, int n, int k) {
    __shared__ float smem_a[TILE_M][TILE_K];

    auto cluster = cooperative_groups::this_cluster();
    auto block = cooperative_groups::this_thread_block();

    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    if (threadIdx.x == 0) {
        init(&barrier, 1);
    }
    __syncthreads();

    // Block 0 broadcasts matrix tile to all blocks in cluster
    if (block.block_rank() == 0 && threadIdx.x == 0) {
        uint16_t mcast_mask = 0xFFFF; // All blocks
        cuda::memcpy_async(smem_a, map_a,
                          cuda::aligned_size_t<128>(sizeof(smem_a)),
                          barrier, mcast_mask);
    }

    // All blocks wait for broadcast
    cluster.sync();
    barrier.arrive_and_wait();

    // All blocks now have smem_a populated
    // Each block processes different output tiles
    matmul_compute(smem_a, c, block.block_rank());
}
```

### Pattern 3: Multi-Block Pipeline

```cpp
__global__ void __cluster_dims__(4, 1, 1)
pipeline_stages(float* input, float* output) {
    __shared__ float stage_data[256];

    auto cluster = cooperative_groups::this_cluster();
    int block_rank = cluster.block_rank();

    // Each block is a pipeline stage
    if (block_rank == 0) {
        // Stage 0: Load
        load_data(stage_data, input);
        cluster.sync();
    }
    else if (block_rank == 1) {
        // Stage 1: Transform
        cluster.sync();
        float* prev_data = (float*)cluster.map_shared_rank(
            stage_data, 0);
        transform_data(stage_data, prev_data);
        cluster.sync();
    }
    else if (block_rank == 2) {
        // Stage 2: Process
        cluster.sync();
        cluster.sync();
        float* prev_data = (float*)cluster.map_shared_rank(
            stage_data, 1);
        process_data(stage_data, prev_data);
        cluster.sync();
    }
    else if (block_rank == 3) {
        // Stage 3: Store
        cluster.sync();
        cluster.sync();
        cluster.sync();
        float* prev_data = (float*)cluster.map_shared_rank(
            stage_data, 2);
        store_data(output, prev_data);
    }
}
```

---

## Memory Access Optimization

### Coalesced Access for Distributed Shared Memory

Follow the same rules as global memory:
- **Align to 32-byte segments**
- **Maintain unit stride** when possible
- **128-byte alignment** for optimal performance

```cpp
// GOOD: Coalesced access
__global__ void __cluster_dims__(2, 1, 1) good_access() {
    __shared__ __align__(128) float data[256];

    auto cluster = cooperative_groups::this_cluster();

    // Aligned, stride-1 access
    int tid = threadIdx.x;
    data[tid] = tid; // Good pattern

    cluster.sync();

    // Access other block with same pattern
    float* other = (float*)cluster.map_shared_rank(data, 1);
    float value = other[tid]; // Still coalesced
}

// BAD: Strided access
__global__ void __cluster_dims__(2, 1, 1) bad_access() {
    __shared__ float data[256];

    auto cluster = cooperative_groups::this_cluster();

    // Strided access
    int tid = threadIdx.x;
    data[tid * 2] = tid; // Wastes bandwidth

    cluster.sync();

    float* other = (float*)cluster.map_shared_rank(data, 1);
    float value = other[tid * 2]; // Still strided, slow
}
```

---

## Performance Considerations

### Occupancy Impact

Larger clusters may reduce occupancy:

```cpp
// Check occupancy impact
int max_clusters;
int max_cluster_size;
cudaOccupancyMaxPotentialClusterSize(
    &max_clusters,
    &max_cluster_size,
    my_kernel,
    256,  // threads per block
    0     // dynamic smem
);

printf("Max cluster size: %d blocks\n", max_cluster_size);
```

### When to Use Clusters

**Good Use Cases:**
- Halo exchanges (stencil computations)
- Multi-block reductions
- Data sharing between related blocks
- Pipeline parallelism across blocks
- Collaborative algorithms (sorting, prefix sum across blocks)

**Avoid Clusters When:**
- Blocks don't need inter-communication
- Memory footprint per block is already high
- Occupancy is critical and clusters reduce it significantly

---

## Debugging

### Assertions for Cluster Kernels

```cpp
__global__ void __cluster_dims__(2, 2, 1) debug_kernel() {
    auto cluster = cooperative_groups::this_cluster();

    // Verify cluster configuration
    assert(cluster.dim_blocks().x == 2);
    assert(cluster.dim_blocks().y == 2);
    assert(cluster.num_blocks() == 4);

    // Verify block rank
    unsigned int rank = cluster.block_rank();
    assert(rank < 4);

    // Verify block index
    dim3 idx = cluster.block_index();
    assert(idx.x < 2 && idx.y < 2);
}
```

### Compute Sanitizer

```bash
# Check for cluster-related issues
compute-sanitizer --tool memcheck ./my_cluster_app
compute-sanitizer --tool racecheck ./my_cluster_app
```

---

## Comparison with Traditional Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Global Memory Communication** | Simple, works everywhere | Slow, high latency |
| **Atomic Operations** | Synchronization primitive | Serialization bottleneck |
| **Persistent Kernels** | Maximum flexibility | Complex, manual scheduling |
| **Thread Block Clusters** | Hardware-accelerated, fast | Hopper+ only, occupancy impact |

---

## Example: 2D Stencil with Clusters

```cpp
#define TILE_X 32
#define TILE_Y 32

__global__ void __cluster_dims__(2, 2, 1)
stencil_2d(float* input, float* output, int width, int height) {
    __shared__ float tile[TILE_Y + 2][TILE_X + 2]; // Include halo

    auto cluster = cooperative_groups::this_cluster();
    dim3 cluster_idx = cluster.block_index();

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global position
    int gx = blockIdx.x * TILE_X + tx;
    int gy = blockIdx.y * TILE_Y + ty;

    // Load center data
    if (gx < width && gy < height) {
        tile[ty + 1][tx + 1] = input[gy * width + gx];
    }

    cluster.sync();

    // Load halos from neighbor blocks in cluster
    if (tx == 0 && cluster_idx.x > 0) {
        // Left halo
        unsigned int left_rank = cluster.block_rank() - 1;
        float* left_tile = (float*)cluster.map_shared_rank(tile, left_rank);
        tile[ty + 1][0] = left_tile[ty + 1][TILE_X];
    }

    if (tx == TILE_X - 1 && cluster_idx.x < cluster.dim_blocks().x - 1) {
        // Right halo
        unsigned int right_rank = cluster.block_rank() + 1;
        float* right_tile = (float*)cluster.map_shared_rank(tile, right_rank);
        tile[ty + 1][TILE_X + 1] = right_tile[ty + 1][1];
    }

    // Similar for top/bottom halos...

    cluster.sync();

    // Apply stencil
    if (gx < width && gy < height) {
        float result = 0.25f * (
            tile[ty][tx + 1] +     // top
            tile[ty + 2][tx + 1] + // bottom
            tile[ty + 1][tx] +     // left
            tile[ty + 1][tx + 2]   // right
        );
        output[gy * width + gx] = result;
    }
}
```

---

## Further Reading

- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [Cooperative Groups Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
- [CUTLASS Cluster Examples](https://github.com/NVIDIA/cutlass)
