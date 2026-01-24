# AMD GPU Intrinsics and HipKittens Framework Guide

## Table of Contents
1. [AMD GPU Intrinsics Overview](#intrinsics-overview)
2. [__builtin_amdgcn_* Functions](#builtin-functions)
3. [Common Intrinsic Patterns](#common-patterns)
4. [HipKittens Framework](#hipkittens)
5. [Tile-Based Programming](#tile-programming)
6. [HipKittens vs Alternatives](#hipkittens-comparison)
7. [Practical Examples](#practical-examples)

---

## AMD GPU Intrinsics Overview {#intrinsics-overview}

AMD provides a rich set of compiler intrinsics for accessing GPU hardware features directly.

### Why Use Intrinsics?

```
┌──────────────────────────────────────────────────────────────┐
│          PROGRAMMING ABSTRACTION LAYERS                       │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Level 4: High-level libraries (rocBLAS, AITER)        │  │
│  │           • Easiest to use                              │  │
│  │           • Limited flexibility                         │  │
│  │           • May not expose all hardware features        │  │
│  └────────────────────────────────────────────────────────┘  │
│                        ↓                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Level 3: HIP C++ (Standard GPU programming)           │  │
│  │           • Good portability                            │  │
│  │           • Compiler handles optimization               │  │
│  │           • May miss hardware-specific optimizations    │  │
│  └────────────────────────────────────────────────────────┘  │
│                        ↓                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Level 2: Compiler intrinsics (__builtin_amdgcn_*)  ← │  │
│  │           • Direct hardware feature access              │  │
│  │           • Better optimization control                 │  │
│  │           • Compiler understands semantics              │  │
│  │           ✅ RECOMMENDED FOR PERFORMANCE               │  │
│  └────────────────────────────────────────────────────────┘  │
│                        ↓                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Level 1: Inline assembly                              │  │
│  │           • Maximum control                             │  │
│  │           • Brittle and error-prone                     │  │
│  │           • Manual hazard management                    │  │
│  │           ❌ NOT RECOMMENDED                            │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Intrinsic Categories

```
┌──────────────────────────────────────────────────────────────┐
│           AMD GPU INTRINSIC CATEGORIES                        │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  1. THREAD/WAVE INTRINSICS                                    │
│     • __builtin_amdgcn_workitem_id_*                          │
│     • __builtin_amdgcn_workgroup_id_*                         │
│     • __builtin_amdgcn_wave_id                                │
│     • __builtin_amdgcn_wave_barrier                           │
│                                                                │
│  2. MEMORY INTRINSICS                                         │
│     • __builtin_amdgcn_ds_*        (LDS operations)           │
│     • __builtin_amdgcn_global_*    (Global memory)            │
│     • __builtin_amdgcn_flat_*      (Flat addressing)          │
│     • __builtin_amdgcn_buffer_*    (Buffer operations)        │
│                                                                │
│  3. MATH INTRINSICS                                           │
│     • __builtin_amdgcn_fma_*       (Fused multiply-add)       │
│     • __builtin_amdgcn_rcp_*       (Reciprocal)               │
│     • __builtin_amdgcn_rsq_*       (Reciprocal sqrt)          │
│     • __builtin_amdgcn_sin_*       (Trigonometric)            │
│                                                                │
│  4. MATRIX INTRINSICS (CDNA)                                  │
│     • __builtin_amdgcn_mfma_*      (Matrix operations)        │
│     • __builtin_amdgcn_wmma_*      (Wave matrix - RDNA3+)     │
│                                                                │
│  5. WARP/LANE INTRINSICS                                      │
│     • __builtin_amdgcn_ds_bpermute (Cross-lane data)          │
│     • __builtin_amdgcn_ds_permute                             │
│     • __builtin_amdgcn_mov_dpp     (Data parallel primitives) │
│     • __builtin_amdgcn_update_dpp                             │
│                                                                │
│  6. ATOMIC INTRINSICS                                         │
│     • __builtin_amdgcn_ds_atomic_* (LDS atomics)              │
│     • __builtin_amdgcn_global_atomic_*                        │
│     • __builtin_amdgcn_flat_atomic_*                          │
│                                                                │
│  7. SYNCHRONIZATION                                           │
│     • __builtin_amdgcn_s_barrier                              │
│     • __builtin_amdgcn_fence                                  │
│     • __builtin_amdgcn_s_waitcnt                              │
└──────────────────────────────────────────────────────────────┘
```

---

## __builtin_amdgcn_* Functions {#builtin-functions}

### Thread and Wave Identification

```cpp
// Get thread/workitem IDs within workgroup
__device__ int get_thread_x() {
    return __builtin_amdgcn_workitem_id_x();  // 0 to workgroup_size_x-1
}

__device__ int get_thread_y() {
    return __builtin_amdgcn_workitem_id_y();
}

__device__ int get_thread_z() {
    return __builtin_amdgcn_workitem_id_z();
}

// Get workgroup/block IDs
__device__ int get_block_x() {
    return __builtin_amdgcn_workgroup_id_x();
}

__device__ int get_block_y() {
    return __builtin_amdgcn_workgroup_id_y();
}

__device__ int get_block_z() {
    return __builtin_amdgcn_workgroup_id_z();
}

// Useful derived quantities
__device__ int get_global_id() {
    int local_id = __builtin_amdgcn_workitem_id_x();
    int group_id = __builtin_amdgcn_workgroup_id_x();
    int group_size = __builtin_amdgcn_workgroup_size_x();
    return group_id * group_size + local_id;
}

// Lane ID within wavefront (0-63 for CDNA, 0-31 for RDNA3)
__device__ int get_lane_id() {
    return __builtin_amdgcn_workitem_id_x() % __AMDGCN_WAVEFRONT_SIZE;
}
```

### Wave-Level Operations

```cpp
// Wave barrier (synchronizes all threads in wavefront)
__device__ void wave_barrier() {
    __builtin_amdgcn_wave_barrier();
}

// Read execution mask (64-bit for Wave64, 32-bit for Wave32)
__device__ uint64_t read_exec_mask() {
    return __builtin_amdgcn_read_exec();
}

// Ballot: Get mask of lanes where condition is true
__device__ uint64_t wave_ballot(int predicate) {
    return __builtin_amdgcn_ballot_w64(predicate);
}

// Count active lanes
__device__ int wave_active_count_one(int predicate) {
    return __builtin_popcountll(wave_ballot(predicate));
}

// Example: Wave-level reduction
__device__ float wave_reduce_sum(float value) {
    // Use DPP (Data Parallel Primitives) for efficient reductions
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        // Row shift (shift within 64-element wave)
        value += __builtin_amdgcn_mov_dpp(
            value,
            0x111 + offset,  // DPP control
            15,              // Row mask
            15,              // Bank mask
            false            // Bound control
        );
    }
    return value;
}
```

### LDS (Local Data Share) Intrinsics

```cpp
// LDS read (32-bit)
__device__ float lds_read_f32(unsigned int offset) {
    return __builtin_amdgcn_ds_read_b32(
        (__attribute__((address_space(3))) int*)(offset)
    );
}

// LDS write (32-bit)
__device__ void lds_write_f32(unsigned int offset, float value) {
    __builtin_amdgcn_ds_write_b32(
        (__attribute__((address_space(3))) int*)(offset),
        __builtin_bit_cast(int, value)
    );
}

// LDS atomic add
__device__ float lds_atomic_add_f32(unsigned int offset, float value) {
    return __builtin_amdgcn_ds_fadd_f32(
        (__attribute__((address_space(3))) float*)(offset),
        value
    );
}

// LDS permute operations (cross-lane communication via LDS hardware)
__device__ int ds_permute(int index, int src) {
    // Read from lane 'index'
    return __builtin_amdgcn_ds_bpermute(index << 2, src);
}

__device__ int ds_bpermute(int index, int src) {
    // Write to lane 'index'
    return __builtin_amdgcn_ds_permute(index << 2, src);
}

// Example: Efficient transpose using permute
__device__ void warp_transpose_4x4(float data[4]) {
    // Transpose a 4x4 matrix distributed across wavefront
    int lane = get_lane_id();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int dest_lane = (lane & ~3) | i;  // Target lane
        float temp = __builtin_amdgcn_ds_bpermute(
            dest_lane << 2,
            __builtin_bit_cast(int, data[i])
        );
        data[i] = __builtin_bit_cast(float, temp);
    }
}
```

### Fast Math Intrinsics

```cpp
// Fast reciprocal (1/x)
__device__ float fast_rcp(float x) {
    return __builtin_amdgcn_rcpf(x);  // ~4 cycles
}

// Fast reciprocal square root (1/sqrt(x))
__device__ float fast_rsqrt(float x) {
    return __builtin_amdgcn_rsqf(x);  // ~4 cycles
}

// Fast division
__device__ float fast_div(float a, float b) {
    return a * __builtin_amdgcn_rcpf(b);
}

// Fast square root
__device__ float fast_sqrt(float x) {
    return x * __builtin_amdgcn_rsqf(x);
}

// Fused multiply-add (single instruction, higher precision)
__device__ float fma(float a, float b, float c) {
    return __builtin_amdgcn_fmaf(a, b, c);  // a*b + c
}

// Fast trigonometric functions
__device__ float fast_sin(float x) {
    return __builtin_amdgcn_sinf(x);
}

__device__ float fast_cos(float x) {
    return __builtin_amdgcn_cosf(x);
}

// Example: Softmax with fast intrinsics
__device__ void fast_softmax(float* x, int N) {
    // Find max
    float max_val = x[0];
    for (int i = 1; i < N; i++) {
        max_val = fmaxf(max_val, x[i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        x[i] = __builtin_amdgcn_expf(x[i] - max_val);  // Fast exp
        sum += x[i];
    }

    // Normalize using fast reciprocal
    float inv_sum = __builtin_amdgcn_rcpf(sum);
    for (int i = 0; i < N; i++) {
        x[i] *= inv_sum;
    }
}
```

### MFMA Intrinsics (Comprehensive)

```cpp
// MFMA intrinsic template (reminder from assembly guide)
// d = __builtin_amdgcn_mfma_CDFmt_MxNxKABFmt(a, b, c, cbsz, abid, blgp)

// Common MFMA operations:

// 1. FP32 16×16×4 (most common)
using float4 = __attribute__((ext_vector_type(4))) float;
using float16 = __attribute__((ext_vector_type(16))) float;

__device__ float16 mfma_f32_16x16x4(
    float4 a, float4 b, float16 c
) {
    return __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
}

// 2. FP16 input → FP32 accumulator (higher throughput)
using half8 = __attribute__((ext_vector_type(8))) __fp16;

__device__ float16 mfma_f32_32x32x8_fp16(
    half8 a, half8 b, float16 c
) {
    return __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);
}

// 3. BF16 input (good for ML)
using bf16_8 = __attribute__((ext_vector_type(8))) __bf16;

__device__ float16 mfma_f32_32x32x8_bf16(
    bf16_8 a, bf16_8 b, float16 c
) {
    return __builtin_amdgcn_mfma_f32_32x32x8bf16(a, b, c, 0, 0, 0);
}

// 4. INT8 (quantized models)
using int4 = __attribute__((ext_vector_type(4))) int;
using int16 = __attribute__((ext_vector_type(16))) int;

__device__ int16 mfma_i32_16x16x16_i8(
    int4 a_packed, int4 b_packed, int16 c
) {
    return __builtin_amdgcn_mfma_i32_16x16x16i8(a_packed, b_packed, c, 0, 0, 0);
}

// 5. FP64 (scientific computing)
using double4 = __attribute__((ext_vector_type(4))) double;

__device__ double4 mfma_f64_16x16x4(
    double a, double b, double4 c
) {
    return __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0);
}

// Example: Small GEMM using MFMA
__global__ void gemm_mfma_16x16(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int K
) {
    float16 acc = {0};  // Accumulator (16 elements)

    // Tile over K dimension
    for (int k = 0; k < K; k += 4) {
        // Load A and B tiles (4 elements each)
        float4 a_tile = load_float4(&A[k]);
        float4 b_tile = load_float4(&B[k]);

        // MFMA: acc += A @ B
        acc = __builtin_amdgcn_mfma_f32_16x16x4f32(
            a_tile, b_tile, acc, 0, 0, 0
        );
    }

    // Store results (16 elements from accumulator)
    store_float16(C, acc);
}
```

### Synchronization and Memory Fence

```cpp
// Workgroup barrier (all threads in block)
__device__ void block_barrier() {
    __builtin_amdgcn_s_barrier();
}

// Memory fence (ensure memory visibility)
__device__ void memory_fence_global() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
}

__device__ void memory_fence_lds() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
}

// Wait for memory operations to complete
__device__ void wait_vmcnt(int count) {
    // Wait for vector memory operations
    __builtin_amdgcn_s_waitcnt(count);  // vmcnt
}

__device__ void wait_lgkmcnt(int count) {
    // Wait for LDS/GDS/scalar memory operations
    __builtin_amdgcn_s_waitcnt(count << 8);  // lgkmcnt
}

__device__ void wait_all_memory() {
    __builtin_amdgcn_s_waitcnt(0);  // Wait for everything
}

// Example: Producer-consumer pattern with proper synchronization
__global__ void producer_consumer_kernel() {
    __shared__ float buffer[1024];

    int tid = threadIdx.x;

    // Producer phase
    if (tid < 512) {
        buffer[tid] = compute_value(tid);
        lds_write_f32(tid * 4, buffer[tid]);
    }

    // Synchronize: Ensure all producers finish writing
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_s_waitcnt(0);  // Wait for LDS writes

    // Consumer phase
    if (tid >= 512) {
        int read_idx = tid - 512;
        float value = lds_read_f32(read_idx * 4);
        process_value(value);
    }
}
```

---

## Common Intrinsic Patterns {#common-patterns}

### Pattern 1: Warp Shuffle Reduction

```cpp
// Efficient reduction using DPP (Data Parallel Primitives)
__device__ float warp_reduce_sum_dpp(float value) {
    // DPP allows efficient cross-lane communication
    // without going through LDS

    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        float other = __builtin_amdgcn_mov_dpp(
            value,
            0x100 + offset,  // row_shr:offset
            0xf,             // row_mask (all rows)
            0xf,             // bank_mask (all banks)
            false            // bound_ctrl
        );
        value += other;
    }
    return value;  // Lane 0 has sum
}

// Usage in block reduction
__device__ float block_reduce_sum(float value) {
    __shared__ float shared[32];  // One per warp

    int lane = get_lane_id();
    int warp = threadIdx.x / 64;

    // Reduce within warp
    float warp_sum = warp_reduce_sum_dpp(value);

    // First thread in warp writes to shared memory
    if (lane == 0) {
        shared[warp] = warp_sum;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp == 0 && lane < (blockDim.x / 64)) {
        value = shared[lane];
        value = warp_reduce_sum_dpp(value);
    }

    return value;  // Thread 0 has result
}
```

### Pattern 2: Vectorized Memory Access

```cpp
// Load 128-bit aligned vector (4 floats)
__device__ float4 vectorized_load(const float* addr) {
    // Cast to vector pointer
    using float4 = __attribute__((ext_vector_type(4))) float;
    const float4* vec_addr = reinterpret_cast<const float4*>(addr);
    return *vec_addr;
}

// Store 128-bit aligned vector
__device__ void vectorized_store(float* addr, float4 value) {
    using float4 = __attribute__((ext_vector_type(4))) float;
    float4* vec_addr = reinterpret_cast<float4*>(addr);
    *vec_addr = value;
}

// Example: Vectorized copy
__global__ void vectorized_copy(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx + 3 < N) {
        float4 data = vectorized_load(&src[vec_idx]);
        vectorized_store(&dst[vec_idx], data);
    }
}
```

### Pattern 3: Software Pipelining with Intrinsics

```cpp
__global__ void pipelined_gemm_kernel(
    float* C,
    const float* A,
    const float* B,
    int M, int N, int K
) {
    // Allocate registers for pipelining
    float4 a_reg[2];  // Double buffer
    float4 b_reg[2];
    float16 acc = {0};

    int buf = 0;  // Current buffer

    // Prologue: Issue first load
    a_reg[buf] = vectorized_load(&A[0]);
    b_reg[buf] = vectorized_load(&B[0]);

    // Main loop with software pipelining
    for (int k = 0; k < K; k += 4) {
        int next_buf = 1 - buf;

        // Issue next load (overlaps with compute)
        if (k + 4 < K) {
            a_reg[next_buf] = vectorized_load(&A[k + 4]);
            b_reg[next_buf] = vectorized_load(&B[k + 4]);
        }

        // Compute using current buffer
        acc = __builtin_amdgcn_mfma_f32_16x16x4f32(
            a_reg[buf], b_reg[buf], acc, 0, 0, 0
        );

        // Swap buffers
        buf = next_buf;
    }

    // Store result
    store_float16(C, acc);
}
```

---

## HipKittens Framework {#hipkittens}

### Overview

HipKittens (HK) is a modern C++ framework for writing high-performance AMD GPU kernels using tile-based abstractions, developed by Stanford's Hazy Research lab.

```
┌──────────────────────────────────────────────────────────────┐
│               HIPKITTENS ARCHITECTURE                         │
│                                                                │
│  Problem: Writing fast AMD kernels is hard                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  ❌ Hand-optimized assembly: Brittle, requires experts │  │
│  │  ❌ PyTorch/libraries: Underperform, limited coverage  │  │
│  │  ❌ Triton/compilers: Portable but sacrifice 20-40%    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Solution: HipKittens                                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  ✅ C++ embedded DSL (Domain Specific Language)        │  │
│  │  ✅ Tile-based abstractions (like ThunderKittens)      │  │
│  │  ✅ Hardware-aware scheduling                           │  │
│  │  ✅ Matches or beats AITER assembly performance        │  │
│  │  ✅ <500 lines of code for complex kernels             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Key Principles:                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  1. Tile abstraction (proven on NVIDIA, works on AMD)  │  │
│  │  2. Architecture-specific backends (AMD ≠ NVIDIA)      │  │
│  │  3. Hardware-aware scheduling (wave-level reasoning)   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Performance Results

```
┌──────────────────────────────────────────────────────────────┐
│          HIPKITTENS PERFORMANCE BENCHMARKS                    │
│                    (AMD MI250X/MI300)                         │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Attention Kernels (Flash Attention):                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  HipKittens:  ████████████████████████████ 95% peak   │  │
│  │  AITER (asm): ████████████████████████   85% peak     │  │
│  │  Triton:      ██████████████            60% peak       │  │
│  │  PyTorch:     ████████                  35% peak       │  │
│  │                                                          │  │
│  │  Code size:   ~500 lines vs 2000+ lines assembly       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  GEMM Kernels:                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  HipKittens:   ███████████████████████████ 92% peak   │  │
│  │  HipBLASLT:    ████████████████████████   80% peak    │  │
│  │  AITER (asm):  ███████████████████████    83% peak    │  │
│  │                                                          │  │
│  │  Code size:    <100 lines core loop                    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Additional Kernels:                                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  • Backward attention: Beats AITER (30% → 70% peak)    │  │
│  │  • Rotary embeddings: 90%+ peak                         │  │
│  │  • Fused dropout-residual-layernorm: 85%+ peak         │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Installation and Setup

```bash
# Clone HipKittens repository
git clone https://github.com/HazyResearch/HipKittens.git
cd HipKittens

# Install dependencies
# Requires: ROCm 6.0+, HIP compiler, CMake 3.20+

# Build examples
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run tests
./tests/test_attention
./tests/test_gemm

# Explore examples
cd ../examples
# Check out: flash_attention.cu, gemm.cu, layernorm.cu
```

---

## Tile-Based Programming {#tile-programming}

### HipKittens Tile Abstraction

```cpp
#include <hip/hip_runtime.h>
#include <hipkittens/hipkittens.hpp>

namespace hk = hipkittens;

/*
 * HipKittens Tile Types:
 *
 * hk::tile<DataType, Height, Width>
 *   - Represents a 2D tile of data
 *   - Lives in registers or shared memory
 *   - DataType: float, __fp16, __bf16, int8_t, etc.
 *   - Height, Width: Compile-time dimensions
 */

// Example: 16×16 tile of FP16 data
using tile_16x16_fp16 = hk::tile<__fp16, 16, 16>;

// Example: 32×32 tile of FP32 data
using tile_32x32_fp32 = hk::tile<float, 32, 32>;

/*
 * Tile Operations:
 *   - load(tile, global_ptr, stride)    : Load from global memory
 *   - store(global_ptr, tile, stride)   : Store to global memory
 *   - gemm(C, A, B)                      : Matrix multiply C = A @ B
 *   - add(C, A, B)                       : Element-wise C = A + B
 *   - mul(C, A, B)                       : Element-wise C = A * B
 *   - softmax(tile)                      : In-place softmax
 *   - layernorm(tile, gamma, beta)       : Layer normalization
 */
```

### HipKittens Flash Attention Example

```cpp
#include <hipkittens/hipkittens.hpp>

namespace hk = hipkittens;

// Flash Attention kernel using HipKittens
__global__ void flash_attention_hipkittens(
    float* output,           // [B, N, D]
    const __fp16* Q,         // [B, N, D]
    const __fp16* K,         // [B, N, D]
    const __fp16* V,         // [B, N, D]
    int B, int N, int D
) {
    // Tile dimensions
    constexpr int Br = 64;  // Query tile rows
    constexpr int Bc = 64;  // KV tile columns
    constexpr int d = 64;   // Head dimension

    // Define tile types
    using Q_tile = hk::tile<__fp16, Br, d>;
    using K_tile = hk::tile<__fp16, d, Bc>;  // Transposed
    using V_tile = hk::tile<__fp16, Bc, d>;
    using S_tile = hk::tile<float, Br, Bc>;  // Attention scores
    using O_tile = hk::tile<float, Br, d>;   // Output accumulator

    // Allocate tiles (automatically placed in registers/LDS)
    Q_tile q_tile;
    K_tile k_tile;
    V_tile v_tile;
    S_tile s_tile;
    O_tile o_tile;

    // Running statistics
    hk::vector<float, Br> max_vec;  // Per-row max
    hk::vector<float, Br> sum_vec;  // Per-row sum

    // Initialize
    hk::fill(o_tile, 0.0f);
    hk::fill(max_vec, -INFINITY);
    hk::fill(sum_vec, 0.0f);

    // Calculate block indices
    int batch_idx = blockIdx.z;
    int q_block = blockIdx.x;

    // Load Q tile (stays in registers throughout)
    hk::load(q_tile, Q + batch_idx * N * D + q_block * Br * D, D);

    // Iterate over KV tiles
    int num_kv_blocks = (N + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        // Load K and V tiles
        hk::load(k_tile, K + batch_idx * N * D + kv_block * Bc * D, D);
        hk::load(v_tile, V + batch_idx * N * D + kv_block * Bc * D, D);

        // Compute attention scores: S = Q @ K^T
        hk::gemm(s_tile, q_tile, k_tile);  // Hardware MFMA used internally

        // Apply scaling
        hk::mul(s_tile, s_tile, 1.0f / sqrtf(d));

        // Online softmax update
        hk::rowmax(max_vec, s_tile);       // Find row max
        hk::softmax_online_update(
            o_tile,      // Accumulator (rescaled)
            s_tile,      // Current scores
            max_vec,     // Running max
            sum_vec      // Running sum
        );

        // Accumulate: O += P @ V (P = softmax(S))
        hk::gemm_accumulate(o_tile, s_tile, v_tile);
    }

    // Final normalization
    hk::normalize_rows(o_tile, sum_vec);

    // Store output
    hk::store(output + batch_idx * N * D + q_block * Br * D, o_tile, D);
}

/*
 * Key advantages:
 *   1. Automatic memory management (register/LDS allocation)
 *   2. Hardware-aware operator fusion
 *   3. Optimal MFMA instruction usage
 *   4. Clean, readable code (~100 lines vs 1000+ assembly)
 *   5. Performance matches or beats hand-written assembly
 */
```

### HipKittens GEMM Example

```cpp
#include <hipkittens/hipkittens.hpp>

namespace hk = hipkittens;

// High-performance GEMM using HipKittens
// C = A @ B (all FP16)
__global__ void gemm_hipkittens(
    __fp16* C,              // [M, N]
    const __fp16* A,        // [M, K]
    const __fp16* B,        // [K, N]
    int M, int N, int K
) {
    // Tile sizes (optimized for MI300)
    constexpr int BM = 128;  // M-dimension tile
    constexpr int BN = 128;  // N-dimension tile
    constexpr int BK = 32;   // K-dimension tile

    // Define tile types
    using A_tile = hk::tile<__fp16, BM, BK>;
    using B_tile = hk::tile<__fp16, BK, BN>;
    using C_tile = hk::tile<float, BM, BN>;  // FP32 accumulator

    // Allocate tiles
    A_tile a_tile;
    B_tile b_tile;
    C_tile c_tile;

    // Initialize accumulator
    hk::fill(c_tile, 0.0f);

    // Calculate output tile position
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;

    // Main GEMM loop (tile over K dimension)
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // Load A and B tiles from global memory
        hk::load(a_tile, A + block_m * BM * K + k_tile, K);
        hk::load(b_tile, B + k_tile * N + block_n * BN, N);

        // Matrix multiplication: C += A @ B
        // Internally uses optimal sequence of MFMA instructions
        hk::gemm_accumulate(c_tile, a_tile, b_tile);
    }

    // Convert FP32 accumulator to FP16 and store
    hk::tile<__fp16, BM, BN> c_tile_fp16;
    hk::convert(c_tile_fp16, c_tile);
    hk::store(C + block_m * BM * N + block_n * BN, c_tile_fp16, N);
}

// Kernel launch
void launch_gemm_hipkittens(
    __fp16* C, const __fp16* A, const __fp16* B,
    int M, int N, int K
) {
    constexpr int BM = 128;
    constexpr int BN = 128;

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(256);  // HipKittens auto-tunes thread count

    hipLaunchKernelGGL(
        gemm_hipkittens,
        grid, block, 0, 0,
        C, A, B, M, N, K
    );
}
```

### HipKittens Fused Operations

```cpp
// Fused LayerNorm + Residual + Dropout
__global__ void fused_layernorm_residual_dropout_hk(
    float* output,
    const float* input,
    const float* residual,
    const float* gamma,
    const float* beta,
    float dropout_prob,
    int N, int D
) {
    using vec_tile = hk::tile<float, 1, 128>;  // 1D tile

    vec_tile input_tile, residual_tile, output_tile;
    vec_tile gamma_tile, beta_tile;

    int idx = blockIdx.x;

    // Load data
    hk::load(input_tile, input + idx * D, D);
    hk::load(residual_tile, residual + idx * D, D);
    hk::load(gamma_tile, gamma, D);
    hk::load(beta_tile, beta, D);

    // Fused operations (single pass through data!)
    hk::layernorm(output_tile, input_tile, gamma_tile, beta_tile);
    hk::add(output_tile, output_tile, residual_tile);  // Residual
    hk::dropout(output_tile, output_tile, dropout_prob);  // Dropout

    // Store result
    hk::store(output + idx * D, output_tile, D);
}

// Performance: 3-4x faster than unfused (eliminates HBM round-trips)
```

---

## HipKittens vs Alternatives {#hipkittens-comparison}

```
┌────────────────────────────────────────────────────────────────┐
│         COMPARISON: HIPKITTENS VS OTHER APPROACHES             │
├──────────────┬─────────┬────────┬───────────┬────────┬────────┤
│ Approach     │ Perf    │ Code   │ Portability│Expertise│Maintain│
├──────────────┼─────────┼────────┼───────────┼────────┼────────┤
│ HipKittens   │ 90-95%  │ ~100 L │ AMD only  │ Medium │ Easy   │
│              │         │        │           │        │        │
│ AITER (asm)  │ 80-90%  │ ~2000L │ AMD only  │ Expert │ Hard   │
│              │         │        │           │        │        │
│ Triton       │ 60-75%  │ ~50 L  │ Multi-GPU │ Low    │ Easy   │
│              │         │        │           │        │        │
│ CK (Compose) │ 75-85%  │ ~500 L │ AMD only  │ High   │ Medium │
│              │         │        │           │        │        │
│ PyTorch      │ 30-50%  │ N/A    │ Multi-GPU │ Low    │ N/A    │
│              │         │        │           │        │        │
│ HipBLASLT    │ 80-85%  │ N/A    │ AMD only  │ Low    │ N/A    │
│ (library)    │         │        │           │        │        │
└──────────────┴─────────┴────────┴───────────┴────────┴────────┘

Legend:
  Perf: % of peak hardware performance
  Code: Lines of code for typical kernel
  Expertise: Required programming skill level
  Maintain: Code maintenance burden
```

---

## Practical Examples {#practical-examples}

### Example 1: Vectorized Copy with Intrinsics

```cpp
__global__ void optimized_copy_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int N
) {
    int idx = __builtin_amdgcn_workitem_id_x() +
              __builtin_amdgcn_workgroup_id_x() *
              __builtin_amdgcn_workgroup_size_x();

    // Vectorized copy (4 floats at a time)
    int vec_idx = idx * 4;

    if (vec_idx + 3 < N) {
        using float4 = __attribute__((ext_vector_type(4))) float;

        const float4* src_vec = (const float4*)(&src[vec_idx]);
        float4* dst_vec = (float4*)(&dst[vec_idx]);

        *dst_vec = *src_vec;
    }
}
```

### Example 2: Reduction with Wave Intrinsics

```cpp
__global__ void sum_reduction_intrinsics(
    float* output,
    const float* input,
    int N
) {
    __shared__ float smem[32];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int lane = tid % 64;
    int warp = tid / 64;

    // Load and initial reduction
    float value = (gid < N) ? input[gid] : 0.0f;

    // Wave-level reduction using DPP
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        float other = __builtin_amdgcn_mov_dpp(
            value, 0x100 + offset, 0xf, 0xf, false
        );
        value += other;
    }

    // First lane writes to shared memory
    if (lane == 0) {
        smem[warp] = value;
    }
    __builtin_amdgcn_s_barrier();

    // Final reduction by first warp
    if (warp == 0 && lane < (blockDim.x / 64)) {
        value = smem[lane];

        // Reduce across warps
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __builtin_amdgcn_mov_dpp(
                value, 0x100 + offset, 0xf, 0xf, false
            );
            value += other;
        }

        if (lane == 0) {
            atomicAdd(output, value);
        }
    }
}
```

### Example 3: Matrix Transpose with LDS

```cpp
__global__ void transpose_lds_intrinsics(
    float* output,       // [N, M]
    const float* input,  // [M, N]
    int M, int N
) {
    constexpr int TILE = 32;

    // LDS tile with padding to avoid bank conflicts
    __shared__ float tile[TILE][TILE + 1];

    int bx = blockIdx.x * TILE;
    int by = blockIdx.y * TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load tile from global memory (coalesced)
    int in_idx = (by + ty) * N + (bx + tx);
    if ((by + ty) < M && (bx + tx) < N) {
        tile[ty][tx] = input[in_idx];
    }

    __builtin_amdgcn_s_barrier();

    // Write transposed tile to global memory (coalesced)
    int out_idx = (bx + ty) * M + (by + tx);
    if ((bx + ty) < N && (by + tx) < M) {
        output[out_idx] = tile[tx][ty];
    }
}
```

---

## Summary and Best Practices

### When to Use Each Approach

```
┌──────────────────────────────────────────────────────────────┐
│              DECISION TREE                                    │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Need maximum performance on AMD?                             │
│  ├─ YES → Consider HipKittens (90-95% peak, clean code)      │
│  │   ├─ Kernel complexity high? → HipKittens                 │
│  │   └─ Kernel complexity low? → Intrinsics directly         │
│  │                                                             │
│  └─ NO → Other options:                                       │
│      ├─ Need portability? → Triton or PyTorch                │
│      ├─ Standard operation? → Use AITER or rocBLAS           │
│      └─ Learning/research? → Start with HipKittens           │
│                                                                │
│  Expertise level:                                             │
│  ├─ Beginner: PyTorch, Triton                                │
│  ├─ Intermediate: HipKittens, Intrinsics                     │
│  └─ Expert: Assembly (only if really necessary)              │
└──────────────────────────────────────────────────────────────┘
```

### Key Recommendations

```
✅ DO:
  • Use HipKittens for complex kernels (attention, GEMM, etc.)
  • Use intrinsics for fine-grained control within kernels
  • Profile actual hardware performance
  • Start with HipKittens examples and modify
  • Leverage MFMA/WMMA instructions for matrix ops
  • Use vectorized loads/stores (float4, etc.)
  • Employ wave-level primitives (DPP, permute)

❌ DON'T:
  • Write inline assembly unless absolutely necessary
  • Assume code will be portable across architectures
  • Ignore bank conflicts in LDS access
  • Forget memory synchronization (__syncthreads, waitcnt)
  • Overlook alignment requirements for vectorization
  • Reinvent the wheel (check AITER/rocBLAS first)
```

### Learning Resources

```
Official Documentation:
  • AMD ROCm Docs: https://rocm.docs.amd.com
  • HipKittens GitHub: https://github.com/HazyResearch/HipKittens
  • AMD Matrix Instruction Calculator

Research Papers:
  • HipKittens: arXiv 2511.08083 (2025)
  • Flash Attention: arXiv 2205.14135 (2022)
  • Flash Attention-2: arXiv 2307.08691 (2023)

Community:
  • ROCm GitHub Discussions
  • AMD GPU Open Forums
  • HipKittens Issues/Discussions
```

---

**Congratulations!** You now have comprehensive knowledge of HIP kernel optimization, from fundamentals through assembly-level programming, ML-specific techniques, kernel fusion, and modern frameworks like HipKittens. These guides provide the foundation for writing world-class AMD GPU kernels.
