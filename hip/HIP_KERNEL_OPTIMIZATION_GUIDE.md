# HIP Kernel Optimization Guide: From Fundamentals to Advanced Techniques

## Table of Contents
1. [Introduction to HIP Architecture](#introduction)
2. [Memory Hierarchy and Optimization](#memory-hierarchy)
3. [Register Optimization](#register-optimization)
4. [LDS (Local Data Share) Optimization](#lds-optimization)
5. [Memory Coalescing Patterns](#memory-coalescing)
6. [Instruction-Level Optimization](#instruction-level)
7. [Occupancy and Wave Management](#occupancy)
8. [Advanced Performance Patterns](#advanced-patterns)

---

## Introduction to HIP Architecture {#introduction}

HIP (Heterogeneous Interface for Portability) is AMD's programming interface for GPU computing, compatible with both AMD ROCm and NVIDIA CUDA platforms.

### GPU Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      AMD GPU ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │  Compute Unit │  │  Compute Unit │  │  Compute Unit │  ...  │
│  ├───────────────┤  ├───────────────┤  ├───────────────┤       │
│  │ • VALU x4     │  │ • VALU x4     │  │ • VALU x4     │       │
│  │ • SALU        │  │ • SALU        │  │ • SALU        │       │
│  │ • LDS (64KB)  │  │ • LDS (64KB)  │  │ • LDS (64KB)  │       │
│  │ • VGPR (256KB)│  │ • VGPR (256KB)│  │ • VGPR (256KB)│       │
│  │ • SGPR        │  │ • SGPR        │  │ • SGPR        │       │
│  │ • Matrix Core │  │ • Matrix Core │  │ • Matrix Core │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
│         │                   │                   │                │
│         └───────────────────┴───────────────────┘                │
│                             │                                     │
│                    ┌────────▼────────┐                           │
│                    │   L2 Cache      │                           │
│                    │   (8-32 MB)     │                           │
│                    └────────┬────────┘                           │
│                             │                                     │
│                    ┌────────▼────────┐                           │
│                    │  HBM Memory     │                           │
│                    │  (32-128 GB)    │                           │
│                    └─────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### Execution Model: Wavefronts

AMD GPUs execute work in **wavefronts** (analogous to CUDA warps):

```
CDNA Architecture:         RDNA Architecture:
┌──────────────┐          ┌──────────────┐
│  Wavefront   │          │  Wavefront   │
│  64 threads  │          │  32 threads  │
│  in lockstep │          │  in lockstep │
└──────────────┘          └──────────────┘
     │                         │
     └─────┬───────┬─────┬────┴─── ... (64 lanes)
     Lane0 Lane1   Lane2  Lane3
```

**Key Characteristics:**
- CDNA (MI100/MI200/MI300): 64-wide wavefronts
- RDNA (Radeon 6000/7000): 32-wide wavefronts (Wave32 mode)
- All threads execute same instruction (SIMT model)
- Divergent branches reduce efficiency

---

## Memory Hierarchy and Optimization {#memory-hierarchy}

### Memory Performance Hierarchy

```
┌────────────────────────────────────────────────────────────────┐
│                    MEMORY HIERARCHY                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Registers (VGPR/SGPR)                                     │  │
│  │ • Latency: ~0 cycles                                      │  │
│  │ • Bandwidth: Unlimited                                    │  │
│  │ • Size: ~2KB per thread                                   │  │
│  │ • Scope: Per-thread (VGPR) / Per-wavefront (SGPR)        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ↓ 10-100x slower                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Local Data Share (LDS) - Shared Memory                   │  │
│  │ • Latency: ~30-40 cycles                                 │  │
│  │ • Bandwidth: ~10 TB/s (per CU)                           │  │
│  │ • Size: 64KB per CU                                      │  │
│  │ • Scope: Workgroup (shared among threads)                │  │
│  │ • Banks: 32 banks × 4 bytes                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ↓ 5-10x slower                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ L1 Cache                                                  │  │
│  │ • Latency: ~80 cycles                                     │  │
│  │ • Size: 16KB per CU                                       │  │
│  │ • Scope: Compute Unit                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ↓ 2-3x slower                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ L2 Cache                                                  │  │
│  │ • Latency: ~200-300 cycles                                │  │
│  │ • Size: 8-32 MB (chip-wide)                               │  │
│  │ • Scope: Global (all CUs)                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ↓ 3-5x slower                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ HBM (High Bandwidth Memory)                               │  │
│  │ • Latency: ~450-550 cycles                                │  │
│  │ • Bandwidth: 1.6-5.3 TB/s (chip-wide)                     │  │
│  │ • Size: 32-192 GB                                         │  │
│  │ • Scope: Device                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Memory Access Patterns - Good vs Bad

```cpp
// ❌ BAD: Strided access (non-coalesced)
__global__ void bad_copy(float* out, const float* in, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in[idx * stride];  // Non-contiguous access
}

// ✅ GOOD: Sequential access (coalesced)
__global__ void good_copy(float* out, const float* in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in[idx];  // Contiguous access
}
```

**Coalescing Visualization:**

```
Thread Access Pattern (Sequential - COALESCED):
Memory: [0][1][2][3][4][5][6][7][8][9][10][11]...
         ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
Thread:  T0 T1 T2 T3 T4 T5 T6 T7
Result: 1 memory transaction (128 bytes)

Thread Access Pattern (Strided - NON-COALESCED):
Memory: [0]...[8]...[16]...[24]...[32]...
         ↑     ↑      ↑      ↑      ↑
Thread:  T0    T1     T2     T3     T4
Result: 8 memory transactions (wasted bandwidth)
```

---

## Register Optimization {#register-optimization}

### Register Types and Allocation

AMD GPUs have two primary register types:

```
┌─────────────────────────────────────────────────────────┐
│              REGISTER ARCHITECTURE                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  VGPR (Vector General Purpose Registers)                │
│  ┌────────────────────────────────────┐                 │
│  │ • Per-thread registers             │                 │
│  │ • 256 registers × 32 bits          │                 │
│  │ • Total: 256 KB per CU             │                 │
│  │ • Allocation: 4-register groups    │                 │
│  │ • Formula: (count - 1) / 4         │                 │
│  └────────────────────────────────────┘                 │
│                                                          │
│  SGPR (Scalar General Purpose Registers)                │
│  ┌────────────────────────────────────┐                 │
│  │ • Per-wavefront registers          │                 │
│  │ • 102-106 registers × 32 bits      │                 │
│  │ • Shared across all threads        │                 │
│  │ • Allocation: 16-register groups   │                 │
│  │ • Formula: (count - 1) / 16        │                 │
│  └────────────────────────────────────┘                 │
│                                                          │
│  AGPR (Accumulation Registers - CDNA only)              │
│  ┌────────────────────────────────────┐                 │
│  │ • Matrix operation accumulation    │                 │
│  │ • 256 registers × 32 bits          │                 │
│  │ • Used by MFMA instructions        │                 │
│  │ • Additional 256 KB per CU         │                 │
│  └────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

### Register Pressure and Occupancy

**Occupancy Formula:**
```
Max_Wavefronts_Per_CU = min(
    256KB_VGPR / (VGPR_per_thread × 256),
    SGPR_limit / SGPR_per_wavefront,
    LDS_limit / LDS_per_workgroup,
    Workgroup_limit
)
```

**Example:**
```cpp
// High register pressure (70 VGPRs)
__global__ void high_pressure() {
    float data[70];  // Stored in registers if possible
    // ... computation ...
}
// Result: Only 3-4 wavefronts per CU (poor occupancy)

// Low register pressure (20 VGPRs)
__global__ void low_pressure() {
    float data[20];
    // ... computation ...
}
// Result: 8-10 wavefronts per CU (good occupancy)
```

### Register Spilling

When register usage exceeds available VGPRs, **register spilling** occurs:

```
┌──────────────────────────────────────────────────────────┐
│              REGISTER SPILLING                            │
│                                                            │
│  Without Spilling:                With Spilling:          │
│  ┌──────────────┐               ┌──────────────┐         │
│  │  Registers   │               │  Registers   │         │
│  │  [Fast]      │               │  [Fast]      │         │
│  │  V0 - V63    │               │  V0 - V127   │         │
│  └──────────────┘               └──────┬───────┘         │
│       0 cycles                         │                  │
│                                        ▼                  │
│                              ┌──────────────────┐         │
│                              │ Scratch Memory   │         │
│                              │ [Slow - Global]  │         │
│                              │ 400+ cycles      │         │
│                              └──────────────────┘         │
│                                                            │
│  Performance: Excellent       Performance: Poor           │
└──────────────────────────────────────────────────────────┘
```

**Detection and Analysis:**
```bash
# Check register usage
hipcc -c kernel.cpp -Rpass-analysis=kernel-resource-usage

# Output example:
# remark: Function Name: myKernel
#   VGPRs: 42
#   SGPRs: 24
#   Scratch: 0 bytes (no spilling)
#   Occupancy: 8 waves/SIMD
```

---

## LDS (Local Data Share) Optimization {#lds-optimization}

### LDS Bank Architecture

LDS on AMD GPUs has **32 banks**, each 4 bytes wide:

```
┌──────────────────────────────────────────────────────────────┐
│                   LDS BANK ARCHITECTURE                       │
│                                                                │
│  Bank 0   Bank 1   Bank 2   Bank 3   ...   Bank 31           │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐           ┌────┐            │
│  │ 0  │  │ 4  │  │ 8  │  │ 12 │    ...    │124 │  Bytes 0-3 │
│  │128 │  │132 │  │136 │  │140 │    ...    │252 │  Bytes 4-7 │
│  │256 │  │260 │  │264 │  │268 │    ...    │380 │  Bytes 8-11│
│  │... │  │... │  │... │  │... │    ...    │... │            │
│  └────┘  └────┘  └────┘  └────┘           └────┘            │
│                                                                │
│  Bank Mapping: address_in_bytes % 128 / 4 = bank_number      │
└──────────────────────────────────────────────────────────────┘
```

### Bank Conflicts

**Conflict Scenario:**
```cpp
// ❌ BAD: 32-way bank conflict
__shared__ float lds[1024];
int tid = threadIdx.x;  // 0 to 31

// All threads access bank 0!
float value = lds[tid * 32];  // 0, 32, 64, 96...
// Each maps to: (0, 128, 256, 384...) % 128 = 0, 0, 0, 0...
// Result: 32x serialization
```

**Conflict-Free Access:**
```cpp
// ✅ GOOD: No bank conflicts
__shared__ float lds[1024];
int tid = threadIdx.x;  // 0 to 31

// Sequential access across banks
float value = lds[tid];  // 0, 1, 2, 3...
// Maps to banks: 0, 1, 2, 3... (all different)
// Result: Parallel access
```

### XOR-Based Swizzle Technique

Advanced technique to eliminate bank conflicts in complex patterns:

```cpp
// XOR-based address swizzle for GEMM tiles
__shared__ float lds[64][64];

// Traditional indexing (may have conflicts)
#define LDS_INDEX_NAIVE(m, k) ((m) * 64 + (k))

// XOR swizzle indexing (conflict-free)
#define XOR_SWIZZLE 3  // XOR with lower bits
#define LDS_INDEX_SWIZZLE(m, k) \
    ((m) * 64 + ((k) ^ ((m) & XOR_SWIZZLE)))

// Usage
__global__ void gemm_kernel() {
    __shared__ float tile[64 * 64];
    int m = threadIdx.y;
    int k = threadIdx.x;

    // Load with swizzle pattern
    int idx = LDS_INDEX_SWIZZLE(m, k);
    tile[idx] = global_data[...];
}
```

**Visualization of XOR Swizzle:**
```
Without XOR:          With XOR Swizzle:
M=0: [0][1][2][3]    M=0: [0^0][1^0][2^0][3^0] = [0][1][2][3]
M=1: [64][65][66]    M=1: [64^1][65^1][66^1]   = [65][64][67][66]
M=2: [128][129]      M=2: [128^2][129^2]       = [130][131][128][129]
M=3: [192][193]      M=3: [192^3][193^3]       = [195][194][193][192]

Bank mapping now distributes accesses evenly!
```

### Padding Technique

```cpp
// ❌ Without padding: 2-way bank conflict
__shared__ float tile[64][64];  // Total: 64×64 = 4096 floats

// ✅ With padding: Conflict-free
__shared__ float tile[64][65];  // Extra column shifts access pattern
```

---

## Memory Coalescing Patterns {#memory-coalescing}

### 2D Array Access Optimization

```cpp
// 2D array: A[HEIGHT][WIDTH]
__global__ void process_2d_array(float* A, int width, int height) {
    // ✅ GOOD: Coalesced (threads access consecutive columns)
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // Column
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // Row

    if (x < width && y < height) {
        int idx = y * width + x;  // Row-major
        float value = A[idx];
        // Adjacent threads: (y,0), (y,1), (y,2)... access [idx], [idx+1], [idx+2]
    }
}
```

**Memory Layout Visualization:**
```
Array Memory Layout (Row-Major):
┌────────────────────────────────────────┐
│ Row 0: [0][1][2][3][4][5][6][7]...     │ ← Thread 0, 1, 2, 3...
│ Row 1: [W][W+1][W+2][W+3]...           │ ← (consecutive access)
│ Row 2: [2W][2W+1][2W+2]...             │
└────────────────────────────────────────┘
```

### Transpose Operations

```cpp
// Naive transpose (non-coalesced reads OR writes)
__global__ void naive_transpose(float* out, const float* in, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Coalesced read, non-coalesced write
    out[x * N + y] = in[y * N + x];
}

// Optimized transpose using shared memory
__global__ void optimized_transpose(float* out, const float* in, int N) {
    __shared__ float tile[32][33];  // Note: 33 to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // Coalesced read from global memory
    tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    // Transpose coordinates
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    // Coalesced write to global memory
    out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}
```

---

## Instruction-Level Optimization {#instruction-level}

### AMD ISA Instruction Types

```
┌─────────────────────────────────────────────────────────────┐
│                   INSTRUCTION CATEGORIES                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  SCALAR INSTRUCTIONS (s_prefix)                              │
│  ┌────────────────────────────────────────┐                 │
│  │ • s_add_i32, s_sub_i32    - Arithmetic │                 │
│  │ • s_mov_b32, s_mov_b64    - Data move  │                 │
│  │ • s_cmp_*                  - Compare    │                 │
│  │ • s_branch, s_cbranch_*   - Branching  │                 │
│  │ • s_load_dword*           - Memory     │                 │
│  │ • s_waitcnt               - Sync       │                 │
│  │ Execution: 1 per wavefront, 1 cycle    │                 │
│  └────────────────────────────────────────┘                 │
│                                                               │
│  VECTOR INSTRUCTIONS (v_prefix)                              │
│  ┌────────────────────────────────────────┐                 │
│  │ • v_add_f32, v_mul_f32    - FP32 math  │                 │
│  │ • v_fma_f32, v_mad_f32    - Fused ops  │                 │
│  │ • v_mov_b32               - Data move  │                 │
│  │ • v_cmp_*                  - Compare    │                 │
│  │ Execution: Per thread, 4 VALUs/CU      │                 │
│  │ Throughput: 1 instruction/cycle/VALU   │                 │
│  └────────────────────────────────────────┘                 │
│                                                               │
│  MEMORY INSTRUCTIONS                                         │
│  ┌────────────────────────────────────────┐                 │
│  │ • global_load_dword*      - Global mem │                 │
│  │ • global_store_dword*     - Global mem │                 │
│  │ • ds_read_b32, ds_write_* - LDS        │                 │
│  │ • flat_load_*, flat_store - Flat       │                 │
│  │ • buffer_load_*, buffer_  - Buffer     │                 │
│  └────────────────────────────────────────┘                 │
│                                                               │
│  MATRIX INSTRUCTIONS (CDNA)                                  │
│  ┌────────────────────────────────────────┐                 │
│  │ • v_mfma_f32_*            - FP32 MFMA  │                 │
│  │ • v_mfma_f64_*            - FP64 MFMA  │                 │
│  │ • v_mfma_i8_*             - INT8 MFMA  │                 │
│  │ • v_accvgpr_*             - AGPR ops   │                 │
│  │ Throughput: 256-1024 ops/cycle         │                 │
│  └────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline and Latency Hiding

```
GPU Pipeline Execution Model:
┌────────────────────────────────────────────────────────────┐
│ Cycle │ Wave0  │ Wave1  │ Wave2  │ Wave3  │ Wave4  │...   │
├───────┼────────┼────────┼────────┼────────┼────────┼──────┤
│   0   │ VALU   │        │        │        │        │      │
│   1   │  wait  │ VALU   │        │        │        │      │
│   2   │  wait  │  wait  │ VALU   │        │        │      │
│   3   │  wait  │  wait  │  wait  │ VALU   │        │      │
│   4   │  wait  │  wait  │  wait  │  wait  │ VALU   │      │
│   5   │ result │  wait  │  wait  │  wait  │  wait  │ VALU │
│   6   │ VALU   │ result │  wait  │  wait  │  wait  │ wait │
│   7   │  wait  │ VALU   │ result │  wait  │  wait  │ wait │
└────────────────────────────────────────────────────────────┘

Key: Multiple waves hide latency through interleaving
```

### Instruction Throughput Optimization

```cpp
// ❌ SLOW: Using division
__global__ void slow_division(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in[idx] / 32.0f;  // Division: ~20-30 cycles
}

// ✅ FAST: Using multiplication by reciprocal
__global__ void fast_division(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in[idx] * 0.03125f;  // Multiplication: ~4 cycles
}

// ✅ FASTER: Using FMA (Fused Multiply-Add)
__global__ void fma_optimization(float* out, const float* a,
                                  const float* b, const float* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // One instruction instead of separate multiply and add
    out[idx] = __fmaf_rn(a[idx], b[idx], c[idx]);  // a*b + c
}
```

---

## Occupancy and Wave Management {#occupancy}

### Occupancy Formula and Limits

```
┌────────────────────────────────────────────────────────────┐
│              OCCUPANCY CALCULATION                          │
│                                                              │
│  Theoretical Max Wavefronts per CU: 40 (CDNA2)             │
│                                                              │
│  Limiting Factors:                                          │
│  ┌──────────────────────────────────────────────┐          │
│  │ 1. VGPR: 256KB / (VGPR_per_thread × 256)    │          │
│  │ 2. SGPR: ~800 / SGPR_per_wavefront           │          │
│  │ 3. LDS:  64KB / LDS_per_workgroup            │          │
│  │ 4. Workgroups: 32 max per CU                 │          │
│  └──────────────────────────────────────────────┘          │
│                                                              │
│  Example 1: High VGPR usage                                │
│    VGPRs = 80 per thread                                   │
│    Max waves = 256KB / (80 × 256) = 12.8 → 12 waves       │
│    Occupancy = 12 / 40 = 30%                               │
│                                                              │
│  Example 2: Balanced usage                                 │
│    VGPRs = 32 per thread                                   │
│    Max waves = 256KB / (32 × 256) = 32 waves               │
│    Occupancy = 32 / 40 = 80%                               │
└────────────────────────────────────────────────────────────┘
```

### Checking Occupancy

```bash
# Method 1: Compilation report
hipcc -c kernel.cpp -Rpass-analysis=kernel-resource-usage

# Method 2: Using rocProfiler
rocprof --stats kernel.exe

# Method 3: Check ISA
hipcc -c --save-temps -g kernel.cpp
# Then inspect kernel.s assembly file
```

### Loop Unrolling Impact

```cpp
// No unrolling: Low register pressure, high occupancy
__global__ void no_unroll(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < 64; i++) {
        sum += in[idx + i * N];
    }
    out[idx] = sum;
}
// VGPRs: ~15, Occupancy: 95%+

// Full unrolling: High register pressure, low occupancy
__global__ void full_unroll(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        sum += in[idx + i * N];
    }
    out[idx] = sum;
}
// VGPRs: ~80, Occupancy: 30%

// Partial unrolling: Balanced
__global__ void partial_unroll(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    #pragma unroll 8  // Unroll by factor of 8
    for (int i = 0; i < 64; i++) {
        sum += in[idx + i * N];
    }
    out[idx] = sum;
}
// VGPRs: ~25, Occupancy: 70%, Better ILP
```

---

## Advanced Performance Patterns {#advanced-patterns}

### Double Buffering

```cpp
__global__ void double_buffer_gemm(float* C, const float* A,
                                    const float* B, int N) {
    __shared__ float tileA[2][32][32];  // Double buffer
    __shared__ float tileB[2][32][32];

    int buf = 0;  // Current buffer

    // Load first tile
    tileA[buf][ty][tx] = A[...];
    tileB[buf][ty][tx] = B[...];
    __syncthreads();

    for (int k = 0; k < N/32; k++) {
        int next_buf = 1 - buf;

        // Prefetch next tile while computing current
        if (k + 1 < N/32) {
            tileA[next_buf][ty][tx] = A[...];  // Async load
            tileB[next_buf][ty][tx] = B[...];
        }

        // Compute using current buffer
        for (int i = 0; i < 32; i++) {
            C_reg += tileA[buf][ty][i] * tileB[buf][i][tx];
        }

        __syncthreads();
        buf = next_buf;  // Swap buffers
    }
}
```

### Warp Shuffle Operations

```cpp
// Efficient reduction using warp shuffle
__device__ float warp_reduce_sum(float val) {
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void block_reduce(float* out, const float* in, int N) {
    __shared__ float shared[32];  // One per warp

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load and reduce within warp
    float val = (idx < N) ? in[idx] : 0.0f;
    val = warp_reduce_sum(val);

    // Write warp result to shared memory
    if (tid % 64 == 0) {
        shared[tid / 64] = val;
    }
    __syncthreads();

    // Final reduction
    if (tid < blockDim.x / 64) {
        val = shared[tid];
        val = warp_reduce_sum(val);
        if (tid == 0) out[blockIdx.x] = val;
    }
}
```

### Software Pipelining

```cpp
__global__ void software_pipeline(float* out, const float* in, int N) {
    float reg[4];

    // Stage 1: Load
    reg[0] = in[idx];
    reg[1] = in[idx + stride];

    for (int i = 0; i < N; i++) {
        // Stage 2: Compute (while next load happens)
        float result = process(reg[0], reg[1]);

        // Stage 3: Store previous result
        if (i > 0) out[idx - stride] = prev_result;

        // Stage 1: Load next (overlaps with compute)
        reg[2] = in[idx + 2*stride];
        reg[3] = in[idx + 3*stride];

        prev_result = result;

        // Rotate registers
        reg[0] = reg[2];
        reg[1] = reg[3];
    }
}
```

---

## Performance Analysis Tools

### Using rocProfiler

```bash
# Profile kernel execution
rocprof --stats ./my_kernel

# Detailed metrics
rocprof --timestamp on --stats ./my_kernel

# Memory hierarchy counters
rocprof -i metrics.txt ./my_kernel

# Generate traces
rocprof --hip-trace --hsa-trace ./my_kernel
```

### Key Metrics to Monitor

```
┌──────────────────────────────────────────────────────────┐
│           CRITICAL PERFORMANCE METRICS                    │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  Occupancy:              Target > 50%                     │
│  Memory Bandwidth:       Target > 80% of theoretical      │
│  VALU Utilization:       Target > 70%                     │
│  LDS Bank Conflicts:     Target < 1%                      │
│  Register Spilling:      Target = 0                       │
│  Cache Hit Rate (L2):    Target > 90%                     │
│  Branch Divergence:      Target < 10%                     │
│  Kernel Launch Overhead: Target < 5% of runtime           │
└──────────────────────────────────────────────────────────┘
```

---

## Summary: Optimization Checklist

**Memory:**
- ✅ Coalesce global memory accesses
- ✅ Use LDS for data reuse
- ✅ Avoid LDS bank conflicts (use XOR swizzle or padding)
- ✅ Align data structures to cache lines

**Registers:**
- ✅ Minimize register usage to maximize occupancy
- ✅ Balance loop unrolling
- ✅ Avoid register spilling to scratch memory

**Execution:**
- ✅ Maximize occupancy (aim for 50%+ of theoretical)
- ✅ Use FMA instructions
- ✅ Minimize divergent branches
- ✅ Hide latency with sufficient wavefronts

**Advanced:**
- ✅ Use double buffering for computation/memory overlap
- ✅ Leverage warp shuffle for reductions
- ✅ Software pipeline independent operations
- ✅ Profile and iterate based on actual metrics

---

**Next:** See `HIP_ASSEMBLY_MFMA_GUIDE.md` for assembly-level optimization and matrix core programming.
