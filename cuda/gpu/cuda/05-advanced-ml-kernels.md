# Advanced ML Kernels: Flash Attention and Modern Techniques

## Overview

Modern deep learning requires specialized kernels that go beyond basic GEMM and convolution. This guide covers advanced kernels critical for transformer models and large language models (LLMs).

**Key Resources:**
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [GPU MODE Lecture 12: Flash Attention](https://christianjmills.com/posts/cuda-mode-notes/lecture-012/)
- [A Case Study in CUDA Kernel Fusion: FlashAttention-2](https://arxiv.org/pdf/2312.11918)

---

## Part 1: Attention Mechanism

### Standard Attention (Baseline)

**Formula**: **Attention(Q, K, V) = softmax(QK^T / √d) × V**

Where:
- Q (Query): [N, d] - N tokens, d dimensions
- K (Key): [N, d]
- V (Value): [N, d]
- Output: [N, d]

### Naive Implementation (Memory Intensive)

```cpp
__global__ void attention_naive(
    const float* Q,  // [N, d]
    const float* K,  // [N, d]
    const float* V,  // [N, d]
    float* O,        // [N, d] output
    int N, int d
) {
    // This is PSEUDOCODE showing the conceptual flow
    // Actual implementation requires multiple kernels

    // Step 1: Compute S = Q × K^T
    // S is [N, N] - HUGE memory for large N!
    float* S = new float[N * N];  // NOT EFFICIENT
    gemm(Q, K_transpose, S, N, N, d);

    // Step 2: Scale by √d
    scale(S, 1.0f / sqrtf(d), N * N);

    // Step 3: Softmax(S) - row-wise
    softmax_rows(S, N, N);

    // Step 4: P × V where P = softmax(S)
    gemm(S, V, O, N, d, N);
}
```

**Problems:**
- **Memory**: O(N²) for attention matrix S
- **HBM Traffic**: Multiple roundtrips to global memory
- **Speed**: For N=4096, d=128: S requires 64MB!
- **Scalability**: Fails for long sequences (N > 8K)

---

## Part 2: FlashAttention (Online Softmax + Tiling)

### Key Innovation: IO-Aware Algorithm

**Core Ideas:**
1. **Never materialize** the full N×N attention matrix
2. **Tiling**: Process in blocks that fit in SRAM (shared memory)
3. **Online softmax**: Compute softmax incrementally without seeing all values
4. **Kernel fusion**: Single kernel for entire attention operation

### Online Softmax Algorithm

Standard softmax requires two passes:
```
Pass 1: max_val = max(x)
Pass 2: sum = Σ exp(x - max_val)
Output: exp(x - max_val) / sum
```

**Online Softmax** (single pass with running statistics):

```cpp
struct SoftmaxState {
    float m;  // running max
    float l;  // running sum of exponentials
};

__device__ SoftmaxState update_softmax_state(
    SoftmaxState state,
    float new_val
) {
    float m_new = fmaxf(state.m, new_val);
    float l_new = state.l * expf(state.m - m_new) +
                  expf(new_val - m_new);

    return {m_new, l_new};
}

__device__ float apply_softmax(float val, SoftmaxState state) {
    return expf(val - state.m) / state.l;
}
```

### FlashAttention Algorithm

```cpp
// Conceptual pseudocode for FlashAttention
__global__ void flash_attention(
    const float* Q,  // [N, d]
    const float* K,  // [N, d]
    const float* V,  // [N, d]
    float* O,        // [N, d]
    int N, int d
) {
    // Block sizes
    const int Br = 64;  // Block size for Q rows
    const int Bc = 64;  // Block size for K/V rows

    // Shared memory for tiles
    __shared__ float Qi[Br][d];
    __shared__ float Kj[Bc][d];
    __shared__ float Vj[Bc][d];
    __shared__ float Sij[Br][Bc];

    // Thread block processes one Qi block
    int block_row = blockIdx.x;

    // Load Qi into SRAM
    load_tile(Q, Qi, block_row * Br, d);

    // Initialize output accumulator and softmax state
    float Oi[d] = {0.0f};
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Loop over K, V blocks (tiling dimension)
    for (int j = 0; j < (N + Bc - 1) / Bc; j++) {
        // Load Kj, Vj into SRAM
        load_tile(K, Kj, j * Bc, d);
        load_tile(V, Vj, j * Bc, d);

        __syncthreads();

        // Compute Sij = Qi @ Kj^T (on chip)
        // Sij is [Br, Bc] - fits in shared memory!
        matmul_tile(Qi, Kj, Sij, Br, Bc, d);

        // Scale by 1/√d
        scale_tile(Sij, 1.0f / sqrtf(d), Br, Bc);

        // Update running max
        float m_ij = row_max(Sij, Br, Bc);
        float m_i_new = fmaxf(m_i, m_ij);

        // Compute softmax numerator: exp(Sij - m_i_new)
        exp_subtract_tile(Sij, m_i_new, Br, Bc);

        // Update running sum with correction factor
        float l_i_new = l_i * expf(m_i - m_i_new) +
                        row_sum(Sij, Br, Bc);

        // Update output: O_i = O_i * correction + Sij @ Vj
        float correction = expf(m_i - m_i_new);
        scale(Oi, correction, d);

        matmul_accumulate(Sij, Vj, Oi, Br, Bc, d);

        // Update softmax state
        m_i = m_i_new;
        l_i = l_i_new;

        __syncthreads();
    }

    // Final normalization: O_i = O_i / l_i
    scale(Oi, 1.0f / l_i, d);

    // Write back to global memory
    store_tile(O, Oi, block_row * Br, d);
}
```

### FlashAttention Implementation (Simplified)

```cpp
#define Br 64
#define Bc 64
#define BLOCK_DIM 128

__global__ void flash_attention_kernel(
    const half* Q,
    const half* K,
    const half* V,
    float* O,
    int N, int d
) {
    __shared__ half Qi_smem[Br * d];
    __shared__ half Kj_smem[Bc * d];
    __shared__ half Vj_smem[Bc * d];

    int tid = threadIdx.x;
    int block_row_idx = blockIdx.x;

    // Each block processes Br rows of Q
    int q_row_start = block_row_idx * Br;
    int q_row_end = min(q_row_start + Br, N);

    // Load Q block
    for (int i = tid; i < Br * d; i += blockDim.x) {
        int row = i / d;
        int col = i % d;
        if (q_row_start + row < N) {
            Qi_smem[i] = Q[(q_row_start + row) * d + col];
        }
    }

    // Initialize accumulators
    float O_local[8] = {0.0f};  // Assume d=128, each thread handles 8 elements
    float m_local = -INFINITY;
    float l_local = 0.0f;

    // Tile over K, V
    for (int kv_block = 0; kv_block < (N + Bc - 1) / Bc; kv_block++) {
        int kv_start = kv_block * Bc;
        int kv_end = min(kv_start + Bc, N);

        // Load K, V tiles
        for (int i = tid; i < Bc * d; i += blockDim.x) {
            int row = i / d;
            int col = i % d;
            if (kv_start + row < N) {
                Kj_smem[i] = K[(kv_start + row) * d + col];
                Vj_smem[i] = V[(kv_start + row) * d + col];
            }
        }

        __syncthreads();

        // Compute attention scores and update
        // (Detailed computation omitted for brevity)
        // Each thread computes partial dot products, accumulates

        __syncthreads();
    }

    // Write output
    for (int i = 0; i < 8; i++) {
        int out_idx = (q_row_start + threadIdx.x / 16) * d +
                      (threadIdx.x % 16) * 8 + i;
        if (out_idx < N * d) {
            O[out_idx] = O_local[i] / l_local;
        }
    }
}
```

**Performance Gains:**
- **Memory**: O(N) instead of O(N²)
- **Speed**: 2-4x faster than standard attention
- **Scalability**: Enables sequence lengths up to 64K+

---

## Part 3: FlashAttention-2 (Work Partitioning)

### Key Improvements Over FlashAttention-1

1. **Reduced Non-Matmul FLOPs**: Minimize rescaling operations
2. **Parallelism over Sequence Length**: Better warp utilization
3. **Better Work Partitioning**: Split work between warps within block

### Warp-Specialized Kernel

```cpp
// FlashAttention-2 uses warp-level programming
__global__ void flash_attention_v2(
    const half* Q,
    const half* K,
    const half* V,
    float* O,
    int N, int d
) {
    // Warp-level tiling
    const int NUM_WARPS = 4;
    const int Br = 64;
    const int Bc = 64;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    __shared__ half Qi[Br][d];
    __shared__ half Kj[Bc][d];
    __shared__ half Vj[Bc][d];

    // Each warp handles a subset of Q rows
    int warp_row_start = blockIdx.x * Br + warp_id * (Br / NUM_WARPS);
    int warp_row_end = warp_row_start + (Br / NUM_WARPS);

    // Warp-level accumulators
    float O_warp[4][d / 32];  // Each thread in warp handles d/32 elements
    float m_warp = -INFINITY;
    float l_warp = 0.0f;

    // Load Q tile (collaborative across warps)
    // Each warp loads its portion
    for (int i = lane_id; i < (Br / NUM_WARPS) * d; i += 32) {
        int local_row = i / d;
        int col = i % d;
        int global_row = warp_row_start + local_row;
        if (global_row < N) {
            Qi[warp_id * (Br / NUM_WARPS) + local_row][col] =
                Q[global_row * d + col];
        }
    }

    __syncthreads();

    // Tile over K, V
    for (int kv_tile = 0; kv_tile < (N + Bc - 1) / Bc; kv_tile++) {
        // Load K, V collaboratively
        load_kv_tile(K, V, Kj, Vj, kv_tile, Bc, d, warp_id, lane_id, N);

        __syncthreads();

        // Compute S = Q @ K^T for this warp's rows
        // Use warp-level matrix multiply

        // Update softmax statistics and output
        // (Detailed computation...)

        __syncthreads();
    }

    // Write warp's output
    for (int i = 0; i < 4; i++) {
        for (int j = lane_id; j < d / 32; j += 32) {
            int row = warp_row_start + i;
            int col = j;
            if (row < N && col < d) {
                O[row * d + col] = O_warp[i][j] / l_warp;
            }
        }
    }
}
```

**Improvements:**
- **1.5-2x faster** than FlashAttention-1
- Better GPU utilization (80%+ on A100)
- Reduced shared memory bank conflicts

---

## Part 4: FlashAttention-3 (Hopper Optimizations)

### Leveraging Hopper H100 Features

1. **TMA (Tensor Memory Accelerator)**: Asynchronous data movement
2. **WGMMA**: Warpgroup matrix multiply (4 warps = 128 threads)
3. **Low-Precision**: FP8 with hardware block quantization
4. **Asynchronous Pipeline**: Overlap data transfer and compute

### Key Techniques

#### 1. Asynchronous Data Loading with TMA

```cpp
__global__ void flash_attention_v3(
    const __grid_constant__ CUtensorMap Q_map,
    const __grid_constant__ CUtensorMap K_map,
    const __grid_constant__ CUtensorMap V_map,
    float* O,
    int N, int d
) {
    __shared__ half Qi[Br][d];
    __shared__ half Kj[Bc][d];
    __shared__ half Vj[Bc][d];

    // Barrier for TMA synchronization
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    if (threadIdx.x == 0) {
        init(&barrier, 1);
    }
    __syncthreads();

    // Producer thread (single thread issues TMA)
    if (threadIdx.x == 0) {
        // Asynchronously load Q tile
        cuda::memcpy_async(Qi, Q_map,
                          cuda::aligned_size_t<128>(sizeof(Qi)),
                          barrier);
    }

    // Consumer threads can do other work while data loads
    // ...

    // Wait for TMA completion
    barrier.arrive_and_wait();

    // Now Qi is ready
    // Proceed with computation...
}
```

#### 2. Warpgroup GEMM (WGMMA)

```cpp
__device__ void warpgroup_matmul(
    const half* A,  // In shared memory
    const half* B,  // In shared memory
    float* C,       // Accumulator
    int M, int N, int K
) {
    // WGMMA instruction: 4 warps collaborate
    // Performs 64×256×16 matrix multiply in hardware

    // Create descriptors
    uint64_t desc_a = make_smem_desc(A);
    uint64_t desc_b = make_smem_desc(B);

    // Issue warpgroup matrix multiply
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0, %1, %2, ..., %63}, %64, %65;"
        : "+f"(C[0]), "+f"(C[1]), ... "+f"(C[63])  // 64 accumulators
        : "l"(desc_a), "l"(desc_b)
    );

    // Wait for completion
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
}
```

#### 3. FP8 with Block Quantization

```cpp
__global__ void flash_attention_fp8(
    const __nv_fp8_e4m3* Q,  // FP8 quantized
    const float* Q_scale,     // Block scales
    const __nv_fp8_e4m3* K,
    const float* K_scale,
    const __nv_fp8_e4m3* V,
    const float* V_scale,
    float* O,
    int N, int d
) {
    // Load FP8 data and scales
    __shared__ __nv_fp8_e4m3 Qi_fp8[Br][d];
    __shared__ float Qi_scale[Br / 16];  // One scale per 16 elements

    // TMA load FP8 data
    tma_load(Qi_fp8, Q, ...);

    // Dequantize on-the-fly during GEMM
    // Hopper WGMMA handles this automatically!

    // Compute with FP8 inputs, FP32 accumulation
    wgmma_fp8_f32(Qi_fp8, Qi_scale, Kj_fp8, Kj_scale, ...);

    // Output remains FP32 for accuracy
}
```

### Performance Results

**FlashAttention-3 on H100:**
- **1.5-2x faster** than FlashAttention-2
- **6x faster** than FlashAttention-1
- **9.5x faster** than PyTorch eager attention
- Enables **context lengths up to 128K+**

---

## Part 5: Other Advanced Kernels

### 1. LayerNorm / RMSNorm

```cpp
__global__ void rms_norm(
    const half* input,
    const half* weight,
    half* output,
    int N, int d
) {
    int row = blockIdx.x;
    if (row >= N) return;

    // Compute RMS: sqrt(mean(x^2))
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float val = __half2float(input[row * d + i]);
        sum_sq += val * val;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Broadcast to all threads
    __shared__ float rms;
    if (threadIdx.x == 0) {
        rms = rsqrtf(sum_sq / d + 1e-6f);
    }
    __syncthreads();

    // Normalize and scale
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float val = __half2float(input[row * d + i]);
        float w = __half2float(weight[i]);
        output[row * d + i] = __float2half(val * rms * w);
    }
}
```

### 2. Rotary Position Embedding (RoPE)

```cpp
__global__ void rope_forward(
    const half* input,   // [N, num_heads, d]
    half* output,
    const float* freqs,  // [d/2]
    int N, int num_heads, int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * num_heads * d;

    if (idx >= total) return;

    int pos = idx / (num_heads * d);
    int head = (idx / d) % num_heads;
    int dim = idx % d;

    int pair_idx = dim / 2;
    float freq = freqs[pair_idx];
    float angle = pos * freq;

    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    float x0 = __half2float(input[pos * num_heads * d + head * d + pair_idx * 2]);
    float x1 = __half2float(input[pos * num_heads * d + head * d + pair_idx * 2 + 1]);

    if (dim % 2 == 0) {
        output[idx] = __float2half(x0 * cos_val - x1 * sin_val);
    } else {
        output[idx] = __float2half(x0 * sin_val + x1 * cos_val);
    }
}
```

### 3. Grouped Query Attention (GQA)

```cpp
__global__ void gqa_attention(
    const half* Q,  // [N, num_q_heads, d]
    const half* K,  // [N, num_kv_heads, d]
    const half* V,  // [N, num_kv_heads, d]
    half* O,        // [N, num_q_heads, d]
    int N, int num_q_heads, int num_kv_heads, int d
) {
    // num_q_heads > num_kv_heads (e.g., 32 Q heads, 4 KV heads)
    int group_size = num_q_heads / num_kv_heads;

    int q_head = blockIdx.y;
    int kv_head = q_head / group_size;

    // Perform attention for Q head q_head with KV head kv_head
    // (Use FlashAttention algorithm here)
    flash_attention_single_head(
        Q + q_head * N * d,
        K + kv_head * N * d,
        V + kv_head * N * d,
        O + q_head * N * d,
        N, d
    );
}
```

---

## Performance Optimization Checklist

### Memory Optimization
- ✓ Minimize HBM traffic (use tiling)
- ✓ Maximize shared memory reuse
- ✓ Vectorized loads/stores (float4, half2)
- ✓ Coalesced memory access

### Compute Optimization
- ✓ Kernel fusion (fuse multiple ops)
- ✓ Use Tensor Cores (WMMA/WGMMA)
- ✓ Asynchronous operations (TMA on Hopper)
- ✓ Low-precision compute (FP16, BF16, FP8)

### Occupancy Optimization
- ✓ Balance registers vs shared memory
- ✓ Tune block size (multiples of 32)
- ✓ Limit register spilling
- ✓ Profile with Nsight Compute

---

## Benchmarking Flash Attention

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

void benchmark_attention(int N, int d) {
    // Allocate memory
    half *Q, *K, *V, *O;
    cudaMalloc(&Q, N * d * sizeof(half));
    cudaMalloc(&K, N * d * sizeof(half));
    cudaMalloc(&V, N * d * sizeof(half));
    cudaMalloc(&O, N * d * sizeof(half));

    // Warm-up
    for (int i = 0; i < 10; i++) {
        flash_attention_kernel<<<...>>>(Q, K, V, O, N, d);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        flash_attention_kernel<<<...>>>(Q, K, V, O, N, d);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(end - start).count() / iterations;

    // Compute FLOPs
    float flops = 4.0f * N * N * d;  // Approximate
    float tflops = (flops / ms) / 1e9;

    std::cout << "N=" << N << ", d=" << d << std::endl;
    std::cout << "Time: " << ms << " ms" << std::endl;
    std::cout << "TFLOPS: " << tflops << std::endl;

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O);
}
```

---

## Further Reading

- [FlashAttention-3 Paper](https://arxiv.org/abs/2407.08608)
- [CUTLASS Flash Attention Examples](https://github.com/NVIDIA/cutlass/tree/main/examples/flash_attention)
- [xFormers Library](https://github.com/facebookresearch/xformers)
- [vLLM Inference Engine](https://github.com/vllm-project/vllm)
- [GPU MODE Discord Community](https://discord.gg/gpumode)
