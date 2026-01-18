# Code Generation and Autotuning for GPU Kernels

## Overview

Modern GPU programming uses **code generation** and **autotuning** to achieve optimal performance without manual kernel tuning for every hardware configuration.

**Key Frameworks:**
- **CUTLASS** (NVIDIA): C++ template metaprogramming for GEMM
- **Triton** (OpenAI): Python DSL for GPU kernels
- **TVM/Apache TVM**: End-to-end ML compiler
- **MLIR**: Multi-level intermediate representation

**Resources:**
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [Triton Documentation](https://triton-lang.org/)
- [CUTLASS 4.2 Autotuning](https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/)

---

## Part 1: What is Autotuning?

### The Problem

**Different hardware needs different parameters:**

```cpp
// A100 optimal: 128x128 tiles
gemm_kernel<128, 128, 8><<<...>>>(A, B, C);

// H100 optimal: 256x128 tiles
gemm_kernel<256, 128, 16><<<...>>>(A, B, C);

// RTX 4090 optimal: 64x256 tiles
gemm_kernel<64, 256, 8><<<...>>>(A, B, C);
```

**Autotuning automat**ically finds the best configuration!

### Tunable Parameters

1. **Tile sizes**: Block_M, Block_N, Block_K
2. **Thread block size**: Number of threads
3. **Register blocking**: Elements per thread
4. **Pipeline depth**: Number of stages
5. **Swizzle patterns**: Memory layout optimizations

---

## Part 2: Simple C++ Template Autotuner

### Complete Example: Autotuning GEMM Tile Sizes

```cpp
// autotune_gemm.cu - Simple GEMM Autotuner
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Templated GEMM kernel with tunable tile sizes
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void gemm_tiled(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Bs[BLOCK_K][BLOCK_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_M + ty;
    int col = blockIdx.x * BLOCK_N + tx;

    float sum = 0.0f;

    // Tile over K dimension
    for (int t = 0; t < (K + BLOCK_K - 1) / BLOCK_K; t++) {
        // Load tiles collaboratively
        if (row < M && t * BLOCK_K + tx < K) {
            As[ty][tx] = A[row * K + t * BLOCK_K + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && t * BLOCK_K + ty < K) {
            Bs[ty][tx] = B[(t * BLOCK_K + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Configuration struct
struct GemmConfig {
    int block_m, block_n, block_k;
    int threads_x, threads_y;
    float time_ms;

    GemmConfig(int m, int n, int k)
        : block_m(m), block_n(n), block_k(k),
          threads_x(k), threads_y(m), time_ms(0.0f) {}
};

// Benchmark a specific configuration
template<int BM, int BN, int BK>
float benchmark_config(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K,
    int iterations = 100
) {
    dim3 block(BK, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // Warm-up
    gemm_tiled<BM, BN, BK><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        gemm_tiled<BM, BN, BK><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iterations;
}

// Autotune: try all configurations and pick the best
std::vector<GemmConfig> autotune_gemm(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K
) {
    std::vector<GemmConfig> configs;

    // Define search space
    std::vector<int> tile_sizes = {16, 32, 64, 128};

    std::cout << "Autotuning GEMM for M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Testing " << tile_sizes.size() * tile_sizes.size() * tile_sizes.size()
              << " configurations..." << std::endl;

    for (int bm : tile_sizes) {
        for (int bn : tile_sizes) {
            for (int bk : tile_sizes) {
                GemmConfig config(bm, bn, bk);

                try {
                    // Dispatch to correct template instantiation
                    if (bm == 16 && bn == 16 && bk == 16) {
                        config.time_ms = benchmark_config<16, 16, 16>(d_A, d_B, d_C, M, N, K);
                    } else if (bm == 16 && bn == 16 && bk == 32) {
                        config.time_ms = benchmark_config<16, 16, 32>(d_A, d_B, d_C, M, N, K);
                    } else if (bm == 16 && bn == 16 && bk == 64) {
                        config.time_ms = benchmark_config<16, 16, 64>(d_A, d_B, d_C, M, N, K);
                    } else if (bm == 16 && bn == 16 && bk == 128) {
                        config.time_ms = benchmark_config<16, 16, 128>(d_A, d_B, d_C, M, N, K);
                    }
                    // ... (add all combinations or use preprocessor magic)
                    else if (bm == 128 && bn == 128 && bk == 16) {
                        config.time_ms = benchmark_config<128, 128, 16>(d_A, d_B, d_C, M, N, K);
                    } else {
                        continue;  // Skip unsupported combinations
                    }

                    configs.push_back(config);

                    float gflops = (2.0f * M * N * K) / (config.time_ms * 1e6);
                    std::cout << "BM=" << bm << ", BN=" << bn << ", BK=" << bk
                              << " : " << config.time_ms << " ms (" << gflops << " GFLOPS)"
                              << std::endl;

                } catch (...) {
                    std::cerr << "Failed: BM=" << bm << ", BN=" << bn << ", BK=" << bk << std::endl;
                }
            }
        }
    }

    // Sort by time (ascending)
    std::sort(configs.begin(), configs.end(),
        [](const GemmConfig& a, const GemmConfig& b) {
            return a.time_ms < b.time_ms;
        });

    return configs;
}

int main() {
    const int M = 2048, N = 2048, K = 2048;

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    for (int i = 0; i < M * K; i++) h_A[i] = ((float)rand() / RAND_MAX);
    for (int i = 0; i < K * N; i++) h_B[i] = ((float)rand() / RAND_MAX);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Run autotuner
    auto configs = autotune_gemm(d_A, d_B, d_C, M, N, K);

    if (!configs.empty()) {
        std::cout << "\n=== BEST CONFIGURATION ===" << std::endl;
        auto best = configs[0];
        float gflops = (2.0f * M * N * K) / (best.time_ms * 1e6);
        std::cout << "Block_M=" << best.block_m
                  << ", Block_N=" << best.block_n
                  << ", Block_K=" << best.block_k << std::endl;
        std::cout << "Time: " << best.time_ms << " ms" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    }

    delete[] h_A; delete[] h_B; delete[] h_C;
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```

**Compile & Run:**
```bash
nvcc -o autotune_gemm autotune_gemm.cu -O3 -arch=sm_80
./autotune_gemm
```

---

## Part 3: Triton - Python DSL for GPU Kernels

### What is Triton?

**Triton** lets you write GPU kernels in Python with automatic optimization!

### Complete Triton Example: Vector Addition

```python
# triton_vecadd.py
import torch
import triton
import triton.language as tl

@triton.jit
def vecadd_kernel(
    x_ptr,  # Pointer to input X
    y_ptr,  # Pointer to input Y
    z_ptr,  # Pointer to output Z
    N,      # Size of vectors
    BLOCK_SIZE: tl.constexpr  # Compile-time constant
):
    # Get program ID
    pid = tl.program_id(0)

    # Compute offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask for bounds checking
    mask = offsets < N

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    z = x + y

    # Store result
    tl.store(z_ptr + offsets, z, mask=mask)

def vecadd_triton(x, y):
    N = x.numel()
    z = torch.empty_like(x)

    # Define search space for autotuning
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        ],
        key=['N']  # Autotune based on input size
    )
    @triton.jit
    def vecadd_autotuned(x_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        z = x + y
        tl.store(z_ptr + offsets, z, mask=mask)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    vecadd_autotuned[grid](x, y, z, N)

    return z

# Test
if __name__ == '__main__':
    N = 1024 * 1024
    x = torch.randn(N, device='cuda')
    y = torch.randn(N, device='cuda')

    # Warmup
    z_triton = vecadd_triton(x, y)

    # Benchmark
    import time
    start = time.time()
    for _ in range(1000):
        z_triton = vecadd_triton(x, y)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Triton VecAdd: {(end - start) / 1000 * 1000:.3f} ms")

    # Verify
    z_torch = x + y
    assert torch.allclose(z_triton, z_torch)
    print("Correctness verified!")
```

**Run:**
```bash
python triton_vecadd.py
```

### Triton Flash Attention Example

```python
# triton_flash_attention.py
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Pointers to Q
    q_ptrs = Q + pid_z * stride_qz + pid_h * stride_qh + \
             offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk

    # Initialize output accumulators
    o = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    m = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')

    # Load Q
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

    # Loop over K, V
    for start_n in range(0, N, BLOCK_N):
        # Load K
        k_ptrs = K + pid_z * stride_kz + pid_h * stride_kh + \
                 (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < N, other=0.0)

        # Compute scores S = Q @ K^T
        s = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]

        # Update statistics (online softmax)
        m_new = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, 1)
        m = m_new

        # Load V
        v_ptrs = V + pid_z * stride_vz + pid_h * stride_vh + \
                 (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N, other=0.0)

        # Update output
        o = o * alpha[:, None] + tl.dot(p.to(v.dtype), v)

    # Final normalization
    o = o / l[:, None]

    # Store output
    o_ptrs = O + pid_z * stride_oz + pid_h * stride_oh + \
             offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, o.to(O.dtype.element_ty), mask=offs_m[:, None] < M)

def flash_attention_triton(Q, K, V):
    Z, H, M, D = Q.shape
    N = K.shape[2]

    O = torch.empty_like(Q)

    BLOCK_M, BLOCK_N = 64, 64

    grid = (triton.cdiv(M, BLOCK_M), Z, H)

    flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        Z, H, M, N, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
    )

    return O

# Test
if __name__ == '__main__':
    Z, H, M, D = 1, 8, 1024, 64

    Q = torch.randn(Z, H, M, D, device='cuda', dtype=torch.float16)
    K = torch.randn(Z, H, M, D, device='cuda', dtype=torch.float16)
    V = torch.randn(Z, H, M, D, device='cuda', dtype=torch.float16)

    O_triton = flash_attention_triton(Q, K, V)

    print(f"Flash Attention output shape: {O_triton.shape}")
    print(f"Output sample: {O_triton[0, 0, :5, :5]}")
```

---

## Part 4: CUTLASS - Template Metaprogramming

### CUTLASS Basics

**CUTLASS** uses C++ templates to generate optimized GEMM kernels.

### Simple CUTLASS GEMM Example

```cpp
// cutlass_gemm.cu
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include <iostream>

int main() {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator
    >;

    int M = 2048;
    int N = 2048;
    int K = 2048;

    float alpha = 1.0f;
    float beta = 0.0f;

    // Allocate device memory
    ElementA *A, *B;
    ElementC *C;

    cudaMalloc(&A, M * K * sizeof(ElementA));
    cudaMalloc(&B, K * N * sizeof(ElementB));
    cudaMalloc(&C, M * N * sizeof(ElementC));

    // Initialize (omitted for brevity)

    // Create GEMM operator
    Gemm gemm_op;

    // Setup arguments
    Gemm::Arguments args{
        {M, N, K},     // Problem size
        {A, K},        // Tensor A
        {B, N},        // Tensor B
        {C, N},        // Tensor C (source)
        {C, N},        // Tensor D (destination)
        {alpha, beta}  // Scalars
    };

    // Initialize
    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to initialize CUTLASS GEMM" << std::endl;
        return -1;
    }

    // Run
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to run CUTLASS GEMM" << std::endl;
        return -1;
    }

    cudaDeviceSynchronize();

    std::cout << "CUTLASS GEMM completed successfully!" << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
```

**Compile:**
```bash
nvcc -o cutlass_gemm cutlass_gemm.cu -I${CUTLASS_DIR}/include -O3 -arch=sm_80
```

---

## Summary Table

| Framework | Language | Ease of Use | Performance | Autotuning |
|-----------|----------|-------------|-------------|------------|
| **Raw CUDA** | C++ | Hard | Highest (manual) | Manual |
| **CUTLASS** | C++ Templates | Medium | Highest | Partial |
| **Triton** | Python | Easy | High | Built-in |
| **TVM** | Python | Medium | High | Built-in |

**Recommendation:**
- **Prototype**: Use Triton
- **Production**: Use CUTLASS or hand-tuned CUDA
- **Research**: Use TVM for end-to-end compilation
