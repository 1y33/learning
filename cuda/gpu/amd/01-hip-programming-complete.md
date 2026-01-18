# Complete HIP Programming Guide for AMD GPUs

## Overview

**HIP (Heterogeneous-compute Interface for Portability)** is AMD's answer to CUDA. It allows you to write GPU code that runs on both **AMD** and **NVIDIA** GPUs from a single source.

**Key Resources:**
- [HIP Documentation](https://rocm.docs.amd.com/projects/HIP/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/programming_guide.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)

---

## Part 1: HIP vs CUDA - Side by Side Comparison

### API Mapping

| CUDA | HIP | Description |
|------|-----|-------------|
| `cudaMalloc` | `hipMalloc` | Allocate device memory |
| `cudaMemcpy` | `hipMemcpy` | Copy memory |
| `cudaFree` | `hipFree` | Free device memory |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | Sync device |
| `__global__` | `__global__` | Kernel qualifier |
| `__device__` | `__device__` | Device function |
| `__host__` | `__host__` | Host function |
| `threadIdx.x` | `threadIdx.x` | Thread index |
| `blockIdx.x` | `blockIdx.x` | Block index |
| `blockDim.x` | `blockDim.x` | Block dimension |
| `gridDim.x` | `gridDim.x` | Grid dimension |
| `__syncthreads()` | `__syncthreads()` | Block sync |
| `warpSize` | `warpSize` | 32 (NVIDIA), 64 (AMD) |

**Key Difference:** AMD GPUs have **wavefront size = 64** (vs CUDA's warp size = 32)

---

## Part 2: Complete Vector Addition in HIP

```cpp
// vecadd_hip.cpp - Complete HIP Vector Addition
#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// HIP kernel (same syntax as CUDA!)
__global__ void vecadd_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1024 * 1024;
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_HIP(hipMalloc(&d_A, bytes));
    CHECK_HIP(hipMalloc(&d_B, bytes));
    CHECK_HIP(hipMalloc(&d_C, bytes));

    // Copy to device
    CHECK_HIP(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    hipLaunchKernelGGL(vecadd_kernel, dim3(blocks), dim3(threads), 0, 0,
                       d_A, d_B, d_C, N);
    CHECK_HIP(hipGetLastError());

    // Synchronize
    CHECK_HIP(hipDeviceSynchronize());

    // Copy result back
    CHECK_HIP(hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost));

    // Verify
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            correct = false;
            break;
        }
    }

    std::cout << "Result: " << (correct ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));

    return 0;
}
```

**Compile for AMD:**
```bash
hipcc -o vecadd_hip vecadd_hip.cpp -O3
./vecadd_hip
```

**Compile for NVIDIA (HIP works on CUDA too!):**
```bash
hipcc -o vecadd_hip vecadd_hip.cpp -O3 --cuda
./vecadd_hip
```

---

## Part 3: Complete Matrix Multiplication in HIP

```cpp
// matmul_hip.cpp - Optimized Matrix Multiplication in HIP
#include <hip/hip_runtime.h>
#include <iostream>

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define TILE_SIZE 16

// Tiled matrix multiplication kernel
__global__ void matmul_tiled(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile from B
        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 1024, N = 1024, K = 1024;

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // Initialize
    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CHECK_HIP(hipMalloc(&d_A, M * K * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_B, K * N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_C, M * N * sizeof(float)));

    CHECK_HIP(hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Warm-up
    hipLaunchKernelGGL(matmul_tiled, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
    CHECK_HIP(hipDeviceSynchronize());

    // Benchmark
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < 100; i++) {
        hipLaunchKernelGGL(matmul_tiled, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float ms;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    ms /= 100;

    float gflops = (2.0f * M * N * K) / (ms * 1e6);
    std::cout << "Matrix Multiplication: " << ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    CHECK_HIP(hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

    delete[] h_A; delete[] h_B; delete[] h_C;
    CHECK_HIP(hipFree(d_A)); CHECK_HIP(hipFree(d_B)); CHECK_HIP(hipFree(d_C));
    CHECK_HIP(hipEventDestroy(start)); CHECK_HIP(hipEventDestroy(stop));

    return 0;
}
```

**Compile:**
```bash
hipcc -o matmul_hip matmul_hip.cpp -O3
./matmul_hip
```

---

## Part 4: AMD-Specific Optimizations

### Wavefront Size = 64

AMD GPUs have **64 threads per wavefront** (vs 32 for NVIDIA).

```cpp
// Wavefront-aware reduction
__global__ void reduce_amd(const float* input, float* output, int N) {
    __shared__ float sdata[64];  // One per wavefront lane

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < N) ? input[idx] : 0.0f;

    // Wavefront reduction (64 threads)
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }

    if (tid % 64 == 0) {
        sdata[tid / 64] = val;
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < (blockDim.x + 63) / 64; i++) {
            sum += sdata[i];
        }
        atomicAdd(output, sum);
    }
}
```

### LDS (Local Data Share) - AMD's Shared Memory

```cpp
__global__ void use_lds() {
    // LDS = Local Data Share (AMD term for shared memory)
    __shared__ float lds[256];  // Same as CUDA shared memory

    int tid = threadIdx.x;
    lds[tid] = tid;

    __syncthreads();

    // Use LDS data
    float val = lds[tid];
}
```

---

## Part 5: ROCm Libraries (AMD's cuBLAS/cuDNN equivalents)

### rocBLAS Example

```cpp
// rocblas_gemm.cpp
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>

int main() {
    const int M = 1024, N = 1024, K = 1024;

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, M * K * sizeof(float));
    hipMalloc(&d_B, K * N * sizeof(float));
    hipMalloc(&d_C, M * N * sizeof(float));

    hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice);

    // Create rocBLAS handle
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    float alpha = 1.0f, beta = 0.0f;

    // GEMM: C = alpha * A * B + beta * C
    rocblas_sgemm(
        handle,
        rocblas_operation_none, rocblas_operation_none,
        M, N, K,
        &alpha,
        d_A, M,
        d_B, K,
        &beta,
        d_C, M
    );

    hipDeviceSynchronize();

    hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);

    std::cout << "rocBLAS GEMM completed. First result: " << h_C[0] << std::endl;

    rocblas_destroy_handle(handle);
    delete[] h_A; delete[] h_B; delete[] h_C;
    hipFree(d_A); hipFree(d_B); hipFree(d_C);

    return 0;
}
```

**Compile:**
```bash
hipcc -o rocblas_gemm rocblas_gemm.cpp -lrocblas
./rocblas_gemm
```

---

## Part 6: Converting CUDA to HIP

### Automatic Conversion with hipify

```bash
# Convert CUDA file to HIP
hipify-perl your_cuda_code.cu > your_hip_code.cpp

# Or use hipify-clang (more accurate)
hipify-clang your_cuda_code.cu -- -x cuda --cuda-path=/usr/local/cuda
```

### Manual Conversion

**CUDA Code:**
```cuda
// cuda_example.cu
#include <cuda_runtime.h>

__global__ void kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

int main() {
    float *d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    kernel<<<256, 4>>>(d_data, 1024);
    cudaDeviceSynchronize();
    cudaFree(d_data);
}
```

**HIP Code (converted):**
```cpp
// hip_example.cpp
#include <hip/hip_runtime.h>

__global__ void kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

int main() {
    float *d_data;
    hipMalloc(&d_data, 1024 * sizeof(float));
    hipLaunchKernelGGL(kernel, dim3(256), dim3(4), 0, 0, d_data, 1024);
    hipDeviceSynchronize();
    hipFree(d_data);
}
```

---

## Part 7: AMD GPU Architecture Basics

### GCN/RDNA/CDNA Architecture

**Key Concepts:**
- **Compute Unit (CU)**: Like NVIDIA's SM
- **Wavefront**: Group of 64 threads (vs CUDA's warp = 32)
- **LDS**: Local Data Share (shared memory)
- **GDS**: Global Data Share
- **Scalar Unit**: Handles uniform operations

### Matrix Cores (CDNA Architecture)

AMD's Matrix Cores (similar to Tensor Cores):

```cpp
// Use rocWMMA for matrix core acceleration
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

__global__ void mfma_kernel(
    const half* A,
    const half* B,
    float* C,
    int M, int N, int K
) {
    // 16x16x16 matrix multiply on Matrix Cores
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    load_matrix_sync(a_frag, A, K);
    load_matrix_sync(b_frag, B, K);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(C, c_frag, N, mem_row_major);
}
```

---

## Comparison: CUDA vs HIP Features

| Feature | CUDA | HIP (AMD) | Notes |
|---------|------|-----------|-------|
| **Warp/Wavefront Size** | 32 | 64 | AMD = 2x |
| **Shared Memory** | Up to 227 KB | Up to 64 KB (CDNA2) | Per CU |
| **Tensor/Matrix Cores** | Yes (Volta+) | Yes (CDNA) | MFMA instructions |
| **Async Copy** | Yes (Ampere+) | Limited | Different API |
| **Unified Memory** | Yes | Yes | Similar |
| **Dynamic Parallelism** | Yes | Limited | Kernel launch from kernel |
| **Half Precision** | FP16 | FP16 | Similar |
| **Library Ecosystem** | Extensive | Growing | rocBLAS, MIOpen, etc. |

---

## Makefile for HIP Projects

```makefile
# Makefile for HIP projects
HIPCC = hipcc
HIPFLAGS = -O3 -std=c++17

# Auto-detect AMD or NVIDIA
GPU_ARCH := $(shell hipconfig --platform)

ifeq ($(GPU_ARCH),amd)
    HIPFLAGS += --amdgpu-target=gfx90a  # MI200 series
else
    HIPFLAGS += --cuda
endif

all: vecadd matmul rocblas_gemm

vecadd: vecadd_hip.cpp
	$(HIPCC) $(HIPFLAGS) -o $@ $<

matmul: matmul_hip.cpp
	$(HIPCC) $(HIPFLAGS) -o $@ $<

rocblas_gemm: rocblas_gemm.cpp
	$(HIPCC) $(HIPFLAGS) -lrocblas -o $@ $<

clean:
	rm -f vecadd matmul rocblas_gemm

.PHONY: all clean
```

---

## Further Reading

- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/programming_guide.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [hipify Tools](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/)
- [rocBLAS Documentation](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/)
- [MIOpen (AMD's cuDNN)](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
