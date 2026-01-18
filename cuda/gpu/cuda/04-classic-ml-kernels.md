# Classic ML Kernels: GEMM and Convolution

## Overview

Matrix multiplication (GEMM - General Matrix Multiply) and convolution are the two fundamental operations in deep learning. Optimizing these kernels is essential for achieving high performance in neural network training and inference.

**Key Resources:**
- [CUTLASS Efficient GEMM Documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html)
- [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- [How to Optimize a CUDA Matmul](https://siboehm.com/articles/22/CUDA-MMM)

---

## Part 1: GEMM (General Matrix Multiply)

### Problem Definition

Compute: **C = α × A × B + β × C**

Where:
- A: M × K matrix
- B: K × N matrix
- C: M × N matrix (input and output)
- α, β: scalar multipliers

### Optimization Hierarchy

Performance optimization follows a multi-level hierarchy:

1. **Grid-level**: Schedule thread blocks cache-friendly
2. **Block-level**: Use shared memory for tiles
3. **Warp-level**: Coalesced and vectorized global loads
4. **Thread-level**: Enough arithmetic to hide latency

---

### Naive Implementation

```cpp
__global__ void gemm_naive(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Launch
dim3 blockDim(16, 16);
dim3 gridDim((N + 15) / 16, (M + 15) / 16);
gemm_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K);
```

**Problems:**
- Uncoalesced global memory access
- No data reuse (each element accessed K times)
- ~1-2% of peak performance

---

### Level 1: Shared Memory Tiling

**Key Idea:** Load tiles into shared memory to enable data reuse.

```cpp
#define TILE_SIZE 16

__global__ void gemm_shared_tiled(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
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
```

**Improvements:**
- Coalesced global memory access
- Each element loaded once per tile
- ~10-20x faster than naive

**Remaining Issues:**
- Shared memory bank conflicts
- Not using vectorized loads
- Limited register reuse

---

### Level 2: Vectorized Global Memory Access

**Key Idea:** Load 4 floats at once using `float4`.

```cpp
__global__ void gemm_vectorized(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < K / TILE_SIZE; t++) {
        // Vectorized load from A (4 floats at once)
        if (row < M && (tx % 4 == 0)) {
            float4 a_vec = *reinterpret_cast<const float4*>(
                &A[row * K + t * TILE_SIZE + tx]);
            As[ty][tx + 0] = a_vec.x;
            As[ty][tx + 1] = a_vec.y;
            As[ty][tx + 2] = a_vec.z;
            As[ty][tx + 3] = a_vec.w;
        }

        // Vectorized load from B
        if (col < N && (ty % 4 == 0)) {
            float4 b_vec = *reinterpret_cast<const float4*>(
                &B[(t * TILE_SIZE + ty) * N + col]);
            Bs[ty + 0][tx] = b_vec.x;
            Bs[ty + 1][tx] = b_vec.y;
            Bs[ty + 2][tx] = b_vec.z;
            Bs[ty + 3][tx] = b_vec.w;
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Improvements:**
- 4x fewer memory transactions
- ~1.5-2x faster than basic tiling

---

### Level 3: Register Tiling (2D Thread Tiling)

**Key Idea:** Each thread computes a small tile (e.g., 8×8) instead of a single element.

```cpp
#define BM 128  // Block tile M
#define BN 128  // Block tile N
#define BK 8    // Block tile K
#define TM 8    // Thread tile M
#define TN 8    // Thread tile N

__global__ void gemm_2d_blocktiling(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    // Shared memory for tiles
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_idx = ty * blockDim.x + tx;

    // Block indices
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Each thread computes TM × TN elements
    float thread_results[TM][TN] = {0.0f};

    // Position in output
    int c_row = block_row * BM + (thread_idx / (BN / TN)) * TM;
    int c_col = block_col * BN + (thread_idx % (BN / TN)) * TN;

    // Loop over K dimension in chunks of BK
    for (int k_idx = 0; k_idx < K; k_idx += BK) {
        // Load tile from A into shared memory
        for (int i = 0; i < BM; i += blockDim.x * blockDim.y / BK) {
            int a_row = i + thread_idx / BK;
            int a_col = thread_idx % BK;
            if (block_row * BM + a_row < M && k_idx + a_col < K) {
                As[a_row][a_col] = A[(block_row * BM + a_row) * K +
                                      k_idx + a_col];
            } else {
                As[a_row][a_col] = 0.0f;
            }
        }

        // Load tile from B into shared memory
        for (int i = 0; i < BK * BN; i += blockDim.x * blockDim.y) {
            int b_row = (thread_idx + i) / BN;
            int b_col = (thread_idx + i) % BN;
            if (k_idx + b_row < K && block_col * BN + b_col < N) {
                Bs[b_row][b_col] = B[(k_idx + b_row) * N +
                                      block_col * BN + b_col];
            } else {
                Bs[b_row][b_col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute: Each thread processes its TM × TN tile
        for (int k = 0; k < BK; k++) {
            // Load into registers
            float a_regs[TM];
            float b_regs[TN];

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                a_regs[i] = As[(thread_idx / (BN / TN)) * TM + i][k];
            }

            #pragma unroll
            for (int j = 0; j < TN; j++) {
                b_regs[j] = Bs[k][(thread_idx % (BN / TN)) * TN + j];
            }

            // Outer product
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    thread_results[i][j] += a_regs[i] * b_regs[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            if (c_row + i < M && c_col + j < N) {
                C[(c_row + i) * N + c_col + j] = thread_results[i][j];
            }
        }
    }
}
```

**Improvements:**
- Massive register reuse (TM × TN = 64 values per thread)
- Reduced shared memory traffic
- ~50-70% of cuBLAS performance

---

### Level 4: Warp-Level Tiling

**Key Idea:** Organize work at warp granularity for better control.

```cpp
#define WARP_SIZE 32
#define WARP_TILE_M 64
#define WARP_TILE_N 64
#define WARP_TILE_K 8

__device__ void warp_gemm(
    const float* As,  // Shared memory A
    const float* Bs,  // Shared memory B
    float* C_warp,    // Warp accumulator
    int warp_row,
    int warp_col,
    int lane_id
) {
    // Each warp computes a 64×64 tile
    // Each thread (lane) handles 2×2 elements

    float reg_a[2];
    float reg_b[2];
    float reg_c[2][2] = {0.0f};

    for (int k = 0; k < WARP_TILE_K; k++) {
        // Load from shared memory
        int row_offset = (lane_id / 8) * 2;
        int col_offset = (lane_id % 8) * 2;

        reg_a[0] = As[warp_row + row_offset + 0][k];
        reg_a[1] = As[warp_row + row_offset + 1][k];

        reg_b[0] = Bs[k][warp_col + col_offset + 0];
        reg_b[1] = Bs[k][warp_col + col_offset + 1];

        // Outer product
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                reg_c[i][j] += reg_a[i] * reg_b[j];
            }
        }
    }

    // Store back (implementation depends on layout)
}
```

---

### Level 5: Tensor Core Acceleration

**Key Idea:** Use Tensor Cores for 16×16×16 matrix multiply.

```cpp
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void gemm_wmma(
    const half* A,
    const half* B,
    float* C,
    int M, int N, int K
) {
    // Warp and lane IDs
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K
    for (int k = 0; k < K; k += WMMA_K) {
        int a_row = warpM * WMMA_M;
        int a_col = k;
        int b_row = k;
        int b_col = warpN * WMMA_N;

        if (a_row < M && a_col < K && b_row < K && b_col < N) {
            // Load matrices
            wmma::load_matrix_sync(a_frag, A + a_row * K + a_col, K);
            wmma::load_matrix_sync(b_frag, B + b_row * N + b_col, N);

            // Perform matrix multiply-accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Store result
    int c_row = warpM * WMMA_M;
    int c_col = warpN * WMMA_N;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N,
                               wmma::mem_row_major);
    }
}

// Launch: One warp per 16×16 output tile
dim3 blockDim(32, 1);  // 1 warp per block
dim3 gridDim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
gemm_wmma<<<gridDim, blockDim>>>(A, B, C, M, N, K);
```

**Performance:**
- 10-20x faster than Level 3
- 80-90% of cuBLAS with careful tuning

---

## Part 2: Convolution

### Problem Definition

2D Convolution: **Y[n][c][h][w] = Σ X[n][k][h+i][w+j] × W[c][k][i][j]**

Where:
- X: Input [N, C_in, H_in, W_in]
- W: Weights [C_out, C_in, K_h, K_w]
- Y: Output [N, C_out, H_out, W_out]

---

### Method 1: Direct Convolution

```cpp
__global__ void conv2d_direct(
    const float* input,   // [N, C_in, H, W]
    const float* weight,  // [C_out, C_in, K, K]
    float* output,        // [N, C_out, H_out, W_out]
    int N, int C_in, int C_out,
    int H, int W, int K,
    int H_out, int W_out
) {
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int h_out = blockIdx.x / W_out;
    int w_out = blockIdx.x % W_out;

    float sum = 0.0f;

    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                int h_in = h_out + kh;
                int w_in = w_out + kw;

                if (h_in < H && w_in < W) {
                    int input_idx = ((n * C_in + c_in) * H + h_in) * W + w_in;
                    int weight_idx = ((c_out * C_in + c_in) * K + kh) * K + kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[output_idx] = sum;
}
```

**Problems:**
- Poor memory access patterns
- Limited parallelism
- Slow for large kernels

---

### Method 2: Im2Col + GEMM

**Key Idea:** Transform convolution into matrix multiplication.

**Step 1: Im2Col** - Unfold input into columns:

```cpp
__global__ void im2col_kernel(
    const float* input,  // [N, C, H, W]
    float* output,       // [K*K*C, H_out*W_out]
    int C, int H, int W,
    int K, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * K * C * H_out * W_out;

    if (idx < total) {
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int kh = (idx / (W_out * H_out * C)) / K;
        int kw = (idx / (W_out * H_out * C)) % K;

        int h_in = h_out + kh;
        int w_in = w_out + kw;

        int input_idx = (c * H + h_in) * W + w_in;
        output[idx] = input[input_idx];
    }
}
```

**Step 2: GEMM**
```
Weight [C_out, K*K*C_in] × Im2Col [K*K*C_in, H_out*W_out] = Output [C_out, H_out*W_out]
```

**Advantages:**
- Leverage highly optimized GEMM
- Good for large feature maps
- Widely used (cuDNN default for many cases)

**Disadvantages:**
- High memory overhead (K² × C_in × H_out × W_out)
- Extra kernel for im2col transformation

---

### Method 3: Winograd Convolution

**Key Idea:** Reduce multiplications using mathematical transforms.

For 3×3 kernel, Winograd F(2×2, 3×3) reduces multiplies from 9 to 4 per output.

**Algorithm:**
1. Transform input: **U = B^T × d × B**
2. Transform filter: **V = G × g × G^T**
3. Element-wise product: **M = U ⊙ V**
4. Transform output: **Y = A^T × M × A**

```cpp
// Winograd F(2x2, 3x3) transform matrices
__constant__ float B[4][4] = {
    {1, 0, -1, 0},
    {0, 1, 1, 0},
    {0, -1, 1, 0},
    {0, 1, 0, -1}
};

__constant__ float G[4][3] = {
    {1, 0, 0},
    {0.5, 0.5, 0.5},
    {0.5, -0.5, 0.5},
    {0, 0, 1}
};

__constant__ float A[2][4] = {
    {1, 1, 1, 0},
    {0, 1, -1, -1}
};

__global__ void winograd_transform_input(
    const float* input,
    float* transformed,
    int H, int W, int C
) {
    // Transform each 4×4 input tile
    // Y = B^T × input_tile × B
    // ...implementation...
}

__global__ void winograd_transform_filter(
    const float* filter,
    float* transformed,
    int C_in, int C_out
) {
    // Transform each 3×3 filter
    // U = G × filter × G^T
    // ...implementation...
}

__global__ void winograd_elementwise_product(
    const float* input_transformed,
    const float* filter_transformed,
    float* output_transformed,
    int C_in, int C_out, int num_tiles
) {
    // M = U ⊙ V (element-wise product)
    // ...implementation...
}

__global__ void winograd_transform_output(
    const float* transformed,
    float* output,
    int H_out, int W_out, int C_out
) {
    // output = A^T × M × A
    // ...implementation...
}
```

**Performance:**
- 2-3x faster than direct for 3×3 kernels
- Accuracy issues with FP16 (use carefully)
- Limited to small kernels (3×3, 5×5)

---

### Method 4: FFT-Based Convolution

**Key Idea:** Convolution in spatial domain = multiplication in frequency domain.

```cpp
#include <cufft.h>

void conv2d_fft(
    const float* input,
    const float* filter,
    float* output,
    int N, int C, int H, int W, int K
) {
    cufftHandle plan;
    cufftComplex *input_freq, *filter_freq, *output_freq;

    // Allocate frequency domain buffers
    cudaMalloc(&input_freq, ...);
    cudaMalloc(&filter_freq, ...);
    cudaMalloc(&output_freq, ...);

    // Forward FFT
    cufftPlan2d(&plan, H, W, CUFFT_R2C);
    cufftExecR2C(plan, input, input_freq);
    cufftExecR2C(plan, filter, filter_freq);

    // Element-wise multiplication in frequency domain
    elementwise_multiply<<<...>>>(input_freq, filter_freq, output_freq);

    // Inverse FFT
    cufftPlan2d(&plan, H, W, CUFFT_C2R);
    cufftExecC2R(plan, output_freq, output);

    // Cleanup
    cufftDestroy(plan);
}
```

**Best For:**
- Very large kernels (K > 11)
- Large feature maps
- Multiple convolutions with same input

---

## Performance Comparison

| Method | Small Kernels (3×3) | Large Kernels (7×7+) | Memory Overhead | Accuracy |
|--------|---------------------|----------------------|-----------------|----------|
| **Direct** | Slow | Very slow | Low | Exact |
| **Im2Col+GEMM** | Fast | Fast | High (K²) | Exact |
| **Winograd** | Very fast | N/A | Medium | ~FP16 issues |
| **FFT** | Slow overhead | Fast | High | Numerical precision |

---

## cuDNN Usage (Recommended for Production)

```cpp
#include <cudnn.h>

void conv2d_cudnn(
    const float* input,
    const float* filter,
    float* output,
    int N, int C, int H, int W,
    int C_out, int K
) {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // Input descriptor
    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT, N, C, H, W);

    // Filter descriptor
    cudnnFilterDescriptor_t filter_desc;
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW, C_out, C, K, K);

    // Convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);

    // Output descriptor
    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    int out_n, out_c, out_h, out_w;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc,
                                          filter_desc, &out_n, &out_c,
                                          &out_h, &out_w);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT, out_n, out_c,
                               out_h, out_w);

    // Find best algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc, filter_desc,
                                           conv_desc, output_desc, 1, 0, &algo);

    // Get workspace size
    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc,
                                            conv_desc, output_desc, algo,
                                            &workspace_size);

    // Allocate workspace
    void* workspace;
    cudaMalloc(&workspace, workspace_size);

    // Execute convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(handle, &alpha, input_desc, input,
                           filter_desc, filter, conv_desc, algo,
                           workspace, workspace_size, &beta,
                           output_desc, output);

    // Cleanup
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroy(handle);
}
```

---

## Key Takeaways

### GEMM Optimization Principles
1. **Tiling**: Block → Warp → Thread hierarchy
2. **Memory**: Vectorized loads, coalescing, shared memory
3. **Compute**: Register reuse, ILP, Tensor Cores

### Convolution Strategy Selection
- **Small kernels (3×3)**: Winograd or Im2Col+GEMM
- **Large kernels (7×7+)**: FFT or Im2Col+GEMM
- **General case**: Use cuDNN (automatically selects best method)

### Production Recommendations
- **Always use cuDNN** for convolutions when possible
- **Use cuBLAS/CUTLASS** for GEMM
- **Custom kernels** only for specialized cases not covered by libraries

---

## Further Reading

- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/)
- [How to Optimize GEMM](https://siboehm.com/articles/22/CUDA-MMM)
- [Winograd Convolution](https://arxiv.org/abs/1509.09308)
