# Advanced ML Kernels: Complete Implementations

## Complete FlashAttention Implementation

This file contains FULL, RUNNABLE code for advanced ML kernels.

---

## Part 1: Complete Standard Attention (Baseline)

```cpp
// standard_attention.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel 1: Compute attention scores S = Q @ K^T and scale
__global__ void compute_qk_scores(
    const float* Q,
    const float* K,
    float* S,
    int N, int d,
    float scale
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < d; k++) {
            sum += Q[row * d + k] * K[col * d + k];
        }
        S[row * N + col] = sum * scale;
    }
}

// Kernel 2: Row-wise softmax
__global__ void softmax_kernel(
    float* S,
    int N
) {
    int row = blockIdx.x;
    if (row >= N) return;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, S[row * N + i]);
    }

    // Warp-level reduction for max
    __shared__ float shared_max[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    if (lane_id == 0) {
        shared_max[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (lane_id < (blockDim.x + 31) / 32) ? shared_max[lane_id] : -INFINITY;
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if (lane_id == 0) {
            shared_max[0] = max_val;
        }
    }
    __syncthreads();
    max_val = shared_max[0];

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = expf(S[row * N + i] - max_val);
        S[row * N + i] = val;
        sum += val;
    }

    // Warp-level reduction for sum
    __shared__ float shared_sum[32];
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + 31) / 32) ? shared_sum[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0) {
            shared_sum[0] = sum;
        }
    }
    __syncthreads();
    sum = shared_sum[0];

    // Normalize
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        S[row * N + i] /= sum;
    }
}

// Kernel 3: Compute output O = S @ V
__global__ void compute_output(
    const float* S,
    const float* V,
    float* O,
    int N, int d
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < d) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += S[row * N + k] * V[k * d + col];
        }
        O[row * d + col] = sum;
    }
}

// Host function
void standard_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N, int d
) {
    float scale = 1.0f / sqrtf((float)d);

    // Allocate attention matrix S [N, N]
    float* S;
    CHECK_CUDA(cudaMalloc(&S, N * N * sizeof(float)));

    // Step 1: Compute Q @ K^T and scale
    dim3 block1(16, 16);
    dim3 grid1((N + 15) / 16, (N + 15) / 16);
    compute_qk_scores<<<grid1, block1>>>(Q, K, S, N, d, scale);
    CHECK_CUDA(cudaGetLastError());

    // Step 2: Softmax
    int threads = 256;
    int blocks = N;
    softmax_kernel<<<blocks, threads>>>(S, N);
    CHECK_CUDA(cudaGetLastError());

    // Step 3: Compute S @ V
    dim3 block3(16, 16);
    dim3 grid3((d + 15) / 16, (N + 15) / 16);
    compute_output<<<grid3, block3>>>(S, V, O, N, d);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(S));
}

// Main test function
int main() {
    const int N = 1024;  // Sequence length
    const int d = 64;    // Head dimension

    // Allocate host memory
    float *h_Q = new float[N * d];
    float *h_K = new float[N * d];
    float *h_V = new float[N * d];
    float *h_O = new float[N * d];

    // Initialize with random data
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_K[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_V[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, N * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, N * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, N * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_O, N * d * sizeof(float)));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, N * d * sizeof(float), cudaMemcpyHostToDevice));

    // Run attention
    standard_attention(d_Q, d_K, d_V, d_O, N, d);

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_O, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Standard Attention completed for N=" << N << ", d=" << d << std::endl;
    std::cout << "First output value: " << h_O[0] << std::endl;

    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));

    return 0;
}
```

**Compile:**
```bash
nvcc -o standard_attention standard_attention.cu -lcublas
./standard_attention
```

---

## Part 2: Complete FlashAttention Implementation

```cpp
// flash_attention.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cmath>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64
#define THREADS 128

// Complete FlashAttention kernel
__global__ void flash_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    float* L,  // Softmax denominators
    float* M,  // Softmax max values
    int N, int d
) {
    const int Br = BLOCK_M;
    const int Bc = BLOCK_N;

    // Shared memory for tiles
    __shared__ float Qi[BLOCK_M][BLOCK_K + 4];  // +4 to avoid bank conflicts
    __shared__ float Kj[BLOCK_N][BLOCK_K + 4];
    __shared__ float Vj[BLOCK_N][BLOCK_K + 4];
    __shared__ float Sij[BLOCK_M][BLOCK_N + 4];

    const int tid = threadIdx.x;
    const int block_row = blockIdx.x;
    const int row_start = block_row * Br;
    const int row_end = min(row_start + Br, N);

    const float scale = 1.0f / sqrtf((float)d);

    // Register storage for output accumulator
    float O_thread[BLOCK_K];
    float m_thread = -INFINITY;
    float l_thread = 0.0f;

    // Initialize output
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i++) {
        O_thread[i] = 0.0f;
    }

    // Load Q tile into shared memory
    for (int i = tid; i < Br * d; i += blockDim.x) {
        int local_row = i / d;
        int col = i % d;
        int global_row = row_start + local_row;

        if (global_row < N && col < d) {
            Qi[local_row][col] = Q[global_row * d + col];
        } else {
            Qi[local_row][col] = 0.0f;
        }
    }
    __syncthreads();

    // Loop over K, V tiles
    for (int tile_idx = 0; tile_idx < (N + Bc - 1) / Bc; tile_idx++) {
        const int col_start = tile_idx * Bc;
        const int col_end = min(col_start + Bc, N);

        // Load K and V tiles
        for (int i = tid; i < Bc * d; i += blockDim.x) {
            int local_row = i / d;
            int col = i % d;
            int global_row = col_start + local_row;

            if (global_row < N && col < d) {
                Kj[local_row][col] = K[global_row * d + col];
                Vj[local_row][col] = V[global_row * d + col];
            } else {
                Kj[local_row][col] = 0.0f;
                Vj[local_row][col] = 0.0f;
            }
        }
        __syncthreads();

        // Compute Sij = Qi @ Kj^T (tiled matmul)
        for (int i = tid; i < Br * Bc; i += blockDim.x) {
            int row = i / Bc;
            int col = i % Bc;

            if (row_start + row < N && col_start + col < N) {
                float sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    sum += Qi[row][k] * Kj[col][k];
                }
                Sij[row][col] = sum * scale;
            } else {
                Sij[row][col] = -INFINITY;
            }
        }
        __syncthreads();

        // Compute row-wise max for each thread's assigned rows
        int rows_per_thread = (Br + blockDim.x - 1) / blockDim.x;
        for (int r = 0; r < rows_per_thread; r++) {
            int row = tid + r * blockDim.x;
            if (row >= Br || row_start + row >= N) continue;

            float row_max = -INFINITY;
            for (int c = 0; c < Bc && col_start + c < N; c++) {
                row_max = fmaxf(row_max, Sij[row][c]);
            }

            // Update running max
            float m_prev = m_thread;
            float m_new = fmaxf(m_thread, row_max);

            // Compute exp(Sij - m_new) and update l
            float exp_sum = 0.0f;
            for (int c = 0; c < Bc && col_start + c < N; c++) {
                float exp_val = expf(Sij[row][c] - m_new);
                Sij[row][c] = exp_val;
                exp_sum += exp_val;
            }

            // Update running sum with correction
            float l_new = l_thread * expf(m_prev - m_new) + exp_sum;

            // Update output with correction factor
            float correction = expf(m_prev - m_new);

            // O = O * correction + Sij @ Vj
            for (int k = 0; k < d; k++) {
                float sum = 0.0f;
                for (int c = 0; c < Bc && col_start + c < N; c++) {
                    sum += Sij[row][c] * Vj[c][k];
                }
                O_thread[k] = O_thread[k] * correction + sum;
            }

            m_thread = m_new;
            l_thread = l_new;
        }
        __syncthreads();
    }

    // Write output
    int rows_per_thread = (Br + blockDim.x - 1) / blockDim.x;
    for (int r = 0; r < rows_per_thread; r++) {
        int local_row = tid + r * blockDim.x;
        int global_row = row_start + local_row;

        if (global_row < N) {
            for (int k = 0; k < d; k++) {
                O[global_row * d + k] = O_thread[k] / l_thread;
            }
            if (L) L[global_row] = l_thread;
            if (M) M[global_row] = m_thread;
        }
    }
}

// Host wrapper
void flash_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N, int d
) {
    float *L, *M;
    CHECK_CUDA(cudaMalloc(&L, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&M, N * sizeof(float)));

    int num_blocks = (N + BLOCK_M - 1) / BLOCK_M;
    flash_attention_kernel<<<num_blocks, THREADS>>>(Q, K, V, O, L, M, N, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(L));
    CHECK_CUDA(cudaFree(M));
}

int main() {
    const int N = 2048;
    const int d = 64;

    // Allocate host memory
    float *h_Q = new float[N * d];
    float *h_K = new float[N * d];
    float *h_V = new float[N * d];
    float *h_O = new float[N * d];

    // Initialize
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_K[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_V[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, N * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, N * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, N * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_O, N * d * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, N * d * sizeof(float), cudaMemcpyHostToDevice));

    // Warm-up
    flash_attention(d_Q, d_K, d_V, d_O, N, d);

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        flash_attention(d_Q, d_K, d_V, d_O, N, d);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iterations;

    CHECK_CUDA(cudaMemcpy(h_O, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost));

    float tflops = (4.0f * N * N * d) / (ms * 1e9);
    std::cout << "FlashAttention N=" << N << ", d=" << d << std::endl;
    std::cout << "Time: " << ms << " ms" << std::endl;
    std::cout << "TFLOPS: " << tflops << std::endl;
    std::cout << "First output: " << h_O[0] << std::endl;

    // Cleanup
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O;
    CHECK_CUDA(cudaFree(d_Q)); CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V)); CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```

**Compile:**
```bash
nvcc -o flash_attention flash_attention.cu -O3 -arch=sm_80
./flash_attention
```

---

## Part 3: Complete RMSNorm Implementation

```cpp
// rmsnorm.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void rmsnorm_kernel(
    const float* input,
    const float* weight,
    float* output,
    int N, int d,
    float eps = 1e-6f
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x = input + row * d;
    float* y = output + row * d;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Block-level reduction
    __shared__ float shared_sum[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + 31) / 32) ? shared_sum[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (lane_id == 0) {
            shared_sum[0] = sum_sq;
        }
    }
    __syncthreads();

    // Compute RMS
    float rms = rsqrtf(shared_sum[0] / d + eps);

    // Normalize and apply weight
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        y[i] = x[i] * rms * weight[i];
    }
}

int main() {
    const int N = 4096;  // Batch size
    const int d = 4096;  // Hidden dimension

    float *h_input = new float[N * d];
    float *h_weight = new float[d];
    float *h_output = new float[N * d];

    // Initialize
    for (int i = 0; i < N * d; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < d; i++) {
        h_weight[i] = 1.0f;
    }

    float *d_input, *d_weight, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weight, d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * d * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight, d * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threads = 256;
    int blocks = N;
    rmsnorm_kernel<<<blocks, threads>>>(d_input, d_weight, d_output, N, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * d * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "RMSNorm completed for N=" << N << ", d=" << d << std::endl;
    std::cout << "First output: " << h_output[0] << std::endl;

    delete[] h_input; delete[] h_weight; delete[] h_output;
    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_weight)); CHECK_CUDA(cudaFree(d_output));

    return 0;
}
```

**Compile:**
```bash
nvcc -o rmsnorm rmsnorm.cu -O3
./rmsnorm
```

---

## Part 4: Complete Rotary Position Embedding (RoPE)

```cpp
// rope.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void rope_kernel(
    const float* input,
    float* output,
    const float* freqs_cos,
    const float* freqs_sin,
    int N, int num_heads, int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * num_heads * d;

    if (idx >= total) return;

    int pos = idx / (num_heads * d);
    int head = (idx / d) % num_heads;
    int dim = idx % d;
    int pair_idx = dim / 2;

    int cos_sin_idx = pos * (d / 2) + pair_idx;
    float cos_val = freqs_cos[cos_sin_idx];
    float sin_val = freqs_sin[cos_sin_idx];

    int base_idx = pos * num_heads * d + head * d;
    float x0 = input[base_idx + pair_idx * 2];
    float x1 = input[base_idx + pair_idx * 2 + 1];

    if (dim % 2 == 0) {
        output[idx] = x0 * cos_val - x1 * sin_val;
    } else {
        output[idx] = x0 * sin_val + x1 * cos_val;
    }
}

void precompute_freqs(float* freqs_cos, float* freqs_sin, int max_seq_len, int d, float theta = 10000.0f) {
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < d / 2; i++) {
            float freq = 1.0f / powf(theta, (2.0f * i) / d);
            float angle = pos * freq;
            freqs_cos[pos * (d / 2) + i] = cosf(angle);
            freqs_sin[pos * (d / 2) + i] = sinf(angle);
        }
    }
}

int main() {
    const int N = 1024;      // Sequence length
    const int num_heads = 8;
    const int d = 64;        // Head dimension

    float *h_input = new float[N * num_heads * d];
    float *h_output = new float[N * num_heads * d];
    float *h_freqs_cos = new float[N * (d / 2)];
    float *h_freqs_sin = new float[N * (d / 2)];

    // Initialize input
    for (int i = 0; i < N * num_heads * d; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Precompute frequencies
    precompute_freqs(h_freqs_cos, h_freqs_sin, N, d);

    float *d_input, *d_output, *d_freqs_cos, *d_freqs_sin;
    CHECK_CUDA(cudaMalloc(&d_input, N * num_heads * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * num_heads * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_freqs_cos, N * (d / 2) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_freqs_sin, N * (d / 2) * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * num_heads * d * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_freqs_cos, h_freqs_cos, N * (d / 2) * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_freqs_sin, h_freqs_sin, N * (d / 2) * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N * num_heads * d + threads - 1) / threads;
    rope_kernel<<<blocks, threads>>>(d_input, d_output, d_freqs_cos, d_freqs_sin, N, num_heads, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * num_heads * d * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "RoPE completed for N=" << N << ", heads=" << num_heads << ", d=" << d << std::endl;

    delete[] h_input; delete[] h_output; delete[] h_freqs_cos; delete[] h_freqs_sin;
    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_freqs_cos)); CHECK_CUDA(cudaFree(d_freqs_sin));

    return 0;
}
```

**Compile:**
```bash
nvcc -o rope rope.cu -O3
./rope
```

---

## Makefile for All Kernels

```makefile
# Makefile
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_80
LIBS = -lcublas

all: standard_attention flash_attention rmsnorm rope

standard_attention: standard_attention.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LIBS)

flash_attention: flash_attention.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

rmsnorm: rmsnorm.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

rope: rope.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f standard_attention flash_attention rmsnorm rope

.PHONY: all clean
```

**Usage:**
```bash
make all
./standard_attention
./flash_attention
./rmsnorm
./rope
```
