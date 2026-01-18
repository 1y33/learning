# Quantization Techniques for Neural Networks

## Overview

Quantization reduces model size and accelerates inference by representing weights and activations with lower precision. This guide covers all major quantization formats from FP32 down to 1-bit, with CUDA implementations.

**Key Resources:**
- [NVIDIA FP4 Quantization](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [INT4 for AI Inference](https://developer.nvidia.com/blog/int4-for-ai-inference/)
- [BitNet: 1-bit LLMs](https://github.com/microsoft/BitNet)
- [GPU MODE Lecture 7: Advanced Quantization](https://christianjmills.com/posts/cuda-mode-notes/lecture-007/)

---

## Quantization Spectrum

| Format | Bits | Range | Use Case | Hardware Support |
|--------|------|-------|----------|------------------|
| **FP32** | 32 | Full | Training, Reference | Universal |
| **TF32** | 19 | Reduced mantissa | Training | Ampere+ |
| **BF16** | 16 | Wide range | Training | Ampere+ |
| **FP16** | 16 | Standard | Training/Inference | Volta+ |
| **FP8 E4M3** | 8 | ML-optimized | Training/Inference | Hopper+ |
| **FP8 E5M2** | 8 | Dynamic range | Gradients | Hopper+ |
| **FP6** | 6 | Experimental | Inference | Custom |
| **NVFP4** | 4 | Blockwise FP | Inference | Blackwell+ |
| **INT8** | 8 | -128 to 127 | Inference | Turing+ |
| **INT4** | 4 | -8 to 7 | Inference | Turing+ (limited) |
| **INT2** | 2 | -2 to 1 | Extreme compression | Custom |
| **INT1** | 1 | -1 or 1 | Binary networks | Custom |
| **1.58-bit** | ~1.6 | {-1, 0, 1} | Ternary networks | Custom |

---

## Part 1: Floating-Point Quantization

### FP16 (Half Precision)

**Format**: 1 sign + 5 exponent + 10 mantissa

```cpp
__global__ void gemm_fp16(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        half sum = __float2half(0.0f);

        for (int k = 0; k < K; k++) {
            // FP16 arithmetic
            sum = __hadd(sum, __hmul(A[row * K + k], B[k * N + col]));
        }

        C[row * N + col] = sum;
    }
}
```

**Using Tensor Cores:**

```cpp
#include <mma.h>
using namespace nvcuda;

__global__ void gemm_fp16_tensorcore(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    wmma::fill_fragment(c_frag, __float2half(0.0f));

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + warpM * 16 * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpN * 16, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpM * 16 * N + warpN * 16, c_frag, N,
                           wmma::mem_row_major);
}
```

---

### BF16 (BFloat16)

**Format**: 1 sign + 8 exponent + 7 mantissa (same range as FP32, less precision)

```cpp
__global__ void gemm_bf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;  // Accumulate in FP32 for accuracy

        for (int k = 0; k < K; k++) {
            float a = __bfloat162float(A[row * K + k]);
            float b = __bfloat162float(B[k * N + col]);
            sum += a * b;
        }

        C[row * N + col] = __float2bfloat16(sum);
    }
}
```

---

### FP8 (8-bit Floating Point) - Hopper+

**Two formats:**
- **E4M3**: 1 sign + 4 exponent + 3 mantissa (values: better precision)
- **E5M2**: 1 sign + 5 exponent + 2 mantissa (gradients: better range)

```cpp
#include <cuda_fp8.h>

__global__ void gemm_fp8_e4m3(
    const __nv_fp8_e4m3* A,
    const float* A_scale,  // Per-tensor or per-block scale
    const __nv_fp8_e4m3* B,
    const float* B_scale,
    float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; k++) {
            // Dequantize on-the-fly
            float a = float(A[row * K + k]) * A_scale[0];
            float b = float(B[k * N + col]) * B_scale[0];
            sum += a * b;
        }

        C[row * N + col] = sum;
    }
}

// Quantization kernel
__global__ void quantize_fp8_e4m3(
    const float* input,
    __nv_fp8_e4m3* output,
    float* scale,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute scale (absmax / max_fp8)
    __shared__ float max_val;
    if (threadIdx.x == 0) {
        float abs_max = 0.0f;
        for (int i = 0; i < N; i++) {
            abs_max = fmaxf(abs_max, fabsf(input[i]));
        }
        max_val = abs_max / 448.0f;  // 448 is FP8 E4M3 max value
        scale[0] = max_val;
    }
    __syncthreads();

    if (idx < N) {
        output[idx] = __nv_fp8_e4m3(input[idx] / max_val);
    }
}
```

---

### NVFP4 (4-bit Floating Point) - Blackwell+

**Format**: Block floating point with E2M1 per element + shared FP8 scale per block

```cpp
// NVFP4: Each block of 16 FP4 values shares one FP8 scale

struct NVFP4Block {
    __nv_fp8_e5m2 scale;           // Shared scale (FP8)
    unsigned char values[8];        // 16 FP4 values packed (2 per byte)
};

__device__ float dequantize_nvfp4(unsigned char packed, int idx, float scale) {
    // Extract 4-bit value (E2M1 format)
    unsigned char fp4_val = (idx == 0) ? (packed & 0x0F) : (packed >> 4);

    // E2M1 decoding: 1 sign + 2 exponent + 1 mantissa
    int sign = (fp4_val >> 3) & 1;
    int exp = (fp4_val >> 1) & 3;
    int mant = fp4_val & 1;

    float value;
    if (exp == 0) {
        // Subnormal
        value = (mant == 0) ? 0.0f : 0.5f;
    } else if (exp == 3) {
        // Special (infinity/NaN)
        value = INFINITY;
    } else {
        // Normal: (-1)^sign * 2^(exp-1) * (1 + mant)
        value = powf(2.0f, exp - 1) * (1.0f + mant);
    }

    return (sign ? -value : value) * scale;
}

__global__ void gemm_nvfp4(
    const NVFP4Block* A,  // [M, K/16] blocks
    const NVFP4Block* B,  // [K/16, N] blocks
    float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k_block = 0; k_block < K / 16; k_block++) {
            // Load A block
            NVFP4Block a_block = A[row * (K / 16) + k_block];
            float a_scale = __half2float(a_block.scale);

            // Load B blocks
            for (int i = 0; i < 16; i++) {
                int k = k_block * 16 + i;
                NVFP4Block b_block = B[(k / 16) * N + col];
                float b_scale = __half2float(b_block.scale);

                // Dequantize and accumulate
                float a = dequantize_nvfp4(a_block.values[i / 2], i % 2, a_scale);
                float b = dequantize_nvfp4(b_block.values[i / 2], i % 2, b_scale);

                sum += a * b;
            }
        }

        C[row * N + col] = sum;
    }
}

// Quantization with block-wise scaling
__global__ void quantize_nvfp4(
    const float* input,
    NVFP4Block* output,
    int N
) {
    int block_idx = blockIdx.x;
    int start = block_idx * 16;

    // Find absmax in this block
    float abs_max = 0.0f;
    for (int i = 0; i < 16; i++) {
        abs_max = fmaxf(abs_max, fabsf(input[start + i]));
    }

    // Compute scale (normalize to FP4 range)
    float scale = abs_max / 6.0f;  // FP4 E2M1 max ≈ 6
    output[block_idx].scale = __float2half(scale);

    // Quantize each value
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        float val = input[start + i] / scale;

        // Quantize to FP4 E2M1
        unsigned char fp4 = quantize_to_fp4_e2m1(val);

        // Pack two FP4 values per byte
        int byte_idx = i / 2;
        int nibble_idx = i % 2;

        if (nibble_idx == 0) {
            output[block_idx].values[byte_idx] = fp4;
        } else {
            output[block_idx].values[byte_idx] |= (fp4 << 4);
        }
    }
}
```

**Performance**: NVFP4 on Blackwell B200 achieves **3.6x speedup** over FP16 for GEMM.

---

## Part 2: Integer Quantization

### INT8 Quantization

**Most Common**: Symmetric or asymmetric quantization

```cpp
// Symmetric quantization: range [-127, 127]
__global__ void quantize_int8_symmetric(
    const float* input,
    int8_t* output,
    float* scale,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute scale (absmax / 127)
    __shared__ float max_val;
    if (threadIdx.x == 0) {
        float abs_max = 0.0f;
        for (int i = 0; i < N; i++) {
            abs_max = fmaxf(abs_max, fabsf(input[i]));
        }
        max_val = abs_max / 127.0f;
        scale[0] = max_val;
    }
    __syncthreads();

    if (idx < N) {
        float scaled = input[idx] / max_val;
        output[idx] = (int8_t)roundf(fmaxf(-127.0f, fminf(127.0f, scaled)));
    }
}

// INT8 GEMM using DP4A (Turing+)
__global__ void gemm_int8_dp4a(
    const int8_t* A,
    const int8_t* B,
    int32_t* C,  // Accumulate in INT32
    float A_scale,
    float B_scale,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int32_t sum = 0;

        for (int k = 0; k < K; k += 4) {
            // Pack 4 INT8 values into INT32
            int32_t a_packed = *reinterpret_cast<const int32_t*>(&A[row * K + k]);
            int32_t b_packed = *reinterpret_cast<const int32_t*>(&B[k * N + col]);

            // DP4A: 4-way dot product
            asm volatile("dp4a.s32.s32 %0, %1, %2, %0;"
                        : "+r"(sum)
                        : "r"(a_packed), "r"(b_packed));
        }

        // Dequantize to float
        float result = sum * A_scale * B_scale;
        C[row * N + col] = result;
    }
}
```

### INT8 with Tensor Cores (Turing+)

```cpp
#include <mma.h>
using namespace nvcuda;

__global__ void gemm_int8_tensorcore(
    const int8_t* A,
    const int8_t* B,
    int32_t* C,
    int M, int N, int K
) {
    wmma::fragment<wmma::matrix_a, 8, 8, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 16, int8_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 16, int32_t> c_frag;

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    wmma::fill_fragment(c_frag, 0);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + warpM * 8 * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpN * 8, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpM * 8 * N + warpN * 8, c_frag, N,
                           wmma::mem_row_major);
}
```

---

### INT4 Quantization

**W4A8**: 4-bit weights, 8-bit activations (most common for LLMs)

```cpp
// INT4 stored as packed (2 values per byte)
__global__ void dequantize_int4_to_fp16(
    const unsigned char* weight_int4,  // Packed INT4
    const half* scales,                // Per-channel or per-group scales
    half* weight_fp16,
    int N, int group_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int byte_idx = idx / 2;
        int nibble_idx = idx % 2;

        // Extract 4-bit value
        int8_t val;
        if (nibble_idx == 0) {
            val = (int8_t)(weight_int4[byte_idx] & 0x0F);
        } else {
            val = (int8_t)((weight_int4[byte_idx] >> 4) & 0x0F);
        }

        // Sign extend from 4 bits
        if (val & 0x08) val |= 0xF0;

        // Dequantize
        int group_idx = idx / group_size;
        float scale = __half2float(scales[group_idx]);
        weight_fp16[idx] = __float2half(val * scale);
    }
}

// INT4 GEMM (weights INT4, activations INT8)
__global__ void gemm_int4_int8(
    const unsigned char* A_int4,  // Weights [M, K] packed
    const half* A_scales,         // [M / group_size]
    const int8_t* B_int8,         // Activations [K, N]
    const half* B_scale,          // Scalar
    half* C,
    int M, int N, int K,
    int group_size
) {
    // Step 1: Dequantize INT4 weights to FP16 on-the-fly
    // Step 2: Dequantize INT8 activations to FP16
    // Step 3: FP16 GEMM with Tensor Cores
    // (Implementation similar to above)
}
```

**PyTorch INT4 Optimization Results**: Up to **1.9x speedup** on H100 for LLM inference.

---

## Part 3: Extreme Quantization

### 2-Bit Quantization

**Format**: 4 values in range [-2, -1, 1, 2] or [0, 1, 2, 3]

```cpp
__global__ void quantize_2bit(
    const float* input,
    unsigned char* output,  // 4 values per byte
    float* scale,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute scale
    __shared__ float s;
    if (threadIdx.x == 0) {
        float max_val = 0.0f;
        for (int i = 0; i < N; i++) {
            max_val = fmaxf(max_val, fabsf(input[i]));
        }
        s = max_val / 2.0f;  // Map to [-2, 2]
        scale[0] = s;
    }
    __syncthreads();

    if (idx < N / 4) {
        unsigned char packed = 0;

        for (int i = 0; i < 4; i++) {
            float val = input[idx * 4 + i] / s;
            int quantized = (int)roundf(fmaxf(-2.0f, fminf(2.0f, val)));

            // Map to [0, 3]
            quantized += 2;

            // Pack into 2 bits
            packed |= (quantized << (i * 2));
        }

        output[idx] = packed;
    }
}
```

---

### 1.58-Bit Quantization (Ternary: {-1, 0, 1})

**BitNet b1.58**: Weights in {-1, 0, 1}, activations in INT8

```cpp
__device__ int8_t quantize_ternary(float val) {
    // Quantize to {-1, 0, 1} using absmean threshold
    if (val > 0.5f) return 1;
    else if (val < -0.5f) return -1;
    else return 0;
}

__global__ void quantize_bitnet_1_58(
    const float* input,
    int8_t* output,  // Ternary values
    float* scale,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute absmean
    __shared__ float absmean;
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += fabsf(input[i]);
        }
        absmean = sum / N;
        scale[0] = absmean;
    }
    __syncthreads();

    if (idx < N) {
        float normalized = input[idx] / absmean;
        output[idx] = quantize_ternary(normalized);
    }
}

// Ternary GEMM using lookup tables
__global__ void gemm_ternary(
    const int8_t* A,  // {-1, 0, 1}
    const int8_t* B,  // INT8 activations
    int32_t* C,
    float A_scale,
    float B_scale,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int32_t sum = 0;

        for (int k = 0; k < K; k++) {
            int8_t a = A[row * K + k];
            int8_t b = B[k * N + col];

            // Ternary multiply is just conditional add/subtract
            if (a == 1) sum += b;
            else if (a == -1) sum -= b;
            // If a == 0, no operation
        }

        C[row * N + col] = sum * A_scale * B_scale;
    }
}
```

**Performance**: BitNet achieves **1.37-5.07x speedup** on ARM CPUs, **2.37-6.17x** on x86.

---

### 1-Bit Quantization (Binary Neural Networks)

**Format**: Weights and activations in {-1, 1}

```cpp
// Binary representation: 32 values packed into uint32
__global__ void binarize(
    const float* input,
    uint32_t* output,  // 32 bits = 32 binary values
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N / 32) {
        uint32_t packed = 0;

        for (int i = 0; i < 32; i++) {
            float val = input[idx * 32 + i];
            // Binarize: > 0 → 1, ≤ 0 → 0
            if (val > 0) {
                packed |= (1U << i);
            }
        }

        output[idx] = packed;
    }
}

// Binary GEMM using XNOR and popcount
__global__ void gemm_binary(
    const uint32_t* A,  // Binary weights
    const uint32_t* B,  // Binary activations
    int32_t* C,
    int M, int N, int K_packed  // K_packed = K / 32
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int32_t sum = 0;

        for (int k = 0; k < K_packed; k++) {
            uint32_t a = A[row * K_packed + k];
            uint32_t b = B[k * N + col];

            // XNOR: 1 if bits match, 0 if different
            uint32_t xnor = ~(a ^ b);

            // Popcount: count matching bits
            sum += __popc(xnor);
        }

        // Convert from [0, K] to [-K, K]
        C[row * N + col] = 2 * sum - (K_packed * 32);
    }
}
```

---

## Part 4: Quantization-Aware Training (QAT)

### Fake Quantization

```cpp
__device__ float fake_quantize_int8(float val, float scale) {
    // Simulate INT8 quantization during training
    float quantized = roundf(val / scale);
    quantized = fmaxf(-127.0f, fminf(127.0f, quantized));
    return quantized * scale;
}

__global__ void qat_forward(
    const float* input,
    const float* weight,
    float* output,
    float input_scale,
    float weight_scale,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; k++) {
            // Fake quantize inputs
            float in_quant = fake_quantize_int8(input[row * K + k], input_scale);
            float w_quant = fake_quantize_int8(weight[k * N + col], weight_scale);

            sum += in_quant * w_quant;
        }

        output[row * N + col] = sum;
    }
}
```

---

## Performance Comparison

| Format | Memory | Speed vs FP32 | Accuracy Loss | Hardware |
|--------|--------|---------------|---------------|----------|
| **FP32** | 1.0x | 1.0x | 0% | Universal |
| **BF16** | 0.5x | 1.5-2x | <1% | Ampere+ |
| **FP16** | 0.5x | 2-3x | <1% | Volta+ |
| **FP8** | 0.25x | 3-4x | <2% | Hopper+ |
| **NVFP4** | 0.125x | 3.6x | 1-2% | Blackwell+ |
| **INT8** | 0.25x | 2-4x | 1-3% | Turing+ |
| **INT4** | 0.125x | 1.5-2x | 2-5% | Custom |
| **2-bit** | 0.0625x | 1.2-1.5x | 5-10% | Custom |
| **1.58-bit** | ~0.05x | 2-6x (CPU!) | 3-8% | Custom |
| **1-bit** | 0.03125x | Variable | 10-20% | Custom |

---

## Further Reading

- [NVIDIA TensorRT Quantization](https://docs.nvidia.com/deeplearning/tensorrt/)
- [LLM.int8() Paper](https://arxiv.org/abs/2208.07339)
- [GPTQ Quantization](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [BitNet b1.58 Paper](https://arxiv.org/abs/2402.17764)
