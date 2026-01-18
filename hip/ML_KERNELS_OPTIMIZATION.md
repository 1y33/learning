# Machine Learning Kernels: Advanced Optimization Techniques

## Table of Contents
1. [Flash Attention Architecture](#flash-attention)
2. [Quantization Techniques](#quantization)
3. [Flash Decoding for Long Contexts](#flash-decoding)
4. [KV Cache Optimization](#kv-cache)
5. [Mixture of Experts (MoE) Kernels](#moe-kernels)
6. [Fused Operators](#fused-operators)
7. [Inference Engine Optimizations](#inference-optimizations)

---

## Flash Attention Architecture {#flash-attention}

Flash Attention revolutionizes transformer attention by reducing memory bandwidth requirements through **tiling** and **recomputation**.

### Standard Attention vs Flash Attention

```
STANDARD ATTENTION (Memory-Bound):
┌──────────────────────────────────────────────────────────────┐
│                  Attention = softmax(QK^T / √d) V            │
│                                                                │
│  Step 1: Compute S = QK^T          Step 2: Compute P = softmax(S)│
│  ┌───────┐   ┌───────┐            ┌─────────────────┐        │
│  │   Q   │ × │  K^T  │  →  HBM   │   Softmax       │        │
│  │ [N×d] │   │ [d×N] │  →  [N×N] │   [N×N]         │        │
│  └───────┘   └───────┘            └─────────────────┘        │
│       ↓                                     ↓                  │
│    Load from HBM                         Load from HBM        │
│    Bandwidth: O(N²)                      Bandwidth: O(N²)     │
│                                                                │
│  Step 3: Compute O = PV                                       │
│  ┌─────────────────┐   ┌───────┐                             │
│  │        P        │ × │   V   │  → Output [N×d]             │
│  │      [N×N]      │   │ [N×d] │                             │
│  └─────────────────┘   └───────┘                             │
│                                                                │
│  Memory Requirement: O(N²) for attention matrix               │
│  I/O Complexity: O(N²d + N²) HBM accesses                    │
└──────────────────────────────────────────────────────────────┘

FLASH ATTENTION (Compute-Bound):
┌──────────────────────────────────────────────────────────────┐
│           Tile-Based Computation in SRAM                      │
│                                                                │
│  Key Innovation: Never materialize full N×N matrix            │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Algorithm: Tiling + Online Softmax                    │  │
│  │                                                          │  │
│  │  for block_i in Q_blocks:                              │  │
│  │      Load Q_i from HBM → SRAM (size: Br × d)           │  │
│  │                                                          │  │
│  │      for block_j in KV_blocks:                          │  │
│  │          Load K_j, V_j from HBM → SRAM (size: Bc × d)  │  │
│  │                                                          │  │
│  │          # Compute in SRAM (fast!)                      │  │
│  │          S_ij = Q_i @ K_j^T            [Br × Bc]        │  │
│  │          P_ij = softmax(S_ij)          [Br × Bc]        │  │
│  │          O_i += P_ij @ V_j             [Br × d]         │  │
│  │                                                          │  │
│  │      Store O_i to HBM                                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Memory Requirement: O(N) (linear in sequence length!)        │
│  I/O Complexity: O(N²d²/M) where M = SRAM size               │
│  Speedup: 2-4x faster than standard attention                │
└──────────────────────────────────────────────────────────────┘
```

### Flash Attention Tiling Strategy

```
Sequence Length N = 1024, Head Dimension d = 64
Tile Sizes: Br = 64 (Q block rows), Bc = 64 (KV block cols)

┌────────────────────────────────────────────────────────────┐
│                 ATTENTION MATRIX TILING                     │
│                                                              │
│         K^T (and V) dimension [N = 1024]                    │
│         ┌───────┬───────┬───────┬───────┐                  │
│         │  Bc   │  Bc   │  Bc   │  Bc   │ (64 each)        │
│         │   0   │   1   │   2   │   3   │                  │
│         ├───────┼───────┼───────┼───────┤                  │
│      Br │ Block │ Block │ Block │ Block │                  │
│  Q   0  │ (0,0) │ (0,1) │ (0,2) │ (0,3) │                  │
│      64 ├───────┼───────┼───────┼───────┤                  │
│         │ Block │ Block │ Block │ Block │                  │
│      Br │ (1,0) │ (1,1) │ (1,2) │ (1,3) │                  │
│      1  ├───────┼───────┼───────┼───────┤                  │
│         │ Block │ Block │ Block │ Block │                  │
│      Br │ (2,0) │ (2,1) │ (2,2) │ (2,3) │                  │
│      2  ├───────┼───────┼───────┼───────┤                  │
│         │ Block │ Block │ Block │ Block │                  │
│      Br │ (3,0) │ (3,1) │ (3,2) │ (3,3) │                  │
│      3  └───────┴───────┴───────┴───────┘                  │
│                                                              │
│  Processing Order:                                          │
│    For each Q block (0,1,2,3):                             │
│      For each KV block (0,1,2,3):                          │
│        Compute block attention in SRAM                      │
│        Update running statistics                            │
│                                                              │
│  SRAM Usage per Block:                                      │
│    Q_block: Br × d = 64 × 64 = 4KB (FP16)                  │
│    K_block: Bc × d = 64 × 64 = 4KB                         │
│    V_block: Bc × d = 64 × 64 = 4KB                         │
│    S_block: Br × Bc = 64 × 64 = 8KB (FP32)                 │
│    Total: ~20KB per block (fits in LDS!)                   │
└────────────────────────────────────────────────────────────┘
```

### Online Softmax Algorithm

The key innovation enabling tiling is **online softmax** with rescaling:

```
┌──────────────────────────────────────────────────────────────┐
│              ONLINE SOFTMAX ALGORITHM                         │
│                                                                │
│  Problem: Compute softmax across blocks without storing full  │
│           attention matrix                                     │
│                                                                │
│  Standard Softmax: softmax(x)_i = exp(x_i) / Σⱼ exp(x_j)     │
│                                                                │
│  Numerically Stable: softmax(x)_i = exp(x_i - max(x)) / Z     │
│    where Z = Σⱼ exp(x_j - max(x))                             │
│                                                                │
│  Online Update (block-by-block):                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Initialize: m⁽⁰⁾ = -∞, ℓ⁽⁰⁾ = 0, O⁽⁰⁾ = 0              │  │
│  │                                                          │  │
│  │  For block j = 1 to num_blocks:                         │  │
│  │    1. Compute scores: S_j = Q @ K_j^T                   │  │
│  │                                                          │  │
│  │    2. Update max: m⁽ʲ⁾ = max(m⁽ʲ⁻¹⁾, rowmax(S_j))      │  │
│  │                                                          │  │
│  │    3. Rescale previous statistics:                      │  │
│  │       α = exp(m⁽ʲ⁻¹⁾ - m⁽ʲ⁾)                            │  │
│  │       ℓ⁽ʲ⁾ = α × ℓ⁽ʲ⁻¹⁾                                  │  │
│  │       O⁽ʲ⁾ = α × O⁽ʲ⁻¹⁾                                  │  │
│  │                                                          │  │
│  │    4. Compute new block contribution:                   │  │
│  │       P_j = exp(S_j - m⁽ʲ⁾)                             │  │
│  │       ℓ⁽ʲ⁾ += rowsum(P_j)                                │  │
│  │       O⁽ʲ⁾ += P_j @ V_j                                  │  │
│  │                                                          │  │
│  │  Final normalization: O = O⁽ⁿᵘᵐ_ᵇˡᵒᶜᵏˢ⁾ / ℓ⁽ⁿᵘᵐ_ᵇˡᵒᶜᵏˢ⁾│  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Key Insight: Rescaling factor α maintains numerical           │
│               stability while processing blocks incrementally  │
└──────────────────────────────────────────────────────────────┘
```

### Flash Attention HIP Implementation Sketch

```cpp
__global__ void flash_attention_forward(
    float* O,           // Output [B, N, d]
    const __fp16* Q,    // Query [B, N, d]
    const __fp16* K,    // Key [B, N, d]
    const __fp16* V,    // Value [B, N, d]
    float scale,        // 1/sqrt(d)
    int B, int N, int d
) {
    // Tile sizes
    constexpr int Br = 64;  // Q block rows
    constexpr int Bc = 64;  // KV block cols

    // Shared memory (LDS) allocation
    __shared__ __fp16 Q_smem[Br][d];      // Q tile
    __shared__ __fp16 K_smem[Bc][d];      // K tile
    __shared__ __fp16 V_smem[Bc][d];      // V tile
    __shared__ float S_smem[Br][Bc];      // Attention scores

    // Thread block processes one Q tile
    int batch_idx = blockIdx.z;
    int q_block_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Running statistics per row
    float m[Br];    // Max value
    float l[Br];    // Sum of exponentials
    float O_local[Br][d];  // Accumulated output

    // Initialize
    #pragma unroll
    for (int i = 0; i < Br; i++) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < d; j++) {
            O_local[i][j] = 0.0f;
        }
    }

    // Load Q tile to shared memory
    load_tile_to_smem(Q_smem, Q, batch_idx, q_block_idx, Br, d);
    __syncthreads();

    // Iterate over KV tiles
    int num_kv_blocks = (N + Bc - 1) / Bc;

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        // Load K and V tiles
        load_tile_to_smem(K_smem, K, batch_idx, kv_block_idx, Bc, d);
        load_tile_to_smem(V_smem, V, batch_idx, kv_block_idx, Bc, d);
        __syncthreads();

        // Compute S = Q @ K^T (using thread-level MFMA if available)
        compute_qk_matmul(S_smem, Q_smem, K_smem, scale, Br, Bc, d);

        // Apply causal mask if needed
        // if (causal) apply_causal_mask(S_smem, q_block_idx, kv_block_idx);

        // Online softmax update
        #pragma unroll
        for (int i = tid; i < Br; i += blockDim.x) {
            // Find row max
            float row_max = -INFINITY;
            for (int j = 0; j < Bc; j++) {
                row_max = fmaxf(row_max, S_smem[i][j]);
            }

            // Update global max
            float m_new = fmaxf(m[i], row_max);
            float m_diff_old = m[i] - m_new;
            float m_diff_new = row_max - m_new;

            // Rescale previous statistics
            float alpha = expf(m_diff_old);
            l[i] = alpha * l[i];
            for (int j = 0; j < d; j++) {
                O_local[i][j] *= alpha;
            }

            // Compute softmax for current block
            float row_sum = 0.0f;
            for (int j = 0; j < Bc; j++) {
                float p_ij = expf(S_smem[i][j] - m_new);
                S_smem[i][j] = p_ij;  // Overwrite S with P
                row_sum += p_ij;
            }

            // Update statistics
            l[i] += row_sum;
            m[i] = m_new;
        }
        __syncthreads();

        // Accumulate: O += P @ V
        accumulate_pv_matmul(O_local, S_smem, V_smem, Br, Bc, d);
        __syncthreads();
    }

    // Final normalization and write output
    #pragma unroll
    for (int i = tid; i < Br; i += blockDim.x) {
        for (int j = 0; j < d; j++) {
            O_local[i][j] /= l[i];
        }
    }

    write_output(O, O_local, batch_idx, q_block_idx, Br, d);
}
```

### Flash Attention Performance Characteristics

```
┌──────────────────────────────────────────────────────────────┐
│         FLASH ATTENTION PERFORMANCE ANALYSIS                  │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Complexity Comparison:                                       │
│  ┌────────────────────────┬───────────┬──────────────────┐   │
│  │ Metric                 │ Standard  │ Flash Attention  │   │
│  ├────────────────────────┼───────────┼──────────────────┤   │
│  │ Memory                 │ O(N²)     │ O(N)             │   │
│  │ HBM Accesses           │ O(N²d)    │ O(N²d²/M)        │   │
│  │ Compute (FLOPs)        │ O(N²d)    │ O(N²d)           │   │
│  │ Numerical Precision    │ Exact     │ Exact (same!)    │   │
│  └────────────────────────┴───────────┴──────────────────┘   │
│                                                                │
│  Real-World Speedups (AMD MI250X):                            │
│    Sequence Length 512:   1.5-2x                              │
│    Sequence Length 2K:    2-3x                                │
│    Sequence Length 8K:    3-4x                                │
│    Sequence Length 128K:  5-7x                                │
│                                                                │
│  Why Flash Attention Wins:                                    │
│    • GPUs are memory-bandwidth limited for attention          │
│    • HBM bandwidth: ~1.6 TB/s                                 │
│    • SRAM bandwidth: ~10 TB/s (per CU)                        │
│    • Reducing HBM accesses = massive speedup                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Quantization Techniques {#quantization}

Quantization reduces memory footprint and increases throughput by using lower-precision data types.

### Quantization Data Types

```
┌──────────────────────────────────────────────────────────────┐
│              PRECISION FORMATS FOR ML                         │
├─────────┬────────┬──────────────┬───────────┬───────────────┤
│ Format  │ Bits   │ Range        │ Precision │ Use Case      │
├─────────┼────────┼──────────────┼───────────┼───────────────┤
│ FP32    │ 32     │ ±3.4e38      │ ~7 digits │ Training      │
│ (IEEE)  │        │              │           │ (baseline)    │
├─────────┼────────┼──────────────┼───────────┼───────────────┤
│ FP16    │ 16     │ ±65,504      │ ~3 digits │ Mixed-prec    │
│ (IEEE)  │        │              │           │ training      │
├─────────┼────────┼──────────────┼───────────┼───────────────┤
│ BF16    │ 16     │ ±3.4e38      │ ~2 digits │ Training,     │
│ (Brain) │ 1:8:7  │ (same as FP32│           │ inference     │
├─────────┼────────┼──────────────┼───────────┼───────────────┤
│ FP8     │ 8      │ ±448 (E4M3) │ <1 digit  │ Inference,    │
│ E4M3    │ 1:4:3  │ ±57,344(E5M2│           │ training      │
│ E5M2    │ 1:5:2  │              │           │               │
├─────────┼────────┼──────────────┼───────────┼───────────────┤
│ INT8    │ 8      │ -128 to 127  │ Integer   │ Inference     │
│         │        │              │           │ (quantized)   │
├─────────┼────────┼──────────────┼───────────┼───────────────┤
│ INT4    │ 4      │ -8 to 7      │ Integer   │ Ultra-low     │
│         │        │              │           │ memory        │
└─────────┴────────┴──────────────┴───────────┴───────────────┘

Bit Layout:
┌─────────────────────────────────────────────────────────┐
│ FP32: [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]           │
│       1  8 bits exp 23 bits mantissa                    │
│                                                          │
│ FP16: [S][EEEEE][MMMMMMMMMM]                           │
│       1  5 bits  10 bits                                │
│                                                          │
│ BF16: [S][EEEEEEEE][MMMMMMM]                           │
│       1  8 bits     7 bits (truncated FP32)            │
│                                                          │
│ FP8 E4M3: [S][EEEE][MMM]                               │
│           1  4 bits 3 bits                              │
│                                                          │
│ FP8 E5M2: [S][EEEEE][MM]                               │
│           1  5 bits  2 bits                             │
│                                                          │
│ INT8: [SMMMMMMM]                                        │
│       1 sign + 7 magnitude bits                         │
└─────────────────────────────────────────────────────────┘
```

### Weight-Only Quantization

```
┌──────────────────────────────────────────────────────────────┐
│           WEIGHT-ONLY QUANTIZATION                            │
│                                                                │
│  Concept: Quantize weights to low precision (INT8/INT4),     │
│           keep activations in higher precision (FP16)         │
│                                                                │
│  Forward Pass:                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Weights (Storage)     Activations                     │  │
│  │  ┌───────────┐        ┌───────────┐                   │  │
│  │  │  INT8/4   │  →     │   FP16    │                   │  │
│  │  │ W_quant   │  Dequant│     X     │                   │  │
│  │  └─────┬─────┘    ↓   └───────────┘                   │  │
│  │        │      ┌───────┐                                │  │
│  │        └─────→│ FP16  │                                │  │
│  │               │   W   │                                │  │
│  │               └───┬───┘                                │  │
│  │                   │                                     │  │
│  │                   ▼                                     │  │
│  │              ┌────────┐                                │  │
│  │              │ MatMul │  Y = W @ X                     │  │
│  │              │ (FP16) │                                │  │
│  │              └────────┘                                │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Quantization Formula (Symmetric):                            │
│    W_quant = round(W / scale)                                 │
│    scale = max(|W|) / 127  (for INT8)                        │
│                                                                │
│  Dequantization:                                              │
│    W_dequant = W_quant × scale                                │
│                                                                │
│  Benefits:                                                    │
│    • 4x memory reduction (FP32 → INT8)                        │
│    • 8x memory reduction (FP32 → INT4)                        │
│    • Lower memory bandwidth during inference                  │
│    • Minimal accuracy loss (<1% typically)                    │
└──────────────────────────────────────────────────────────────┘
```

### Quantization Granularity

```
┌──────────────────────────────────────────────────────────────┐
│         QUANTIZATION GRANULARITY TRADEOFFS                    │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  PER-TENSOR QUANTIZATION (Coarsest)                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Weight Matrix [M×N]                                   │  │
│  │  ┌───────────────────────────────────┐                │  │
│  │  │ All elements share ONE scale      │                │  │
│  │  │                                    │                │  │
│  │  │      scale = max(|W|) / 127       │                │  │
│  │  └───────────────────────────────────┘                │  │
│  │  Storage: M×N INT8 + 1 FP32 scale                     │  │
│  │  Accuracy: Lowest                                      │  │
│  │  Speed: Fastest                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  PER-CHANNEL QUANTIZATION (Medium)                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Weight Matrix [M×N]                                   │  │
│  │  ┌──┬──┬──┬──┬──┬──┬──┬──┐                            │  │
│  │  │s0│  │  │  │  │  │  │  │  Each row has scale        │  │
│  │  ├──┼──┼──┼──┼──┼──┼──┼──┤                            │  │
│  │  │s1│  │  │  │  │  │  │  │                            │  │
│  │  ├──┼──┼──┼──┼──┼──┼──┼──┤                            │  │
│  │  │s2│  │  │  │  │  │  │  │                            │  │
│  │  └──┴──┴──┴──┴──┴──┴──┴──┘                            │  │
│  │  Storage: M×N INT8 + M FP32 scales                    │  │
│  │  Accuracy: Medium                                      │  │
│  │  Speed: Medium (vectorizable)                          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  GROUP-WISE QUANTIZATION (Fine)                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Weight Matrix [M×N], Group Size G=64                  │  │
│  │  ┌──┬──┬──┬──┐┌──┬──┬──┬──┐                           │  │
│  │  │ s0  Group 0││ s1  Group 1│                          │  │
│  │  │64 elements ││64 elements │                          │  │
│  │  └──┴──┴──┴──┘└──┴──┴──┴──┘                           │  │
│  │  Storage: M×N INT8 + (M×N/G) FP32 scales              │  │
│  │  Accuracy: High                                        │  │
│  │  Speed: Slower (more scales to load)                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  BLOCK-SCALE QUANTIZATION (Advanced)                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  2D blocking: [B×B] tiles each with own scale         │  │
│  │  ┌────┬────┬────┐                                      │  │
│  │  │ s00│ s01│ s02│  Typical: B=32 or B=64               │  │
│  │  ├────┼────┼────┤                                      │  │
│  │  │ s10│ s11│ s12│  AITER uses this (2x speedup)       │  │
│  │  ├────┼────┼────┤                                      │  │
│  │  │ s20│ s21│ s22│                                      │  │
│  │  └────┴────┴────┘                                      │  │
│  │  Storage: M×N INT8 + (M/B)×(N/B) FP32 scales          │  │
│  │  Accuracy: Very High                                   │  │
│  │  Speed: Optimized for GPU tiles                        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### INT8 Quantized GEMM Kernel

```cpp
__global__ void int8_gemm_with_dequant(
    float* C,                    // Output [M×N] FP32
    const int8_t* A_quant,       // Input [M×K] INT8
    const int8_t* B_quant,       // Weights [K×N] INT8
    const float* A_scale,        // Per-row scales [M]
    const float* B_scale,        // Per-column scales [N]
    int M, int K, int N
) {
    // Use MFMA INT8 instructions for high throughput
    using int4 = __attribute__((ext_vector_type(4))) int;
    using int16 = __attribute__((ext_vector_type(16))) int;

    __shared__ int8_t A_smem[64][128];
    __shared__ int8_t B_smem[128][64];

    int tid = threadIdx.x;
    int m = blockIdx.x * 64 + threadIdx.y;
    int n = blockIdx.y * 64 + threadIdx.z;

    // Accumulator (INT32)
    int16 acc = {0};

    // Main loop over K
    for (int k_tile = 0; k_tile < K; k_tile += 128) {
        // Load INT8 tiles to LDS
        load_int8_tile(A_smem, A_quant, m, k_tile, M, K);
        load_int8_tile(B_smem, B_quant, k_tile, n, K, N);
        __syncthreads();

        // INT8 Matrix multiplication using MFMA
        // v_mfma_i32_16x16x16i8: Computes 16×16 INT32 output from INT8 inputs
        for (int k = 0; k < 128; k += 16) {
            int4 a_frag = load_fragment_int8(A_smem, k);
            int4 b_frag = load_fragment_int8(B_smem, k);

            // MFMA: acc = a × b + acc (INT8 → INT32)
            acc = __builtin_amdgcn_mfma_i32_16x16x16i8(
                a_frag, b_frag, acc, 0, 0, 0
            );
        }
        __syncthreads();
    }

    // Dequantization: Convert INT32 → FP32 and apply scales
    float scale_a = A_scale[m];
    float scale_b = B_scale[n];
    float combined_scale = scale_a * scale_b;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        C[m * N + n + i] = static_cast<float>(acc[i]) * combined_scale;
    }
}
```

### FP8 Quantization (CDNA3+)

```
┌──────────────────────────────────────────────────────────────┐
│              FP8 FORMATS (AMD MI300)                          │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Two FP8 formats with different tradeoffs:                    │
│                                                                │
│  E4M3 (1:4:3) - HIGHER PRECISION, SMALLER RANGE              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Sign: 1 bit                                           │  │
│  │  Exponent: 4 bits (range: 2^-6 to 2^8)                │  │
│  │  Mantissa: 3 bits                                      │  │
│  │  Range: ±448                                           │  │
│  │  Smallest positive: ~0.015625                          │  │
│  │  Use case: Activations, forward pass                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  E5M2 (1:5:2) - LARGER RANGE, LOWER PRECISION                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Sign: 1 bit                                           │  │
│  │  Exponent: 5 bits (range: 2^-14 to 2^16)              │  │
│  │  Mantissa: 2 bits                                      │  │
│  │  Range: ±57,344                                        │  │
│  │  Smallest positive: ~0.0000610                         │  │
│  │  Use case: Gradients, backward pass                    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Hardware Support:                                            │
│    • CDNA3 (MI300): Native FP8 MFMA instructions              │
│    • Throughput: 2x FP16 (1950 TFLOPS FP8 vs 975 TFLOPS FP16)│
│    • Memory: 2x reduction vs FP16                             │
└──────────────────────────────────────────────────────────────┘
```

---

## Flash Decoding for Long Contexts {#flash-decoding}

### The Decoding Problem

```
┌──────────────────────────────────────────────────────────────┐
│         ATTENTION DURING AUTOREGRESSIVE DECODING              │
│                                                                │
│  Prefill Phase (Process entire prompt):                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Q: [N_prompt, d]        Full parallelism               │  │
│  │  KV: [N_prompt, d]       Batch size: Large (128+)       │  │
│  │  Compute-bound: Good GPU utilization                    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Decode Phase (Generate one token at a time):                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Q: [1, d]               Single query!                  │  │
│  │  KV: [N_context, d]      Growing context (could be 128K)│  │
│  │  Memory-bound: Poor GPU utilization                     │  │
│  │  Bottleneck: Reading entire KV cache for one token      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Problem Scaling:                                             │
│    Context N = 128K, Batch B = 1                              │
│    KV cache reads per token: 2 × 128K × d × 2 bytes           │
│                             = 128K × 128 × 2 = 32 MB          │
│    For 100 tokens/sec: 3.2 GB/s bandwidth (underutilizes GPU)│
└──────────────────────────────────────────────────────────────┘
```

### Flash Decoding Algorithm

Flash Decoding parallelizes across the sequence dimension during decoding:

```
┌──────────────────────────────────────────────────────────────┐
│              FLASH DECODING STRATEGY                          │
│                                                                │
│  Key Insight: Parallelize over KV sequence, not batch        │
│                                                                │
│  Standard Decoding (Sequential over K):                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Single GPU thread block processes:                    │  │
│  │    score[i] = Q @ K[i] for i=0...N                     │  │
│  │    softmax(scores)                                      │  │
│  │    output = Σ prob[i] × V[i]                           │  │
│  │                                                          │  │
│  │  Parallelism: Batch size only (small during decode)    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Flash Decoding (Parallel over K):                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Launch blocks = ceil(N_context / block_size)          │  │
│  │                                                          │  │
│  │  Each block i computes:                                 │  │
│  │    scores[i] = Q @ K[i*B : (i+1)*B]                    │  │
│  │    max[i] = max(scores[i])                             │  │
│  │    sum[i] = Σ exp(scores[i] - max[i])                  │  │
│  │    partial_out[i] = Σ exp(scores[i] - max[i]) × V[i]  │  │
│  │                                                          │  │
│  │  Reduction phase:                                       │  │
│  │    global_max = max(max[0], max[1], ..., max[P])      │  │
│  │    global_sum = Σ exp(max[i] - global_max) × sum[i]   │  │
│  │    output = Σ exp(max[i] - global_max) × partial_out[i]│  │
│  │             / global_sum                                │  │
│  │                                                          │  │
│  │  Parallelism: N_context / block_size (massive!)        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Speedup: 5-10x for long contexts (N > 32K)                  │
└──────────────────────────────────────────────────────────────┘
```

### Flash Decoding Kernel Sketch

```cpp
// Phase 1: Parallel computation over KV blocks
__global__ void flash_decode_phase1(
    float* partial_out,       // [num_blocks, d]
    float* max_vals,          // [num_blocks]
    float* sum_vals,          // [num_blocks]
    const __fp16* Q,          // [1, d]  (single query)
    const __fp16* K_cache,    // [N, d]  (long context)
    const __fp16* V_cache,    // [N, d]
    int N, int d
) {
    constexpr int BLOCK_SIZE = 256;
    int block_idx = blockIdx.x;
    int start_idx = block_idx * BLOCK_SIZE;
    int end_idx = min(start_idx + BLOCK_SIZE, N);

    __shared__ __fp16 Q_smem[128];
    __shared__ __fp16 K_smem[BLOCK_SIZE][128];
    __shared__ __fp16 V_smem[BLOCK_SIZE][128];
    __shared__ float scores[BLOCK_SIZE];

    // Load Q to shared memory
    load_to_smem(Q_smem, Q, d);

    // Load K, V block
    load_block_to_smem(K_smem, K_cache, start_idx, end_idx, d);
    load_block_to_smem(V_smem, V_cache, start_idx, end_idx, d);
    __syncthreads();

    // Compute attention scores
    int tid = threadIdx.x;
    for (int i = tid; i < (end_idx - start_idx); i += blockDim.x) {
        float score = 0.0f;
        for (int j = 0; j < d; j++) {
            score += float(Q_smem[j]) * float(K_smem[i][j]);
        }
        scores[i] = score / sqrtf(d);
    }
    __syncthreads();

    // Find block max (for numerical stability)
    __shared__ float block_max;
    if (tid == 0) {
        float max_val = -INFINITY;
        for (int i = 0; i < (end_idx - start_idx); i++) {
            max_val = fmaxf(max_val, scores[i]);
        }
        block_max = max_val;
        max_vals[block_idx] = max_val;
    }
    __syncthreads();

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < (end_idx - start_idx); i += blockDim.x) {
        float exp_val = expf(scores[i] - block_max);
        scores[i] = exp_val;
        sum += exp_val;
    }

    // Reduce sum across threads
    sum = block_reduce_sum(sum);
    if (tid == 0) {
        sum_vals[block_idx] = sum;
    }
    __syncthreads();

    // Compute partial output: Σ exp(score) × V
    float partial[128] = {0};
    for (int i = 0; i < (end_idx - start_idx); i++) {
        float weight = scores[i];
        for (int j = tid; j < d; j += blockDim.x) {
            partial[j] += weight * float(V_smem[i][j]);
        }
    }

    // Write partial results
    write_partial_output(partial_out + block_idx * d, partial, d);
}

// Phase 2: Reduce partial results
__global__ void flash_decode_phase2(
    float* output,            // [d]
    const float* partial_out, // [num_blocks, d]
    const float* max_vals,    // [num_blocks]
    const float* sum_vals,    // [num_blocks]
    int num_blocks, int d
) {
    // Find global max
    float global_max = max_vals[0];
    for (int i = 1; i < num_blocks; i++) {
        global_max = fmaxf(global_max, max_vals[i]);
    }

    // Compute rescaled sum
    float global_sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        global_sum += expf(max_vals[i] - global_max) * sum_vals[i];
    }

    // Aggregate partial outputs with rescaling
    int tid = threadIdx.x;
    for (int j = tid; j < d; j += blockDim.x) {
        float result = 0.0f;
        for (int i = 0; i < num_blocks; i++) {
            float scale = expf(max_vals[i] - global_max);
            result += scale * partial_out[i * d + j];
        }
        output[j] = result / global_sum;
    }
}
```

---

## KV Cache Optimization {#kv-cache}

### KV Cache Memory Layout

```
┌──────────────────────────────────────────────────────────────┐
│              KV CACHE ORGANIZATION                            │
│                                                                │
│  Dimensions:                                                  │
│    B = Batch size                                             │
│    L = Sequence length (dynamic, grows during generation)     │
│    H = Number of heads                                        │
│    D = Head dimension                                         │
│                                                                │
│  Naive Layout: [B, L, H, D]                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Poor for cache efficiency!                            │  │
│  │  Reading one head requires strided access               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Optimized Layout: [B, H, L, D]                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Better! Each head is contiguous in memory             │  │
│  │  ┌─────────────────────────────────┐                   │  │
│  │  │ Head 0: [L×D contiguous block]  │                   │  │
│  │  ├─────────────────────────────────┤                   │  │
│  │  │ Head 1: [L×D contiguous block]  │                   │  │
│  │  ├─────────────────────────────────┤                   │  │
│  │  │ ...                              │                   │  │
│  │  └─────────────────────────────────┘                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Page-Based Layout (vLLM style):                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Sequence divided into fixed-size pages (e.g., 16)     │  │
│  │                                                          │  │
│  │  Page Table:  Seq 0 → [Page 5, Page 12, Page 3, ...]  │  │
│  │               Seq 1 → [Page 8, Page 1, Page 15, ...]   │  │
│  │                                                          │  │
│  │  Physical Memory:                                       │  │
│  │  ┌────┬────┬────┬────┬────┬────┐                       │  │
│  │  │ P0 │ P1 │ P2 │ P3 │ P4 │... │                       │  │
│  │  └────┴────┴────┴────┴────┴────┘                       │  │
│  │                                                          │  │
│  │  Benefits:                                              │  │
│  │    • No memory fragmentation                            │  │
│  │    • Efficient sequence packing                         │  │
│  │    • Shared prefixes (system prompts)                   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### KV Cache Quantization

```cpp
// INT8 quantized KV cache
struct QuantizedKVCache {
    int8_t* K_quant;      // [B, H, L, D] quantized keys
    int8_t* V_quant;      // [B, H, L, D] quantized values
    float* K_scale;       // [B, H, L] per-token scales for K
    float* V_scale;       // [B, H, L] per-token scales for V

    // Quantize and store
    __device__ void store_kv(
        const __fp16* K_new,
        const __fp16* V_new,
        int batch, int head, int seq_pos
    ) {
        // Quantize K
        float k_max = find_abs_max(K_new, D);
        float k_scale = k_max / 127.0f;
        K_scale[index(batch, head, seq_pos)] = k_scale;

        for (int d = 0; d < D; d++) {
            int8_t k_quant = round(float(K_new[d]) / k_scale);
            K_quant[index(batch, head, seq_pos, d)] = k_quant;
        }

        // Quantize V (similar)
        // ...
    }

    // Load and dequantize
    __device__ void load_kv(
        __fp16* K_out,
        __fp16* V_out,
        int batch, int head, int seq_pos
    ) {
        float k_scale = K_scale[index(batch, head, seq_pos)];

        for (int d = 0; d < D; d++) {
            int8_t k_quant = K_quant[index(batch, head, seq_pos, d)];
            K_out[d] = __fp16(float(k_quant) * k_scale);
        }

        // Dequantize V (similar)
        // ...
    }
};

// Memory savings:
//   FP16: 2 bytes per element
//   INT8: 1 byte per element + scale overhead
//   Compression: ~2x (accounting for scales)
//   Quality: <0.5% accuracy loss with per-token quantization
```

---

## Mixture of Experts (MoE) Kernels {#moe-kernels}

### MoE Architecture

```
┌──────────────────────────────────────────────────────────────┐
│         MIXTURE OF EXPERTS COMPUTATION                        │
│                                                                │
│  Input: X [batch×seq, hidden_dim]                            │
│                                                                │
│  Step 1: ROUTING                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Router Network: X → scores [batch×seq, num_experts]   │  │
│  │  TopK Selection: Select top K experts per token        │  │
│  │                                                          │  │
│  │  Example (K=2, E=8 experts):                            │  │
│  │    Token 0: experts [3, 7] with weights [0.6, 0.4]     │  │
│  │    Token 1: experts [1, 3] with weights [0.7, 0.3]     │  │
│  │    Token 2: experts [0, 5] with weights [0.5, 0.5]     │  │
│  └────────────────────────────────────────────────────────┘  │
│                        ↓                                       │
│  Step 2: SORTING AND GROUPING                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Group tokens by expert assignment:                     │  │
│  │    Expert 0: [Token 2]                                  │  │
│  │    Expert 1: [Token 1]                                  │  │
│  │    Expert 3: [Token 0, Token 1]                         │  │
│  │    Expert 5: [Token 2]                                  │  │
│  │    Expert 7: [Token 0]                                  │  │
│  │                                                          │  │
│  │  Challenges:                                            │  │
│  │    • Dynamic batch sizes per expert (load imbalance)    │  │
│  │    • Sorting overhead                                   │  │
│  │    • Non-coalesced memory access                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                        ↓                                       │
│  Step 3: EXPERT COMPUTATION (Parallel)                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  For each expert e:                                     │  │
│  │    tokens_e = gather(X, assignments[e])                │  │
│  │    output_e = FFN_expert_e(tokens_e)                   │  │
│  │                                                          │  │
│  │  FFN: X → W1(X) → activation → W2(X)                   │  │
│  │       [d] → [4d] → [4d] → [d]                          │  │
│  └────────────────────────────────────────────────────────┘  │
│                        ↓                                       │
│  Step 4: COMBINE RESULTS                                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  For each token i:                                      │  │
│  │    output[i] = Σ weight[i,k] × expert_output[e_k,i]   │  │
│  │                                                          │  │
│  │  Scatter results back to original token positions      │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Fused MoE Kernel

AITER achieves 3x speedup by fusing operations:

```cpp
__global__ void fused_moe_kernel(
    float* output,              // [num_tokens, hidden_dim]
    const float* input,         // [num_tokens, hidden_dim]
    const float* router_logits, // [num_tokens, num_experts]
    const float** expert_w1,    // [num_experts] -> [hidden_dim, ffn_dim]
    const float** expert_w2,    // [num_experts] -> [ffn_dim, hidden_dim]
    int num_tokens,
    int hidden_dim,
    int ffn_dim,
    int num_experts,
    int top_k
) {
    // FUSION 1: Router + TopK + Softmax (fused)
    __shared__ int expert_ids[BLOCK_SIZE][MAX_TOPK];
    __shared__ float expert_weights[BLOCK_SIZE][MAX_TOPK];

    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx < num_tokens) {
        // Fused routing computation
        float logits[MAX_EXPERTS];
        for (int e = 0; e < num_experts; e++) {
            logits[e] = router_logits[token_idx * num_experts + e];
        }

        // TopK selection (efficient partial sort)
        topk_select(expert_ids[threadIdx.x], expert_weights[threadIdx.x],
                    logits, num_experts, top_k);

        // Softmax normalization (fused)
        float sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            expert_weights[threadIdx.x][k] = expf(expert_weights[threadIdx.x][k]);
            sum += expert_weights[threadIdx.x][k];
        }
        for (int k = 0; k < top_k; k++) {
            expert_weights[threadIdx.x][k] /= sum;
        }
    }
    __syncthreads();

    // FUSION 2: Expert computation + Weighted combination (fused)
    // Instead of: compute all experts → combine
    // Do: compute and accumulate on-the-fly

    float result[HIDDEN_DIM] = {0};

    for (int k = 0; k < top_k; k++) {
        int expert_id = expert_ids[threadIdx.x][k];
        float weight = expert_weights[threadIdx.x][k];

        // Load expert weights (coalesced)
        const float* w1 = expert_w1[expert_id];
        const float* w2 = expert_w2[expert_id];

        // FUSION 3: FFN layers fused
        // Traditional: X → W1 → store → activation → load → W2
        // Fused: X → W1 → activation → W2 (all in registers!)

        float intermediate[FFN_DIM];

        // W1: hidden_dim → ffn_dim
        for (int i = 0; i < ffn_dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hidden_dim; j++) {
                sum += input[token_idx * hidden_dim + j] * w1[j * ffn_dim + i];
            }
            // Fused activation (SwiGLU)
            intermediate[i] = sum * sigmoidf(sum);  // Swish activation
        }

        // W2: ffn_dim → hidden_dim (accumulate directly to result)
        for (int i = 0; i < hidden_dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < ffn_dim; j++) {
                sum += intermediate[j] * w2[j * hidden_dim + i];
            }
            result[i] += weight * sum;  // Weighted combination fused!
        }
    }

    // Write final result
    for (int i = 0; i < hidden_dim; i++) {
        output[token_idx * hidden_dim + i] = result[i];
    }
}

// Performance benefits:
//   1. Eliminates intermediate buffers (saves memory bandwidth)
//   2. Keeps data in registers/LDS (10x faster than HBM)
//   3. Reduces kernel launch overhead (1 kernel vs 5+)
//   4. Better instruction-level parallelism
//   Result: 3x speedup reported by AITER
```

---

## Fused Operators {#fused-operators}

### Common Fusion Patterns

```
┌──────────────────────────────────────────────────────────────┐
│           OPERATOR FUSION PATTERNS                            │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  1. ELEMENTWISE FUSION                                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Unfused: X → LayerNorm → GELU → Dropout → output     │  │
│  │           [Store]       [Store] [Store]                │  │
│  │                                                          │  │
│  │  Fused: X → [LayerNorm+GELU+Dropout] → output         │  │
│  │           All in registers                              │  │
│  │  Speedup: 2-3x (eliminates 2 HBM round-trips)         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  2. GEMM + EPILOGUE FUSION                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Unfused: Y = MatMul(A, B)                             │  │
│  │           Y = Y + bias                                  │  │
│  │           Y = activation(Y)                             │  │
│  │                                                          │  │
│  │  Fused: Y = activation(MatMul(A, B) + bias)            │  │
│  │  ┌─────────┐                                            │  │
│  │  │  GEMM   │ → [+bias] → [activation] → output         │  │
│  │  │ Compute │   (fused epilogue)                         │  │
│  │  └─────────┘                                            │  │
│  │  Speedup: 1.5-2x                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  3. NORMALIZATION + RESIDUAL FUSION                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Unfused: Y = LayerNorm(X)                             │  │
│  │           Z = Y + residual                              │  │
│  │                                                          │  │
│  │  Fused: Z = LayerNorm(X) + residual                    │  │
│  │  Algorithm:                                             │  │
│  │    mean = Σ X[i] / N                                   │  │
│  │    var = Σ (X[i] - mean)² / N                          │  │
│  │    Y[i] = (X[i] - mean) / √(var + ε) × γ + β          │  │
│  │    Z[i] = Y[i] + residual[i]  ← fused!                │  │
│  │  Speedup: 1.3-1.5x                                      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  4. ATTENTION + SOFTMAX FUSION (Flash Attention)             │
│  └────────────────────────────────────────────────────────┘  │
│     Covered in Flash Attention section (2-4x speedup)        │
└──────────────────────────────────────────────────────────────┘
```

### Fused LayerNorm + Residual Kernel

```cpp
__global__ void fused_layernorm_residual(
    float* output,           // [N] output
    const float* input,      // [N] input
    const float* residual,   // [N] residual connection
    const float* gamma,      // [N] scale parameters
    const float* beta,       // [N] shift parameters
    int N,
    float epsilon = 1e-5f
) {
    __shared__ float smem_mean;
    __shared__ float smem_var;

    int tid = threadIdx.x;
    int idx = blockIdx.x * N + tid;

    // Load input
    float x = input[idx];

    // Parallel reduction for mean
    float sum = x;
    sum = block_reduce_sum(sum);  // Warp shuffle + shared memory
    if (tid == 0) {
        smem_mean = sum / N;
    }
    __syncthreads();
    float mean = smem_mean;

    // Parallel reduction for variance
    float diff = x - mean;
    float sq = diff * diff;
    sq = block_reduce_sum(sq);
    if (tid == 0) {
        smem_var = sq / N;
    }
    __syncthreads();
    float var = smem_var;

    // Normalize
    float inv_std = rsqrtf(var + epsilon);  // Fast inverse sqrt
    float normalized = diff * inv_std;

    // Apply affine transform
    float y = normalized * gamma[tid] + beta[tid];

    // FUSED: Add residual (no separate kernel!)
    y += residual[idx];

    // Write output
    output[idx] = y;
}

// Performance impact:
//   Unfused: 3 kernel launches, 5 HBM accesses per element
//   Fused: 1 kernel launch, 3 HBM accesses per element
//   Speedup: ~1.4x
```

---

## Inference Engine Optimizations {#inference-optimizations}

### Continuous Batching

```
┌──────────────────────────────────────────────────────────────┐
│           CONTINUOUS BATCHING STRATEGY                        │
│                                                                │
│  Traditional Static Batching:                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Batch 1: [Seq A (100 tokens)]                         │  │
│  │           [Seq B (500 tokens)]  ← Wait for longest!    │  │
│  │           [Seq C (50 tokens)]                           │  │
│  │  All finish at t=500 (bottlenecked by Seq B)           │  │
│  │  GPU utilization: Poor after short sequences finish     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Continuous Batching (vLLM, AITER):                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  t=0:   [Seq A] [Seq B] [Seq C]                        │  │
│  │  t=50:  [Seq A] [Seq B] [Seq D] ← C finishes, D joins  │  │
│  │  t=100: [Seq E] [Seq B] [Seq D] ← A finishes, E joins  │  │
│  │  t=200: [Seq E] [Seq F] [Seq D] ← Continuous!          │  │
│  │                                                          │  │
│  │  Benefits:                                              │  │
│  │    • 2-3x higher throughput                             │  │
│  │    • Better GPU utilization (>90%)                      │  │
│  │    • Lower latency for short sequences                  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Speculative Decoding

```
┌──────────────────────────────────────────────────────────────┐
│           SPECULATIVE DECODING                                │
│                                                                │
│  Idea: Use small "draft" model to speculate multiple tokens, │
│        verify with large "target" model in parallel           │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Step 1: Draft model generates K tokens speculatively  │  │
│  │    Small model (1B params): Fast, lower quality        │  │
│  │    Token sequence: [t1, t2, t3, t4, t5]                │  │
│  │                                                          │  │
│  │  Step 2: Target model verifies all K tokens in parallel│  │
│  │    Large model (70B params): Slow, high quality        │  │
│  │    Verification: [✓, ✓, ✓, ✗, -]                       │  │
│  │    Accept first 3, reject 4th                           │  │
│  │                                                          │  │
│  │  Step 3: Target model generates correct token for t4   │  │
│  │    Continue from t4 with new draft sequence            │  │
│  │                                                          │  │
│  │  Expected speedup: 1.5-2.5x                             │  │
│  │    Depends on draft model accuracy                      │  │
│  │    Best case: Accept all K tokens (K×speedup)          │  │
│  │    Worst case: Reject all (no speedup)                 │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Summary: ML Kernel Optimization Checklist

```
✅ Flash Attention:
  • Use tiled computation to reduce HBM accesses
  • Implement online softmax for numerical stability
  • Target SRAM/LDS for intermediate results
  • Expected: 2-4x speedup over standard attention

✅ Quantization:
  • INT8 weight-only: 4x memory reduction, <1% accuracy loss
  • FP8 (CDNA3): 2x compute throughput, 2x memory savings
  • Use per-channel or block-scale for best accuracy
  • Fuse dequantization with GEMM computation

✅ Flash Decoding:
  • Parallelize over sequence dimension during decode
  • Critical for long-context inference (N > 32K)
  • Expected: 5-10x speedup for long contexts

✅ KV Cache:
  • Use page-based memory allocation
  • Quantize to INT8/FP8 (2-4x memory savings)
  • Optimize layout for cache efficiency

✅ MoE Kernels:
  • Fuse routing + expert computation + combination
  • Minimize sorting/gathering overhead
  • Balance load across experts
  • Expected: 2-3x speedup with fusion

✅ Operator Fusion:
  • Fuse elementwise operations
  • GEMM epilogue fusion (bias, activation)
  • Normalization + residual connections
  • Keep intermediate values in registers/LDS
```

---

**Next:** See `KERNEL_FUSION_ADVANCED_PATTERNS.md` for deep dive into kernel fusion techniques, including RL-based fusion strategies, actor-critic methods, and advanced composition patterns.
