# HIP Assembly and MFMA Programming Guide: Deep Dive into GPU ISA

## Table of Contents
1. [AMD GPU ISA Architecture](#isa-architecture)
2. [Reading and Understanding AMDGCN Assembly](#reading-assembly)
3. [Register Architecture Deep Dive](#register-architecture)
4. [MFMA (Matrix Fused Multiply-Add) Instructions](#mfma-instructions)
5. [WMMA (Wave Matrix Multiply-Accumulate)](#wmma)
6. [Writing Inline Assembly in HIP](#inline-assembly)
7. [Assembly-Level Optimizations](#assembly-optimizations)
8. [rocWMMA Library Usage](#rocwmma)
9. [AITER Assembly Kernels](#aiter)

---

## AMD GPU ISA Architecture {#isa-architecture}

### Supported ISA Versions

AMD provides multiple GPU architecture families, each with distinct ISA specifications:

```
┌────────────────────────────────────────────────────────────────┐
│                   AMD GPU ARCHITECTURE TIMELINE                 │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GCN (Graphics Core Next) - Legacy                              │
│  ├─ GCN 1.0 (gfx600-gfx601): Southern Islands                  │
│  ├─ GCN 2.0 (gfx700-gfx701): Sea Islands                       │
│  ├─ GCN 3.0 (gfx803-gfx805): Volcanic Islands / Fiji           │
│  └─ GCN 4.0 (gfx900-gfx906): Vega                              │
│                                                                  │
│  CDNA (Compute DNA) - HPC/AI Focused                           │
│  ├─ CDNA 1 (gfx908): MI100                                     │
│  │   • Wave64, Matrix Cores, AGPRs                             │
│  │   • MFMA: FP64, FP32, FP16, BF16, INT8                      │
│  ├─ CDNA 2 (gfx90a): MI200 Series                              │
│  │   • Enhanced Matrix Cores (256 FLOPS/cycle FP32)            │
│  │   • Unified Memory Architecture                              │
│  │   • Infinity Fabric                                          │
│  └─ CDNA 3 (gfx940-gfx942): MI300 Series                       │
│      • Pipelined SFU operations                                 │
│      • 1024 FLOPS/cycle FP32 MFMA                               │
│      • FP8 support                                              │
│                                                                  │
│  RDNA (Radeon DNA) - Gaming/Graphics                           │
│  ├─ RDNA 1 (gfx1010-gfx1012): RX 5000                          │
│  │   • Wave32 mode                                              │
│  │   • WGP (dual CU) architecture                               │
│  ├─ RDNA 2 (gfx1030-gfx1034): RX 6000                          │
│  │   • Ray tracing acceleration                                 │
│  │   • Infinity Cache                                           │
│  ├─ RDNA 3 (gfx1100-gfx1103): RX 7000                          │
│  │   • WMMA support (tensor cores)                              │
│  │   • Dual-issue Wave32 SIMDs                                  │
│  └─ RDNA 4 (gfx1200+): RX 9000                                 │
│      • Enhanced WMMA instructions                                │
└────────────────────────────────────────────────────────────────┘
```

### Wavefront Execution Model

```
┌──────────────────────────────────────────────────────────────┐
│               WAVEFRONT EXECUTION DETAILS                     │
│                                                                │
│  CDNA Architecture (Wave64):                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                   Wavefront (64 threads)                │  │
│  │  ┌──┬──┬──┬──┬──┬──┬──┬──┐                            │  │
│  │  │L0│L1│L2│L3│..│61│62│63│  Lane IDs                   │  │
│  │  └──┴──┴──┴──┴──┴──┴──┴──┘                            │  │
│  │          │                                               │  │
│  │          ▼                                               │  │
│  │  ┌──────────────────────┐                               │  │
│  │  │   EXEC Mask (64-bit)  │                               │  │
│  │  │ 1111111111111111...   │  (1=active, 0=inactive)      │  │
│  │  └──────────────────────┘                               │  │
│  │          │                                               │  │
│  │          ▼                                               │  │
│  │  ┌──────────────────────┐                               │  │
│  │  │  VCC (64-bit cond)   │  Compare results             │  │
│  │  │  SCC (1-bit scalar)  │  Scalar compare              │  │
│  │  └──────────────────────┘                               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  RDNA Architecture (Wave32):                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                   Wavefront (32 threads)                │  │
│  │  ┌──┬──┬──┬──┬──┬──┬──┬──┐                            │  │
│  │  │L0│L1│L2│L3│..│29│30│31│  Lane IDs                   │  │
│  │  └──┴──┴──┴──┴──┴──┴──┴──┘                            │  │
│  │  • Reduced divergence overhead                          │  │
│  │  • Lower register pressure per wavefront                │  │
│  │  • Dual-issue capability (RDNA3+)                       │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Reading and Understanding AMDGCN Assembly {#reading-assembly}

### Generating Assembly from HIP

```bash
# Method 1: Generate assembly during compilation
hipcc -c kernel.hip --save-temps -g
# Produces: kernel.s (assembly file)

# Method 2: Disassemble object file
hipcc -c kernel.hip -o kernel.o
llvm-objdump -d kernel.o

# Method 3: Extract from bundled binary
extractkernel --extract kernel.co

# Method 4: Using roc-obj-ls and roc-obj
roc-obj-ls kernel.co
roc-obj -d kernel.co -o kernel.asm
```

### Assembly Structure

```asm
; ============================================================
; KERNEL METADATA AND SETUP
; ============================================================
.amdgcn_target "amdgcn-amd-amdhsa--gfx90a"  ; Target ISA

; Kernel descriptor
.amd_kernel_code_t
    amd_code_version_major = 1
    amd_code_version_minor = 2
    amd_machine_kind = 1
    amd_machine_version_major = 9
    amd_machine_version_minor = 0
    amd_machine_version_stepping = 10

    ; Resource allocation
    kernarg_segment_byte_size = 64      ; Kernel argument size
    workitem_vgpr_count = 32            ; VGPRs per thread
    wavefront_sgpr_count = 24           ; SGPRs per wavefront
    compute_pgm_rsrc1_vgprs = 7         ; (32-1)/4 = 7
    compute_pgm_rsrc1_sgprs = 2         ; (24-1)/8 = 2
    compute_pgm_rsrc2_user_sgpr = 8     ; User SGPRs

    ; LDS configuration
    workgroup_group_segment_byte_size = 4096  ; 4KB LDS

    ; Execution mode
    enable_sgpr_kernarg_segment_ptr = 1
    enable_sgpr_workgroup_id_x = 1
    enable_sgpr_workgroup_id_y = 1
    enable_vgpr_workitem_id = 1
.end_amd_kernel_code_t

; ============================================================
; KERNEL ENTRY POINT
; ============================================================
my_kernel:
    ; Register initialization state:
    ; s[0:1]   = Kernel arguments pointer (kernarg)
    ; s2       = Workgroup ID X
    ; s3       = Workgroup ID Y
    ; s4       = Workgroup ID Z
    ; v0       = Thread ID X (within workgroup)
    ; v1       = Thread ID Y
    ; v2       = Thread ID Z

    ; ========================================================
    ; LOAD KERNEL ARGUMENTS
    ; ========================================================
    ; Load pointer to input array from kernarg
    s_load_dwordx2 s[4:5], s[0:1], 0x0    ; Load 64-bit pointer at offset 0
    s_load_dwordx2 s[6:7], s[0:1], 0x8    ; Load output pointer at offset 8
    s_load_dword s8, s[0:1], 0x10          ; Load N (int) at offset 16

    ; Wait for scalar loads to complete
    s_waitcnt lgkmcnt(0)

    ; ========================================================
    ; CALCULATE GLOBAL THREAD ID
    ; ========================================================
    ; Global ID = workgroup_id * workgroup_size + thread_id
    s_lshl_b32 s2, s2, 6           ; s2 = workgroup_id_x * 64
    v_add_u32 v3, s2, v0           ; v3 = global_id_x

    ; ========================================================
    ; BOUNDS CHECK
    ; ========================================================
    v_cmp_lt_u32 vcc, v3, s8       ; Compare global_id < N
    s_and_saveexec_b64 s[10:11], vcc  ; Save exec mask, disable out-of-bounds

    ; ========================================================
    ; COMPUTE ADDRESS AND LOAD DATA
    ; ========================================================
    v_lshlrev_b32 v4, 2, v3        ; v4 = global_id * 4 (byte offset)
    v_add_co_u32 v5, vcc, s4, v4   ; v5 = base_ptr_lo + offset
    v_addc_co_u32 v6, vcc, s5, 0, vcc  ; v6 = base_ptr_hi + carry

    ; Load from global memory
    global_load_dword v7, v[5:6], off  ; Load input[global_id]

    ; Wait for global load
    s_waitcnt vmcnt(0)

    ; ========================================================
    ; COMPUTATION
    ; ========================================================
    v_mul_f32 v7, v7, 2.0          ; v7 = v7 * 2.0
    v_add_f32 v7, v7, 1.0          ; v7 = v7 + 1.0

    ; ========================================================
    ; STORE RESULT
    ; ========================================================
    v_add_co_u32 v5, vcc, s6, v4   ; Calculate output address
    v_addc_co_u32 v6, vcc, s7, 0, vcc

    global_store_dword v[5:6], v7, off  ; Store result

    ; ========================================================
    ; KERNEL EXIT
    ; ========================================================
    s_waitcnt vmcnt(0)             ; Wait for stores to complete
    s_mov_b64 exec, s[10:11]       ; Restore exec mask
    s_endpgm                        ; End program
```

### Instruction Format Breakdown

```
┌────────────────────────────────────────────────────────────┐
│           AMDGCN INSTRUCTION ENCODING                       │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  SCALAR ARITHMETIC: s_<op>_<type> dst, src0, src1          │
│  ┌────────────────────────────────────────────────────┐    │
│  │ s_add_i32  s2, s0, s1    ; s2 = s0 + s1 (32-bit)  │    │
│  │ s_sub_u32  s5, s3, s4    ; s5 = s3 - s4 (unsigned)│    │
│  │ s_mul_i32  s8, s6, s7    ; s8 = s6 × s7           │    │
│  │ s_lshl_b32 s9, s9, 2     ; s9 = s9 << 2 (shift)   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  VECTOR ARITHMETIC: v_<op>_<type> dst, src0, src1          │
│  ┌────────────────────────────────────────────────────┐    │
│  │ v_add_f32  v2, v0, v1    ; FP32 addition          │    │
│  │ v_mul_f32  v3, v1, s0    ; VGPR × SGPR (broadcast)│    │
│  │ v_fma_f32  v4, v0, v1, v2 ; v4 = v0×v1 + v2 (FMA)│    │
│  │ v_mad_f32  v5, v1, v2, v3 ; Similar to FMA        │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  MEMORY OPERATIONS:                                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │ s_load_dword     s0, s[2:3], 0x0  ; Scalar load   │    │
│  │ s_load_dwordx2   s[0:1], s[2:3], 0x10 ; 64-bit    │    │
│  │ global_load_dword v0, v[1:2], off  ; Vector load  │    │
│  │ global_store_dword v[0:1], v2, off ; Vector store │    │
│  │ ds_read_b32      v0, v1            ; LDS read     │    │
│  │ ds_write_b32     v0, v1            ; LDS write    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  SYNCHRONIZATION:                                           │
│  ┌────────────────────────────────────────────────────┐    │
│  │ s_waitcnt lgkmcnt(0)  ; Wait for LDS/GDS/scalar   │    │
│  │ s_waitcnt vmcnt(0)    ; Wait for vector memory    │    │
│  │ s_barrier             ; Workgroup barrier         │    │
│  └────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────┘
```

---

## Register Architecture Deep Dive {#register-architecture}

### Register File Organization

```
┌──────────────────────────────────────────────────────────────┐
│                 REGISTER FILE HIERARCHY                       │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  VECTOR REGISTERS (VGPR)                                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Thread 0:  [V0][V1][V2]...[V255]                      │  │
│  │  Thread 1:  [V0][V1][V2]...[V255]                      │  │
│  │  Thread 2:  [V0][V1][V2]...[V255]                      │  │
│  │  ...                                                    │  │
│  │  Thread 63: [V0][V1][V2]...[V255]                      │  │
│  │                                                          │  │
│  │  • 256 registers per thread                             │  │
│  │  • 32-bit per register                                  │  │
│  │  • Total: 256KB per CU (64 threads × 256 × 32 bits)    │  │
│  │  • Allocated in groups of 4                             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  SCALAR REGISTERS (SGPR)                                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Wavefront: [S0][S1][S2]...[S101]                      │  │
│  │                                                          │  │
│  │  • 102-106 registers per wavefront                      │  │
│  │  • 32-bit per register                                  │  │
│  │  • Shared by all 64 threads                             │  │
│  │  • Allocated in groups of 16 (CDNA)                    │  │
│  │                                                          │  │
│  │  Special registers:                                     │  │
│  │    VCC[0:1]       - Vector condition code (64-bit)     │  │
│  │    EXEC[0:1]      - Execution mask (64-bit)            │  │
│  │    FLAT_SCRATCH   - Flat addressing base               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  ACCUMULATION REGISTERS (AGPR) - CDNA Only                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Thread 0:  [A0][A1][A2]...[A255]                      │  │
│  │  Thread 1:  [A0][A1][A2]...[A255]                      │  │
│  │  ...                                                    │  │
│  │                                                          │  │
│  │  • 256 registers per thread                             │  │
│  │  • Dedicated to MFMA output accumulation                │  │
│  │  • Additional 256KB per CU                              │  │
│  │  • Transfer with v_accvgpr_read/write                   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Register Pair and Tuple Usage

```asm
; 64-bit operations use register pairs
; Notation: s[n:m] for consecutive registers

; Load 64-bit pointer
s_load_dwordx2 s[4:5], s[0:1], 0x0
; s4 = low 32 bits, s5 = high 32 bits

; 128-bit vector operations
v_pk_add_f32 v[0:1], v[2:3], v[4:5]  ; Packed dual FP32

; 512-bit MFMA accumulator (16 registers)
v_mfma_f32_16x16x4f32 a[0:15], v0, v1, a[0:15]
```

### Register Allocation Example

```cpp
// HIP kernel
__global__ void example(float* out, float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a = in[idx];
    float b = a * 2.0f;
    float c = b + 1.0f;
    out[idx] = c;
}
```

```asm
; Resulting assembly register usage
; VGPR Usage:
;   v0  = threadIdx.x
;   v1  = (unused/temp)
;   v2  = global_id
;   v3  = byte_offset
;   v4:v5 = address (64-bit)
;   v6  = loaded value (a)
;   v7  = intermediate (b)
;   v8  = final result (c)
; Total VGPRs: 9 → rounds to 12 (allocated in groups of 4)

; SGPR Usage:
;   s[0:1] = kernarg pointer
;   s2  = workgroup_id
;   s[4:5] = input pointer
;   s[6:7] = output pointer
;   s8  = N
;   s[10:11] = saved exec mask
; Total SGPRs: 12 → rounds to 16 (allocated in groups of 16)
```

---

## MFMA (Matrix Fused Multiply-Add) Instructions {#mfma-instructions}

### MFMA Instruction Format

```
Instruction: v_mfma_<CDFmt>_<M>x<N>x<K><ABFmt>

Components:
  CDFmt  = C/D matrix format (f32, f64, i32)
  M×N    = Output matrix dimensions
  K      = Inner dimension (reduction)
  ABFmt  = A/B matrix format (f32, f64, f16, bf16, i8)

Example: v_mfma_f32_16x16x4f32
  • Output (C/D): 16×16 FP32 matrix
  • Input (A/B):  16×4 and 4×16 FP32 matrices
  • Computes: D = A × B + C
```

### MFMA Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│               MFMA EXECUTION MODEL                              │
│                                                                  │
│  Wavefront (64 threads):                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Lane  │  A Fragment  │  B Fragment  │  C/D Accumulator  │  │
│  ├───────┼──────────────┼──────────────┼───────────────────┤  │
│  │  0    │   A[0][:]    │   B[:][0]    │   D[0][0]         │  │
│  │  1    │   A[0][:]    │   B[:][1]    │   D[0][1]         │  │
│  │  2    │   A[0][:]    │   B[:][2]    │   D[0][2]         │  │
│  │  ...  │   ...        │   ...        │   ...             │  │
│  │  63   │   A[15][:]   │   B[:][15]   │   D[15][15]       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Matrix Distribution (16×16×4f32 example):                     │
│                                                                  │
│    A Matrix (16×4)      B Matrix (4×16)      C/D Matrix(16×16) │
│   ┌───┬───┬───┬───┐   ┌──────────────┐    ┌──────────────┐   │
│ 0 │ • │ • │ • │ • │   │ • • • • ...  │  0 │ ▓ ▓ ▓ ▓ ...  │   │
│ 1 │ • │ • │ • │ • │   │ • • • • ...  │  1 │ ▓ ▓ ▓ ▓ ...  │   │
│ 2 │ • │ • │ • │ • │   │ • • • • ...  │  2 │ ▓ ▓ ▓ ▓ ...  │   │
│...│ • │ • │ • │ • │   │ • • • • ...  │... │ ▓ ▓ ▓ ▓ ...  │   │
│15 │ • │ • │ • │ • │   │ • • • • ...  │ 15 │ ▓ ▓ ▓ ▓ ...  │   │
│   └───┴───┴───┴───┘   └──────────────┘    └──────────────┘   │
│    4 VGPRs per lane     4 VGPRs per lane   4 AGPRs per lane   │
│                                                                  │
│  Throughput (CDNA2 MI250X):                                    │
│    FP32: 256 FLOPS/cycle/CU                                    │
│    FP16: 1024 FLOPS/cycle/CU                                   │
│    INT8: 1024 OPS/cycle/CU                                     │
└────────────────────────────────────────────────────────────────┘
```

### Available MFMA Variants (CDNA2)

```
┌──────────────────────────────────────────────────────────────┐
│              MFMA INSTRUCTION VARIANTS                        │
├────────┬─────────┬─────────┬────────┬────────┬──────────────┤
│ Instr  │ M×N×K   │ A/B Fmt │ C/D Fmt│ Blocks │ FLOPS/cycle  │
├────────┼─────────┼─────────┼────────┼────────┼──────────────┤
│ FP64   │ 16×16×4 │ FP64    │ FP64   │   1    │ 256 (DP)     │
│ FP64   │ 4×4×4   │ FP64    │ FP64   │   4    │ 256 (DP)     │
├────────┼─────────┼─────────┼────────┼────────┼──────────────┤
│ FP32   │ 32×32×8 │ FP16    │ FP32   │   1    │ 1024         │
│ FP32   │ 16×16×16│ FP16    │ FP32   │   1    │ 1024         │
│ FP32   │ 32×32×4 │ FP32    │ FP32   │   2    │ 256          │
│ FP32   │ 16×16×4 │ FP32    │ FP32   │   4    │ 256          │
│ FP32   │ 16×16×1 │ FP32    │ FP32   │   16   │ 256          │
├────────┼─────────┼─────────┼────────┼────────┼──────────────┤
│ FP32   │ 32×32×8 │ BF16    │ FP32   │   1    │ 1024         │
│ FP32   │ 16×16×16│ BF16    │ FP32   │   1    │ 1024         │
│ FP32   │ 32×32×4 │ BF16    │ FP32   │   2    │ 512          │
│ FP32   │ 16×16×8 │ BF16    │ FP32   │   1    │ 512          │
├────────┼─────────┼─────────┼────────┼────────┼──────────────┤
│ INT32  │ 32×32×16│ INT8    │ INT32  │   1    │ 1024         │
│ INT32  │ 16×16×32│ INT8    │ INT32  │   1    │ 1024         │
│ INT32  │ 32×32×8 │ INT8    │ INT32  │   2    │ 512          │
│ INT32  │ 16×16×16│ INT8    │ INT8   │   4    │ 512          │
└────────┴─────────┴─────────┴────────┴────────┴──────────────┘
```

### MFMA Compiler Intrinsic Syntax

```cpp
// Generic format:
d = __builtin_amdgcn_mfma_CDFmt_MxNxKABFmt(a, b, c, cbsz, abid, blgp)

// Parameters:
// a, b   : Input matrices (VGPRs)
// c      : Input accumulator (AGPRs)
// d      : Output accumulator (AGPRs)
// cbsz   : C broadcast size (0 = no broadcast)
// abid   : A/B broadcast ID for multi-block operations
// blgp   : B lane group pattern (0-7 for swizzling)

// Example 1: Basic FP32 16×16×4
using float4 = __attribute__((ext_vector_type(4))) float;
using float16 = __attribute__((ext_vector_type(16))) float;

__device__ float16 mfma_example(float4 a, float4 b, float16 c) {
    // Compute: D = A × B + C
    return __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
}

// Example 2: FP16 input, FP32 accumulator
using half4 = __attribute__((ext_vector_type(4))) __fp16;
using half8 = __attribute__((ext_vector_type(8))) __fp16;

__device__ float16 mfma_fp16_to_fp32(half8 a, half8 b, float16 c) {
    return __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);
}

// Example 3: INT8 matrix multiplication
using int4 = __attribute__((ext_vector_type(4))) int;
using int16 = __attribute__((ext_vector_type(16))) int;

__device__ int16 mfma_int8(int4 a_packed, int4 b_packed, int16 c) {
    // Each int contains 4 packed INT8 values
    return __builtin_amdgcn_mfma_i32_16x16x16i8(a_packed, b_packed, c, 0, 0, 0);
}
```

### MFMA Assembly Example

```asm
; GEMM kernel using MFMA instructions
; Compute: C[M×N] = A[M×K] × B[K×N] + C[M×N]

.global mfma_gemm_kernel
mfma_gemm_kernel:
    ; ================================================
    ; SETUP AND REGISTER ALLOCATION
    ; ================================================
    ; VGPRs:
    ;   v[0:3]   - A matrix fragment (4×FP32)
    ;   v[4:7]   - B matrix fragment (4×FP32)
    ;   a[0:15]  - C/D accumulator (16×FP32)

    ; Initialize accumulator to zero
    v_mov_b32 v16, 0
    .rept 16
        v_accvgpr_write a[.rept_index], v16
    .endr

    ; ================================================
    ; MAIN LOOP: Iterate over K dimension
    ; ================================================
k_loop:
    ; Load A fragment from global memory
    global_load_dwordx4 v[0:3], v[addr_a:addr_a+1], off
    ; Load B fragment from global memory
    global_load_dwordx4 v[4:7], v[addr_b:addr_b+1], off

    ; Wait for loads
    s_waitcnt vmcnt(0)

    ; ================================================
    ; MFMA INSTRUCTION
    ; ================================================
    ; D[16×16] = A[16×4] × B[4×16] + C[16×16]
    v_mfma_f32_16x16x4f32 a[0:15], v[0:3], v[4:7], a[0:15]

    ; Note: MFMA has latency ~64 cycles on CDNA2
    ; Must wait before using results

    ; Update pointers and loop
    v_add_u32 v[addr_a], v[addr_a], 16    ; Move to next A block
    v_add_u32 v[addr_b], v[addr_b], 16    ; Move to next B block
    s_sub_i32 s[k_remaining], s[k_remaining], 4
    s_cmp_gt_i32 s[k_remaining], 0
    s_cbranch_scc1 k_loop

    ; ================================================
    ; WRITE RESULTS
    ; ================================================
    ; Must transfer from AGPRs to VGPRs before storing
    .rept 16
        v_accvgpr_read v[.rept_index], a[.rept_index]
    .endr

    ; Wait for MFMA completion and AGPR reads
    s_waitcnt vmcnt(0)

    ; Store results to global memory
    .rept 4
        global_store_dwordx4 v[addr_c:addr_c+1], v[.rept_index*4:.rept_index*4+3], off
        v_add_u32 v[addr_c], v[addr_c], 16
    .endr

    s_waitcnt vmcnt(0)
    s_endpgm
```

### MFMA Performance Considerations

```
┌──────────────────────────────────────────────────────────────┐
│           MFMA OPTIMIZATION GUIDELINES                        │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  1. LATENCY MANAGEMENT                                        │
│     • MFMA latency: 64 cycles (CDNA2)                        │
│     • Solution: Software pipelining, multiple waves           │
│                                                                │
│  2. DATA MOVEMENT                                             │
│     • AGPR ↔ VGPR transfers: 2-4 cycles                      │
│     • Minimize transfers by keeping data in AGPRs             │
│     • Use v_accvgpr_read/write strategically                  │
│                                                                │
│  3. INSTRUCTION SCHEDULING                                    │
│     • Issue MFMA early in pipeline                            │
│     • Overlap with global memory loads                        │
│     • Keep at least 4 independent MFMA operations in-flight   │
│                                                                │
│  4. OCCUPANCY                                                 │
│     • AGPRs count toward register budget                      │
│     • Balance AGPR usage vs occupancy                         │
│     • Typical: 4-8 waves per CU for MFMA kernels             │
│                                                                │
│  5. MEMORY BANDWIDTH                                          │
│     • MFMA throughput: 1024 FLOPS/cycle (FP16)               │
│     • Memory bandwidth: ~1.6 TB/s (MI250X per GCD)           │
│     • Arithmetic intensity target: > 100 FLOPS/byte          │
│                                                                │
│  6. MATRIX DIMENSIONS                                         │
│     • Prefer larger tiles (32×32×8) for efficiency           │
│     • Smaller tiles (16×16×4) for flexibility                │
│     • Multi-block variants for batching                       │
└──────────────────────────────────────────────────────────────┘
```

---

## WMMA (Wave Matrix Multiply-Accumulate) {#wmma}

### WMMA vs MFMA

```
┌────────────────────────────────────────────────────────────┐
│                  WMMA vs MFMA Comparison                    │
├────────────────────┬───────────────────┬───────────────────┤
│ Feature            │ MFMA (CDNA)       │ WMMA (RDNA3+)     │
├────────────────────┼───────────────────┼───────────────────┤
│ Architecture       │ CDNA 1/2/3        │ RDNA 3/4          │
│ Wavefront Size     │ 64 threads        │ 32 threads        │
│ Accumulator Regs   │ AGPR (separate)   │ VGPR (unified)    │
│ Instruction Prefix │ v_mfma_*          │ v_wmma_*          │
│ Typical Dims       │ 16×16, 32×32      │ 16×16             │
│ Target Workload    │ HPC, Large AI     │ Gaming, Small AI  │
│ Peak FP16 (CU)     │ 1024 FLOPS/cycle  │ 256 FLOPS/cycle   │
└────────────────────┴───────────────────┴───────────────────┘
```

### WMMA Instruction Format

```cpp
// WMMA intrinsic format (RDNA 3):
d = __builtin_amdgcn_wmma_CDFmt_MxNxKABFmt(a, b, c)

// Example: FP16 input, FP32 output, 16×16×16
using half16 = __attribute__((ext_vector_type(16))) __fp16;
using float8 = __attribute__((ext_vector_type(8))) float;

__device__ float8 wmma_fp16_example(half16 a, half16 b, float8 c) {
    // Compute 16×16×16 WMMA: D = A × B + C
    return __builtin_amdgcn_wmma_f32_16x16x16_f16(a, b, c);
}

// Available WMMA variants (RDNA3):
// v_wmma_f32_16x16x16_f16   - FP16 → FP32
// v_wmma_f32_16x16x16_bf16  - BF16 → FP32
// v_wmma_f16_16x16x16_f16   - FP16 → FP16
// v_wmma_bf16_16x16x16_bf16 - BF16 → BF16
// v_wmma_i32_16x16x16_iu8   - INT8 → INT32
```

### WMMA Memory Layout

```
WMMA 16×16×16 Layout (Wave32):

A Matrix (16×16):           B Matrix (16×16):
Lane  Ownership             Lane Ownership
┌─────────────────┐        ┌─────────────────┐
│ 0  0  1  1 ...  │        │ 0  0  1  1 ...  │
│ 0  0  1  1 ...  │        │ 0  0  1  1 ...  │
│ 2  2  3  3 ...  │        │ 2  2  3  3 ...  │
│ 2  2  3  3 ...  │        │ 2  2  3  3 ...  │
│ ...             │        │ ...             │
│ 30 30 31 31 ... │        │ 30 30 31 31 ... │
└─────────────────┘        └─────────────────┘

Each lane owns: 16 FP16 values (8×2 block)
C/D Accumulator: 8 FP32 values per lane
```

---

## Writing Inline Assembly in HIP {#inline-assembly}

### Basic Inline Assembly Syntax

```cpp
__device__ void inline_asm_example() {
    int a = 10, b = 20, c;

    // Basic syntax: asm("instruction" : outputs : inputs : clobbers);
    asm volatile(
        "s_add_i32 %0, %1, %2"  // Instruction
        : "=s"(c)                // Output: scalar register
        : "s"(a), "s"(b)        // Inputs: scalar registers
        : /* no clobbers */
    );
    // Result: c = a + b
}
```

### Register Constraints

```cpp
// Register constraint specifiers:
// "v" = VGPR (vector register)
// "s" = SGPR (scalar register)
// "a" = AGPR (accumulation register, CDNA only)
// "=r" = output register
// "+r" = input/output register

__device__ float vector_add_asm(float a, float b) {
    float result;
    asm volatile(
        "v_add_f32 %0, %1, %2"
        : "=v"(result)           // Output in VGPR
        : "v"(a), "v"(b)        // Inputs in VGPRs
    );
    return result;
}

__device__ float fma_asm(float a, float b, float c) {
    float result;
    asm volatile(
        "v_fma_f32 %0, %1, %2, %3"
        : "=v"(result)
        : "v"(a), "v"(b), "v"(c)
    );
    return result;  // result = a * b + c
}
```

### MFMA Inline Assembly

```cpp
// MFMA 16×16×4 using inline assembly
__device__ void mfma_inline_asm() {
    // Input/output fragments
    float a[4], b[4], c[16], d[16];

    // Initialize inputs and accumulator
    // ... (initialization code) ...

    // Execute MFMA
    asm volatile(
        "v_mfma_f32_16x16x4f32 %0, %1, %2, %3"
        : "=a"(d[0:15])         // Output: 16 AGPRs
        : "v"(a[0:3]),          // Input A: 4 VGPRs
          "v"(b[0:3]),          // Input B: 4 VGPRs
          "a"(c[0:15])          // Input C: 16 AGPRs
    );

    // Transfer results from AGPRs to VGPRs
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        asm volatile(
            "v_accvgpr_read %0, %1"
            : "=v"(d[i])
            : "a"(i)
        );
    }
}
```

### LDS Operations with Assembly

```cpp
__device__ float lds_read_asm(int offset) {
    float value;
    asm volatile(
        "ds_read_b32 %0, %1"
        : "=v"(value)            // Output
        : "v"(offset)            // LDS address offset
        : "memory"               // Clobbers memory
    );
    // Need to wait for LDS operation
    asm volatile("s_waitcnt lgkmcnt(0)");
    return value;
}

__device__ void lds_write_asm(int offset, float value) {
    asm volatile(
        "ds_write_b32 %0, %1"
        : /* no outputs */
        : "v"(offset), "v"(value)
        : "memory"
    );
}
```

### Why Inline Assembly is NOT Recommended

```
┌────────────────────────────────────────────────────────────┐
│         PROBLEMS WITH INLINE ASSEMBLY                       │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  ❌ Compiler doesn't understand semantics                   │
│     • Cannot optimize across inline assembly                │
│     • Cannot detect data hazards                            │
│     • Cannot perform register allocation optimally          │
│                                                              │
│  ❌ Manual hazard management required                       │
│     • Must insert s_waitcnt manually                        │
│     • MFMA latency not tracked by compiler                  │
│     • Easy to introduce subtle bugs                         │
│                                                              │
│  ❌ Portability issues                                      │
│     • ISA-specific code                                     │
│     • Different instruction sets across architectures       │
│     • Won't work on NVIDIA GPUs                             │
│                                                              │
│  ✅ BETTER ALTERNATIVES:                                    │
│     1. Compiler intrinsics (__builtin_amdgcn_*)            │
│     2. rocWMMA library (portable, optimized)                │
│     3. Composable Kernels (CK) framework                    │
│     4. Let compiler generate assembly from HIP              │
└────────────────────────────────────────────────────────────┘
```

---

## rocWMMA Library Usage {#rocwmma}

### rocWMMA Overview

rocWMMA is a C++ library providing portable access to matrix multiplication hardware:

```
┌──────────────────────────────────────────────────────────┐
│              rocWMMA Architecture                         │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  User Code (C++ Templates)                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ rocwmma::fragment<...>                            │    │
│  │ rocwmma::load_matrix_sync(...)                    │    │
│  │ rocwmma::mma_sync(...)                            │    │
│  │ rocwmma::store_matrix_sync(...)                   │    │
│  └──────────────────────────────────────────────────┘    │
│                     ↓                                      │
│  rocWMMA Abstraction Layer                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ • Fragment management                             │    │
│  │ • Layout transformations                          │    │
│  │ • Architecture dispatch                           │    │
│  └──────────────────────────────────────────────────┘    │
│                     ↓                                      │
│  Hardware-Specific Implementations                        │
│  ┌───────────────────────┬──────────────────────────┐    │
│  │ CDNA Backend          │ RDNA Backend             │    │
│  │ • MFMA instructions   │ • WMMA instructions      │    │
│  │ • AGPR management     │ • VGPR only              │    │
│  │ • Wave64              │ • Wave32                 │    │
│  └───────────────────────┴──────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### Basic rocWMMA GEMM Example

```cpp
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

// Matrix dimensions
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

__global__ void rocwmma_gemm(
    float* C,
    const __fp16* A,
    const __fp16* B,
    int lda, int ldb, int ldc
) {
    // Define fragment types
    // fragment<MatrixType, M, N, K, DataType, LayoutType>
    fragment<matrix_a, M, N, K, __fp16, row_major> fragA;
    fragment<matrix_b, M, N, K, __fp16, col_major> fragB;
    fragment<accumulator, M, N, K, float> fragAcc;
    fragment<accumulator, M, N, K, float> fragC;

    // Calculate tile position
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Initialize accumulator to zero
    fill_fragment(fragAcc, 0.0f);

    // Main loop over K dimension
    for (int k = 0; k < K; k += K) {
        // Load fragments from global memory
        load_matrix_sync(fragA, A + warpM * M * lda + k, lda);
        load_matrix_sync(fragB, B + k * ldb + warpN * N, ldb);

        // Perform matrix multiplication
        // fragAcc = fragA × fragB + fragAcc
        mma_sync(fragAcc, fragA, fragB, fragAcc);
    }

    // Store result
    store_matrix_sync(C + warpM * M * ldc + warpN * N, fragAcc, ldc, mem_row_major);
}
```

### rocWMMA Fragment Types

```cpp
// Fragment type specifications:
// fragment<Usage, M, N, K, DataType, Layout>

// Matrix A fragment (input)
using FragA_FP16 = fragment<
    matrix_a,        // Matrix A role
    16,              // M dimension
    16,              // N dimension
    16,              // K dimension
    __fp16,          // Data type
    row_major        // Memory layout
>;

// Matrix B fragment (input)
using FragB_FP16 = fragment<
    matrix_b,        // Matrix B role
    16, 16, 16,
    __fp16,
    col_major
>;

// Accumulator fragment (input/output)
using FragAcc_FP32 = fragment<
    accumulator,     // Accumulator role
    16, 16, 16,
    float,           // Accumulation precision
    void             // No layout for accumulator
>;

// Mixed-precision fragments
using FragA_BF16 = fragment<matrix_a, 16, 16, 16, __bf16, row_major>;
using FragB_INT8 = fragment<matrix_b, 16, 16, 16, int8_t, col_major>;
using FragAcc_INT32 = fragment<accumulator, 16, 16, 16, int32_t>;
```

### rocWMMA API Functions

```cpp
// 1. LOAD MATRIX
// Loads data from global/shared memory into fragment
load_matrix_sync(fragment, pointer, stride);

// 2. STORE MATRIX
// Stores fragment data to global/shared memory
store_matrix_sync(pointer, fragment, stride, layout);

// 3. FILL FRAGMENT
// Initialize fragment with constant value
fill_fragment(fragment, value);

// 4. MMA SYNC
// Perform matrix multiply-accumulate
// D = A × B + C
mma_sync(fragD, fragA, fragB, fragC);

// 5. SYNCHRONIZE WORKGROUP
// Synchronization primitive
synchronize_workgroup();

// Example complete workflow:
__global__ void wmma_workflow(float* out, const __fp16* A, const __fp16* B) {
    fragment<matrix_a, 16, 16, 16, __fp16, row_major> a;
    fragment<matrix_b, 16, 16, 16, __fp16, col_major> b;
    fragment<accumulator, 16, 16, 16, float> c, d;

    // Step 1: Initialize
    fill_fragment(c, 0.0f);

    // Step 2: Load
    load_matrix_sync(a, A, 16);
    load_matrix_sync(b, B, 16);

    // Step 3: Compute
    mma_sync(d, a, b, c);

    // Step 4: Store
    store_matrix_sync(out, d, 16, mem_row_major);
}
```

---

## Assembly-Level Optimizations {#assembly-optimizations}

### Software Pipelining for MFMA

```asm
; Optimized MFMA loop with software pipelining
; Goal: Overlap computation with memory loads

mfma_pipelined_loop:
    ; ================================================
    ; PROLOGUE: Issue first loads
    ; ================================================
    global_load_dwordx4 v[0:3], v[addr_a], off     ; Load A[0]
    global_load_dwordx4 v[4:7], v[addr_b], off     ; Load B[0]

    ; Advance addresses for next iteration
    v_add_u32 v[addr_a], v[addr_a], 16
    v_add_u32 v[addr_b], v[addr_b], 16

    ; Issue second loads (will complete while first MFMA executes)
    global_load_dwordx4 v[8:11], v[addr_a], off    ; Load A[1]
    global_load_dwordx4 v[12:15], v[addr_b], off   ; Load B[1]

    ; Wait for first loads only
    s_waitcnt vmcnt(2)  ; Wait for v[0:7], allow v[8:15] to continue

    ; ================================================
    ; MAIN LOOP
    ; ================================================
loop_body:
    ; MFMA using current data (64-cycle latency)
    v_mfma_f32_16x16x4f32 a[0:15], v[0:3], v[4:7], a[0:15]

    ; While MFMA executes, advance addresses
    v_add_u32 v[addr_a], v[addr_a], 16
    v_add_u32 v[addr_b], v[addr_b], 16

    ; Issue next loads (for iteration i+2)
    global_load_dwordx4 v[0:3], v[addr_a], off     ; Load A[i+2]
    global_load_dwordx4 v[4:7], v[addr_b], off     ; Load B[i+2]

    ; Wait for previous loads (A[i+1], B[i+1])
    s_waitcnt vmcnt(2)

    ; Second MFMA using prefetched data
    v_mfma_f32_16x16x4f32 a[0:15], v[8:11], v[12:15], a[0:15]

    ; Move prefetched data to working registers
    ; (Actually just swap register names for next iteration)
    ; v[0:7] ↔ v[8:15] (conceptual swap via renaming)

    ; Loop control
    s_sub_i32 s[k_count], s[k_count], 2
    s_cmp_gt_i32 s[k_count], 0
    s_cbranch_scc1 loop_body

    ; ================================================
    ; EPILOGUE: Handle remaining data
    ; ================================================
    s_waitcnt vmcnt(0)  ; Wait for all loads
    ; Final MFMA operations...
```

### Register Pressure Optimization

```asm
; BAD: High register pressure, low occupancy
high_register_kernel:
    ; Using 80 VGPRs
    v_mov_b32 v0, ...
    v_mov_b32 v1, ...
    ; ... (many temporary variables)
    v_mov_b32 v79, ...

    ; Computation using many registers
    ; Result: Only 3 wavefronts per CU (30% occupancy)

; GOOD: Optimized register usage
low_register_kernel:
    ; Reuse registers strategically
    v_mov_b32 v0, ...
    ; Compute and immediately use v0
    v_add_f32 v1, v0, ...

    ; Free v0 for reuse
    v_mov_b32 v0, new_value

    ; Result: 8 wavefronts per CU (80% occupancy)
```

### LDS Banking Optimization

```asm
; Optimized LDS access with XOR swizzle
.macro LDS_SWIZZLE_ADDR, out, m, k, xor_mask
    ; Calculate base address: m * 64 + k
    s_lshl_b32 \out, \m, 6           ; m * 64
    v_add_u32 \out, \out, \k         ; + k

    ; Apply XOR swizzle: addr ^ (m & xor_mask)
    v_and_b32 v_temp, \m, \xor_mask
    v_xor_b32 \out, \out, v_temp

    ; Convert to byte address
    v_lshlrev_b32 \out, 2, \out      ; * 4 bytes
.endm

; Usage:
lds_optimized_access:
    LDS_SWIZZLE_ADDR v_lds_addr, v_m, v_k, 3

    ; Access LDS with swizzled address (conflict-free!)
    ds_read_b32 v_data, v_lds_addr
```

---

## AITER Assembly Kernels {#aiter}

AITER (AI Tensor Engine for ROCm) represents AMD's highest-performance kernel implementations, written in hand-optimized assembly by expert engineers.

### AITER Architecture

```
┌────────────────────────────────────────────────────────────┐
│              AITER IMPLEMENTATION STACK                     │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  Python/C++ API Layer                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ • High-level operator interfaces                      │  │
│  │ • Auto-tuning and kernel selection                    │  │
│  │ • PyTorch integration                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  Kernel Dispatch Layer                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ • Architecture detection (CDNA2/3, RDNA3/4)          │  │
│  │ • Precision selection (FP32/FP16/BF16/FP8/INT8)      │  │
│  │ • Tile size optimization                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  Implementation Backends                                    │
│  ┌──────────────┬──────────────┬────────────┬───────────┐  │
│  │   Assembly   │    Triton    │     CK     │    HIP    │  │
│  │              │              │            │           │  │
│  │ • Highest    │ • Flexible   │ • Modular  │ • Portable│  │
│  │   perf       │ • Productive │ • Template │ • Baseline│  │
│  │ • Manual     │ • DSL        │   based    │           │  │
│  │ • Brittle    │ • Good perf  │ • Good     │           │  │
│  └──────────────┴──────────────┴────────────┴───────────┘  │
└────────────────────────────────────────────────────────────┘
```

### Key AITER Operators

```
┌──────────────────────────────────────────────────────────────┐
│           AITER OPERATOR CATALOG                              │
├───────────────────────┬──────────────────────────────────────┤
│ Category              │ Operators                             │
├───────────────────────┼──────────────────────────────────────┤
│ Matrix Multiplication │ • GEMM (General Matrix Multiply)     │
│                       │ • Block-scale GEMM (2x speedup)      │
│                       │ • Batched GEMM                        │
│                       │ • FP8/INT8 quantized GEMM            │
├───────────────────────┼──────────────────────────────────────┤
│ Attention             │ • Flash Attention (prefill)          │
│                       │ • Flash Attention (decode)           │
│                       │ • MHA (Multi-Head Attention)         │
│                       │ • MLA (Multi-head Latent Attention)  │
│                       │ • GQA (Grouped Query Attention)      │
├───────────────────────┼──────────────────────────────────────┤
│ Normalization         │ • LayerNorm                          │
│                       │ • RMSNorm                             │
│                       │ • Fused LayerNorm+Residual           │
├───────────────────────┼──────────────────────────────────────┤
│ Activation            │ • RoPE (Rotary Position Embedding)   │
│                       │ • SiLU, GELU, Swish                   │
│                       │ • Fused activation + GEMM            │
├───────────────────────┼──────────────────────────────────────┤
│ MoE (Mixture of       │ • TopK routing                        │
│  Experts)             │ • Expert sorting and tiling          │
│                       │ • Fused MoE (3x speedup)             │
├───────────────────────┼──────────────────────────────────────┤
│ Quantization          │ • Dynamic quantization               │
│                       │ • Static quantization                │
│                       │ • Dequantization kernels             │
└───────────────────────┴──────────────────────────────────────┘
```

### AITER Performance Results

From real-world deployments:

```
DeepSeek v3/r1 Integration (MI300X):
  Before AITER: 6,484 tokens/second
  After AITER:  13,704 tokens/second
  Speedup:      2.11x

Individual Operator Speedups:
  • Block-scale GEMM:  2x
  • Fused MoE:         3x
  • MLA decode:        17x
  • MHA prefill:       14x
```

---

## Summary and Best Practices

### Assembly Programming Guidelines

```
✅ DO:
  • Use compiler intrinsics (__builtin_amdgcn_*)
  • Use rocWMMA for matrix operations
  • Let compiler generate assembly from HIP
  • Study generated assembly to understand performance
  • Use profiling tools (rocprof, omniperf)

❌ DON'T:
  • Write inline assembly unless absolutely necessary
  • Manually manage instruction scheduling
  • Assume assembly will be faster than compiler output
  • Write architecture-specific code without abstraction
```

### Learning Path

```
1. Foundation → Read this guide and HIP_KERNEL_OPTIMIZATION_GUIDE.md
2. Practice   → Write HIP kernels, examine generated assembly
3. Profile    → Use rocprof to identify bottlenecks
4. Optimize   → Apply intrinsics and rocWMMA
5. Advanced   → Study AITER implementations
6. Expert     → Contribute optimized kernels to community
```

---

**Next:** See `ML_KERNELS_OPTIMIZATION.md` for machine learning-specific kernel optimization techniques including flash attention, quantization, and fusion patterns.
