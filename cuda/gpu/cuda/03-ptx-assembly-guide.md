# PTX Assembly Programming Guide

## Overview

PTX (Parallel Thread Execution) is NVIDIA's low-level virtual machine and instruction set architecture. It serves as a stable, portable ISA that spans multiple GPU generations, enabling:

- **Forward compatibility**: PTX code runs on future GPUs
- **Performance tuning**: Direct control over instruction scheduling and memory operations
- **Access to new features**: Use hardware capabilities before high-level CUDA support

**Key Resources:**
- [PTX ISA 9.1 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Inline PTX Assembly in CUDA](https://docs.nvidia.com/cuda/pdf/Inline_PTX_Assembly.pdf)
- [PTX Writers Guide to Interoperability](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/)

---

## Compilation Pipeline

```
CUDA C++ Source (.cu)
        ↓
    nvcc compiler
        ↓
    PTX Assembly (.ptx)
        ↓
    ptxas assembler
        ↓
SASS (GPU-specific binary)
```

**Why PTX?**
- Application distribution: Ship PTX, JIT-compile on user's GPU
- Performance analysis: Inspect compiler output
- Custom optimizations: Hand-tune critical sections
- New hardware features: Access before CUDA intrinsics available

---

## Execution Model

### Thread Organization

```ptx
// Special registers for thread identification
mov.u32 %r0, %tid.x;      // Thread ID in block (x dimension)
mov.u32 %r1, %tid.y;
mov.u32 %r2, %tid.z;

mov.u32 %r3, %ntid.x;     // Number of threads in block (x dim)
mov.u32 %r4, %ctaid.x;    // Block ID in grid (x dimension)
mov.u32 %r5, %nctaid.x;   // Number of blocks in grid (x dim)
```

**Thread Hierarchy:**
- **Thread**: Individual execution unit
- **Warp**: 32 threads executing SIMT (Single Instruction, Multiple Thread)
- **CTA (Cooperative Thread Array)**: Thread block
- **Cluster** (sm_90+): Group of CTAs
- **Grid**: All CTAs launched by a kernel

### SIMT Architecture

Threads in a warp:
- Execute the same instruction simultaneously
- Can diverge (different control flow paths)
- Divergence causes serialization
- Reconverge after divergent region

```ptx
// Example: Divergent execution
setp.gt.s32 p, %r0, 10;    // Predicate: r0 > 10?
@p add.s32 %r1, %r1, 1;    // Only execute if predicate true
@!p sub.s32 %r1, %r1, 1;   // Execute if predicate false
```

---

## Memory Hierarchy

### State Spaces

PTX defines multiple memory regions:

```ptx
.reg .b32 r0;              // Register (per-thread, fastest)
.local .b32 local_var;     // Local memory (per-thread, slower)
.shared .b32 shared_var;   // Shared memory (per-block)
.global .b32 global_var;   // Global memory (device-wide)
.const .b32 const_var;     // Constant memory (read-only)
.param .b32 param_var;     // Kernel parameters
```

**Performance Hierarchy:**
1. **Registers**: ~1 cycle latency, limited quantity
2. **Shared Memory**: ~20-30 cycles, 48-227 KB per SM
3. **L1/L2 Cache**: Automatic, varies by access pattern
4. **Global Memory**: 200-400 cycles, large capacity

### Memory Instructions

```ptx
// Load from global memory
ld.global.f32 %f0, [%r0];

// Store to shared memory
st.shared.f32 [%r1], %f1;

// Load with cache hints (Volta+)
ld.global.ca.f32 %f0, [%r0];    // Cache at all levels
ld.global.cg.f32 %f0, [%r0];    // Cache global (L2 only)
ld.global.cs.f32 %f0, [%r0];    // Cache streaming

// Volatile access (no caching)
ld.volatile.global.f32 %f0, [%r0];

// Aligned loads (better performance)
ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%r0]; // Load 4 floats at once
```

---

## Register Types and Data Types

### Register Declarations

```ptx
.reg .b8  r8;     // 8-bit register
.reg .b16 r16;    // 16-bit register
.reg .b32 r32;    // 32-bit register
.reg .b64 r64;    // 64-bit register

.reg .f32 f32;    // 32-bit float
.reg .f64 f64;    // 64-bit float
.reg .pred p;     // Predicate (boolean)

// Vector registers
.reg .v4 .f32 vec;  // 4-element float vector
```

### Type Conversions

```ptx
// Float to int
cvt.rni.s32.f32 %r0, %f0;    // Round to nearest integer

// Int to float
cvt.rn.f32.s32 %f0, %r0;     // Round to nearest

// Float precision conversion
cvt.f64.f32 %fd0, %f0;       // f32 -> f64
cvt.rn.f32.f64 %f0, %fd0;    // f64 -> f32 (round to nearest)

// Bit-cast (reinterpret bits)
mov.b32 %r0, %f0;            // Treat float bits as int
```

---

## Arithmetic Instructions

### Integer Arithmetic

```ptx
add.s32 %r0, %r1, %r2;       // r0 = r1 + r2 (signed)
sub.u32 %r0, %r1, %r2;       // r0 = r1 - r2 (unsigned)
mul.lo.s32 %r0, %r1, %r2;    // r0 = (r1 * r2) & 0xFFFFFFFF
mul.hi.s32 %r0, %r1, %r2;    // r0 = (r1 * r2) >> 32
mad.lo.s32 %r0, %r1, %r2, %r3; // r0 = r1 * r2 + r3

div.s32 %r0, %r1, %r2;       // r0 = r1 / r2 (slow!)
rem.s32 %r0, %r1, %r2;       // r0 = r1 % r2
```

### Floating-Point Arithmetic

```ptx
add.f32 %f0, %f1, %f2;       // f0 = f1 + f2
sub.f32 %f0, %f1, %f2;
mul.f32 %f0, %f1, %f2;
fma.rn.f32 %f0, %f1, %f2, %f3; // f0 = f1 * f2 + f3 (fused)

div.approx.f32 %f0, %f1, %f2;  // Fast approximate division
sqrt.approx.f32 %f0, %f1;      // Fast approximate sqrt

// Math functions
ex2.approx.f32 %f0, %f1;     // 2^x
lg2.approx.f32 %f0, %f1;     // log2(x)
sin.approx.f32 %f0, %f1;
cos.approx.f32 %f0, %f1;
```

### Bitwise Operations

```ptx
and.b32 %r0, %r1, %r2;       // r0 = r1 & r2
or.b32  %r0, %r1, %r2;       // r0 = r1 | r2
xor.b32 %r0, %r1, %r2;       // r0 = r1 ^ r2
not.b32 %r0, %r1;            // r0 = ~r1

shl.b32 %r0, %r1, 4;         // r0 = r1 << 4
shr.u32 %r0, %r1, 4;         // r0 = r1 >> 4 (logical)
shr.s32 %r0, %r1, 4;         // r0 = r1 >> 4 (arithmetic)
```

---

## Control Flow

### Predicates and Conditional Execution

```ptx
// Set predicate based on comparison
setp.gt.s32 p, %r0, %r1;     // p = (r0 > r1)
setp.eq.f32 p, %f0, 0.0;     // p = (f0 == 0.0)

// Conditional execution
@p add.s32 %r2, %r2, 1;      // Execute only if p is true
@!p sub.s32 %r2, %r2, 1;     // Execute only if p is false

// Predicate logic
and.pred p0, p1, p2;         // p0 = p1 && p2
or.pred  p0, p1, p2;         // p0 = p1 || p2
not.pred p0, p1;             // p0 = !p1
```

### Branches and Loops

```ptx
// Unconditional branch
bra LABEL;

// Conditional branch
@p bra LABEL;                // Branch if p is true

// Example: Simple loop
    mov.u32 %r0, 0;          // i = 0
LOOP:
    setp.ge.s32 p, %r0, 100; // p = (i >= 100)
    @p bra END;              // if (i >= 100) break
    // ... loop body ...
    add.u32 %r0, %r0, 1;     // i++
    bra LOOP;
END:
```

### Function Calls

```ptx
// Declare function
.func (.param .b32 result) add_func (
    .param .b32 a,
    .param .b32 b
)
{
    .reg .b32 %r<3>;
    ld.param.b32 %r0, [a];
    ld.param.b32 %r1, [b];
    add.s32 %r2, %r0, %r1;
    st.param.b32 [result], %r2;
    ret;
}

// Call function
    .param .b32 param_a;
    .param .b32 param_b;
    .param .b32 param_result;

    st.param.b32 [param_a], %r0;
    st.param.b32 [param_b], %r1;
    call (param_result), add_func, (param_a, param_b);
    ld.param.b32 %r2, [param_result];
```

---

## Inline PTX in CUDA

### Basic Syntax

```cuda
__global__ void kernel() {
    int x, y;

    // Inline PTX assembly
    asm("add.s32 %0, %1, %2;"
        : "=r"(x)              // Output: x in register
        : "r"(10), "r"(20)     // Inputs: literals 10, 20
    );
    // x = 30

    // Multiple instructions
    asm("add.s32 %0, %1, %2;\n\t"
        "mul.lo.s32 %0, %0, %3;"
        : "=r"(y)
        : "r"(x), "r"(5), "r"(2)
    );
    // y = (x + 5) * 2
}
```

### Constraint Letters

```cuda
"r" - 32-bit register (.b32)
"l" - 64-bit register (.b64)
"f" - 32-bit float register (.f32)
"d" - 64-bit float register (.f64)
"h" - 16-bit register (.b16)
"c" - 8-bit register (.b8)
```

### Example: Fast Integer Division by Constant

```cuda
__device__ __forceinline__
uint32_t div_by_5(uint32_t n) {
    uint32_t result;

    // Magic number method: n / 5 = (n * 0xCCCCCCCD) >> 34
    asm("{\n\t"
        "  mul.hi.u32 %0, %1, 0xCCCCCCCD;\n\t"
        "  shr.u32 %0, %0, 2;\n\t"
        "}"
        : "=r"(result)
        : "r"(n)
    );

    return result;
}
```

### Example: Accessing Special Registers

```cuda
__device__ __forceinline__
uint32_t get_smid() {
    uint32_t sm_id;
    asm("mov.u32 %0, %%smid;" : "=r"(sm_id));
    return sm_id;
}

__device__ __forceinline__
uint32_t get_warpid() {
    uint32_t warp_id;
    asm("mov.u32 %0, %%warpid;" : "=r"(warp_id));
    return warp_id;
}

__device__ __forceinline__
uint32_t get_laneid() {
    uint32_t lane_id;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}
```

---

## Advanced Memory Operations

### Async Copy (Ampere+)

```ptx
// cp.async: Asynchronous global to shared copy
cp.async.ca.shared.global [%r0], [%r1], 16;

// Wait for async copies to complete
cp.async.wait_group 0;  // Wait for all groups
cp.async.wait_all;      // Wait for all pending copies

// Commit current group
cp.async.commit_group;
```

### TMA (Tensor Memory Accelerator - Hopper)

```ptx
// Load using TMA
cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [%r_dest], [%r_tensormap, {%r_coord}], [%r_mbarrier];

// Store using TMA
cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group
    [%r_tensormap, {%r_coord}], [%r_src];
```

---

## Synchronization

### Thread Synchronization

```ptx
// Barrier: Synchronize all threads in block
bar.sync 0;

// Barrier with arrival count
bar.arrive %r0, 0;       // Decrement barrier count
bar.sync 0;              // Wait for barrier

// Memory fence
membar.cta;              // Fence for block (shared memory)
membar.gl;               // Fence for global memory
membar.sys;              // System-wide fence
```

### Warp-Level Operations

```ptx
// Warp shuffle
shfl.sync.bfly.b32 %r0|%p0, %r1, %r2, 0x1f, 0xffffffff;

// Warp vote
vote.all.pred p, q;      // p = all threads in warp have q true
vote.any.pred p, q;      // p = any thread in warp has q true
vote.uni.pred p, q;      // p = all threads have same q value

// Warp match
match.any.sync.b32 %r0|%p0, %r1, 0xffffffff;
```

---

## Tensor Core Instructions (Compute Capability 7.0+)

### WMMA (Warp Matrix Multiply-Accumulate)

```ptx
// Load matrix fragment
wmma.load.a.sync.aligned.row.m16n16k16.f16
    {%r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7},
    [%r_ptr], %r_stride;

// Matrix multiply-accumulate
wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16.f32
    {%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7},   // D (accumulator)
    {%r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7},   // A
    {%r8, %r9, %r10, %r11, %r12, %r13, %r14, %r15}, // B
    {%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7};   // C

// Store result
wmma.store.d.sync.aligned.row.m16n16k16.f32
    [%r_ptr], {%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7},
    %r_stride;
```

### MMA (Hopper Async Matrix Multiply)

```ptx
// Warpgroup matrix multiply (Hopper)
wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16
    {%f0, %f1, %f2, ..., %f127},  // 128 f32 accumulators
    %r_desc_a,                      // Descriptor for A
    %r_desc_b;                      // Descriptor for B

// Wait for wgmma to complete
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;
```

---

## Performance Optimization Techniques

### 1. Instruction-Level Parallelism (ILP)

```ptx
// BAD: Dependencies stall pipeline
add.f32 %f0, %f1, %f2;
mul.f32 %f0, %f0, %f3;   // Waits for previous add
sub.f32 %f0, %f0, %f4;   // Waits for previous mul

// GOOD: Independent operations
add.f32 %f0, %f1, %f2;
add.f32 %f5, %f6, %f7;   // Independent, can execute in parallel
add.f32 %f8, %f9, %f10;  // Independent
mul.f32 %f0, %f0, %f3;   // Now has time to resolve
```

### 2. Memory Coalescing

```ptx
// GOOD: Coalesced access (stride-1)
ld.global.f32 %f0, [%r0];           // Thread 0: addr + 0
ld.global.f32 %f0, [%r0 + 4];       // Thread 1: addr + 4
ld.global.f32 %f0, [%r0 + 8];       // Thread 2: addr + 8
// Single 128-byte transaction

// BAD: Strided access
ld.global.f32 %f0, [%r0];           // Thread 0: addr + 0
ld.global.f32 %f0, [%r0 + 128];     // Thread 1: addr + 128
ld.global.f32 %f0, [%r0 + 256];     // Thread 2: addr + 256
// Multiple transactions
```

### 3. Shared Memory Bank Conflicts

```ptx
// 32 banks, 4-byte width

// GOOD: No conflicts (different banks)
st.shared.f32 [base + 4 * tid], %f0;  // tid 0->bank 0, tid 1->bank 1

// BAD: 2-way conflict
st.shared.f32 [base + 8 * (tid/2)], %f0;  // tid 0,1 -> same bank

// GOOD: Padding to avoid conflicts
.shared .align 128 .b8 smem[33][32];  // 33 instead of 32 to avoid conflicts
```

---

## Complete Kernel Example

```ptx
.version 8.5
.target sm_90
.address_size 64

// Vector addition kernel: C = A + B
.visible .entry vecadd_kernel(
    .param .u64 param_A,
    .param .u64 param_B,
    .param .u64 param_C,
    .param .u32 param_N
)
{
    .reg .pred %p<2>;
    .reg .b32  %r<10>;
    .reg .b64  %rd<8>;
    .reg .f32  %f<3>;

    // Load parameters
    ld.param.u64 %rd0, [param_A];
    ld.param.u64 %rd1, [param_B];
    ld.param.u64 %rd2, [param_C];
    ld.param.u32 %r0,  [param_N];

    // Calculate global thread ID
    mov.u32 %r1, %ctaid.x;      // blockIdx.x
    mov.u32 %r2, %ntid.x;       // blockDim.x
    mov.u32 %r3, %tid.x;        // threadIdx.x

    mad.lo.u32 %r4, %r1, %r2, %r3;  // tid = blockIdx.x * blockDim.x + threadIdx.x

    // Bounds check
    setp.ge.u32 %p0, %r4, %r0;  // if (tid >= N)
    @%p0 bra EXIT;               // return

    // Calculate addresses
    mul.wide.u32 %rd3, %r4, 4;  // offset = tid * sizeof(float)
    add.u64 %rd4, %rd0, %rd3;   // &A[tid]
    add.u64 %rd5, %rd1, %rd3;   // &B[tid]
    add.u64 %rd6, %rd2, %rd3;   // &C[tid]

    // Load A[tid] and B[tid]
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];

    // C[tid] = A[tid] + B[tid]
    add.f32 %f2, %f0, %f1;

    // Store result
    st.global.f32 [%rd6], %f2;

EXIT:
    ret;
}
```

---

## Viewing Generated PTX

### Method 1: Compile to PTX

```bash
# Generate PTX file
nvcc -ptx kernel.cu -o kernel.ptx

# With optimization
nvcc -ptx -O3 --use_fast_math kernel.cu -o kernel.ptx

# For specific architecture
nvcc -ptx -arch=sm_90 kernel.cu -o kernel.ptx
```

### Method 2: View from Cubin

```bash
# Generate cubin with PTX embedded
nvcc -cubin -arch=sm_90 kernel.cu -o kernel.cubin

# Disassemble to view PTX/SASS
cuobjdump -ptx kernel.cubin
cuobjdump -sass kernel.cubin
```

### Method 3: Compiler Explorer

Use [Godbolt Compiler Explorer](https://godbolt.org/) with CUDA support to view PTX interactively.

---

## Debugging PTX

### Common Issues

1. **Register Spilling**: Too many registers, data moved to local memory
   ```bash
   nvcc --ptxas-options=-v kernel.cu
   # Look for "spill stores/loads"
   ```

2. **Alignment Violations**: Misaligned memory access
   ```bash
   cuda-memcheck ./program
   ```

3. **Invalid PTX**: Syntax or semantic errors
   ```bash
   ptxas kernel.ptx -o kernel.cubin
   # Errors will be reported
   ```

---

## Further Reading

- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Inline PTX Assembly Guide](https://docs.nvidia.com/cuda/inline-ptx-assembly/)
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
- [cuobjdump Documentation](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#cuobjdump)
