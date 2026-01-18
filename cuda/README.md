# Advanced GPU Programming Learning Path

> Complete guide from basics to expert-level CUDA and HIP programming with full, compilable code examples

---

## 🎯 Learning Path Overview

This repository contains **complete, production-ready code** for learning advanced GPU programming on NVIDIA (CUDA) and AMD (HIP) platforms.

**Skill Levels:**
- 📗 **Beginner**: Basic CUDA concepts
- 📘 **Intermediate**: Optimization techniques
- 📕 **Advanced**: Cutting-edge features (Hopper H100, Quantization, ML Kernels)
- 📙 **Expert**: Code generation, autotuning, custom kernels

---

## 📂 Directory Structure

```
cuda/
├── gpu/
│   ├── cuda/          # NVIDIA CUDA documentation and code
│   └── amd/           # AMD HIP documentation and code
└── README.md          # This file
```

---

## 🎓 CUDA Learning Track (NVIDIA GPUs)

### Foundation (Start Here)

1. **[Async Copies and TMA](gpu/cuda/01-async-copies-and-tma.md)** 📘
   - LDGSTS (Ampere)
   - Tensor Memory Accelerator (Hopper H100)
   - STAS for cluster memory
   - Complete working examples with barriers and pipelines

2. **[Thread Block Clusters](gpu/cuda/02-thread-block-clusters.md)** 📕
   - Hopper H100 distributed shared memory
   - Cluster-wide synchronization
   - TMA multicast examples
   - Full code for halo exchanges and reductions

3. **[PTX Assembly Guide](gpu/cuda/03-ptx-assembly-guide.md)** 📕
   - Low-level GPU programming
   - Inline PTX in CUDA
   - Performance tuning with assembly
   - Complete examples with tensor cores

---

### ML and HPC Kernels

4. **[Classic ML Kernels](gpu/cuda/04-classic-ml-kernels.md)** 📘
   - GEMM optimization (naive → tensor cores)
   - Convolution (direct, im2col, Winograd, FFT)
   - Complete implementations with benchmarks
   - cuBLAS and cuDNN integration

5. **[Advanced ML Kernels - COMPLETE](gpu/cuda/05-advanced-ml-kernels-COMPLETE.md)** 📕
   - **Full FlashAttention implementation**
   - Standard attention (3 kernels)
   - FlashAttention (online softmax + tiling)
   - RMSNorm (complete with warp reductions)
   - RoPE (Rotary Position Embedding)
   - **All code is compilable and runnable!**

---

### Quantization

6. **[Quantization Techniques - COMPLETE](gpu/cuda/06-quantization-COMPLETE.md)** 📕
   - **Complete INT8 implementation** with DP4A
   - **Complete INT4 implementation** with group quantization
   - FP8 E4M3/E5M2 (Hopper)
   - NVFP4 (Blackwell)
   - 1.58-bit ternary (BitNet)
   - **Kernel fusion examples** (GEMM+Bias+ReLU)
   - Full working code with error analysis

---

### Code Generation and Autotuning

7. **[Code Generation & Autotuning](gpu/cuda/07-code-generation-autotuning.md)** 📙
   - **Complete C++ autotuner** for GEMM
   - Triton DSL (Python GPU programming)
   - CUTLASS template metaprogramming
   - Full examples with benchmarking

---

## 🔴 AMD Learning Track (AMD GPUs)

### HIP Programming

1. **[HIP Programming Complete](gpu/amd/01-hip-programming-complete.md)** 📘
   - Vector addition (complete code)
   - Matrix multiplication (tiled, optimized)
   - rocBLAS integration
   - CUDA-to-HIP conversion guide
   - Wavefront size = 64 optimizations
   - Complete Makefile for cross-platform builds

---

## 🚀 Quick Start Guide

### Prerequisites

**For NVIDIA:**
```bash
# Check CUDA installation
nvcc --version

# Required: CUDA Toolkit 11.0+
# Recommended: CUDA 12.0+ for Hopper features
```

**For AMD:**
```bash
# Check ROCm installation
hipcc --version

# Required: ROCm 5.0+
```

### Compile Your First Example

**NVIDIA (FlashAttention):**
```bash
cd gpu/cuda
nvcc -o flash_attention flash_attention.cu -O3 -arch=sm_80
./flash_attention
```

**AMD (Vector Addition):**
```bash
cd gpu/amd
hipcc -o vecadd vecadd_hip.cpp -O3
./vecadd
```

---

## 📊 Learning Roadmap

### Week 1-2: Foundation
- [ ] Read async copies documentation
- [ ] Implement basic tiled GEMM
- [ ] Understand shared memory usage
- [ ] Benchmark your kernels

### Week 3-4: Advanced Features
- [ ] Study FlashAttention algorithm
- [ ] Implement online softmax
- [ ] Use thread block clusters (H100)
- [ ] Profile with Nsight Compute

### Week 5-6: Quantization
- [ ] Implement INT8 quantization
- [ ] Add INT4 with group quantization
- [ ] Measure accuracy vs speed tradeoffs
- [ ] Fuse kernels for better performance

### Week 7-8: Production
- [ ] Use Triton for rapid prototyping
- [ ] Integrate CUTLASS for production
- [ ] Autotune for your hardware
- [ ] Compare with cuBLAS/rocBLAS

---

## 🎯 Code Quality Standards

**All code in this repository:**
- ✅ **Compiles** without errors
- ✅ **Runs** with provided test cases
- ✅ **Includes** error checking
- ✅ **Has** benchmarking code
- ✅ **Provides** correctness verification
- ✅ **Documents** hardware requirements

---

## 📈 Performance Targets

### GEMM Performance (FP32)

| Hardware | Naive | Tiled | Tensor Cores | cuBLAS |
|----------|-------|-------|--------------|--------|
| RTX 3090 | 0.5 | 5.0 | 15.0 | 19.6 TFLOPS |
| A100 | 0.6 | 6.0 | 16.0 | 19.5 TFLOPS |
| H100 | 0.8 | 8.0 | 40.0 | 51.0 TFLOPS |

### Attention Performance (FP16, seq_len=2048)

| Implementation | A100 | H100 |
|----------------|------|------|
| PyTorch Eager | 12 ms | 8 ms |
| Standard Fused | 8 ms | 5 ms |
| FlashAttention-2 | 4 ms | 2.5 ms |
| FlashAttention-3 | - | 1.3 ms |

---

## 🛠️ Tools and Resources

### Profiling
- **Nsight Compute**: Kernel profiling
- **Nsight Systems**: Timeline analysis
- **ROCProfiler**: AMD profiling tool

### Libraries
- **CUTLASS**: NVIDIA GEMM templates
- **cuBLAS**: NVIDIA BLAS library
- **cuDNN**: NVIDIA DNN library
- **rocBLAS**: AMD BLAS library
- **MIOpen**: AMD DNN library

### Learning Resources
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [HIP Documentation](https://rocm.docs.amd.com/projects/HIP/)
- [GPU MODE Discord](https://discord.gg/gpumode)

---

## 📝 File Naming Convention

- `01-*.md`: Foundational concepts
- `02-*.md`: Intermediate techniques
- `03-*.md`: Advanced features
- `*-COMPLETE.md`: Fully working, production-ready code
- `*.cu`: CUDA source files
- `*.cpp`: HIP source files (also works as .cu for CUDA)

---

## 🤝 Contributing

This is a learning repository. Feel free to:
- Add more examples
- Improve documentation
- Fix bugs
- Add support for new GPU architectures

---

## 📜 License

Educational use. Code examples are provided as-is for learning purposes.

---

## 🎓 Credits

Documentation compiled from:
- NVIDIA official documentation (2025)
- AMD ROCm documentation (2025)
- Research papers (FlashAttention, NVFP4, BitNet)
- Community contributions (GPU MODE, CUDA subreddit)

---

## ⚡ Quick Reference Card

### Essential CUDA Syntax
```cpp
// Kernel launch
kernel<<<blocks, threads>>>(args);

// Memory
cudaMalloc(&ptr, size);
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
cudaFree(ptr);

// Synchronization
__syncthreads();           // Block-level
cudaDeviceSynchronize();   // Device-level

// Thread indexing
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### Essential HIP Syntax
```cpp
// Kernel launch
hipLaunchKernelGGL(kernel, blocks, threads, 0, 0, args);

// Memory
hipMalloc(&ptr, size);
hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
hipFree(ptr);

// Synchronization
__syncthreads();           // Block-level
hipDeviceSynchronize();    // Device-level

// Thread indexing (same as CUDA)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

---

**Start Learning:** [Async Copies and TMA →](gpu/cuda/01-async-copies-and-tma.md)

**Questions?** Open an issue or check [GPU MODE Discord](https://discord.gg/gpumode)
