# Advanced HIP GPU Programming: Complete Learning Resources

## Overview

This repository contains comprehensive documentation for mastering HIP (Heterogeneous Interface for Portability) and AMD GPU programming, from fundamental concepts through advanced assembly-level optimization.

**Target Audience:** Intermediate to advanced GPU programmers looking to achieve peak performance on AMD hardware (CDNA and RDNA architectures).

**Hardware Focus:** AMD Instinct MI100/MI200/MI300 (CDNA) and Radeon RX 6000/7000/9000 (RDNA) series GPUs.

---

## 📚 Documentation Structure

### 1. **HIP_KERNEL_OPTIMIZATION_GUIDE.md** (Fundamentals)
**Start here if you're new to HIP optimization**

- Memory hierarchy and optimization strategies
- Register pressure and occupancy management
- LDS (Local Data Share) optimization with bank conflict avoidance
- Memory coalescing patterns with visual examples
- Instruction-level optimization techniques
- Performance analysis tools and metrics

**Key Topics:**
- ✅ Coalesced memory access patterns
- ✅ XOR-based swizzle for LDS bank conflict elimination
- ✅ Double buffering and software pipelining
- ✅ Warp shuffle operations
- ✅ Occupancy calculation and optimization

**Prerequisites:** Basic HIP/CUDA programming knowledge

**Estimated Reading Time:** 45-60 minutes

---

### 2. **HIP_ASSEMBLY_MFMA_GUIDE.md** (Advanced Assembly)
**Deep dive into GPU ISA and matrix operations**

- AMD GPU ISA architecture (GCN, CDNA, RDNA)
- Reading and understanding AMDGCN assembly
- Register architecture (VGPR, SGPR, AGPR)
- MFMA (Matrix Fused Multiply-Add) instructions
- WMMA (Wave Matrix Multiply-Accumulate) for RDNA3+
- Assembly-level optimization techniques
- AITER (AI Tensor Engine) assembly kernels
- rocWMMA library usage

**Key Topics:**
- ✅ Complete MFMA instruction reference with examples
- ✅ Compiler intrinsics (__builtin_amdgcn_*)
- ✅ Software pipelining for MFMA latency hiding
- ✅ CDNA vs RDNA architecture differences
- ✅ Matrix core programming

**Prerequisites:** HIP_KERNEL_OPTIMIZATION_GUIDE.md + Assembly basics

**Estimated Reading Time:** 90-120 minutes

---

### 3. **ML_KERNELS_OPTIMIZATION.md** (Machine Learning Focus)
**Optimizations specific to AI/ML workloads**

- Flash Attention architecture and implementation
- Quantization techniques (INT8, INT4, FP8)
- Flash Decoding for long-context inference
- KV Cache optimization strategies
- Mixture of Experts (MoE) kernels
- Fused operators in deep learning
- Inference engine optimizations

**Key Topics:**
- ✅ Flash Attention with online softmax (2-4x speedup)
- ✅ Weight-only quantization (4x memory reduction)
- ✅ Flash Decoding parallelization (5-10x for long contexts)
- ✅ Continuous batching and speculative decoding
- ✅ Fused MoE kernels (3x speedup)

**Prerequisites:** HIP_KERNEL_OPTIMIZATION_GUIDE.md + Deep learning fundamentals

**Estimated Reading Time:** 75-90 minutes

---

### 4. **KERNEL_FUSION_ADVANCED_PATTERNS.md** (Compiler Optimization)
**Advanced fusion strategies and automation**

- Kernel fusion fundamentals and benefits
- Vertical vs horizontal fusion taxonomy
- RL-based fusion optimization with PPO
- Actor-critic fusion strategies
- Polyhedral optimization techniques
- Fusion in TVM, XLA, PyTorch 2.0
- Multi-level fusion hierarchies

**Key Topics:**
- ✅ Reinforcement learning for automatic fusion (3-4x speedup)
- ✅ Graph Neural Networks for fusion decisions
- ✅ Polyhedral model for loop optimization
- ✅ Real-world fusion examples (ResNet, BERT, Transformers)
- ✅ Operator fusion in production compilers

**Prerequisites:** ML_KERNELS_OPTIMIZATION.md + Compiler basics

**Estimated Reading Time:** 60-75 minutes

---

### 5. **AMD_INTRINSICS_HIPKITTENS.md** (Modern Frameworks)
**Latest tools and intrinsics for AMD GPUs**

- Complete __builtin_amdgcn_* intrinsic reference
- HipKittens tile-based programming framework
- Wave-level operations and DPP (Data Parallel Primitives)
- LDS permute operations for cross-lane communication
- Fast math intrinsics
- Memory fence and synchronization primitives
- HipKittens vs alternatives comparison

**Key Topics:**
- ✅ 100+ intrinsic functions with examples
- ✅ HipKittens achieving 90-95% peak performance
- ✅ Tile-based abstractions for clean code
- ✅ Wave shuffle reductions
- ✅ Production-ready kernel templates

**Prerequisites:** HIP_ASSEMBLY_MFMA_GUIDE.md

**Estimated Reading Time:** 60-90 minutes

---

## 🎯 Learning Paths

### Path 1: Performance Optimization Track
**Goal: Optimize existing HIP kernels**

1. HIP_KERNEL_OPTIMIZATION_GUIDE.md (Fundamentals)
2. AMD_INTRINSICS_HIPKITTENS.md (Intrinsics)
3. HIP_ASSEMBLY_MFMA_GUIDE.md (Assembly insights)

**Time Commitment:** 3-4 hours
**Outcome:** Ability to optimize memory-bound and compute-bound kernels

---

### Path 2: Machine Learning Engineer Track
**Goal: Write high-performance ML kernels**

1. HIP_KERNEL_OPTIMIZATION_GUIDE.md (Fundamentals)
2. ML_KERNELS_OPTIMIZATION.md (ML-specific techniques)
3. AMD_INTRINSICS_HIPKITTENS.md (HipKittens framework)

**Time Commitment:** 3-4 hours
**Outcome:** Implement Flash Attention, quantization, and fused operators

---

### Path 3: Compiler Developer Track
**Goal: Build optimizing compilers for GPUs**

1. HIP_KERNEL_OPTIMIZATION_GUIDE.md (Fundamentals)
2. KERNEL_FUSION_ADVANCED_PATTERNS.md (Fusion techniques)
3. HIP_ASSEMBLY_MFMA_GUIDE.md (Assembly codegen)

**Time Commitment:** 4-5 hours
**Outcome:** Implement automatic fusion and code generation

---

### Path 4: GPU Architecture Expert Track
**Goal: Deep understanding of AMD GPU architecture**

1. HIP_KERNEL_OPTIMIZATION_GUIDE.md
2. HIP_ASSEMBLY_MFMA_GUIDE.md
3. AMD_INTRINSICS_HIPKITTENS.md
4. ML_KERNELS_OPTIMIZATION.md
5. KERNEL_FUSION_ADVANCED_PATTERNS.md

**Time Commitment:** 6-8 hours
**Outcome:** Mastery of AMD GPU programming from HIP to assembly

---

## 🔥 Highlight Features

### ASCII Diagrams Throughout
All documents include detailed ASCII diagrams for:
- Memory hierarchy visualization
- Attention mechanism computation flow
- Register allocation patterns
- Tile-based matrix operations
- Wavefront execution models

### Code Examples
Over **100+ runnable code examples** including:
- Complete kernel implementations
- Intrinsic function usage patterns
- HipKittens tile-based kernels
- Assembly snippets with explanations
- Performance optimization patterns

### Performance Metrics
Real-world benchmarks from:
- AMD MI250X and MI300 GPUs
- AITER (AMD's hand-optimized kernels)
- HipKittens framework results
- Flash Attention implementations
- Production ML inference engines

---

## 📊 Key Technologies Covered

### Hardware Architectures
- ✅ CDNA 1/2/3 (MI100, MI200, MI300)
- ✅ RDNA 2/3/4 (RX 6000/7000/9000)
- ✅ GCN (legacy support)

### AMD Software Stack
- ✅ ROCm 6.0+
- ✅ HIP runtime and compiler
- ✅ rocBLAS, rocWMMA libraries
- ✅ AITER kernels

### Programming Models
- ✅ HIP C++ API
- ✅ Compiler intrinsics (__builtin_amdgcn_*)
- ✅ HipKittens framework
- ✅ Inline assembly (with caveats)
- ✅ rocWMMA matrix operations

### Optimization Techniques
- ✅ Memory coalescing and alignment
- ✅ LDS bank conflict avoidance (XOR swizzle)
- ✅ Register pressure optimization
- ✅ Occupancy tuning
- ✅ Instruction-level parallelism
- ✅ Software pipelining
- ✅ Kernel fusion (vertical & horizontal)
- ✅ Quantization (INT8, INT4, FP8)

---

## 🛠️ Prerequisites

### Required Knowledge
- C++ programming (intermediate level)
- Basic GPU programming concepts (threads, blocks, memory hierarchy)
- Linear algebra (matrix operations)
- For ML track: Deep learning fundamentals

### Software Requirements
- ROCm 6.0 or later
- AMD GPU (Instinct or Radeon)
- CMake 3.20+
- GCC 11+ or Clang 14+

### Optional but Recommended
- PyTorch (for ML examples)
- Triton (for comparison)
- rocprof (for profiling)

---

## 📖 How to Use This Documentation

### For Self-Study
1. Choose a learning path based on your goals
2. Read documents sequentially within the path
3. Try code examples on your AMD GPU
4. Profile and iterate on performance

### For Teams
1. Start with HIP_KERNEL_OPTIMIZATION_GUIDE.md for all team members
2. Specialize based on roles:
   - ML Engineers: ML_KERNELS_OPTIMIZATION.md
   - Compiler Developers: KERNEL_FUSION_ADVANCED_PATTERNS.md
   - Performance Engineers: HIP_ASSEMBLY_MFMA_GUIDE.md

### For Reference
- Use as quick reference for intrinsics and patterns
- Copy-paste optimized code snippets
- Refer to ASCII diagrams for architecture understanding

---

## 🎓 Advanced Topics Coverage

### Instruction Set Architecture
- Wave64 vs Wave32 execution models
- VGPR, SGPR, AGPR register types
- Instruction encoding and scheduling
- ISA differences across CDNA/RDNA

### Matrix Operations
- MFMA instruction variants (16×16, 32×32, etc.)
- FP32, FP16, BF16, INT8, FP64 matrix multiply
- WMMA for RDNA3 (AMD's answer to Tensor Cores)
- Matrix core throughput analysis

### Memory Optimization
- 32-bank LDS architecture on AMD
- XOR-based swizzle patterns
- Padding vs swizzle tradeoffs
- Vectorized memory access (float4, etc.)

### Machine Learning Kernels
- Flash Attention (2-4x speedup)
- Flash Decoding (5-10x for long contexts)
- Quantization (4-8x memory reduction)
- Fused operators (2-3x speedup)

### Compiler Techniques
- RL-based fusion with PPO
- Graph Neural Networks for fusion decisions
- Polyhedral optimization
- Auto-tuning strategies

---

## 🌟 Real-World Applications

These techniques are used in:
- **vLLM**: High-throughput LLM serving
- **SGLang**: Fast inference for language models
- **DeepSeek**: Large-scale model training/inference (13K → 27K tokens/sec with AITER)
- **AMD PyTorch**: ROCm backend optimizations
- **HuggingFace Transformers**: AMD GPU acceleration
- **Triton**: GPU programming language
- **TVM/XLA**: Optimizing compilers

---

## 📈 Expected Performance Gains

By applying techniques from these guides:

**Memory-Bound Kernels:**
- Baseline → Coalesced: 2-3x
- Coalesced → + LDS tiling: 1.5-2x
- + Register optimization: 1.2-1.5x
- **Total: 4-9x speedup**

**Compute-Bound Kernels:**
- Baseline → MFMA usage: 10-20x
- + Occupancy tuning: 1.3-1.5x
- + Software pipelining: 1.2-1.3x
- **Total: 15-40x speedup**

**ML Workloads:**
- Flash Attention: 2-4x vs standard
- Quantization: 2-4x throughput + 4-8x memory
- Kernel fusion: 2-3x vs unfused
- **End-to-end: 5-20x improvement possible**

---

## 🤝 Contributing

Found an error? Have a suggestion? Want to add examples?

These documentation files are designed to be living resources. Contributions welcome for:
- Additional code examples
- Architecture-specific optimizations
- Updated benchmarks
- Clarifications and corrections

---

## 📚 Additional Resources

### Official AMD Documentation
- [ROCm Documentation](https://rocm.docs.amd.com)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP)
- [AMD Instinct ISA Reference](https://www.amd.com/en/support/instinct-accelerators)
- [AMD GPU Architecture Docs](https://gpuopen.com)

### Research Papers
- Flash Attention: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Flash Attention-2: [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- HipKittens: [arXiv:2511.08083](https://arxiv.org/abs/2511.08083)

### Open Source Projects
- [HipKittens](https://github.com/HazyResearch/HipKittens)
- [AITER (ROCm AI Tensor Engine)](https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine)
- [rocWMMA](https://github.com/ROCm/rocWMMA)
- [Composable Kernels (CK)](https://github.com/ROCm/composable_kernel)

### Community
- ROCm GitHub Discussions
- AMD GPU Open Forums
- r/ROCM on Reddit

---

## 🏆 Summary

This documentation suite represents **hundreds of hours of research, testing, and optimization** on AMD GPUs. By working through these guides, you will:

1. **Understand** AMD GPU architecture at a deep level
2. **Write** high-performance kernels using HIP and intrinsics
3. **Optimize** memory access patterns and register usage
4. **Leverage** MFMA/WMMA matrix operations
5. **Implement** state-of-the-art ML kernels (Flash Attention, quantization)
6. **Apply** advanced compiler techniques (fusion, RL optimization)
7. **Use** modern frameworks like HipKittens

**Start with HIP_KERNEL_OPTIMIZATION_GUIDE.md and happy optimizing!**

---

## 📜 License

This documentation is provided as educational material for learning HIP and AMD GPU programming. Code examples may be used freely in your projects.

---

**Last Updated:** January 2026
**AMD Hardware Tested:** MI250X, MI300, RX 7900 XT
**ROCm Version:** 6.0+

---

**Questions? Suggestions? Open an issue or discussion!**
