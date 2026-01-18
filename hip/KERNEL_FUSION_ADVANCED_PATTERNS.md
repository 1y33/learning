# Advanced Kernel Fusion Patterns and Techniques

## Table of Contents
1. [Kernel Fusion Fundamentals](#fusion-fundamentals)
2. [Vertical vs Horizontal Fusion](#vertical-horizontal)
3. [RL-Based Fusion Optimization](#rl-fusion)
4. [Actor-Critic Fusion Strategies](#actor-critic)
5. [Polyhedral Optimization](#polyhedral)
6. [Fusion in Deep Learning Compilers](#dl-compilers)
7. [Advanced Composition Patterns](#composition)
8. [Performance Analysis](#performance-analysis)

---

## Kernel Fusion Fundamentals {#fusion-fundamentals}

### Why Kernel Fusion Matters

```
┌──────────────────────────────────────────────────────────────┐
│            KERNEL FUSION MOTIVATION                           │
│                                                                │
│  GPU Performance Bottlenecks:                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  1. Memory Bandwidth (PRIMARY)                          │  │
│  │     • Modern GPUs: 1-5 TB/s HBM bandwidth               │  │
│  │     • Compute: 100-300 TFLOPS (FP32)                    │  │
│  │     • Arithmetic Intensity = FLOPS / Byte               │  │
│  │     • Most kernels: Memory-bound, not compute-bound     │  │
│  │                                                          │  │
│  │  2. Kernel Launch Overhead                              │  │
│  │     • Each kernel launch: 5-20 μs overhead              │  │
│  │     • For small operations: Overhead > Computation!     │  │
│  │                                                          │  │
│  │  3. Cache Locality                                      │  │
│  │     • Data evicted between kernels                      │  │
│  │     • Repeated loads from HBM                           │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Fusion Benefits:                                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  ✅ Reduced Memory Traffic                              │  │
│  │     Eliminate intermediate buffers in HBM               │  │
│  │                                                          │  │
│  │  ✅ Improved Cache Utilization                          │  │
│  │     Data stays in cache/registers                       │  │
│  │                                                          │  │
│  │  ✅ Fewer Kernel Launches                               │  │
│  │     Reduced CPU-GPU synchronization overhead            │  │
│  │                                                          │  │
│  │  ✅ Higher Arithmetic Intensity                         │  │
│  │     More compute per byte loaded                        │  │
│  │                                                          │  │
│  │  ✅ Better Instruction-Level Parallelism                │  │
│  │     More operations to schedule                         │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Fusibility Analysis

```
┌──────────────────────────────────────────────────────────────┐
│              WHEN CAN KERNELS BE FUSED?                       │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Requirements for Safe Fusion:                                │
│                                                                │
│  1. DATA DEPENDENCIES                                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Producer → Consumer relationship                       │  │
│  │                                                          │  │
│  │  ✅ Fusible:                                            │  │
│  │    Y = Op1(X)                                           │  │
│  │    Z = Op2(Y)     ← Y only used here                    │  │
│  │                                                          │  │
│  │  ❌ Not Fusible:                                        │  │
│  │    Y = Op1(X)                                           │  │
│  │    Z1 = Op2(Y)                                          │  │
│  │    Z2 = Op3(Y)    ← Y has multiple consumers            │  │
│  │    Z3 = Op4(Z1, Z2)                                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  2. COMPUTATIONAL PATTERNS                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Element-wise ops: Highly fusible                       │  │
│  │    Y[i] = f(X[i])                                       │  │
│  │                                                          │  │
│  │  Reductions: Partial fusibility                         │  │
│  │    Y = Σ X[i]     ← Different parallelism pattern       │  │
│  │                                                          │  │
│  │  Matrix operations: Complex fusion                      │  │
│  │    Y = X @ W      ← Tiling required                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  3. MEMORY FOOTPRINT                                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Must fit in shared memory/registers:                   │  │
│  │    Working_Set_Size < LDS_Size (64KB per CU)           │  │
│  │                                                          │  │
│  │  If too large: Tile-based fusion                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  4. SYNCHRONIZATION REQUIREMENTS                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Within-block sync: __syncthreads() supported          │  │
│  │  Cross-block sync: Requires kernel boundaries           │  │
│  │                                                          │  │
│  │  ❌ Cannot fuse ops requiring global synchronization    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Vertical vs Horizontal Fusion {#vertical-horizontal}

### Fusion Taxonomy

```
┌──────────────────────────────────────────────────────────────┐
│              KERNEL FUSION TYPES                              │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  VERTICAL FUSION (Producer-Consumer Chain)                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  Computation Graph:                                     │  │
│  │                                                          │  │
│  │      [Input X]                                          │  │
│  │          ↓                                               │  │
│  │      [Op1: ReLU]                                        │  │
│  │          ↓                                               │  │
│  │      [Op2: BatchNorm]                                   │  │
│  │          ↓                                               │  │
│  │      [Op3: Dropout]                                     │  │
│  │          ↓                                               │  │
│  │      [Output Y]                                         │  │
│  │                                                          │  │
│  │  Fused Kernel:                                          │  │
│  │    Y[i] = dropout(batchnorm(relu(X[i])))               │  │
│  │                                                          │  │
│  │  Memory Access:                                         │  │
│  │    Unfused: 4 HBM reads + 3 HBM writes = 7 accesses    │  │
│  │    Fused:   1 HBM read + 1 HBM write = 2 accesses      │  │
│  │    Speedup: 3.5x (memory-bound case)                    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  HORIZONTAL FUSION (Parallel Operations)                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  Computation Graph:                                     │  │
│  │                                                          │  │
│  │         [Input X]                                       │  │
│  │            ↓                                             │  │
│  │      ┌─────┴─────┐                                      │  │
│  │      ↓           ↓                                       │  │
│  │  [Op1: sin]  [Op2: cos]                                │  │
│  │      ↓           ↓                                       │  │
│  │  [Y1 = sin(X)] [Y2 = cos(X)]                           │  │
│  │                                                          │  │
│  │  Fused Kernel:                                          │  │
│  │    Y1[i], Y2[i] = sin(X[i]), cos(X[i])                │  │
│  │    (Process both in same kernel)                        │  │
│  │                                                          │  │
│  │  Benefits:                                              │  │
│  │    • Shared input loading (1 read instead of 2)        │  │
│  │    • Reduced kernel launch overhead (1 vs 2 launches)  │  │
│  │    • Better instruction scheduling                      │  │
│  │    Speedup: 1.5-2x                                      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  BACKWARD VERTICAL FUSION                                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Reverse dependency chain (consumer → producer)         │  │
│  │                                                          │  │
│  │  Example: Gradient backpropagation                      │  │
│  │    dL/dX = dL/dY × dY/dZ × dZ/dX                       │  │
│  │                                                          │  │
│  │  Fuse gradient computations in reverse order            │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  DIVERGENT HORIZONTAL FUSION                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Fuse operations with different output shapes           │  │
│  │                                                          │  │
│  │  Example:                                               │  │
│  │    Y1 = reduce_sum(X, axis=1)    [N, M] → [N, 1]      │  │
│  │    Y2 = reduce_max(X, axis=0)    [N, M] → [1, M]      │  │
│  │                                                          │  │
│  │  Challenges: Different thread mapping strategies        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Hybrid Fusion Example

```cpp
// Fused Transformer Layer: Multiple fusion types combined
__global__ void fused_transformer_block(
    float* output,              // [B, N, D]
    const float* input,         // [B, N, D]
    const float* qkv_weight,    // [3*D, D]
    const float* out_weight,    // [D, D]
    const float* ln1_gamma,
    const float* ln1_beta,
    const float* ln2_gamma,
    const float* ln2_beta,
    int B, int N, int D
) {
    // This kernel fuses:
    //   1. LayerNorm1
    //   2. QKV projection (vertical fusion)
    //   3. Multi-head attention
    //   4. Output projection
    //   5. Residual connection 1 (vertical fusion)
    //   6. LayerNorm2
    //   7. FFN (Feed-Forward Network)
    //   8. Residual connection 2 (vertical fusion)

    extern __shared__ float smem[];

    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;

    // Load input
    float x[D];
    load_vector(x, input + batch_idx * N * D + seq_idx * D, D);

    // FUSION 1: LayerNorm1 → QKV projection
    float ln1_out[D];
    layer_norm_inplace(ln1_out, x, ln1_gamma, ln1_beta, D);

    // Keep ln1_out in registers, immediately use for QKV
    float q[D], k[D], v[D];
    matmul_3way(q, k, v, ln1_out, qkv_weight, D);  // Fused QKV

    // FUSION 2: Self-attention (using Flash Attention internally)
    float attn_out[D];
    flash_attention_fused(attn_out, q, k, v, N, D);

    // FUSION 3: Output projection + Residual 1
    float proj_out[D];
    matmul_fused_add(proj_out, attn_out, out_weight, x, D);  // Fused

    // FUSION 4: LayerNorm2 → FFN → Residual 2
    float ln2_out[D];
    layer_norm_inplace(ln2_out, proj_out, ln2_gamma, ln2_beta, D);

    // FFN: Two linear layers with GELU activation
    float ffn_hidden[4*D];
    matmul_fused_gelu(ffn_hidden, ln2_out, ffn_w1, 4*D);  // Fused

    float ffn_out[D];
    matmul_fused_add(ffn_out, ffn_hidden, ffn_w2, proj_out, D);  // Fused

    // Write output
    store_vector(output + batch_idx * N * D + seq_idx * D, ffn_out, D);
}

// Performance:
//   Unfused: ~15 kernel launches
//   Fused: 1 kernel launch
//   Speedup: 3-5x (depending on model size and sequence length)
```

---

## RL-Based Fusion Optimization {#rl-fusion}

### The Fusion Search Problem

```
┌──────────────────────────────────────────────────────────────┐
│           FUSION AS A COMBINATORIAL PROBLEM                   │
│                                                                │
│  Given: Computation graph with N operations                   │
│  Goal: Find optimal fusion strategy                           │
│  Search Space: Exponential in N!                              │
│                                                                │
│  Example: 10 operations                                       │
│    Possible fusion patterns: > 10! possibilities              │
│    Hand-coded rules: Suboptimal, brittle                      │
│    Solution: Machine learning!                                │
│                                                                │
│  Challenges:                                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  • Hardware-specific optimal fusion differs            │  │
│  │  • Input shapes affect fusion decisions                │  │
│  │  • Memory constraints vary by workload                 │  │
│  │  • Compile-time vs runtime tradeoffs                   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### RL Formulation for Fusion

```
┌──────────────────────────────────────────────────────────────┐
│      REINFORCEMENT LEARNING FOR KERNEL FUSION                 │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  STATE: s_t                                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  • Computation graph representation                     │  │
│  │    - Node features: op type, shape, memory footprint   │  │
│  │    - Edge features: data dependencies, tensor sizes    │  │
│  │  • Current fusion decisions (partial solution)          │  │
│  │  • Hardware constraints (LDS size, register pressure)   │  │
│  │  • Performance counters (estimated cost so far)         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  ACTION: a_t                                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Decision: Which operations to fuse next?              │  │
│  │                                                          │  │
│  │  Actions space:                                         │  │
│  │    1. Fuse Op_i with Op_j (if valid)                   │  │
│  │    2. Keep Op_i separate                                │  │
│  │    3. Create fusion cluster [Op_i, Op_j, Op_k]         │  │
│  │                                                          │  │
│  │  Validity constraints:                                  │  │
│  │    • Data dependencies respected                        │  │
│  │    • Memory budget not exceeded                         │  │
│  │    • Synchronization requirements met                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  REWARD: r_t                                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Objective: Minimize end-to-end latency                │  │
│  │                                                          │  │
│  │  Reward function:                                       │  │
│  │    r = -latency(fused_program)                         │  │
│  │                                                          │  │
│  │  Measured by:                                           │  │
│  │    • Profiling actual kernel execution                  │  │
│  │    • Or: Cost model (memory accesses, compute)         │  │
│  │                                                          │  │
│  │  Sparse reward: Only at episode end (all ops fused)    │  │
│  │  Dense reward: Intermediate feedback per fusion step   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  POLICY: π(a_t | s_t; θ)                                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Neural network architecture:                           │  │
│  │                                                          │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  Input: Graph embedding                          │  │  │
│  │  │    ↓                                              │  │  │
│  │  │  Graph Neural Network (GNN)                      │  │  │
│  │  │    • Message passing over computation graph      │  │  │
│  │  │    • Node embeddings capture op characteristics  │  │  │
│  │  │    ↓                                              │  │  │
│  │  │  Attention Layer                                  │  │  │
│  │  │    • Focus on fusible operation pairs            │  │  │
│  │  │    ↓                                              │  │  │
│  │  │  Policy Head: Softmax over actions               │  │  │
│  │  │    P(fuse Op_i, Op_j) = softmax(score_ij)       │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                                                          │  │
│  │  Training: PPO (Proximal Policy Optimization)          │  │
│  │    • Sample-efficient                                   │  │
│  │    • Stable training                                    │  │
│  │    • Widely used in compilers (TVM, XLA)               │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Priority-Based Fusion with RL

```python
# Pseudocode for RL-based fusion
class FusionRL:
    def __init__(self, graph_nn, policy_nn):
        self.graph_encoder = graph_nn
        self.policy_network = policy_nn

    def train(self, computation_graphs, num_episodes=10000):
        """Train policy using PPO"""
        optimizer = AdamW(self.policy_network.parameters())

        for episode in range(num_episodes):
            graph = sample(computation_graphs)
            state = self.encode_state(graph)

            trajectory = []  # (state, action, reward) tuples
            done = False

            while not done:
                # Generate fusion action
                action_probs = self.policy_network(state)
                action = sample_action(action_probs)

                # Validate action
                if not self.is_valid_fusion(graph, action):
                    reward = -10  # Penalty for invalid fusion
                    done = True
                else:
                    # Apply fusion
                    graph = self.apply_fusion(graph, action)

                    # Check if all fusible ops are processed
                    if self.is_terminal(graph):
                        # Measure performance of fused program
                        latency = self.profile_kernel(graph)
                        reward = -latency  # Negative for minimization
                        done = True
                    else:
                        reward = 0  # No intermediate reward
                        state = self.encode_state(graph)

                trajectory.append((state, action, reward))

            # Update policy using PPO
            self.ppo_update(trajectory, optimizer)

    def encode_state(self, graph):
        """
        Encode computation graph as GNN embedding
        """
        # Node features: [op_type, input_shape, output_shape, memory_cost]
        node_features = []
        for op in graph.ops:
            features = [
                self.op_type_embedding[op.type],
                op.input_shape,
                op.output_shape,
                self.estimate_memory(op)
            ]
            node_features.append(features)

        # Edge features: data dependencies
        adjacency_matrix = graph.get_adjacency()

        # GNN forward pass
        embeddings = self.graph_encoder(node_features, adjacency_matrix)

        return embeddings

    def is_valid_fusion(self, graph, action):
        """
        Check if proposed fusion satisfies constraints
        """
        op_i, op_j = action

        # Check data dependencies
        if not graph.can_fuse(op_i, op_j):
            return False

        # Check memory constraints
        fused_memory = self.estimate_memory(op_i) + self.estimate_memory(op_j)
        if fused_memory > LDS_SIZE:
            return False

        # Check synchronization requirements
        if requires_global_sync(op_i, op_j):
            return False

        return True

    def profile_kernel(self, fused_graph):
        """
        Compile and profile fused kernel on actual hardware
        """
        kernel_code = self.codegen(fused_graph)
        compiled = hip_compile(kernel_code)
        latency = run_benchmark(compiled)
        return latency
```

### Results from Literature

```
┌──────────────────────────────────────────────────────────────┐
│        RL-BASED FUSION PERFORMANCE (Published Results)        │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  "Learning to Fuse" Paper (Google, 2019):                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Model: GNN + PPO policy                                │  │
│  │  Benchmark: TensorFlow graphs                           │  │
│  │                                                          │  │
│  │  Results:                                               │  │
│  │    • RL policy vs hand-tuned XLA: 8% speedup           │  │
│  │    • RL policy vs no fusion: 3.41x speedup             │  │
│  │    • Training time: 24 hours on 64 TPUs                │  │
│  │    • Generalization: Works on unseen graphs            │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Operator Fusion in XLA (Google, 2018):                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  ResNet-50: 1.48x speedup                               │  │
│  │  Transformer: 1.92x speedup                             │  │
│  │  BERT: 2.13x speedup                                    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Actor-Critic Fusion Strategies {#actor-critic}

### Actor-Critic Architecture for Fusion

```
┌──────────────────────────────────────────────────────────────┐
│          ACTOR-CRITIC FOR KERNEL FUSION                       │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ARCHITECTURE:                                                │
│                                                                │
│                    Computation Graph                          │
│                           ↓                                    │
│                  ┌─────────────────┐                          │
│                  │  Graph Encoder  │                          │
│                  │      (GNN)      │                          │
│                  └────────┬────────┘                          │
│                           │                                    │
│             ┌─────────────┴─────────────┐                     │
│             ↓                           ↓                      │
│     ┌──────────────┐            ┌──────────────┐             │
│     │    ACTOR     │            │    CRITIC    │             │
│     │   Network    │            │   Network    │             │
│     └──────┬───────┘            └──────┬───────┘             │
│            │                           │                      │
│            ↓                           ↓                      │
│    Action Probabilities        State Value V(s)              │
│    π(a|s) ∈ [0,1]^|A|        V(s) ∈ ℝ                       │
│            │                           │                      │
│            │                           │                      │
│            └──────────┬────────────────┘                      │
│                       │                                       │
│                  Training Signal                              │
│             Advantage = Q(s,a) - V(s)                        │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  ACTOR: Learns WHAT actions to take                    │  │
│  │    Policy: π(a|s; θ)                                   │  │
│  │    Output: Probability distribution over fusion actions│  │
│  │    Loss: -log π(a|s) × Advantage                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  CRITIC: Learns HOW GOOD a state is                    │  │
│  │    Value function: V(s; φ)                             │  │
│  │    Output: Expected cumulative reward from state s     │  │
│  │    Loss: MSE(V(s), actual_return)                      │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Advantage Calculation

```
┌──────────────────────────────────────────────────────────────┐
│             ADVANTAGE-BASED POLICY GRADIENT                   │
│                                                                │
│  Problem with vanilla policy gradient:                        │
│    High variance → unstable training                          │
│                                                                │
│  Solution: Advantage function                                 │
│    A(s,a) = Q(s,a) - V(s)                                    │
│                                                                │
│  Interpretation:                                              │
│    "How much better is action 'a' compared to average?"      │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Q(s,a): Action-value function                          │  │
│  │          Expected return after taking action a in s     │  │
│  │                                                          │  │
│  │  V(s):   State-value function                           │  │
│  │          Expected return from state s (any action)      │  │
│  │                                                          │  │
│  │  A(s,a): Advantage                                      │  │
│  │          Positive: action a is better than average      │  │
│  │          Negative: action a is worse than average       │  │
│  │          Zero: action a is average                      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Practical Estimation (GAE - Generalized Advantage Est.):    │
│                                                                │
│    δ_t = r_t + γV(s_{t+1}) - V(s_t)    [TD error]           │
│                                                                │
│    A_t = Σ (γλ)^k δ_{t+k}              [GAE]                 │
│          k=0                                                  │
│                                                                │
│  Where:                                                       │
│    γ = discount factor (0.99)                                │
│    λ = GAE parameter (0.95)                                  │
└──────────────────────────────────────────────────────────────┘
```

### Actor-Critic Training Loop

```python
class ActorCriticFusion:
    def __init__(self):
        self.actor = ActorNetwork()   # Policy π(a|s; θ)
        self.critic = CriticNetwork()  # Value V(s; φ)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=1e-3)

    def train_step(self, graph):
        """Single training step"""
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []

        state = self.encode_graph(graph)
        done = False

        # Collect trajectory
        while not done:
            # Actor: Select action
            action_probs = self.actor(state)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            # Critic: Estimate state value
            value = self.critic(state)

            # Take action
            next_state, reward, done = self.env_step(graph, action)

            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)

            state = next_state

        # Compute returns and advantages
        returns = self.compute_returns(rewards, gamma=0.99)
        advantages = returns - torch.tensor(values)

        # Normalize advantages (stabilizes training)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update Actor (policy gradient)
        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Update Critic (value function)
        critic_loss = F.mse_loss(torch.tensor(values), returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def compute_returns(self, rewards, gamma=0.99):
        """Compute discounted returns"""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return torch.tensor(returns)
```

---

## Polyhedral Optimization {#polyhedral}

### Polyhedral Model for Loop Optimization

The polyhedral model provides a mathematical framework for analyzing and optimizing loop nests:

```
┌──────────────────────────────────────────────────────────────┐
│            POLYHEDRAL MODEL FUNDAMENTALS                      │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Represents loop nests using:                                 │
│    • Iteration domain (polyhedra)                             │
│    • Access functions (affine expressions)                    │
│    • Dependencies (affine relations)                          │
│                                                                │
│  Example Code:                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  for (i = 0; i < N; i++) {                             │  │
│  │      for (j = 0; j < M; j++) {                         │  │
│  │          C[i][j] = A[i][j] + B[i][j];                  │  │
│  │      }                                                   │  │
│  │  }                                                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Polyhedral Representation:                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Iteration Domain D:                                    │  │
│  │    { [i, j] : 0 ≤ i < N ∧ 0 ≤ j < M }                  │  │
│  │                                                          │  │
│  │  Access Functions:                                      │  │
│  │    C[i, j]: write access                                │  │
│  │    A[i, j]: read access                                 │  │
│  │    B[i, j]: read access                                 │  │
│  │                                                          │  │
│  │  Schedule σ: D → Time                                   │  │
│  │    σ([i, j]) = [i, j]  (original order)                │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Transformations (affine scheduling):                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Loop tiling:                                           │  │
│  │    σ([i,j]) = [i/32, j/32, i%32, j%32]                │  │
│  │                                                          │  │
│  │  Loop interchange:                                      │  │
│  │    σ([i,j]) = [j, i]                                   │  │
│  │                                                          │  │
│  │  Loop fusion:                                           │  │
│  │    Merge iteration spaces with compatible schedules    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Automatic Fusion with Polyhedral Analysis

```cpp
// Original unfused code
for (int i = 0; i < N; i++) {
    Y[i] = X[i] + 1.0f;
}
for (int i = 0; i < N; i++) {
    Z[i] = Y[i] * 2.0f;
}

// Polyhedral analysis:
//   Loop 1 domain: { [i] : 0 ≤ i < N }
//   Loop 2 domain: { [i] : 0 ≤ i < N }
//   Dependency: Loop2[i] depends on Loop1[i]
//   Decision: FUSIBLE (same iteration space, producer-consumer)

// Fused code (automatically generated)
for (int i = 0; i < N; i++) {
    Y[i] = X[i] + 1.0f;
    Z[i] = Y[i] * 2.0f;
}

// Further optimization: Eliminate Y array!
for (int i = 0; i < N; i++) {
    float temp = X[i] + 1.0f;
    Z[i] = temp * 2.0f;
}
```

---

## Fusion in Deep Learning Compilers {#dl-compilers}

### TVM/Apache TVM Fusion

```
┌──────────────────────────────────────────────────────────────┐
│              TVM OPERATOR FUSION PASSES                       │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Pass 1: OPERATOR FUSION                                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Groups operators into fusible patterns:                │  │
│  │                                                          │  │
│  │  • Injective: element-wise (broadcasted ops)            │  │
│  │  • Reduction: reduce operations (sum, max)              │  │
│  │  • OutEWiseFusable: e.g., conv2d (can fuse after)      │  │
│  │  • Opaque: complex ops that can't be fused             │  │
│  │                                                          │  │
│  │  Fusion Rules:                                          │  │
│  │    Injective ← Injective: ✅ Always fusible            │  │
│  │    Injective ← Reduction: ✅ Fusible                   │  │
│  │    Reduction ← Injective: ✅ Fusible                   │  │
│  │    OutEWiseFusable ← Injective: ✅ Epilogue fusion     │  │
│  │    Opaque ← Any: ❌ Not fusible                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Pass 2: SCHEDULE GENERATION                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Auto-TVM / AutoScheduler generates GPU schedule:       │  │
│  │    • Thread/block mapping                               │  │
│  │    • Memory hierarchy (shared memory usage)             │  │
│  │    • Loop tiling and unrolling                          │  │
│  │    • Vectorization                                      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  Pass 3: CODE GENERATION                                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Emit fused HIP/CUDA kernel code                        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### PyTorch 2.0 torch.compile

```python
import torch

# Define model
class MyModel(torch.nn.Module):
    def forward(self, x):
        x = x + 1.0              # Op1: Add
        x = torch.nn.functional.relu(x)  # Op2: ReLU
        x = x * 2.0              # Op3: Mul
        x = x.sum(dim=1)         # Op4: Reduce
        return x

# Compile with fusion
model = MyModel()
compiled_model = torch.compile(model, mode="reduce-overhead")

# Behind the scenes:
#   1. Captures computation graph (via TorchDynamo)
#   2. Analyzes fusion opportunities
#   3. Generates fused Triton kernels
#   4. Result: Single fused kernel instead of 4 separate ops
#
# Speedup: 1.5-3x depending on model and hardware
```

---

## Advanced Composition Patterns {#composition}

### Multi-Level Fusion

```
┌──────────────────────────────────────────────────────────────┐
│           HIERARCHICAL FUSION STRATEGY                        │
│                                                                │
│  Level 1: INTRA-LAYER FUSION                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Within a single layer:                                 │  │
│  │    Conv2D + BatchNorm + ReLU → FusedConvBNReLU         │  │
│  │  Speedup: 1.5-2x                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                        ↓                                       │
│  Level 2: INTER-LAYER FUSION                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Across consecutive layers:                             │  │
│  │    Layer1 + Layer2 → FusedBlock                         │  │
│  │  (e.g., ResNet block fusion)                            │  │
│  │  Speedup: 2-3x                                          │  │
│  └────────────────────────────────────────────────────────┘  │
│                        ↓                                       │
│  Level 3: PIPELINE FUSION                                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Entire model stages:                                   │  │
│  │    Encoder + Decoder → FusedEncoderDecoder             │  │
│  │  Requires advanced memory management                    │  │
│  │  Speedup: 3-5x                                          │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Performance Analysis {#performance-analysis}

### Measuring Fusion Impact

```bash
# Profile unfused code
rocprof --stats ./model_unfused

# Profile fused code
rocprof --stats ./model_fused

# Compare metrics:
#   • Kernel count: Expect 50-80% reduction
#   • Memory bandwidth: Expect 30-60% reduction
#   • Execution time: Expect 1.5-4x speedup
#   • Register pressure: May increase slightly
```

---

## Summary

```
KEY TAKEAWAYS:

✅ Fusion Types:
  • Vertical: Producer-consumer chains (highest impact)
  • Horizontal: Parallel operations (moderate impact)
  • Hybrid: Combination strategies

✅ Optimization Techniques:
  • Rule-based: Fast, predictable, limited scope
  • RL-based: Adaptive, generalizable, requires training
  • Polyhedral: Mathematically rigorous, best for loops

✅ Expected Speedups:
  • Element-wise fusion: 2-4x
  • GEMM epilogue fusion: 1.5-2x
  • Full transformer block fusion: 3-5x
  • Multi-level fusion: 5-10x

✅ Tools:
  • TVM/Apache TVM: Open-source ML compiler
  • XLA: Google's optimizing compiler
  • torch.compile: PyTorch 2.0 fusion
  • Triton: Python-based GPU programming
```

---

**Next:** See `AMD_INTRINSICS_HIPKITTENS.md` for comprehensive guide to `__builtin_amdgcn_*` intrinsics and the HipKittens framework for high-performance AMD GPU programming.
