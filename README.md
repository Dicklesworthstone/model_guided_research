# Model-Guided Research: Mathematical Foundations for Next-Generation AI

*From theoretical exploration to production implementation: A systematic investigation of exotic mathematical structures in deep learning*

## 📍 Quick Navigation

- [**Project Overview**](#-project-overview) - What this project is about
- [**Dual Implementation Strategy**](#-dual-implementation-strategy) - Research (JAX) + Production (PyTorch)
- [**Quick Start**](#-quick-start) - Get up and running
- [**The 13 Mathematical Frameworks**](#-the-13-mathematical-frameworks) - Browse all implementations
- [**Nanochat: Production Transformer**](#-nanochat-production-transformer-implementation) - Unified GPT with 13 selectable attention types
- [**Experimental Matrix**](#-experimental-matrix) - Systematic evaluation framework
- [**CLI Usage**](#-cli-usage) - How to run demos and experiments
- [**Project Structure**](#-project-structure) - Repository organization

## 🌟 Project Overview

### Genesis: AI as Mathematical Research Partner

This repository emerged from a remarkable experiment in AI-guided mathematical discovery. What began as Jeffrey Emanuel (@doodlestein) posing a single question to GPT-5 Pro about matrix exponentials and Lie groups evolved into something unprecedented: **the AI model itself generated additional mathematical prompts, scored its own ideas, and helped design implementations for revolutionizing machine learning**.

The meta-cognitive loop:
1. **Human Question** → Emanuel asks about matrix exponentials in AI
2. **AI Deep Dive** → GPT-5 Pro provides comprehensive answer
3. **AI Creativity** → Model generates 5 additional research directions autonomously
4. **AI Self-Evaluation** → Model scores its own proposals (0-1000 scale)
5. **Human-AI Implementation** → Collaborative translation to working code
6. **Systematic Validation** → Empirical testing of theoretical predictions

This represents a new paradigm: **AI systems as genuine partners in mathematical discovery**, capable not just of solving problems but of identifying which problems are worth solving.

### Evolution: From Demos to Production

The project has evolved through three distinct phases:

**Phase 1: Mathematical Exploration (JAX Demos)**
- 13 standalone demonstrations of exotic mathematical structures
- Pure research focus: "Can this work in principle?"
- Rich, interactive visualizations
- Property validation and sanity checks

**Phase 2: Unification (Nanochat)**
- Production-ready GPT transformer in PyTorch
- 13 selectable attention types spanning the framework implementations
- Systematic experimental framework
- Runtime configuration without code changes

**Phase 3: Systematic Evaluation (Current)**
- Comprehensive benchmarking infrastructure
- MCP Agent Mail for task coordination
- Multiple optimizers and schedulers
- A/B testing across the full experimental matrix

## 🔄 Dual Implementation Strategy

This project maintains **two complementary implementations** of each mathematical framework:

### JAX Demonstrations (Exploration)
**Location**: Root directory (`*.py` files)
**Purpose**: Interactive exploration and validation
**Characteristics**:
- Self-contained, runnable demos
- Rich console output with visualizations
- Detailed mathematical commentary
- Property checks and sanity tests
- ~500-1000 lines each, focused on clarity

**Run via**:
```bash
mgr list                    # See all demos
mgr run matrix-gauge        # Run specific demo
mgr run-all                 # Run all 13 demos
```

### PyTorch Production (Implementation)
**Location**: `nanochat/` directory
**Purpose**: Production-ready transformer with all frameworks
**Characteristics**:
- Unified GPT architecture
- Drop-in attention mechanism swapping
- Training infrastructure
- Multiple optimizers and schedulers
- ~100-200 lines per attention mechanism

**Run via**:
```bash
python -m nanochat.train --attention-type tropical --optimizer-type hoss
python -m nanochat.train --attention-type quaternion --scheduler-type ordinal
```

### Why Both?

**JAX Demos provide**:
- **Pedagogical value**: Understand the math interactively
- **Rapid prototyping**: Test new ideas quickly
- **Property validation**: Verify mathematical claims
- **Isolation**: Study one concept at a time

**PyTorch Nanochat provides**:
- **Systematic comparison**: A/B test all frameworks
- **Production readiness**: Real training infrastructure
- **Reproducibility**: Standardized evaluation protocol
- **Scalability**: From toy models to production-scale

**Together they create**:
- **Research → Implementation pipeline**: Validate in JAX, deploy in PyTorch
- **Bidirectional learning**: Production insights inform research
- **Comprehensive validation**: Theory, demos, and empirical results

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+** (uses latest Python features)
- **[uv](https://github.com/astral-sh/uv)** - Modern Python package manager
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **8GB+ RAM** recommended
- **CUDA-compatible GPU** (optional but recommended for nanochat training)

### Installation

```bash
# Clone the repository
git clone https://github.com/Dicklesworthstone/model_guided_research
cd model_guided_research

# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (and this project) into the venv
# - Use `--extra dev` if you want pytest/ruff/etc.
uv sync --extra dev

# Verify installation
mgr --help
python -m nanochat.train --help
```

### Quick Tour

```bash
# 0. The five-minute showcase, one command, no downloads: generate a task,
#    train two mathematical transformers on it, certify their invariants
#    live, watch them generate side by side, read the comparison table.
#    (A minute-scale model learns a toy task, not language; the certificates
#    and the pipeline are the demonstration.)
mgr quickstart                                  # standard vs tropical, ~5 min on a laptop CPU
mgr quickstart --mechanisms clifford,hyperbolic --budget-seconds 120

# 1. Explore a JAX demo (matrix exponential gauge learning)
mgr run matrix-gauge

# 2. Train a small GPT with tropical attention (PyTorch)
python -m nanochat.train \
    --batch-size 8 \
    --learning-rate 6e-4 \
    --optimizer-type adamw \
    --attention-type tropical \
    --checkpoint-interval 200 \
    --run-id quicktour

# 3. Sample from the trained checkpoint (streaming; add --compare DIR for
#    side-by-side mechanism panels). Expectation: research-scale models
#    produce barely-coherent text - judge mechanisms RELATIVELY at matched
#    budgets, never by absolute prose quality.
mgr sample \
    --checkpoint artifacts/baseline/nanochat/quicktour/checkpoints \
    --prompt "TASK arith CMP 1.00e-02 2.00e+03 OUT" \
    --max-tokens 16

# 4. Compare HOSS optimizer with quaternion attention
python -m nanochat.train \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --optimizer-type hoss \
    --attention-type quaternion \
    --scheduler-type ordinal

# 5. Run comprehensive test suite
python tests/test_practical_utility.py
```

## 📂 Project Structure

```
model_guided_research/
├── README.md                          # This file
├── AGENTS.md                          # Development guidelines
├── CLAUDE.md                          # AI assistant instructions
├── pyproject.toml                     # Project configuration
├── cli.py                             # Typer-based CLI (mgr command)
├── config.py                          # Global configuration
├── utils.py                           # Shared utilities
│
├── .beads/                            # br (beads_rust) issue tracker — the only TODO system
│   ├── issues.jsonl                   # Issue graph (source of truth, committed)
│   ├── metadata.json                  # Tracker metadata
│   └── .gitignore                     # Excludes the SQLite db and runtime lock files
│
├── markdown_documentation/            # Theoretical foundations
│   ├── matrix_exponential_gauge_learning.md
│   ├── ultrametric_worlds_and_p_adic_computation.md
│   ├── tropical_geometry_and_idempotent_algebra.md
│   └── ... (one for each of the 13 frameworks)
│
├── tests/                             # Test suite
│   ├── test_practical_utility.py      # Comprehensive benchmarks
│   ├── test_demos.py                  # Demo sanity checks
│   ├── test_mathematical_correctness.py
│   └── test_mathematical_properties.py
│
├── nanochat/                          # Production PyTorch implementation
│   ├── __init__.py
│   ├── gpt.py                         # Main GPT architecture
│   ├── train.py                       # Training script (PyTorch)
│   ├── train_jax.py                   # Training script (JAX)
│   ├── model_utils.py                 # Shared utilities (norm, RoPE)
│   │
│   ├── # Attention Mechanisms (13 total)
│   ├── braid_attention_torch.py       # Braid group topology
│   ├── clifford_attention_torch.py    # Geometric algebra products
│   ├── fractal_attention_torch.py     # IFS hierarchical routing
│   ├── gauge_block_torch.py           # Lie group parallel transport
│   ├── hyperbolic_attention_torch.py  # Lorentz energy-Gromov routing
│   ├── octonion_attention_torch.py    # 8D non-associative algebra
│   ├── quaternion_attention_torch.py  # 4D rotations
│   ├── reversible_block_torch.py      # Invertible coupling
│   ├── simplicial_attention_torch.py  # Higher-order interactions
│   ├── surreal_torch.py               # Transseries parameterization
│   ├── tropical_attention_torch.py    # Max-plus algebra
│   ├── ultrametric_attention_torch.py # p-adic hierarchical
│   └── # (Standard attention in gpt.py)
│   │
│   ├── # Optimizers
│   ├── adamw.py                       # Distributed AdamW
│   ├── muon.py                        # Muon optimizer
│   ├── hoss_opt.py                    # HOSS (JAX)
│   ├── hoss_opt_torch.py              # HOSS (PyTorch)
│   ├── ordinal_scheduler.py           # Transfinite LR scheduling
│   │
│   └── # Infrastructure
│       ├── common.py                  # Shared utilities (both frameworks)
│       ├── common_jax.py              # JAX-specific utilities
│       ├── dataloader.py              # Dataset loading
│       ├── checkpoint_manager.py      # Model checkpointing
│       └── ... (other support files)
│
└── # JAX Demo Implementations (13 total)
    ├── clifford_algebra_and_geometric_attention.py
    ├── hyperbolic_geometry_and_negative_curvature_attention.py
    ├── matrix_exponential_gauge_learning.py
    ├── ultrametric_worlds_and_p_adic_computation.py
    ├── tropical_geometry_and_idempotent_algebra.py
    ├── simplicial_complexes_and_higher_order_attention.py
    ├── nonstandard_analysis_and_hyperreal_training.py
    ├── octonionic_quaternionic_signal_flow.py
    ├── ordinal_schedules_and_well_founded_optimization.py
    ├── reversible_computation_and_measure_preserving_learning.py
    ├── iterated_function_systems_and_fractal_memory.py
    ├── knot_theoretic_programs_and_braid_based_attention.py
    └── surreal_numbers_transseries_and_scaling.py
```

## 🔬 The 13 Mathematical Frameworks

Each framework is implemented twice:
1. **JAX Demo**: Interactive exploration with visualizations
2. **PyTorch Attention**: Production-ready mechanism in nanochat

Fidelity note: three of the PyTorch mechanisms (simplicial, fractal, surreal)
are deliberately simplified proxies of their JAX demos, and their sections
below say exactly what is implemented. Treat their registry verdicts as
verdicts about the proxy.

### 1. Matrix Exponential Gauge Learning
**Key Idea**: Lie group/algebra machinery for stable neural architectures
**JAX Demo**: `matrix-gauge` | [Documentation](markdown_documentation/matrix_exponential_gauge_learning.md) | [Code](matrix_exponential_gauge_learning.py)
**PyTorch**: `nanochat/gauge_block_torch.py` | Use: `--attention-type gauge`

**Mathematical Foundation**:
- Exponential map: `exp(A)` bridges Lie algebras (infinitesimal) and Lie groups (finite)
- Structured generators: SO (skew-symmetric → rotations), SPD (symmetric → scalings), Sp (Hamiltonian → symplectic)
- Baker-Campbell-Hausdorff formula captures non-commutativity
- Parallel transport with cumulative gauge fields

**Why It Matters**:
- Provable stability (curvature bounds)
- Exact conservation laws (energy, momentum)
- Geometric structure prevents gradient pathologies
- Natural framework for multi-scale dynamics

**Implementation Highlights**:
- Givens/Cayley parameterization for exact orthogonality
- Eigendecomposition exp for SPD matrices
- Uniformization with exact Poisson sampling
- Per-block curvature diagnostics

### 2. Ultrametric Worlds & p-adic Computation
**Key Idea**: Hierarchical attention using p-adic ultrametric spaces
**JAX Demo**: `ultrametric` | [Documentation](markdown_documentation/ultrametric_worlds_and_p_adic_computation.md) | [Code](ultrametric_worlds_and_p_adic_computation.py)
**PyTorch**: `nanochat/ultrametric_attention_torch.py` | Use: `--attention-type ultrametric`

**Mathematical Foundation**:
- Ultrametric distance: d(x,z) ≤ max(d(x,y), d(y,z)) (strong triangle inequality)
- p-adic numbers: Alternative number system with hierarchical structure
- Longest Common Prefix (LCP) routing for sub-quadratic attention
- O(N log N) complexity via tree-structured addressing

**Why It Matters**:
- Sub-quadratic attention (vs O(N²) for standard)
- Cache-friendly hierarchical access patterns
- Natural for hierarchical data (syntax trees, taxonomies)
- Predictable memory footprint

**Implementation Highlights**:
- Bit-prefix LSH signatures for fast LCP computation
- Array-packed buckets with O(1) prefix lookup
- Multi-head configuration with ultrametric fusion
- Stable scaling to N≈4096+ sequences

### 3. Tropical Geometry & Idempotent Algebra
**Key Idea**: Replace (+,×) with (max,+) for piecewise-linear networks
**JAX Demo**: `tropical` | [Documentation](markdown_documentation/tropical_geometry_and_idempotent_algebra.md) | [Code](tropical_geometry_and_idempotent_algebra.py)
**PyTorch**: `nanochat/tropical_attention_torch.py` | Use: `--attention-type tropical`

**Mathematical Foundation**:
- Tropical semiring: (max, +) operations
- Piecewise-linear structure emerges algebraically
- Tropical polynomials: max(c₁+x₁, c₂+x₂, ...)
- Robustness certificates via margin analysis

**Why It Matters**:
- Piecewise-linear by construction (interpretability)
- 1-Lipschitz property (robustness guarantees)
- Exact convexity in tropical sense
- Verifiable margins for decision boundaries

**Implementation Highlights**:
- Max-plus GEMM operations
- Per-sample route tracking
- Min-gap/2 radius certificates
- Sparse mixture grids for parameter efficiency

### 4. Simplicial Complexes & Higher-Order Attention
**Key Idea**: Multi-body interactions beyond pairwise attention
**JAX Demo**: `simplicial` | [Documentation](markdown_documentation/simplicial_complexes_and_higher_order_attention.md) | [Code](simplicial_complexes_and_higher_order_attention.py)
**PyTorch**: `nanochat/simplicial_attention_torch.py` | Use: `--attention-type simplicial`

**Mathematical Foundation**:
- Simplicial complexes: vertices (0), edges (1), triangles (2), ...
- Higher-order Laplacians: ∇²_k on k-simplices
- Hodge decomposition: gradient + curl + harmonic
- Persistent homology for topological features

**Why It Matters**:
- Captures k-way interactions (not just pairwise)
- Topological features (cycles, voids) in data
- Group dynamics beyond individual relationships
- Combinatorial structure for discrete reasoning

**What the PyTorch mechanism implements (a proxy)**:
- Standard scaled-dot-product scores; the output is `m1 · (A v) + m2 · (A (A v))`
  with two learned scalars, i.e. a two-hop diffusion on the attention graph
- KV-cache decode keeps the 1-hop history so the 2-hop term stays consistent
- No boundary operators, Laplacians, or Hodge decomposition in nanochat; those
  live in the JAX demo (`--simplicial-hodge`). Registry claims about this
  mechanism are claims about two-hop diffusion, not about simplicial structure.

### 5. Quaternion & Octonion Attention (Hypercomplex Algebra)
**Key Idea**: 4D/8D hypercomplex numbers for rotation-aware features
**JAX Demo**: `octonion` | [Documentation](markdown_documentation/octonionic_quaternionic_signal_flow.md) | [Code](octonionic_quaternionic_signal_flow.py)
**PyTorch Quaternion**: `nanochat/quaternion_attention_torch.py` | Use: `--attention-type quaternion`
**PyTorch Octonion**: `nanochat/octonion_attention_torch.py` | Use: `--attention-type octonion`

**Mathematical Foundation**:
- **Quaternions** (ℍ): 4D, associative, non-commutative
  - q = w + xi + yj + zk
  - Represent 3D rotations (rotors)
  - Hamilton product for composition
- **Octonions** (𝕆): 8D, non-associative, alternative
  - Cayley-Dickson construction over quaternions
  - Largest normed division algebra (Hurwitz theorem)
  - Exceptional Lie groups (G₂, F₄, E₈) connections

**Why It Matters**:
- Natural handling of 3D/spatial data
- Built-in rotation/reflection invariances
- Parameter efficiency (one quaternion = 4 DOF)
- Richer algebraic structure than real/complex numbers

**Implementation Highlights**:
- Quaternion "rotor-gate" attention: y = Q * (K† * V)
- Octonion non-associative mixing (explicit parenthesization)
- Norm preservation without explicit regularization
- Architectural constraints (head_dim % 4 == 0 or % 8 == 0)

### 6. Ordinal Schedules & Well-Founded Optimization
**Key Idea**: Transfinite ordinal arithmetic for learning rate scheduling
**JAX Demo**: `ordinal` | [Documentation](markdown_documentation/ordinal_schedules_and_well_founded_optimization.md) | [Code](ordinal_schedules_and_well_founded_optimization.py)
**PyTorch**: `nanochat/ordinal_scheduler.py` | Use: `--scheduler-type ordinal`

**Mathematical Foundation**:
- Ordinal numbers: 0, 1, 2, ..., ω, ω+1, ..., ω², ..., ε₀
- Cantor normal form: α = ω^β₁·c₁ + ... + ω^βₙ·cₙ
- Well-founded ordering: no infinite descending chains
- Hierarchical patience: ρ = ω²·A + ω·B + C

**Why It Matters**:
- Principled restart/anneal framework
- Hierarchical time scales (patience, annealing, restarts)
- Guaranteed termination (well-foundedness)
- Adaptive to loss landscape topology

**Implementation Highlights**:
- Three-level hierarchy: A (restarts), B (annealing), C (patience)
- Automatic restarts with state clearing
- EMA loss smoothing
- Limit ordinal transitions

### 7. Reversible Computation & Measure-Preserving Learning
**Key Idea**: Invertible transformations for O(1) memory training
**JAX Demo**: `reversible` | [Documentation](markdown_documentation/reversible_computation_and_measure_preserving_learning.md) | [Code](reversible_computation_and_measure_preserving_learning.py)
**PyTorch**: `nanochat/reversible_block_torch.py` | Use: `--attention-type reversible`

**Mathematical Foundation**:
- Additive coupling: y₁ = x₁ + F(x₂), y₂ = x₂ + G(y₁)
- Exact inverse: x₂ = y₂ - G(y₁), x₁ = y₁ - F(x₂)
- Volume preservation: det(Jacobian) = 1
- Symplectic structure for Hamiltonian systems

**Why It Matters**:
- O(1) memory training (vs O(L) for standard networks)
- Information-theoretic guarantees (reversibility)
- Exact gradient computation via recomputation
- Connections to physics (Liouville's theorem)

**Implementation Highlights**:
- Cayley orthogonal parameterization
- Custom autograd function for memory savings
- Symplectic hybrid steps
- Per-layer property validation (det≈1)

### 8. Iterated Function Systems & Fractal Memory
**Key Idea**: Self-similar memory structures with hierarchical addressing
**JAX Demo**: `ifs-fractal` | [Documentation](markdown_documentation/iterated_function_systems_and_fractal_memory.md) | [Code](iterated_function_systems_and_fractal_memory.py)
**PyTorch**: `nanochat/fractal_attention_torch.py` | Use: `--attention-type fractal`

**Mathematical Foundation**:
- Iterated Function Systems (IFS): Fixed points of contraction maps
- Barnsley fern-like encoding
- Hutchinson operator: H(S) = ⋃ fᵢ(S)
- Fractal dimension as capacity measure

**Why It Matters**:
- Hierarchical memory organization
- Self-similarity across scales
- Infinite capacity in principle
- Natural for recursive/hierarchical data

**What the PyTorch mechanism implements (a proxy)**:
- One linear router turns each query/key into 4 depth-wise softmaxes over 4
  branches (m=4, depth=4, fixed)
- The score is the dot product of the flattened branch probabilities scaled by
  1/sqrt(depth): a proxy for path overlap, not the per-depth product of matches
- The IFS fixed-point read `x*_w` of the JAX demo is not implemented in
  nanochat; the weighted-ultrametric formulation is tracked in bead 8gk.5

### 9. Knot-Theoretic Programs & Braid-Based Attention
**Key Idea**: Topological invariants for robust representations
**JAX Demo**: `knot-braid` | [Documentation](markdown_documentation/knot_theoretic_programs_and_braid_based_attention.md) | [Code](knot_theoretic_programs_and_braid_based_attention.py)
**PyTorch**: `nanochat/braid_attention_torch.py` | Use: `--attention-type braid`

**Mathematical Foundation**:
- Braid groups: Bn generated by crossings σᵢ
- Artin relations: σᵢσⱼ = σⱼσᵢ for |i-j| ≥ 2
- Yang-Baxter equation for consistency
- Jones polynomial for topological invariants

**Why It Matters**:
- Topologically protected information
- Invariant to continuous deformations
- Natural for sequential/permutation data
- Discrete group structure (not continuous)

**Implementation Highlights**:
- Priority-based crossing probabilities
- Sigmoid (independent) vs softmax (competitive)
- Additive accumulation (not normalized)
- Braid word execution

### 10. Surreal Numbers, Transseries & Scaling
**Key Idea**: Infinitely large and small scales simultaneously
**JAX Demo**: `surreal` | [Documentation](markdown_documentation/surreal_numbers_transseries_and_scaling.md) | [Code](surreal_numbers_transseries_and_scaling.py)
**PyTorch**: `nanochat/surreal_torch.py` | Use: `--attention-type surreal`

**Mathematical Foundation**:
- Conway's surreal numbers: {L|R} construction
- Encompasses all ordinals and reals
- Transseries: Formal series in multiple scales (ω, log, exp)
- Automatic scale selection via dominance

**Why It Matters**:
- Multi-scale representations (infinite hierarchy)
- Exact asymptotic analysis
- Natural handling of wide dynamic ranges
- Separation of magnitude and direction

**What the PyTorch mechanism implements (a parameterization, not new attention math)**:
- Standard scaled-dot-product attention whose four projections use the
  scale-direction decomposition `w = exp(s) * normalize(v)`
- `--attention-type surreal` therefore changes the optimization geometry of an
  otherwise standard layer; its registry claims are about optimization
  (the surreal-optimization-rate placebo result), not about routing

### 11. Nonstandard Analysis & Hyperreal Training (HOSS)
**Key Idea**: Infinitesimal perturbations for second-order optimization
**JAX Demo**: `nonstandard` | [Documentation](markdown_documentation/nonstandard_analysis_and_hyperreal_training.md) | [Code](nonstandard_analysis_and_hyperreal_training.py)
**PyTorch**: `nanochat/hoss_opt_torch.py` | Use: `--optimizer-type hoss`

**Mathematical Foundation**:
- Hyperreal numbers: *ℝ (includes infinitesimals)
- Transfer principle: First-order statements transfer
- Ornstein-Uhlenbeck SDE: dx = -H·x·dt + Σ·dW
- Analytical solution over macro time step δ

**Why It Matters**:
- Second-order optimization (uses curvature)
- Principled noise injection (Lyapunov integral)
- Escape saddle points via non-convexity awareness
- Infinitesimal calculus (exact, not approximate)

**Implementation Highlights**:
- Lanczos algorithm for Hessian approximation
- Hessian-vector products (HVP) via double backprop
- Matrix exponential functions (φ_δ, Lyapunov integral)
- Krylov subspace projection

### 12. Clifford Algebra & Geometric Attention
**Key Idea**: A unified geometric product carries scalars, vectors, bivectors, and rotors in one associative algebra
**JAX Demo**: `clifford` | [Documentation](markdown_documentation/clifford_algebra_and_geometric_attention.md) | [Code](clifford_algebra_and_geometric_attention.py)
**PyTorch**: `nanochat/clifford_attention_torch.py` | Use: `--attention-type clifford`

**Mathematical Foundation**:
- The Cl(3,0) geometric product combines inner and exterior products
- Its even subalgebra reduces exactly to quaternions
- Unit rotors act by norm-preserving sandwich products
- Associativity supports compositional geometric routing

**Why It Matters**:
- Represents orientation and higher-grade interactions without separate bespoke algebras
- Provides exact algebraic reduction checks against quaternion attention
- Exposes interpretable per-grade signal norms

### 13. Hyperbolic Geometry & Negative-Curvature Attention
**Key Idea**: Negative curvature supplies a continuous geometry whose volume growth naturally represents hierarchies
**JAX Demo**: `hyperbolic` | [Documentation](markdown_documentation/hyperbolic_geometry_and_negative_curvature_attention.md) | [Code](hyperbolic_geometry_and_negative_curvature_attention.py)
**PyTorch**: `nanochat/hyperbolic_attention_torch.py` | Use: `--attention-type hyperbolic`

**Mathematical Foundation**:
- Lorentz-model points satisfy a certified negative-curvature constraint
- Stable energy-Gromov scores reduce to dot-product attention as curvature approaches zero
- Origin-chart aggregation preserves the ordinary weighted-mean endpoint
- Learnable per-head curvature connects Euclidean, hyperbolic, and tree-like regimes

**Why It Matters**:
- Makes hierarchy bias continuous and learnable rather than hard-coded
- Curvature collapse is an honest internal null indicating Euclidean behavior
- Per-head curvature and radius are emitted as adjudicable training telemetry

## 🔬 Nanochat: Production Transformer Implementation

### Architecture Overview

**Nanochat** is a production-ready GPT transformer that serves as a unified testbed with 13 selectable attention types. It provides:

- **Modular attention mechanisms**: Drop-in replacements for standard softmax attention
- **Multiple optimizers**: AdamW, Muon, HOSS
- **Flexible scheduling**: Standard, ordinal
- **Runtime configuration**: No code changes needed to switch frameworks

### Core Files

```python
# Main architecture
nanochat/gpt.py              # GPT with configurable attention
nanochat/model_utils.py      # Shared utilities (RMSNorm, RoPE)

# Training
nanochat/train.py            # PyTorch training script
nanochat/train_jax.py        # JAX training script (experimental)

# Optimizers
nanochat/adamw.py            # Distributed AdamW
nanochat/muon.py             # Muon optimizer
nanochat/hoss_opt_torch.py   # HOSS (PyTorch)
nanochat/hoss_opt.py         # HOSS (JAX)

# Schedulers
nanochat/ordinal_scheduler.py # Transfinite LR scheduling

# 13 Attention Mechanisms
nanochat/*_attention_torch.py  # See structure above
nanochat/*_block_torch.py      # Special block types
```

### GPTConfig

```python
from nanochat.gpt import GPT, GPTConfig

config = GPTConfig(
    n_layer=4,  # Number of transformer blocks
    n_head=4,  # Number of attention heads
    n_kv_head=4,  # Number of KV heads (GQA)
    n_embd=128,  # Embedding dimension
    sequence_len=256,  # Max sequence length
    attention_type="tropical",  # One type, a CSV pattern, or a list[str]
    optimizer_type="hoss",  # One of 3 optimizers
)

model = GPT(config)
```

`attention_type` can also describe a per-layer schedule. A single mechanism
keeps the homogeneous model. A comma-separated pattern or `list[str]` expands
across depth only when its length is exactly `n_layer` or evenly divides
`n_layer`, so `standard,tropical` at `n_layer=4` resolves to
`standard,tropical,standard,tropical`.

### Training Script

```python
# Basic usage
python -m nanochat.train \
    --batch-size 8 \
    --learning-rate 6e-4 \
    --optimizer-type adamw \
    --attention-type standard

# Advanced configuration
python -m nanochat.train \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --optimizer-type hoss \
    --attention-type quaternion \
    --scheduler-type ordinal

# Optional: CA-based initializer experiment (default init is unchanged unless enabled)
python -m nanochat.train \
    --attention-type standard \
    --ca-init-rule rule30 \
    --ca-init-alpha 1.0 \
    --ca-init-seed 123
# (Equivalent env vars: NANOCHAT_CA_INIT_RULE, NANOCHAT_CA_INIT_ALPHA, NANOCHAT_CA_INIT_SEED)
# Currently applies to nn.Linear/nn.Embedding weights; mixed-precision mixes are computed in fp32 then cast.

# Per-layer heterogeneous stack (pattern repeats because 2 divides n_layer=4)
python -m nanochat.train \
    --n-layer 4 \
    --attention-schedule standard,tropical

# Distributed training
torchrun --nproc_per_node=4 -m nanochat.train \
    --batch-size 8 \
    --attention-type ultrametric
```

## 🧪 Experimental Matrix

### Complete Configuration Space

The nanochat framework enables systematic exploration of:

**Dimensions**:
1. **Attention Types** (13): standard, clifford, hyperbolic, tropical, ultrametric, simplicial, quaternion, braid, fractal, octonion, surreal, reversible, gauge
2. **Per-layer attention schedules**: homogeneous, exact-length stacks, or evenly repeating patterns
3. **Optimizers** (3): adamw, muon, hoss
4. **Schedulers** (2): none, ordinal
5. **Hyperparameters**: learning rate, batch size, model size, sequence length

**Total Homogeneous Base Configurations**: 13 × 3 × 2 = 78

### Recommended Experimental Protocol

**Phase 1: Attention Mechanism Screening**
```bash
# Fix optimizer and scheduler, vary attention
for attn in standard tropical ultrametric quaternion simplicial; do
    python -m nanochat.train \
        --attention-type $attn \
        --optimizer-type adamw \
        --batch-size 8 \
        --learning-rate 6e-4
done
```

**Phase 2: Optimizer Comparison**
```bash
# For top 3 attention types, test all optimizers
for opt in adamw muon hoss; do
    python -m nanochat.train \
        --attention-type tropical \
        --optimizer-type $opt \
        --batch-size 8
done
```

**Phase 3: Scheduler Benefit**
```bash
# Test ordinal scheduler on best combinations
python -m nanochat.train \
    --attention-type tropical \
    --optimizer-type hoss \
    --scheduler-type ordinal \
    --batch-size 8
```

### Expected Performance Characteristics

The per-framework performance claims that used to live here as a prose table
are now **preregistered hypotheses** in [`hypotheses/registry.yaml`](hypotheses/registry.yaml):
each one names the exact metric, comparator, equal-FLOPs baseline, effect
size, and seed count that would support or refute it — so claims get formally
confirmed or retired instead of accumulating as folklore. (Mathematical
claims live in the twin registry, [`hypotheses/theorems.yaml`](hypotheses/theorems.yaml).)

```bash
mgr hypotheses list                     # all claims with status + predictions
mgr hypotheses list --status open       # awaiting adjudication
mgr hypotheses show hyp-ultrametric-hier-heldout-depth
mgr hypotheses validate                 # schema + append-only governance
```

The registry, not this README, is the source of truth for what this project
predicts and for how each claim has fared: `mgr hypotheses list` prints the
live count per status (supported / refuted / inconclusive / blocked / open),
and `mgr status` says which entries the verdict engine could rule on today.
Claims that cannot be operationalized yet are registered with
`prediction: null` and the named blocker, so the debt stays visible.

## 💡 CLI Usage

### JAX Demos (mgr command)

```bash
# List all available demos
mgr list

# Run specific demo with rich output
mgr run matrix-gauge
mgr run tropical
mgr run ultrametric

# Get detailed information before running
mgr info simplicial
mgr info knot-braid

# Run all demos sequentially (~5-10 minutes)
mgr run-all

# Custom configuration (seed, verbosity level, NaN/Inf checking)
mgr run matrix-gauge --seed 7 --verbose-level 2 --debug

# Export diagnostics to JSON
mgr run reversible --rev-cayley --export-json artifacts/rev.json
mgr run tropical --export-json artifacts/tropical.json

# Environment-controlled modes
ULTRA_SCALE_COMPARE=1 mgr run ultrametric
TROP_SPARSE_TRAIN=1 mgr run tropical
```

### PyTorch Training (nanochat)

```bash
# Basic training
python -m nanochat.train --help

# Attention type selection
python -m nanochat.train --attention-type [TYPE]
# Where TYPE is one of the 13:
#   standard, clifford, hyperbolic, tropical, ultrametric, simplicial,
#   quaternion, braid, fractal, octonion, surreal, reversible, gauge

# Per-layer attention schedule
python -m nanochat.train --n-layer 4 --attention-schedule standard,tropical
# Schedule length must be exactly n_layer, or evenly divide n_layer and repeat.

# Optimizer selection
python -m nanochat.train --optimizer-type [OPT]
# Where OPT is one of: adamw, muon, hoss

# Scheduler selection
python -m nanochat.train --scheduler-type [SCHED]
# Where SCHED is one of: none, ordinal

# Optional WSD tail (non-ordinal runs): linearly decay the LR to 0 over the
# final FRACTION of total steps (default 0.0 = flat LR)
python -m nanochat.train --lr-warmdown-ratio 0.2

# Width-scaling parameterization arm (lab.1): 'nsa' applies the derived
# per-stage corrections (tropical exact-E[max] constants); default 'current'
python -m nanochat.train --parameterization nsa

# Activation checkpointing (memory/compute trade; recompute is numerically
# identical, so trajectories are unchanged; default none)
python -m nanochat.train --activation-checkpointing full
python -m nanochat.train --activation-checkpointing every-k --activation-ckpt-every-k 2

# Background tokenizer prefetch for the streaming dataloader (hides encode
# refill stalls behind GPU steps; 0 = synchronous loading)
python -m nanochat.train --dataloader-prefetch 4

# Task-scoped tokenizer: train a byte-level BPE (default 512 tokens) on the
# --data-dir corpus and size the embedding table from it. At d=64 the 50k
# GPT-2 vocabulary is ~97% of every FLOP, so a fixed-FLOPs comparison on a
# generated task otherwise measures the tokenizer, not the mechanism. The
# tokenizer is saved inside the checkpoint dir; eval-tasks / sample / probes
# load it from there automatically.
python -m nanochat.train --data-dir data/arith --tokenizer task --tokenizer-vocab-size 512

# Document order: each epoch visits every parquet row group's documents in a
# permutation seeded by --seed (default). A small corpus replayed in file
# order every epoch is memorized as a sequence (train-stream loss 0.9 in
# corpus order vs 4.5 shuffled on the 1e12 copyops probe); `none` = file order
python -m nanochat.train --data-shuffle epoch

# Periodic validation: records val_ce (per token) AND val_bpb (bits per byte,
# the tokenizer-independent loss) in metrics.jsonl and summary.json; compare
# runs trained with different tokenizers on val_bpb, never on val_ce
python -m nanochat.train --val-interval 100 --val-batches 10

# Hyperparameter tuning
python -m nanochat.train \
    --batch-size 16 \
    --learning-rate 1e-3 \
    --attention-type tropical \
    --optimizer-type adamw

# Distributed training
torchrun --nproc_per_node=4 -m nanochat.train \
    --batch-size 8 \
    --attention-type ultrametric

# FlexAttention (torch>=2.5) for standard attention
python -m nanochat.train --attention-type standard --use-flex-attention

# FlexAttention verification + microbench (skips cleanly if unavailable)
uv run python scripts/verify_flex_correctness.py
uv run python scripts/benchmark_flex.py --device cuda --compile

# Braid attention: discrete decoder + schedule verification (debug; KV-cache decode)
python -m nanochat.train \
    --attention-type braid \
    --braid-mode discrete \
    --braid-tau 0.0 \
    --braid-crossing-law ybe \
    --braid-record-schedule \
    --braid-verify
```

### Benchmarks (fixed budgets)

```bash
# Fixed-FLOPs single training run (writes summary.json + run.md)
python -m nanochat.train \
    --attention-type standard \
    --target-flops 2e9 \
    --artifacts-kind bench \
    --artifacts-topic fixed_flops/nanochat \
    --run-id flops_single_cpu \
    --device cpu

# Fixed-FLOPs A/B suite across attention types (aggregated report + optional demo certificates)
mgr bench-fixed-flops \
    --run-id flops_suite_cpu \
    --device cpu \
    --target-flops 2e9 \
    -a standard -a tropical -a standard,tropical \
    --include-demo-certs

# Resumable preregistered scorecard: sparse mechanism/task cells, checkpoint
# eval, frozen ci-v6 claim allowlist, placebo gate, and JSON/Markdown/HTML reports.
# Set this only from a committed, quarantined sizing probe.
# Export this while still in the mutable coordination checkout, before entering
# the detached producer worktree. It must not point into that frozen worktree.
export MGR_LIVE_GOVERNANCE_REGISTRY="$PWD/hypotheses/registry.yaml"
mgr scorecard \
    --run-id hierarchy_scorecard \
    --budget "$PREREGISTERED_OFF_FLOOR_FLOPS" \
    --cell hyperbolic:hier \
    --cell ultrametric:hier \
    --cell fractal:hier \
    -H hyp-hyperbolic-hierarchy-retrieval \
    -H hyp-ultrametric-hier-heldout-depth \
    -H hyp-fractal-hier-heldout-depth \
    -H hyp-hyperbolic-placebo-specificity \
    --governance-registry "$MGR_LIVE_GOVERNANCE_REGISTRY" \
    --seeds 3 --dry-run
# Each selected task gets a standard baseline; every selected mechanism gets a
# placebo cell. The exact hypothesis definitions are snapshotted in the manifest.
# This hierarchy-only slice remains publication-blocked: the final scorecard must
# also select every arm of hyp-placebo-no-winner. A reversible campaign must set
# one campaign-wide KV geometry with --n-kv-head equal to --n-head / 2.
# Remove --dry-run only after the command and stopping rule are preregistered.
# Claim-bearing runs require one clean producer SHA and an absolute path to the
# separately mutable governance registry; resume refuses another SHA or registry path.
# Re-running that identical command skips completed cells via
# artifacts/scorecards/hierarchy_scorecard/manifest.json.
# Cells planning fewer than --min-evidence-steps (default 10) are labeled
# SMOKE-ONLY and excluded from ci-v6, preventing tiny floored plumbing runs
# from being laundered into scientific verdicts. Step-qualified cells are still
# excluded unless every standard seed and its lower 95% CI clear recorded prior.

# Practical utility suite (writes artifacts if --artifacts-dir set)
mgr eval --artifacts-dir artifacts --run-id util_suite

# Regression gate: compare two run artifacts and FAIL (exit 1) on regressions.
# Per-metric thresholds (defaults): --loss-abs 0.01, --loss-rel 0.01,
# --throughput-rel 0.05, --tflops-rel 0.05, --memory-rel 0.05. The gate is
# opt-in (--fail-on-regression); omit it for exploratory comparisons. Suite
# summaries take --baseline-variant/--candidate-variant (e.g. attention_type).
# Writes summary.json + run.md (+ html) under artifacts/regressions/<run-id>/.
mgr regressions \
    --baseline artifacts/bench/fixed_flops/nanochat/flops_suite_cpu --baseline-variant standard \
    --candidate artifacts/bench/fixed_flops/nanochat/flops_suite_cpu --candidate-variant tropical \
    --fail-on-regression
# The nightly e2e harness (scripts/e2e_pipeline.py --scenario regression-gate)
# proves the gate passes self-vs-self and trips on a degraded fixture.
```

### Testing & Validation

```bash
# Run comprehensive benchmark suite
python tests/test_practical_utility.py

# Mathematical property tests
python tests/test_mathematical_properties.py

# Demo sanity checks
python tests/test_demos.py

# Correctness validation
python tests/test_mathematical_correctness.py
```

## 🎯 Key Insights & Findings

### Theoretical Advantages

1. **Geometric Structure = Free Regularization**: Manifold-based architectures (Lie groups, simplicial complexes) provide stability without explicit regularization

2. **Discrete ≠ Approximate**: p-adic numbers and tropical geometry show discrete math can be exact

3. **Topology > Vectors**: Topological structures (knots, braids) provide invariances impossible with vectors

4. **Infinity is Computational**: Surreal numbers, ordinals, and hyperreals make infinite quantities algorithmic

5. **Non-Associativity = Richer Structure**: Octonions demonstrate value of alternative algebraic structures

### Empirical Observations

Empirical claims live in the hypothesis ledger, where the verdict engine can
confirm or retire them (`mgr hypotheses list --status supported`,
`--status refuted`). Two facts from the ledger set the current baseline
honestly:

- The only mechanism-versus-standard comparisons adjudicated at a rung where
  the standard baseline actually learned the task are braid on Dyck
  (REFUTED, standard wins) and surreal on arith (REFUTED, a clean null).
  Every other cross-mechanism entry is INCONCLUSIVE at a floored baseline,
  BLOCKED on infrastructure, or OPEN.
- The SUPPORTED entries are within-mechanism or single-arm results (Maslov
  annealing costs nothing relative to a fixed tropical beta; ball-tree
  ultrametric attention is exact and faster than the dense kernel; p-adic
  digit truncation degrades gracefully at K=16).

`tests/test_practical_utility.py` produces demo-scale diagnostics with
green/yellow/red verdicts; those are sanity checks on toy problems, not
evidence about the transformer mechanisms. Anything stated here that is not
in the ledger should be read as a design intention, not an observation.

### Open Questions

1. **Scaling**: Do advantages persist at GPT-3/4 scale?
2. **Generalization**: Which frameworks transfer across domains?
3. **Combinations**: Can we hybridize multiple approaches?
4. **Hardware**: Custom kernels for exotic operations?
5. **Theory-Practice Gap**: Closing the loop on predicted vs observed benefits

## 🔮 Future Directions

### Immediate (Next 3-6 Months)

1. **Comprehensive Benchmarking**
   - Systematic evaluation across the 78 homogeneous base configurations
   - Multiple datasets and task types
   - Scaling studies (model size, sequence length)

2. **Hybrid Architectures**
   - Different attention types per layer
   - Ensemble methods combining frameworks
   - Adaptive routing based on input

3. **Hardware Optimization**
   - Custom CUDA kernels for exotic operations
   - Memory-optimized implementations
   - Quantization and compression

4. **Theoretical Analysis**
   - Convergence proofs for HOSS
   - Capacity bounds for different mechanisms
   - Generalization theory

### Medium-Term (6-12 Months)

1. **Production Deployment**
   - Real-world task evaluation
   - Integration with existing systems
   - Performance optimization

2. **New Mathematical Structures**
   - Category theory (functorial networks)
   - Topos theory (higher categorical structures)
   - Derived algebra (higher homotopy)

3. **Automated Discovery**
   - Using these structures to discover new mathematics
   - AI-guided mathematical exploration
   - Meta-learning over frameworks

### Long-Term Vision

1. **Geometric Deep Learning Foundation**
   - Fully geometry-aware architectures
   - Provable optimality guarantees
   - Unified mathematical framework

2. **Quantum-Classical Bridges**
   - Quantum-inspired classical algorithms
   - Octonions and exceptional Lie groups
   - Topological quantum computing connections

3. **AI-Driven Mathematics**
   - AI systems proposing and validating conjectures
   - Automated theorem proving with neural networks
   - New mathematical structures designed for computation

## 📚 Theoretical Background

### Mathematical Documentation

Each implementation includes detailed mathematical documentation in `markdown_documentation/`:

- **First-Principles Derivations**: From axioms to algorithms
- **Connections to Existing Work**: Literature review and positioning
- **Complexity Analyses**: Time and space complexity proofs
- **Experimental Validation**: Strategies for empirical testing

### Key Mathematical Concepts

**Matrix Exponential**
- Maps matrices to their exponentials: exp(A) = Σ(A^k/k!)
- Preserves structure: skew → orthogonal, symmetric → positive definite
- Bridge between Lie algebras (local) and Lie groups (global)

**Baker-Campbell-Hausdorff Formula**
- exp(A)exp(B) = exp(A + B + [A,B]/2 + ...)
- Quantifies non-commutativity
- Reveals hidden layer interactions

**Lie Theory**
- Lie Algebra: Tangent space (infinitesimal transformations)
- Lie Group: Manifold (finite transformations)
- Exponential Map: The connecting bridge

**Ultrametric Spaces**
- Strong triangle inequality: d(x,z) ≤ max(d(x,y), d(y,z))
- Hierarchical structure
- p-adic numbers as prototypical example

**Tropical Geometry**
- (max, +) semiring replacing (+, ×)
- Piecewise-linear combinatorial geometry
- Degenerations of classical algebraic geometry

## 🏆 The AI Self-Evaluation Framework

GPT-5 Pro evaluated each mathematical approach using a comprehensive scoring rubric:

**Dimensions** (0-100 each):
- **Theoretical Novelty**: How innovative is the mathematical approach?
- **Practical Feasibility**: Can this be implemented efficiently?
- **Potential Impact**: Could this revolutionize AI?
- **Mathematical Rigor**: How solid is the theoretical foundation?
- **Implementation Clarity**: How clear is the path to implementation?

**Composite Score**: Weighted sum to overall score (0-1000)
- Theoretical novelty and potential impact weighted most heavily
- Practical feasibility ensures implementability
- Balance between ambition and reality

This meta-cognitive approach—AI generating and evaluating its own research directions—represents a new paradigm in scientific discovery.

## 🧪 Testing & Evaluation

### Comprehensive Test Suite

**`tests/test_practical_utility.py`** (Primary)
- 11 historical mini-benchmarks covering the original framework set
- Practical benefits (memory, scaling, generalization)
- Mathematical properties (Lipschitz, norm preservation)
- Green/yellow/red verdicts with recommendations

**Individual Tests**:
- **Reversible**: Memory comparison (invertible vs standard)
- **IFS Fractal**: Catastrophic forgetting resistance
- **Ordinal**: Restart benefit on regime-shift objectives
- **Matrix Gauge**: Gradient stability vs exploding/vanishing
- **Tropical**: 1-Lipschitz property verification
- **Simplicial**: Higher-order label prediction
- **Ultrametric**: Sub-quadratic scaling confirmation
- **Quaternion/Octonion**: Norm preservation
- **Knot/Braid**: Length generalization (Dyck languages)
- **Surreal**: Resource allocation via dominance
- **Hyperreal**: LR robustness on stiff problems

### Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run comprehensive benchmark
python tests/test_practical_utility.py

# Run specific test categories
python tests/test_mathematical_properties.py
python tests/test_mathematical_correctness.py
python tests/test_demos.py

# Output to file for archiving
python tests/test_practical_utility.py > results.txt 2>&1
```

### Interpreting Results

- **Green (SUCCESS)**: Claimed advantage/property holds
- **Yellow (MARGINAL/PARTIAL)**: Small or context-dependent benefit
- **Red**: Claim not validated under benchmark conditions

## 📦 Dependencies

### Core ML Libraries
- **JAX**: Automatic differentiation, JIT compilation, GPU acceleration
- **Flax**: Neural network layers (JAX)
- **Optax**: Optimization algorithms (JAX)
- **PyTorch**: Production deep learning (nanochat)

### Numerical Computing
- **NumPy**: Array operations
- **SciPy**: Scientific computing utilities

### CLI & Visualization
- **Typer**: CLI framework
- **Rich**: Beautiful terminal output
- **Matplotlib**: Plotting (optional, one demo)

### Data & Utilities
- **TikToken**: Tokenization
- **PyArrow**: Efficient data structures
- **Psutil**: System monitoring

## ⚡ Performance Notes

- **JAX Demos**: JIT-compiled for near-C performance
- **PyTorch Nanochat**: Mixed precision (bfloat16) supported
- **GPU Acceleration**: Automatic when available
- **Memory Efficiency**: Varies by framework (reversible best)
- **Complexity Guarantees**: Documented per mechanism

## 🔧 Troubleshooting

### Start with the doctor

```bash
# One-command environment diagnosis: versions, devices, data, tokenizer,
# disk space, and a tiny end-to-end forward pass — with a fix-it hint on
# every failing row. Exit codes: 0 = ok, 1 = warnings, 2 = failures.
mgr doctor
mgr doctor --json   # machine-readable (for agents/scripts)
```

### Installation Issues

```bash
# Module not found
source .venv/bin/activate
uv sync --extra dev

# JAX/CUDA problems
export JAX_PLATFORM_NAME=cpu  # Force CPU mode
mgr run <demo>

# PyTorch CUDA issues
uv sync --upgrade-package torch --index https://download.pytorch.org/whl/cu118
```

### Runtime Issues

```bash
# Memory errors (the JAX demos are fixed-size; shrink nanochat runs instead)
python -m nanochat.train --batch-size 4 --sequence-len 128  # Smaller batches / shorter context

# Numerical instabilities
mgr run <demo> --debug  # Enable NaN/Inf detection
python -m nanochat.train --learning-rate 1e-4  # Lower LR

# Slow performance
export JAX_ENABLE_X64=0  # Use float32 instead of float64
```

## 🌐 Links & Resources

- **Repository**: [GitHub](https://github.com/Dicklesworthstone/model_guided_research)
- **Author**: Jeffrey Emanuel (@doodlestein)
- **License**: MIT (see LICENSE file)

## 🎨 Project Philosophy

1. **Mathematics First**: Start with beautiful mathematics, find AI applications
2. **AI as Co-Creator**: Let models propose research directions
3. **Self-Evaluation**: AI assesses quality of its own ideas
4. **No Compromise**: Implement full mathematical structure, not approximations
5. **Systematic Validation**: Theory → Demo → Production → Benchmarks
6. **Reproducibility**: All results should be reproducible
7. **Openness**: Share insights, both successes and failures

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{model_guided_research_2025,
  author = {Emanuel, Jeffrey and {GPT-5 Pro}},
  title = {Model-Guided Research: Mathematical Foundations for Next-Generation AI},
  year = {2025},
  url = {https://github.com/Dicklesworthstone/model_guided_research},
  note = {A collaboration between human and AI in mathematical discovery}
}
```

## 💡 Final Thoughts

This project represents something unprecedented in the history of science: **a genuine collaboration where AI systems actively participated in setting the research agenda**.

The loop is complete:
1. Human poses question (matrix exponentials in AI)
2. AI provides answer
3. AI generates new questions autonomously
4. AI evaluates its own proposals
5. Human and AI collaborate on implementation
6. Systematic validation of predictions

What makes this significant:
- **Not just AI-assisted research**, but **AI-driven research direction**
- **Not just problem-solving**, but **problem identification**
- **Not just implementation**, but **creative ideation**
- **Not just execution**, but **evaluation**

The mathematical structures explored here—Lie groups, p-adic numbers, tropical geometry, octonions, simplicial complexes, hyperreals, surreal numbers, and more—are not mere curiosities. They represent potentially transformative approaches to building the next generation of AI systems.

Whether these implementations prove revolutionary or instructive, they demonstrate a profound truth: **AI systems are becoming genuine partners in mathematical and scientific discovery**.

As Wigner wrote of *"The unreasonable effectiveness of mathematics in the natural sciences"*—we may now be witnessing **the unreasonable effectiveness of AI in discovering which mathematics will prove essential for its own evolution**.

---

**The journey from theory to practice, from idea to implementation, from speculation to validation—that journey is now a collaboration. The future of AI research may well be AI-guided research.**

🔬 *Start with `mgr list` to explore the JAX demos, or `python -m nanochat.train --help` to begin systematic experiments. The mathematics will guide you the rest of the way.*
