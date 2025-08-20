# Model-Guided Research: Mathematical Foundations for Next-Generation AI

*Exploring the deep connections between abstract mathematics and machine learning through practical implementations*

## 📍 Quick Navigation

- [**Quick Start**](#-quick-start) - Get up and running
- [**The Implementations**](#-the-implementations) - Browse all 11 mathematical demos
- [**Key Concepts**](#-key-mathematical-concepts) - Core mathematical ideas
- [**CLI Usage**](#running-demos) - How to run and explore demos
- [**Project Structure**](#-project-structure) - Repository organization

## 🌟 Project Genesis

This repository contains the practical implementations that emerged from a remarkable experiment in AI-guided mathematical discovery. What began as Jeffrey Emanuel (@doodlestein) posing a single question to GPT-5 Pro about matrix exponentials and Lie groups evolved into something far more ambitious: **the AI model itself generated additional mathematical prompts and scored its own ideas for revolutionizing machine learning**.

### The Original Experiment

Emanuel's initial thread posed a provocative question: Could the mathematical machinery connecting Lie groups and Lie algebras—particularly through the matrix exponential—provide fundamental breakthroughs in AI efficiency and capability? The model's response was so compelling that Emanuel took an unprecedented next step.

### The Meta-Discovery: AI as Mathematical Research Partner

After the success with the matrix exponential prompt, Emanuel challenged GPT-5 Pro to go further—to **create its own similar prompts** exploring other exotic mathematical structures that could transform AI. The model generated additional research directions, each as ambitious as the original:

1. **Ultrametric Worlds & p-adic Computation** - Hierarchical attention using p-adic numbers
2. **Tropical Geometry & Idempotent Algebra** - Max-plus algebra for piecewise-linear networks
3. **Octonionic/Quaternionic Signal Flow** - Non-associative algebra for richer representations
4. **Simplicial Complexes & Higher-Order Attention** - Multi-body interactions beyond pairwise
5. **Nonstandard Analysis & Hyperreal Training** - Infinitesimal perturbations and transfer principles

### The Self-Evaluation Framework

In a fascinating twist, Emanuel then asked GPT-5 Pro to **evaluate its own ideas** using a comprehensive scoring rubric. The model assessed each proposal across multiple dimensions:
- Theoretical Novelty (0-100)
- Practical Feasibility (0-100)
- Potential Impact (0-100)
- Mathematical Rigor (0-100)
- Implementation Clarity (0-100)

These scores were then combined into an overall score (0-1000) using a weighted formula that the model itself devised. This meta-cognitive approach—having the AI both generate and evaluate mathematical research directions—represents a new paradigm in human-AI collaboration.

### From Theory to Implementation

What you see in this repository is the next step: **turning these AI-generated mathematical visions into working code**. Each of the model's proposals has been implemented as a functioning demonstration, allowing us to test whether these exotic mathematical structures truly offer the revolutionary advantages the model predicted.

## 🧮 The Core Insight

The matrix exponential function `exp(A) = Σ(A^k/k!)` serves as a bridge between:
- **Local** (infinitesimal generators in Lie algebras)
- **Global** (finite transformations in Lie groups)

This mathematical structure appears throughout physics, geometry, and optimization, but has been underutilized in modern AI. By leveraging these tools, we can:
- Build **provably stable** neural architectures
- Achieve **exact conservation laws** during training
- Create **geometry-aware** optimization algorithms
- Enable **continuous-depth** networks with perfect reversibility

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+** (we use the latest Python features)
- **[uv](https://github.com/astral-sh/uv)** - Modern Python package manager (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **8GB+ RAM** recommended for running demos
- **CUDA-compatible GPU** (optional, for accelerated computation)

### Installation

```bash
# Clone the repository
git clone https://github.com/Dicklesworthstone/model_guided_research
cd model_guided_research

# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
uv pip install -e .

# Verify installation
mgr --help
```

### Running Demos

The project includes a comprehensive CLI (`mgr`) for exploring all implementations:

```bash
# List all available demos with descriptions
mgr list

# Run a specific demo (use the short name from 'mgr list')
mgr run matrix-gauge
mgr run tropical
mgr run simplicial

# Get detailed information about a demo before running
mgr info ultrametric
mgr info knot-braid

# Run all demos in sequence (takes ~5-10 minutes)
mgr run-all

# Run with custom configuration
mgr run matrix-gauge --verbose --max-iterations 500

# Export JSON artifacts with diagnostics (per demo)
mgr run reversible --rev-cayley --rev-pareto --export-json artifacts/rev.json
mgr run matrix-gauge --export-json artifacts/gauge.json
mgr run tropical --export-json artifacts/tropical.json

# Additional focused runs
# Ultrametric packed vs dict scaling comparison
ULTRA_SCALE_COMPARE=1 mgr run ultrametric
# Tropical sparse-train K grid (sweep top-k supports)
TROP_SPARSE_TRAIN=1 TROP_SPARSE_TRAIN_KS=4,8,16 mgr run tropical --export-json artifacts/tropical_sparse.json

# See all available options
mgr run --help
```

#### Example Output

When you run a demo, you'll see rich, colorful output showing:
- Mathematical theory being demonstrated
- Step-by-step computations with visualizations
- Performance metrics and convergence behavior
- Key insights and takeaways

## 📂 Project Structure

```
model_guided_research/
├── pyproject.toml                    # Project configuration and dependencies
├── cli.py                            # Typer-based CLI interface (mgr command)
├── config.py                         # Global configuration settings
├── utils.py                          # Shared utilities and helpers
├── CLAUDE.md                         # Development guidelines
├── README.md                         # This file
│
├── markdown_documentation/           # Theoretical foundations for each module
│   ├── matrix_exponential_gauge_learning.md
│   ├── ultrametric_worlds_and_p_adic_computation.md
│   └── ... (one for each implementation)
│
├── tests/                           # Test suite
│   └── test_practical_utility.py    # Practical utility + property tests
│
└── *.py                             # Implementation modules (11 demos)
    ├── matrix_exponential_gauge_learning.py
    ├── ultrametric_worlds_and_p_adic_computation.py
    ├── simplicial_complexes_and_higher_order_attention.py
    ├── nonstandard_analysis_and_hyperreal_training.py
    ├── octonionic_quaternionic_signal_flow.py
    ├── ordinal_schedules_and_well_founded_optimization.py
    ├── reversible_computation_and_measure_preserving_learning.py
    ├── iterated_function_systems_and_fractal_memory.py
    ├── knot_theoretic_programs_and_braid_based_attention.py
    ├── surreal_numbers_transseries_and_scaling.py
    └── tropical_geometry_and_idempotent_algebra.py
```

## 🔬 The Implementations

Each implementation below corresponds to either Emanuel's original matrix exponential prompt or one of the mathematical research directions that GPT-5 Pro generated autonomously. The model not only proposed these ideas but also provided detailed theoretical frameworks that have been translated into working code.

For each demo, you can explore:
- **Documentation**: Detailed mathematical theory in `markdown_documentation/`
- **Implementation**: Working code demonstrating the concepts
- **Live Demo**: Run with `mgr run <demo-name>` to see it in action

### What Each Demo Shows

Each demo is a self-contained exploration that:
1. **Introduces** the mathematical concept with visual examples
2. **Implements** the core algorithms from first principles
3. **Demonstrates** advantages over traditional approaches
4. **Visualizes** the results with rich console output
5. **Measures** performance and convergence properties

### 1. **Matrix Exponential Gauge Learning** (`matrix-gauge`)
*Emanuel's original prompt that started it all—demonstrating Lie group/algebra principles in neural networks*

📖 [Mathematical Documentation](markdown_documentation/matrix_exponential_gauge_learning.md) | 💻 [Implementation](matrix_exponential_gauge_learning.py)

- **Key Idea**: Gauge-invariant transport with exact Lie-theoretic maps (skew/symmetric/Hamiltonian generators)
- **Components**:
  - Structured generators: SO (skew via Givens/Cayley), SPD (eigendecomposition exp), and Sp (symplectic Cayley)
  - Banded Markov mixing via uniformization (expmv) with an exact Poisson sampling mode
  - BCH-aware stacking with a compact per-block summary (curvature/commutators) and per-head/ per-block K diagnostics
  - SPD channel gating (exp(S)) and nilpotent upper-band channels for controlled expressivity
- **Why It Matters**: Provides mathematical guarantees for stability and geometric structure

### 2. **Ultrametric Worlds & p-adic Computation** (`ultrametric`)
*Hierarchical attention using ultrametric-prefix trees*

📖 [Mathematical Documentation](markdown_documentation/ultrametric_worlds_and_p_adic_computation.md) | 💻 [Implementation](ultrametric_worlds_and_p_adic_computation.py)

- **Key Idea**: LCP-based routing over ultrametric signatures for sub-quadratic attention
- **Components**:
  - Bit-prefix LSH signatures for LCP routing; per-level packed buckets with O(1) prefix lookup (array-packed with prefix sums)
  - Insert/query timing remains stable up to N≈4096; per-level occupancy and constant-time rank/test utilities
  - Multi-head configuration with ultrametric fusion and head-variance diagnostics; simple p tuner for LCPTree pipelines
- **Why It Matters**: Enables hierarchical, cache-friendly attention with predictable memory

### 3. **Simplicial Complexes & Higher-Order Attention** (`simplicial`)
*Multi-body interactions beyond pairwise attention*

📖 [Mathematical Documentation](markdown_documentation/simplicial_complexes_and_higher_order_attention.md) | 💻 [Implementation](simplicial_complexes_and_higher_order_attention.py)

- **Key Innovation**: Extends attention to k-way interactions using simplicial complexes
- **Components**:
  - Higher-order Laplacians for multi-token relationships
  - Hodge decomposition for gradient-free components
  - Persistent homology for topological features
- **Why It Matters**: Captures group dynamics beyond pairwise interactions; in our toy task the advantage is modest but consistent

### 4. **Nonstandard Analysis & Hyperreal Training** (`nonstandard`)
*Infinitesimal perturbations and transfer principles*

📖 [Mathematical Documentation](markdown_documentation/nonstandard_analysis_and_hyperreal_training.md) | 💻 [Implementation](nonstandard_analysis_and_hyperreal_training.py)

- **Key Innovation**: Uses hyperreal numbers to handle infinite precision and infinitesimal learning rates
- **Components**:
  - Dual-number automatic differentiation
  - Infinitesimal perturbation analysis
  - Standard part extraction for finite answers
- **Why It Matters**: Explores infinitesimal methods and dual-number arithmetic

### 5. **Octonionic/Quaternionic Signal Flow** (`octonion`)
*Non-associative algebra for richer representations*

📖 [Mathematical Documentation](markdown_documentation/octonionic_quaternionic_signal_flow.md) | 💻 [Implementation](octonionic_quaternionic_signal_flow.py)

- **Key Innovation**: Leverages 8D octonions and 4D quaternions for rotation-invariant features
- **Components**:
  - Quaternion convolutions for 3D-aware processing
  - Octonionic attention with non-associative mixing
  - Automatic rotation/reflection invariance
- **Why It Matters**: Natural handling of 3D data, built-in symmetries, richer algebra

### 6. **Ordinal Schedules & Well-Founded Optimization** (`ordinal`)
*Transfinite learning schedules beyond real numbers*

📖 [Mathematical Documentation](markdown_documentation/ordinal_schedules_and_well_founded_optimization.md) | 💻 [Implementation](ordinal_schedules_and_well_founded_optimization.py)

- **Key Innovation**: Uses ordinal numbers to create learning schedules that "restart at infinity"
- **Components**:
  - Ordinal arithmetic for schedule composition
  - Well-founded descent guarantees
  - Limit ordinal checkpointing
- **Why It Matters**: Provides a principled restart/anneal framework; in our noisy toy setup performance is comparable to cosine schedules

### 7. **Reversible Computation & Measure-Preserving Learning** (`reversible`)
*Bijective networks with perfect information preservation*

📖 [Mathematical Documentation](markdown_documentation/reversible_computation_and_measure_preserving_learning.md) | 💻 [Implementation](reversible_computation_and_measure_preserving_learning.py)

- **Key Idea**: Bijective flows with explicit inverse maps and O(1) training memory
- **Components**:
  - Additive coupling with orthogonal mixing (Householder or Givens), exact inverse maps; per-layer det≈1 checks
  - Cayley orthogonal step with custom O(1) JVP; symplectic hybrid and generating-function steps (exact inverse)
  - JAX‑native training valve path (audit path records exact bits); consolidated per‑layer property table
  - Memory–compute Pareto with layers×iterations sweep; ASCII sparklines for forward/inverse trends
- **Why It Matters**: Information-theoretic guarantees through reversibility; on our demo scale we observe a few-fold memory savings, not 10x

### 8. **Iterated Function Systems & Fractal Memory** (`ifs-fractal`)
*Self-similar memory structures with infinite capacity*

📖 [Mathematical Documentation](markdown_documentation/iterated_function_systems_and_fractal_memory.md) | 💻 [Implementation](iterated_function_systems_and_fractal_memory.py)

- **Key Innovation**: Memory organized as attractors of iterated function systems
- **Components**:
  - Barnsley fern-like memory encoding
  - Hutchinson operators for retrieval
  - Fractal dimension as capacity measure
- **Why It Matters**: Self-similar structures for hierarchical memory; current demo shows mixed results on catastrophic forgetting vs a simple baseline

### 9. **Knot-Theoretic Programs & Braid-Based Attention** (`knot-braid`)
*Topological invariants for robust representations*

📖 [Mathematical Documentation](markdown_documentation/knot_theoretic_programs_and_braid_based_attention.md) | 💻 [Implementation](knot_theoretic_programs_and_braid_based_attention.py)

- **Key Innovation**: Information encoded in knot/braid topology rather than vectors
- **Components**:
  - Braid group representations
  - Jones polynomial features
  - Reidemeister move equivariance
- **Why It Matters**: Topologically protected information, invariant features; in length generalization tests we see small improvements rather than perfect generalization

### 10. **Surreal Numbers, Transseries & Scaling** (`surreal`)
*Infinitely large and small scales simultaneously*

📖 [Mathematical Documentation](markdown_documentation/surreal_numbers_transseries_and_scaling.md) | 💻 [Implementation](surreal_numbers_transseries_and_scaling.py)

- **Key Innovation**: Uses Conway's surreal numbers for multi-scale representations
- **Components**:
  - Transseries expansions for asymptotic behavior
  - Surreal arithmetic for infinite hierarchies
  - Automatic scale selection
- **Why It Matters**: Natural handling of multiple scales, exact asymptotics; currently underperforms simple heuristics in our toy allocation test

### 11. **Tropical Geometry & Idempotent Algebra** (`tropical`)
*Max-plus algebra for piecewise-linear deep learning*

📖 [Mathematical Documentation](markdown_documentation/tropical_geometry_and_idempotent_algebra.md) | 💻 [Implementation](tropical_geometry_and_idempotent_algebra.py)

- **Key Idea**: Replace (+,×) with (max,+) to realize piecewise linear networks by construction
- **Components**:
  - Tropical polynomials and max‑plus GEMM; tropical convexity tools
  - Route‑level certificates: per‑sample route min‑gap, per‑node runner-up gaps along routes, and min‑gap/2 radius (plus per-node min-gap/2)
  - Param‑efficient mixtures: sparse mixture grid (k, λ) with accuracy trade-offs and tidy JSON export; optional sparse-support training with per-class sparsity
- **Why It Matters**: Piecewise linear structure emerges naturally from the algebra

## 💡 Key Insights & Findings

### Why These Mathematical Structures Matter

These implementations demonstrate several breakthrough insights:

1. **Geometric Structure = Free Regularization**: By building neural networks on mathematical manifolds (Lie groups, simplicial complexes), we get stability and interpretability without explicit regularization terms.

2. **Discrete ≠ Approximate**: Structures like p-adic numbers and tropical geometry show that discrete mathematics can be exact, not just approximations of continuous math.

3. **Topology > Vectors**: Encoding information in topological structures (knots, braids) provides invariances that vector representations cannot achieve.

4. **Infinity is Computational**: Surreal numbers, ordinals, and hyperreals show that infinite quantities can be manipulated algorithmically, opening new optimization landscapes.

5. **Non-Associativity = Richer Representations**: Octonions demonstrate that giving up associativity yields representations with built-in symmetries impossible in standard linear algebra.

## 🧾 Artifacts & Diagnostics

All demos support exporting JSON artifacts via `--export-json <path>`.

- `reversible`: exports Pareto arrays (`time/mem` vs iterations and depth), per-layer property checks (with thresholds and pass/fail), strict Givens flag, and optional generating‑param norms.
- `matrix-gauge`: exports uniformization K stats (per block/head), curvature summaries, and commutator heatmap rows.
- `matrix-gauge` also exports a compact BCH summary and sampling schedule comparisons (variance and deterministic diff), plus an optional train/eval schedule compare.
- `tropical`: exports route‑level certificate rows (first 10) with min‑gap/2 and median margins.
  - `tropical` also exports node‑wise route margin details and sparse mixture grid results, plus optional sparse-train summary.
- `ultrametric`: exports timing summaries, head variance, constant‑time rank/test samples, occupancy per level, a small scaling block, variance‑reduction deltas, and (optionally) a simple p tuner decision.
  - `ultrametric` can also print a packed vs dict scaling compare table when `ULTRA_SCALE_COMPARE=1`.
 - `reversible`: additionally exports a small `gen_mode` block and a `property_summary` rollup.

### CLI Knobs (selected)

- `reversible`:
  - `--rev-givens`: enforce strict Givens mixing (det=1, exact inverse)
  - `--rev-generating`: enable generating-function symplectic step (exact inverse)
  - `--rev-gen-vjp`: use custom VJP for generating step (O(1) grads; ignores ∂/∂(a,b,c))
  - `--rev-cayley`, `--rev-cayley-iters`, `--rev-inv-iters`, `--rev-symplectic-hybrid`, `--rev-pareto`
- `matrix-gauge`:
  - `--gauge-bch-compact`: only compact per-block summary
  - `--gauge-alt-struct`: alternate structured/unstructured on odd blocks and print compare
- `tropical`:
  - `TROP_SPARSE_MIX=1` with `TROP_SPARSE_KS`/`TROP_SPARSE_LAMBDAS` to sweep sparse mixtures
  - `TROP_SPARSE_TRAIN=1` (env) to run a tiny sparse-support training loop; knobs: `TROP_SPARSE_TRAIN_K`, `TROP_SPARSE_STEPS`, `TROP_SPARSE_LR`
- `ultrametric`:
  - `ULTRA_PACKED_ARRAYS=1` to enable array-packed prefix rank/test; `ULTRA_FUSE=1` for fusion mode
  - `ULTRA_SCALE_COMPARE=1` to print packed vs dict scaling compare table

The CLI automatically collects module diagnostics when present and nests them under `diagnostics.<demo_name>` in the output JSON.

### Theoretical Advantages

These implementations explore theoretical benefits:
- **Improved numerical stability** through geometric structure (matrix exponential)
- **Hierarchical organization** naturally emerging from ultrametric spaces
- **Conservation laws** built into the architecture (symplectic)
- **Information preservation** through reversible operations
- **Self-similar representations** via fractal structures (IFS)

## 🎯 Key Mathematical Concepts

### The Matrix Exponential
The function `exp(A)` maps a matrix to its exponential, preserving crucial algebraic structures:
- Skew-symmetric → Orthogonal (rotations)
- Symmetric → Positive definite (scalings)
- Hamiltonian → Symplectic (phase-space preserving)
- Stochastic generator → Stochastic matrix (probability preserving)

### Baker-Campbell-Hausdorff Formula
The BCH formula `exp(A)exp(B) = exp(A + B + [A,B]/2 + ...)` quantifies non-commutativity, revealing hidden interactions between layers that standard analysis misses.

### Lie Theory Connection
- **Lie Algebra**: The tangent space at identity (infinitesimal transformations)
- **Lie Group**: The manifold of transformations (finite transformations)
- **Exponential Map**: The bridge between them

## 🔮 Future Directions

### Immediate Goals
1. **Benchmarking**: Systematic comparison with standard architectures
2. **Scaling Studies**: Understanding behavior at different model sizes
3. **Hybrid Architectures**: Combining multiple mathematical structures
4. **Hardware Optimization**: Custom kernels for exotic operations

### Long-term Vision
- **Geometric Deep Learning**: Fully geometry-aware neural architectures
- **Quantum-Classical Bridges**: Leveraging quantum-inspired classical algorithms
- **Automated Mathematical Discovery**: Using these structures to discover new mathematics
- **Provably Optimal Networks**: Architectures with mathematical optimality guarantees

## 📚 Theoretical Background

Each implementation is accompanied by detailed mathematical documentation in the `markdown_documentation/` folder. These documents provide:
- First-principles derivations
- Connections to existing ML techniques
- Complexity analyses
- Experimental validation strategies

## 🤝 Contributing

This project emerged from the intersection of mathematical speculation and practical engineering. We welcome contributions that:
- Implement additional mathematical structures
- Improve computational efficiency
- Provide empirical validations
- Extend theoretical understanding

## 🏆 The Scoring Framework

GPT-5 Pro evaluated each mathematical approach using a comprehensive scoring rubric across multiple dimensions:
- **Theoretical Novelty** (0-100): How innovative is the mathematical approach?
- **Practical Feasibility** (0-100): Can this be implemented efficiently?
- **Potential Impact** (0-100): Could this revolutionize AI?
- **Mathematical Rigor** (0-100): How solid is the theoretical foundation?
- **Implementation Clarity** (0-100): How clear is the path to implementation?

The scoring methodology weighted theoretical novelty and potential impact most heavily, while still requiring practical feasibility—reflecting the project's balance between mathematical ambition and engineering reality.

## 📖 References & Acknowledgments

### Original Inspiration
- Jeffrey Emanuel's exploration of matrix exponentials and Lie groups
- GPT-5 Pro's autonomous generation of five additional mathematical research directions
- The meta-experiment of having AI evaluate its own mathematical creativity
- The Baker-Campbell-Hausdorff formula and its implications

### Key Mathematical Sources
- Lie Groups and Lie Algebras (multiple classical texts)
- Conway's "On Numbers and Games" (surreal numbers)
- Mac Lane's "Categories for the Working Mathematician"
- Various papers on tropical geometry, p-adic analysis, and exotic algebras

### Implementation Framework
- JAX for automatic differentiation and JIT compilation
- Flax for neural network layers
- Optax for optimization algorithms
- Rich for beautiful console output
- Typer for CLI interface

## 🎨 Philosophy

This project embodies a unique approach to AI research:
1. **Mathematical First**: Start with beautiful mathematics, then find AI applications
2. **AI as Co-Creator**: Let advanced models propose their own research directions
3. **Self-Evaluation**: Have AI assess the quality of its own ideas
4. **No Compromise**: Implement the full mathematical structure, not approximations
5. **Radical Simplicity**: Complex behavior from simple, elegant rules
6. **Geometric Thinking**: Leverage the geometry of solution spaces
7. **Cross-Pollination**: Connect seemingly unrelated mathematical fields

## ⚡ Performance Notes

- All implementations use JAX for GPU acceleration when available
- JIT compilation provides near-C performance for hot loops
- Memory-efficient algorithms chosen where possible
- Complexity guarantees provided for each method

## 📝 License

MIT License - See LICENSE file for details

## 🔧 Troubleshooting

### Common Issues

**Import errors or module not found**
```bash
# Ensure you're in the virtual environment
source .venv/bin/activate
# Reinstall in editable mode
uv pip install -e .
```

**JAX/CUDA issues**
```bash
# For CPU-only mode (no GPU required)
export JAX_PLATFORM_NAME=cpu
mgr run <demo-name>
```

**Memory errors**
```bash
# Reduce batch size or iterations
mgr run <demo-name> --max-iterations 100
```

**Numerical instabilities**
```bash
# Enable debug mode to catch NaN/Inf
mgr run <demo-name> --debug
```

## 📦 Dependencies

Core dependencies (automatically installed):
- **JAX**: Automatic differentiation and JIT compilation
- **Flax**: Neural network layers and models
- **Optax**: Optimization algorithms
- **NumPy/SciPy**: Numerical computations
- **Rich**: Beautiful terminal output
- **Typer**: CLI interface

## 🌐 Links

- **Repository**: [GitHub](https://github.com/Dicklesworthstone/model_guided_research)
- **Author**: Jeffrey Emanuel (@doodlestein)

## 💡 Final Thoughts

This project represents something unprecedented: **a collaboration where AI not only answered questions but posed its own**. GPT-5 Pro didn't just respond to Emanuel's prompt about matrix exponentials—it generated five additional mathematical research programs, evaluated them systematically, and provided detailed theoretical frameworks that have now been implemented in code.

The meta-cognitive loop here is striking:
1. Human poses mathematical question to AI
2. AI provides detailed answer
3. Human asks AI to generate similar questions
4. AI creates new research directions autonomously
5. AI evaluates its own proposals
6. Human and AI collaborate to implement the ideas

This represents a new paradigm in scientific discovery—one where AI systems don't just assist with predetermined research directions but actively participate in setting the research agenda itself. The mathematical structures explored here—Lie groups, p-adic numbers, tropical geometry, octonions, simplicial complexes, and hyperreals—are not just abstract curiosities but potentially contain the seeds of revolutionary AI architectures.

Whether these implementations prove transformative or merely instructive, they demonstrate something profound: **AI systems are becoming genuine partners in mathematical discovery**, capable not just of solving problems but of identifying which problems are worth solving.

As Wigner wrote of *"The unreasonable effectiveness of mathematics in the natural sciences"*—we may now be witnessing the unreasonable effectiveness of AI in discovering which mathematics will prove essential for its own evolution.

---

**Remember**: Each demo can be run independently, and the mathematical documentation provides deep dives into the theory. Start with whatever captures your imagination—the mathematics will guide you the rest of the way.
## 🧪 Testing & Evaluation

This repo includes a comprehensive, scriptable evaluation that checks both mathematical properties and practical utility across all demos. It uses rich console output for clarity.

- `tests/test_practical_utility.py` (primary): Runs eleven mini-benchmarks — one per demo — and reports:
  - Practical benefits (e.g., memory savings, scaling exponents, generalization)
  - Claimed mathematical properties (e.g., 1‑Lipschitz, norm preservation)
  - A green/yellow/red verdict plus a summary table and recommendations

What each sub‑test does:
- Reversible Computation: Compares peak activation memory of a standard residual MLP vs. an invertible coupling stack with explicit recomputation. Also prints a checkpoint K‑sweep table (store every K‑th activation) for a fair baseline comparison.
- IFS Fractal Memory: Stresses catastrophic forgetting by constraining a FIFO baseline capacity; compares average recall error vs. a contractive IFS store with signature‑based routing.
- Ordinal Schedules: Uses a piecewise‑stationary (regime‑shift) quadratic objective to test whether ordinal restarts/anneals improve final‑window loss over a cosine schedule.
- Matrix Exponential Gauge: Checks gradient stability against a standard deep network (vanish/explode detection), illustrating the conditioning benefits.
- Tropical Geometry: Verifies the 1‑Lipschitz property of a tropical attention adapter via finite‑difference perturbations.
- Simplicial Complexes: Constructs a higher‑order (triangle‑dependent) label and evaluates an incidence‑only flow with a tiny linear readout over a Hodge‑like diffusion.
- Ultrametric Worlds: Estimates scaling exponents on sequence lengths and confirms sub‑quadratic behavior with an LSH‑based prefix‑trie attention.
- Quaternions/Octonions: Measures norm drift after many layers; quaternion layer should preserve norms without additional normalization.
- Knot/Braid: Dyck‑1 (balanced parentheses) length generalization with a simple curriculum; braid model trained on short/medium lengths and tested on longer sequences.
- Surreal Numbers: Resource‑allocation choices via rank/z‑score dominance with small guardbands, preserving ≥ baseline performance on a held‑out split.
- Hyperreal Training: Robustness to learning‑rate choice on a stiff quadratic; compares variance of final losses across rates.

How it works:
- All sub‑tests compute a baseline metric, a proposed metric from the demo module, an improvement ratio, and a boolean `is_better` used for the verdict.
- Output includes per‑test metrics and a final summary table with “Worth Further Investigation?” recommendations. Rich tables and panels make results easy to scan.
- CPU‑only by default (sets `JAX_PLATFORM_NAME=cpu`) for reproducible CI without CUDA.

Running the tests:
- Activate your `uv` venv, then run:
  - `python tests/test_practical_utility.py`
  - Optional: pipe output to a file to archive results.

Interpreting results:
- Green “SUCCESS” means the claimed advantage or property holds under the benchmark conditions.
- Yellow “MARGINAL/PARTIAL” indicates a small or context‑dependent advantage.
- Red indicates the claim did not hold under the benchmark (tweakable via test knobs as desired).

Notes & reproducibility:
- The tests favor determinism and modest sizes; they are not micro‑optimized for speed but are structured to be robust and interpretable.
- Most demos also include lightweight sanity checks or helper functions inside the modules themselves (e.g., `simplicial.sanity_suite`) that can be used interactively.
