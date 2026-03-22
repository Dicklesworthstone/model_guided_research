# Changelog

All notable changes to **Model-Guided Research** are documented in this file.

This project has no formal releases or tags. The changelog is organized by development phase, with entries grouped by capability rather than commit order. Every commit hash links to its GitHub diff.

Repository: <https://github.com/Dicklesworthstone/model_guided_research>

---

## Phase 3: Systematic Evaluation & Production Hardening (2026-01-09 -- present)

### CI/CD & Build

- Add GitHub Actions CI workflow (ruff, mypy, pytest) and release workflow, plus Dependabot configuration for grouped dependency updates ([`17c6ccc`](https://github.com/Dicklesworthstone/model_guided_research/commit/17c6cccc1bb8b59c451cb99cc30592eb03416efa)) -- 3 new files, +295 lines
- Fix CI: use `--extra` instead of `--group` for uv optional dependencies ([`8a019e2`](https://github.com/Dicklesworthstone/model_guided_research/commit/8a019e2fd66098a9dfab472c28643e44d5cd030e))
- Fix CI: resolve ruff lint errors across 56 files ([`a264536`](https://github.com/Dicklesworthstone/model_guided_research/commit/a264536811fea3fb65394f63d8ab68c8c6894879))
- Bump GitHub Actions versions across CI and release workflows ([`8b989f3`](https://github.com/Dicklesworthstone/model_guided_research/commit/8b989f33624ecc55a754548bd39493ad8a6ecfed))
- Dependabot PR (unmerged, on branch): bump the github-actions group with 4 updates ([`284c657`](https://github.com/Dicklesworthstone/model_guided_research/commit/284c6573c1fa79e1a0ceb180f14b0d47f63de05b), [`76ec1ef`](https://github.com/Dicklesworthstone/model_guided_research/commit/76ec1eff9935e928c474a71636895bc8c28bdea6))

### Benchmarking Infrastructure (2026-01-09)

- Add comprehensive benchmarking harness to CLI with statistical analysis -- Mann-Whitney U tests, effect-size calculations, confidence intervals -- enabling systematic comparison of all 11 attention mechanisms (+1,781 lines to `cli.py`) ([`2cfd826`](https://github.com/Dicklesworthstone/model_guided_research/commit/2cfd826e5d13ad37af71ba905a053e14f193c9cc))
- Add benchmark artifacts from initial runs: 130 files covering baseline, fixed-flops, per-head-metrics, CMA-ES, and FlexAttention correctness results ([`5a29380`](https://github.com/Dicklesworthstone/model_guided_research/commit/5a293806cb1d9d96794229abdf7d93a1078de246))
- Add developer scripts: `benchmark_flex.py`, `cmaes_phase1.py`, `verify_flex_correctness.py` and planning documentation covering CMA-ES budget scheduling, GPU environment notes, FlexAttention edge cases, and reproducibility (+2,880 lines across 13 files) ([`40b2d0f`](https://github.com/Dicklesworthstone/model_guided_research/commit/40b2d0fa0549760853b96a4cfed9fe0126dbfc8b))

### Nanochat Core Upgrades (2026-01-09)

- Upgrade GPT core with PyTorch FlexAttention backend and advanced initialization (muP-style, spectral norm) -- +478 lines across `gpt.py`, `common.py`, `model_utils.py` ([`3f76562`](https://github.com/Dicklesworthstone/model_guided_research/commit/3f76562a78605df4755caae96901fc1d3d8728b8))
- Enhance all 11 attention mechanisms with KV-cache decode path and per-head diagnostics (+801 lines across 11 files) ([`e3c17c4`](https://github.com/Dicklesworthstone/model_guided_research/commit/e3c17c4396646fba5cce8a4d5eef655c7828f81c))
- Expand training pipeline with comprehensive metrics, artifact generation, and W&B-compatible logging (+1,231 lines to `train.py`) ([`5c608da`](https://github.com/Dicklesworthstone/model_guided_research/commit/5c608dad48837a1a322ccb7f7837839c213df759))
- Enhance inference engine with robust KV-cache handling and serving improvements ([`b777a51`](https://github.com/Dicklesworthstone/model_guided_research/commit/b777a517288f16248a33c4dd162731731d7a1b36))
- Refactor optimizers (HOSS, AdamW) and schedulers (ordinal) for clarity and correctness ([`9677e9e`](https://github.com/Dicklesworthstone/model_guided_research/commit/9677e9e2e138c534c73f9a86e676184e035e3531))
- Update supporting modules (synaptic dynamics, neuroviz, tokenizer) for production readiness ([`a10bf41`](https://github.com/Dicklesworthstone/model_guided_research/commit/a10bf419dd4fd8f449826db158c943fd1c78bd07))

### JAX Demos & Shared Utilities (2026-01-09)

- Refine all 11 JAX demonstrations with production-ready algorithms and validation ([`62ae775`](https://github.com/Dicklesworthstone/model_guided_research/commit/62ae77577ccc6bd909d3285b83b1dee928bee762))
- Enhance shared utilities with cross-framework (JAX + PyTorch) seeding and diagnostics ([`9f6e44c`](https://github.com/Dicklesworthstone/model_guided_research/commit/9f6e44cca83b82ee71e9eea6869b08fbcc595c6c))

### Testing (2026-01-09)

- Expand test suite with comprehensive framework validation and CMA-ES objective tests (+1,039 lines across 5 files) ([`e7f01f6`](https://github.com/Dicklesworthstone/model_guided_research/commit/e7f01f6a20437457ec0ef98df9bcf066d431ef0a))

### Documentation & Dependencies (2026-01-09)

- Restructure `pyproject.toml` with optional feature groups (`jax`, `torch`, `dev`) ([`61e4a17`](https://github.com/Dicklesworthstone/model_guided_research/commit/61e4a17e63fbfe0a4857cb52fb654e46b60a923d))
- Expand documentation with comprehensive tooling guides and theoretical updates for mathematical frameworks (+346 lines) ([`c5fc7ab`](https://github.com/Dicklesworthstone/model_guided_research/commit/c5fc7abfd50bf825b22474d7a5028ce4b0e69647))

### Major Codebase Refactor (2026-01-24)

- Major refactor touching all 70 files: CLI restructuring, improved type safety, better error handling, refined mathematical operations across all 11 JAX demos, all nanochat modules, and the full test suite (+2,317 / -1,670 lines) ([`9ad9476`](https://github.com/Dicklesworthstone/model_guided_research/commit/9ad9476213ee6d31baded3ff5aa98038c5c70044))

### Licensing & Branding

- Add MIT License ([`a376456`](https://github.com/Dicklesworthstone/model_guided_research/commit/a3764560be58a13e17bf50f2e63eb554cd5a9a08))
- Update license to MIT with OpenAI/Anthropic Rider (+53 lines to `LICENSE`) ([`bf92d23`](https://github.com/Dicklesworthstone/model_guided_research/commit/bf92d23388585367fbd5df95f3af021c95b4d398))
- Add GitHub social preview image (1280x640) ([`042bc3e`](https://github.com/Dicklesworthstone/model_guided_research/commit/042bc3ed932c036130381f416156f25f83e218e2))

### Dependency Updates

- Update Python dependencies to latest stable versions ([`16705be`](https://github.com/Dicklesworthstone/model_guided_research/commit/16705be1544afa8abe232b489ec5645c2eb7ef87))
- Update beads tracking and nanochat training script ([`c6951ae`](https://github.com/Dicklesworthstone/model_guided_research/commit/c6951aea4ed74153c086a1ae02d7bf07ca1f9b78))

### Housekeeping

- Update AGENTS.md with project context ([`5348d74`](https://github.com/Dicklesworthstone/model_guided_research/commit/5348d74627da8f33d6bb0a4e2a8e211f76164de5), [`f331cb6`](https://github.com/Dicklesworthstone/model_guided_research/commit/f331cb6e2940285c89d1528c3bab17c01089977a))
- Gitignore updates for beads ephemeral files and viewer config ([`c8e93d9`](https://github.com/Dicklesworthstone/model_guided_research/commit/c8e93d9a9d6660cac25230f65b497f141de264b6), [`88a4cdd`](https://github.com/Dicklesworthstone/model_guided_research/commit/88a4cdd8ce9eed2f27e29a6aa7a78056911b7449), [`168a06b`](https://github.com/Dicklesworthstone/model_guided_research/commit/168a06bb081787e7354aed628c92420f18df2b92), [`b830b3a`](https://github.com/Dicklesworthstone/model_guided_research/commit/b830b3ad140d9cf32ceeb95833ee06ab3a236bce), [`bb945b3`](https://github.com/Dicklesworthstone/model_guided_research/commit/bb945b3848319bce0b92069a4610f0a4e72dae7f))

---

## Phase 2: Nanochat -- Unified PyTorch Transformer (2025-11-19 -- 2025-12-19)

This phase transformed the project from standalone JAX demonstrations into a production-ready dual-framework system. A complete GPT transformer was built in both JAX/Flax and PyTorch, with all 11 mathematical attention mechanisms ported to PyTorch as drop-in replacements.

### Nanochat Foundation (2025-11-19)

- Add nanochat: JAX/Flax GPT implementation infrastructure for mathematical experiments -- GPT model, synaptic dynamics, tokenizer, training loop, serving UI, visualization, fused kernels, and tropical attention (35 new files, +8,431 lines) ([`ddded58`](https://github.com/Dicklesworthstone/model_guided_research/commit/ddded58da80a4e4c85ed8b8c675ef1cf42810d0d))
- Enable CUDA support and add nanochat requirements to `pyproject.toml` ([`7b2b7f7`](https://github.com/Dicklesworthstone/model_guided_research/commit/7b2b7f7d5b944acdc517940d24312eb84d03044b))
- Extract common JAX utilities to shared `common_jax.py` module ([`0320f78`](https://github.com/Dicklesworthstone/model_guided_research/commit/0320f784e8ea0dc75504373cc474c75fb14cd19e))

### GPT Architecture & Generation (2025-11-19 -- 2025-11-20)

- Fix critical sequence generation bug in GPT JAX ([`c3cc5e3`](https://github.com/Dicklesworthstone/model_guided_research/commit/c3cc5e379670fb1677d08e0ab7e91409fe08bbbe))
- Implement KV caching in tropical attention and fix critical bugs ([`307c4fe`](https://github.com/Dicklesworthstone/model_guided_research/commit/307c4feda4c2a680cabd8dc4e51e6bde1fbd8e47))
- Refactor generation loop: replace JAX `while_loop` with Python `for`-loop for clarity ([`5a14904`](https://github.com/Dicklesworthstone/model_guided_research/commit/5a14904764ca4583f0544de1acce530275f116fb))
- Implement KV caching and fix critical generation bugs in GPT models ([`5c21b3a`](https://github.com/Dicklesworthstone/model_guided_research/commit/5c21b3addbb9c45308c23e0192cddd4d951b9b71))
- Finalize GPT JAX architecture: simplify cache interface and add bfloat16 support ([`2735f03`](https://github.com/Dicklesworthstone/model_guided_research/commit/2735f0399d362b9d679b2d0bc86810bc5552ad9e))
- Align tropical and ultrametric attention with GPT JAX cache interface ([`0d2de74`](https://github.com/Dicklesworthstone/model_guided_research/commit/0d2de7457beb8ff588bb19d3eadfe4bccbb8155e))
- Enhance `apply_rotary_emb` documentation with explicit broadcasting semantics ([`aa77a3f`](https://github.com/Dicklesworthstone/model_guided_research/commit/aa77a3fe81573216c16095803b9632f3918f8fc1))

### PyTorch Port: All 11 Attention Mechanisms (2025-11-20)

Every mathematical framework was ported from JAX to PyTorch as drop-in attention modules:

- Port tropical and ultrametric attention mechanisms ([`518395d`](https://github.com/Dicklesworthstone/model_guided_research/commit/518395d5942ab4dccc5974360f97ebcf6314d390))
- Port hypercomplex algebra attention: octonion and quaternion ([`015a7d7`](https://github.com/Dicklesworthstone/model_guided_research/commit/015a7d75ba09936c56ad836caf547f6413ce49b7))
- Port topological and geometric attention: braid, fractal, simplicial ([`079098e`](https://github.com/Dicklesworthstone/model_guided_research/commit/079098e7ef09a260b1269bb71fb6ff4171ac34d3))
- Port advanced architectural components: surreal blocks, reversible blocks, gauge blocks ([`bd5c101`](https://github.com/Dicklesworthstone/model_guided_research/commit/bd5c10168f7c05b6b52f54ca06540d7b1a1b7560))
- Integrate all 11 experimental attention mechanisms and block types into unified GPT model ([`be9df4d`](https://github.com/Dicklesworthstone/model_guided_research/commit/be9df4d2e92738c3025392227a0abefd723b53b2))
- Integrate experimental attention mechanisms and HOSS optimizer into GPT model ([`e79768f`](https://github.com/Dicklesworthstone/model_guided_research/commit/e79768f18ab4268205c307734033ab6c8ebc9669))
- Refine braid and simplicial attention implementations ([`ebfc59a`](https://github.com/Dicklesworthstone/model_guided_research/commit/ebfc59a0b9a8f8b67a5080980f5903f4ff806663))
- Fix gauge block to use proper cumulative parallel transport ([`6970fd4`](https://github.com/Dicklesworthstone/model_guided_research/commit/6970fd4d845d20ccc0a2102afefe086f0ce8095b))

### HOSS Optimizer (2025-11-20)

The Higher-Order Spectral Stepping optimizer was implemented in JAX, debugged extensively, then ported to PyTorch:

- Add HOSS optimizer and ultrametric attention implementations in JAX ([`d1bfbc0`](https://github.com/Dicklesworthstone/model_guided_research/commit/d1bfbc081fbe2c8c7d7bddd2c876b93e8abf8794))
- Prepare JAX training pipeline for HOSS optimizer integration ([`5015abd`](https://github.com/Dicklesworthstone/model_guided_research/commit/5015abdc8178ba41d97321d910ff83f66c838f81))
- Fix critical bugs: gradient clipping and PyTree registration ([`f2b0e67`](https://github.com/Dicklesworthstone/model_guided_research/commit/f2b0e6733b4b3f1dd8e7c8be34edc1a4f8cb1f59))
- Enforce float32 precision throughout for numerical stability ([`a895c7f`](https://github.com/Dicklesworthstone/model_guided_research/commit/a895c7f15d7f71a3e538961a57baa2640534ba63))
- Add debug instrumentation for numerical analysis ([`59922e1`](https://github.com/Dicklesworthstone/model_guided_research/commit/59922e16193adf15820e06e7e56e386d63324b33))
- Add aggressive NaN detection and debug logging to JAX training loop ([`c378914`](https://github.com/Dicklesworthstone/model_guided_research/commit/c37891414e6ed04a35e89e7eadace092c0235266))
- Add comprehensive traceback printing to JAX training exception handler ([`83e17b0`](https://github.com/Dicklesworthstone/model_guided_research/commit/83e17b0c5b035b5ae13abdd3297ebc6919337255))
- Port HOSS optimizer to PyTorch for production deep learning workflows ([`e129760`](https://github.com/Dicklesworthstone/model_guided_research/commit/e1297606ace99ad50c5d7bde4c2371bdb1aae6da))
- Clean up HOSS optimizer documentation and remove verbose comments ([`fd34558`](https://github.com/Dicklesworthstone/model_guided_research/commit/fd3455804425f04813eda3a35ee60ab9fdf44660))

### Ordinal Learning Rate Scheduler (2025-11-20)

- Port ordinal learning rate scheduler with transfinite ordering to PyTorch ([`896cbd7`](https://github.com/Dicklesworthstone/model_guided_research/commit/896cbd704258587faa1fa6d847e57cf0c369f521))
- Add ordinal scheduler support to PyTorch training script ([`d0458a1`](https://github.com/Dicklesworthstone/model_guided_research/commit/d0458a1b5d87a59f4c0b238febbe7b70c15fb6db))

### Training & Inference Infrastructure (2025-11-20)

- Add PyTorch training script with experimental configuration support ([`422476c`](https://github.com/Dicklesworthstone/model_guided_research/commit/422476cd4fafc364336ea414031b1ab9d84f1c1d))
- Extract shared model utilities (norm, RoPE) to dedicated module ([`439aa52`](https://github.com/Dicklesworthstone/model_guided_research/commit/439aa5201c82a736b7aa18e9796fb6f238cc0c3b))
- Add comprehensive monitoring and CLI configurability to JAX training ([`3e514cf`](https://github.com/Dicklesworthstone/model_guided_research/commit/3e514cf6bb79d4a031d54bffbb496d9287e63bc0))

### Security & Robustness (2025-11-20)

- Replace dangerous `eval()` with AST-based safe expression evaluator in inference engine ([`c29353b`](https://github.com/Dicklesworthstone/model_guided_research/commit/c29353bda5414ec2af19995732d26bc16796739a))
- Migrate tokenizer from pickle to JSON format and strengthen error handling ([`e311eae`](https://github.com/Dicklesworthstone/model_guided_research/commit/e311eae6ef44db7bffdb5acede9aed115d4dbdc5))
- Strengthen data pipeline error handling and validation ([`8678a07`](https://github.com/Dicklesworthstone/model_guided_research/commit/8678a077f451086daf93086fa9d69c3ef5c8337c))
- Improve execution and serving infrastructure with robust error handling ([`7b8fb9b`](https://github.com/Dicklesworthstone/model_guided_research/commit/7b8fb9befced3d7a991a1073de48487556694e34))
- Improve evaluation, visualization, and reporting with better error handling ([`50c5ae7`](https://github.com/Dicklesworthstone/model_guided_research/commit/50c5ae7ae7e3b8d44b092a86d97d6185c8136b31))

### Nanochat Core Refinement (2025-11-19 -- 2025-11-20)

- Refactor nanochat core infrastructure for improved robustness and clarity ([`f24bb56`](https://github.com/Dicklesworthstone/model_guided_research/commit/f24bb564189e28a3bbe03c75e3fc2534e4d6ea25))
- Refine specialized components: synaptic dynamics, fused kernels, and tropical tests ([`5c5e5e2`](https://github.com/Dicklesworthstone/model_guided_research/commit/5c5e5e22c43b61b0d6d2d577aae5af9116919948))
- Refine shared utilities for improved reliability and consistency ([`6933cda`](https://github.com/Dicklesworthstone/model_guided_research/commit/6933cda5e801f69b06d2409725dcffefc0e2827c))
- Clean up code and improve documentation across experimental modules ([`714eb3d`](https://github.com/Dicklesworthstone/model_guided_research/commit/714eb3dd80e4dc16d21b4cd9f49ca85f46085a3d))

### JAX Demo Improvements (2025-11-20)

- Enhance tropical geometry and ultrametric attention demonstrations ([`39eeb76`](https://github.com/Dicklesworthstone/model_guided_research/commit/39eeb76bc379e14d2bf0a1c68588af5ddc766965))
- Refine reversible computation and simplicial complex implementations ([`c867fc7`](https://github.com/Dicklesworthstone/model_guided_research/commit/c867fc72d705dfe1376b8f0d9d00f33280795954))
- Polish remaining mathematical demonstrations: hyperreals, octonions, ordinals, and fractals ([`3e7a5ea`](https://github.com/Dicklesworthstone/model_guided_research/commit/3e7a5ea4a157ec8fbed1eabf9de2a3eb02fa1c66))
- Fix robustness issues in matrix exponential gauge structured blocks ([`2f94125`](https://github.com/Dicklesworthstone/model_guided_research/commit/2f9412586152441749986062b1449963990a2fc5))
- Add debug utility for matrix exponential gauge development ([`e768550`](https://github.com/Dicklesworthstone/model_guided_research/commit/e76855018beb0e78a881816483d312ea959fc598))

### Testing & Validation (2025-11-20)

- Strengthen testing infrastructure with comprehensive validation framework ([`cf65752`](https://github.com/Dicklesworthstone/model_guided_research/commit/cf657523a9a3d15ba5b905f1895ff659c64b5dab))
- Enhance CLI evaluation command with comprehensive test suite integration ([`b2ad162`](https://github.com/Dicklesworthstone/model_guided_research/commit/b2ad16224a6ddd5efd54edacd249a8d8561ad304))

### Documentation & Infrastructure (2025-11-20)

- Comprehensively revise README to reflect dual-implementation (JAX + PyTorch) architecture ([`95ab717`](https://github.com/Dicklesworthstone/model_guided_research/commit/95ab717124a53d816acfdc82d21628e443b811df))
- Add MCP Agent Mail (beads) infrastructure configuration ([`9396497`](https://github.com/Dicklesworthstone/model_guided_research/commit/9396497d7d91c3a143ea90ec6b05be30347bb3a7))
- Update project configuration and add UBS integration documentation ([`93f4b0b`](https://github.com/Dicklesworthstone/model_guided_research/commit/93f4b0b5ce15ed7c4c1b14a59ef48b186c5da030))
- Beads sync: MCP Agent Mail issue tracking updates ([`8e83dc1`](https://github.com/Dicklesworthstone/model_guided_research/commit/8e83dc1a30145e7be1602e0a172d472c17887240), [`d39575d`](https://github.com/Dicklesworthstone/model_guided_research/commit/d39575df6ee3c12b01ac84173749858214204c48))

---

## Phase 1: Mathematical Exploration -- JAX Demos (2025-08-17 -- 2025-08-20)

The project began with a GPT-5 Pro-generated research document on matrix exponentials in AI, then rapidly expanded into 11 standalone mathematical framework implementations in JAX with a Typer-based CLI, a configuration system, and a comprehensive test suite.

### Genesis (2025-08-17)

- Upload the GPT-5 Pro-generated matrix exponential idea assessment report -- the AI-authored research document that started the project ([`6107964`](https://github.com/Dicklesworthstone/model_guided_research/commit/610796402c7211232513732b6000574ddcd2cc9f))

### Initial Codebase (2025-08-18)

- Add project structure with 11 mathematical framework implementations, Typer CLI (`mgr`), configuration system, comprehensive test suite, and full `pyproject.toml` packaging (41 files, +16,274 lines) ([`b60181a`](https://github.com/Dicklesworthstone/model_guided_research/commit/b60181aeb1b047c0b09e3fb4433103b627cab57b))
  - The 11 mathematical frameworks: matrix exponential gauge learning, ultrametric/p-adic computation, tropical geometry, simplicial complexes, quaternion/octonion signal flow, ordinal schedules, reversible computation, iterated function systems (fractal memory), knot-theoretic braid attention, nonstandard analysis (hyperreals), and surreal numbers/transseries
- Enhance configuration handling: string-to-Path conversion, robust Lanczos normalization, cycle indicator improvements ([`7900df1`](https://github.com/Dicklesworthstone/model_guided_research/commit/7900df1541179bc6d92684891db32c65d38778b3))
- Implement enhanced CLI features: `config-gen` and `config-show` commands, improve argument handling with type annotations, add `config.example.json` ([`b1599aa`](https://github.com/Dicklesworthstone/model_guided_research/commit/b1599aac7878aaa4a9ea1bc9c4d36d1aaddc5d55))
- Add JAX pytree registration for `Params` class and enhance mathematical test outputs ([`50f722d`](https://github.com/Dicklesworthstone/model_guided_research/commit/50f722d815e527fdfe18c60305cb57fd6db6c233))

### CLI Feature Expansion (2025-08-19)

- Add AGENTS.md with development guidelines; implement CLI options for JSON export and per-demo flags; add `test_practical_utility.py` (+1,204 lines, +2,419 lines total across 18 files) ([`9b94f05`](https://github.com/Dicklesworthstone/model_guided_research/commit/9b94f055ab9c2df3cc039623a118bf7a1e0f6911))
- Enhance CLI with Cayley computation options, structured gauge blocks, O(1) memory gradients, fixed-point iterations ([`13a6833`](https://github.com/Dicklesworthstone/model_guided_research/commit/13a6833b2d9a114c47ee4857bd78cc2e7999108d))
- Add inverse fixed-point iteration option to CLI; implement optional sparse mixtures of tropical polynomials ([`fd958de`](https://github.com/Dicklesworthstone/model_guided_research/commit/fd958de1bdeff60025289132e2801679e911dafa))
- Fix ruff/mypy issues across modules, correct CLI logic, add symplectic leapfrog, type buckets, remove undefined exports ([`8ecffb4`](https://github.com/Dicklesworthstone/model_guided_research/commit/8ecffb4e70e2addce398762ff681b1702683c441))

### Mathematical Implementations (2025-08-19)

- Introduce symplectic leapfrog integrator in reversible computation module ([`f5ccfae`](https://github.com/Dicklesworthstone/model_guided_research/commit/f5ccfaea79c235f172044f45d6a2bacc28a9ee0d))
- Add Givens mixing with custom JVP to reversible computation module; refactor UltrametricAttention for improved bucket handling ([`cecb872`](https://github.com/Dicklesworthstone/model_guided_research/commit/cecb872e500a86da5a9fe55173feb155a1e43a85))
- Add optional stochastic Poisson uniformization to matrix exponential gauge learning; implement sampling mode for smoothness estimation ([`fb263b6`](https://github.com/Dicklesworthstone/model_guided_research/commit/fb263b6921f7751db51dc51ca25fdb21a7184538))
- Enhance matrix exponential gauge learning with optional uniformization caps, per-block statistics, ASCII visualizations; refactor Cayley iterations ([`e3f945a`](https://github.com/Dicklesworthstone/model_guided_research/commit/e3f945a822eeb0d077523d3755d952500dd8a812))
- Refactor UltrametricAttention to support optional array-backed bucket handling with occupancy tracking ([`23d4c64`](https://github.com/Dicklesworthstone/model_guided_research/commit/23d4c64e1401af3e2d64a422369265a6340d1c2d))

### Diagnostics & Refinements (2025-08-19 -- 2025-08-20)

- Refactor tensor initialization and enhance diagnostics collection; implement module-level diagnostics export ([`4769ea7`](https://github.com/Dicklesworthstone/model_guided_research/commit/4769ea7bc08ca63eb3cd9c703b96594eabb8538b))
- Update demo output to include sampling smoothness variance mean ([`18c73db`](https://github.com/Dicklesworthstone/model_guided_research/commit/18c73db6851dcb8d87012480037da15d1d897ce6))
- Add CLI options for reversible computation: strict Givens mixing, generating-function symplectic steps; enhance matrix-gauge demo with compact BCH summaries and sampling schedule comparisons; add ultrametric attention diagnostics for packed array handling and variance reduction ([`01db611`](https://github.com/Dicklesworthstone/model_guided_research/commit/01db61133bb3198dc4ef8951d571d09294bc345e))

---

## Reference

### The 11 Mathematical Frameworks

Each framework has a JAX demo (root `*.py`) and a PyTorch attention mechanism (`nanochat/*_torch.py`):

| # | Framework | JAX Demo | PyTorch Module | CLI Flag |
|---|-----------|----------|----------------|----------|
| 1 | Matrix Exponential Gauge Learning | `matrix_exponential_gauge_learning.py` | `gauge_block_torch.py` | `--attention-type gauge` |
| 2 | Ultrametric (p-adic) Computation | `ultrametric_worlds_and_p_adic_computation.py` | `ultrametric_attention_torch.py` | `--attention-type ultrametric` |
| 3 | Tropical Geometry & Idempotent Algebra | `tropical_geometry_and_idempotent_algebra.py` | `tropical_attention_torch.py` | `--attention-type tropical` |
| 4 | Simplicial Complexes & Higher-Order Attention | `simplicial_complexes_and_higher_order_attention.py` | `simplicial_attention_torch.py` | `--attention-type simplicial` |
| 5 | Quaternion & Octonion Signal Flow | `octonionic_quaternionic_signal_flow.py` | `quaternion_attention_torch.py`, `octonion_attention_torch.py` | `--attention-type quaternion` / `octonion` |
| 6 | Ordinal Schedules & Well-Founded Optimization | `ordinal_schedules_and_well_founded_optimization.py` | `ordinal_scheduler.py` | `--scheduler-type ordinal` |
| 7 | Reversible Computation & Measure-Preserving Learning | `reversible_computation_and_measure_preserving_learning.py` | `reversible_block_torch.py` | `--attention-type reversible` |
| 8 | Iterated Function Systems & Fractal Memory | `iterated_function_systems_and_fractal_memory.py` | `fractal_attention_torch.py` | `--attention-type fractal` |
| 9 | Knot-Theoretic Programs & Braid Attention | `knot_theoretic_programs_and_braid_based_attention.py` | `braid_attention_torch.py` | `--attention-type braid` |
| 10 | Nonstandard Analysis & Hyperreal Training | `nonstandard_analysis_and_hyperreal_training.py` | *(integrated in GPT)* | -- |
| 11 | Surreal Numbers, Transseries & Scaling | `surreal_numbers_transseries_and_scaling.py` | `surreal_torch.py` | `--attention-type surreal` |

### Key Components (not per-framework)

| Component | Path | First Appeared |
|-----------|------|----------------|
| HOSS Optimizer (JAX) | `nanochat/hoss_opt.py` | 2025-11-20 |
| HOSS Optimizer (PyTorch) | `nanochat/hoss_opt_torch.py` | 2025-11-20 |
| Muon Optimizer | `nanochat/muon.py`, `nanochat/muon_jax.py` | 2025-11-19 |
| AdamW | `nanochat/adamw.py` | 2025-11-19 |
| GPT Model (PyTorch) | `nanochat/gpt.py` | 2025-11-19 |
| GPT Model (JAX) | `nanochat/gpt_jax.py` | 2025-11-19 |
| Synaptic Dynamics | `nanochat/synaptic.py` | 2025-11-19 |
| Training Script (PyTorch) | `nanochat/train.py` | 2025-11-20 |
| Training Script (JAX) | `nanochat/train_jax.py` | 2025-11-19 |
| Benchmarking CLI | `cli.py` (benchmark subcommand) | 2026-01-09 |
| CI/CD Workflows | `.github/workflows/` | 2026-01-17 |
