# CMA-ES Infra Plan for MGR (Objective + Distributed Evaluation)

Bead: `model_guided_research-ajh`

This doc scopes a **CMA-ES** integration for `model_guided_research/` focused on **reproducible**, **fixed-budget**
objective evaluations for:

1) **nanochat** training runs (PyTorch), and
2) a limited set of **math demos** (JAX) where a small number of continuous knobs exist.

No implementation is included in this bead; the goal is to remove ambiguity so the follow-up implementation beads
can be executed mechanically.

---

## Constraints (Project Rules)

- Python **3.13 only**.
- Dependency management via **`uv` + `pyproject.toml`** only (never `pip`).
- Console output should be **Rich-first** (tables, panels, progress, colored errors).
- No brittle “mass rewrite” scripts that modify many code files automatically.
- Avoid file proliferation: new code files only for genuinely new functionality (docs are fine).

---

## Definitions

- **Candidate**: a real-valued vector `x ∈ R^d` sampled by CMA-ES.
- **Decoder**: deterministic mapping `x → config/knobs` used by an objective.
- **Objective**: deterministic(ish) evaluation function returning a scalar **score** (lower is better).
- **Budget**: a fixed compute cap (preferably **FLOPs target**, otherwise `steps`/`tokens`/`seconds`).
- **Run**: one CMA-ES search instance identified by `run_id`.
- **Evaluation**: one objective call for a candidate `x` (may use multiple RNG seeds).

---

## Library Choice (CMA-ES)

Recommended: use a small, maintained “ask/tell” CMA-ES library and keep the outer loop on CPU.

Candidate libraries (implementation bead decides; doc recommendation):
- `cmaes` (lightweight ask/tell; good fit for batch/async evaluation).
- `cma` (feature-rich, heavier; also ask/tell).

Selection criteria:
- Pure Python + NumPy (no compiled deps).
- Clean serialization of state (or state can be reconstructed from `(mean, sigma, cov, rng)`).
- Supports parallel evaluation: `ask(n)` / `tell(pop, fitness)`.

---

## Objective API (Proposed)

All objectives should implement the same interface so the CMA-ES driver can be generic.

### `ObjectiveSpec`

- `name`: string identifier (e.g. `"nanochat_train_loss"`, `"demo_tropical_margin"`).
- `mode`: `"nanochat_train"` | `"demo_run"`.
- `metric`: describes what is minimized (e.g. `"val_ce"`, `"train_ce"`, `"bpb"`, `"negative_margin"`).
- `budget`: `BudgetSpec` (below).
- `base_config`: an immutable baseline config (nanochat config fields or demo args/config).
- `param_space`: ordered list of `ParamSpec` describing vector encoding.
- `eval_seeds`: list of per-candidate seeds (e.g. `[0]` or `[0, 1, 2]` for robust averaging).
- `device_policy`: `"cpu"` | `"cuda"` | `"auto"` plus per-worker GPU selection.

### `BudgetSpec`

Preferred (for fair A/B):
- `target_flops: float` (global compute budget per evaluation).

Fallbacks (when FLOPs estimate is unavailable):
- `max_steps: int`
- `max_tokens: int`
- `max_seconds: float`

### `ObjectiveResult` (Telemetry Schema)

Minimum fields to log (JSON-serializable):

- `run_id: str`
- `objective: str`
- `status: "ok" | "nan" | "oom" | "error" | "timeout"`
- `score: float` (the scalar CMA-ES minimizes)
- `seed: int`
- `x: list[float]` (raw candidate vector)
- `decoded_params: dict[str, float | int | str | bool]`
- `metrics: dict[str, float]` (losses, bpb, margins, timing, etc.)
- `budget: dict` (the resolved realized budget: steps/tokens/flops_est)
- `duration_s: float`
- `device: dict` (cpu/gpu name, cuda version, torch/jax versions if applicable)
- `git: dict` (commit, branch, dirty)
- `artifacts: dict[str, str]` (paths written for this evaluation)
- `error: str | null` (stack trace / summary)

Notes:
- Keep `score` and `metrics` **finite**; if the run explodes, return `status="nan"` and a large penalty score.
- Include enough metadata to reproduce the exact evaluation deterministically.

---

## Parameter Vector Encoding

CMA-ES operates in ℝ, so we standardize a **decoded parameter space** that avoids invalid values.

### `ParamSpec` (proposed)

Each element in `x` corresponds to one `ParamSpec`:

- `name`: canonical string key.
- `kind`: `"log10"` | `"linear"` | `"sigmoid01"` | `"tanh11"` | `"int"` (avoid categorical for CMA-ES phase 1).
- `bounds_x`: `(low, high)` bounds in the **search space** (the raw CMA-ES coordinates after any internal scaling).
- `decode(x_i) -> value`: deterministic.
- `encode(value) -> x_i`: for seeding / resuming.
- `apply_to_config(base_config, value) -> new_config`: pure function, no side effects.

### Recommended transforms

- **Positive scales** (`lr`, weight decay, eps, noise scales): use `log10`.
  - Example: `x_i ∈ [-6, -2]` → `lr = 10**x_i`.
- **Bounded fractions** (`dropout`, mixing weights): use `sigmoid01`.
  - Example: `x_i ∈ [-6, 6]` → `p = sigmoid(x_i)`.
- **Signed bounded** (rare): use `tanh11`.
  - Example: `x_i ∈ [-4, 4]` → `v = tanh(x_i)`.
- **Discrete ints**: avoid unless necessary; if needed, use `int(round(...))` with explicit clamp.

### Validity checks (must be fail-fast)

Decoder should raise on:
- out-of-range values after decoding,
- incompatible combinations (e.g. `n_head` divisibility constraints),
- parameters that would exceed memory constraints given `B,T,model_size`.

This is separate from NaN/oom handling in the objective run.

---

## FLOPs-Budget Harness (nanochat)

We already have model-side FLOPs estimates:
- `nanochat/gpt.py`: `GPT.estimate_flops()`
- `nanochat/gpt_synaptic.py`: `GPTSynaptic.estimate_flops()`

### Budget resolution formula (proposed)

Let:
- `f_tok = model.estimate_flops()` (estimated FLOPs per token for *training*; treated as an approximation).
- `tokens_per_step_global = global_batch_size * sequence_len`
  - where `global_batch_size = batch_size_per_rank * world_size` (DDP) or `batch_size` (single process).

Then:
- `steps = ceil(target_flops / (f_tok * tokens_per_step_global))`
- `max_steps` acts as a hard safety cap even when using `target_flops`.

Telemetry should log:
- `f_tok`, `tokens_per_step_global`, `steps`, and `flops_est = f_tok * tokens_per_step_global * steps`.

### Recommended evaluation metric for CMA-ES

For a fixed-budget objective, do **short training** and score with:
- `validation CE` on a fixed small batch stream, or
- `bpb` via `nanochat/loss_eval.py:evaluate_bpb` if token byte mapping is available.

Avoid using raw `train loss` only unless necessary, because it can be gamed by overfitting tiny batches.

---

## Seed Discipline (Critical)

We need repeatable objective evaluations and comparable search trajectories.

### Seeds to separate

- `search_seed`: RNG seed controlling CMA-ES sampling.
- `eval_seed`: RNG seed(s) for objective evaluation.

Recommended scheme:
- `eval_seed = hash32(search_seed, generation, candidate_index, eval_seed_index)`
- Evaluate each candidate on `k` seeds and use:
  - `score = mean(score_seed_i)` (default), and log variance.

### What to seed

nanochat:
- Python `random`
- NumPy RNG
- `torch.manual_seed` and (if CUDA) `torch.cuda.manual_seed_all`
- Dataloader split seed (must be pinned; see bead `model_guided_research-wiz`)

demos (JAX):
- `jax.random.PRNGKey`
- Any environment variables controlling demo branches must be recorded.

---

## Distributed Evaluation Strategy (Multi-GPU)

Primary goal: evaluate a **population** in parallel across available GPUs without entangling candidates.

### Recommended architecture

- **Coordinator (CPU)**:
  - Runs CMA-ES outer loop (`ask`/`tell`), maintains state, writes run-level telemetry.
- **Workers (one process per GPU)**:
  - Each worker pulls evaluation jobs from a queue, sets `CUDA_VISIBLE_DEVICES` (or uses `torch.device(i)`),
    runs objective evaluation, and returns `ObjectiveResult`.

Key properties:
- Workers are **stateless** beyond caching dataset/tokenizer; each job gets its full decoded config.
- Failures (OOM/NAN) return penalty scores but do not crash the coordinator.
- Allows **async** evaluation: coordinator can `tell` once all population members return, or implement
  asynchronous variants (phase 2).

### DDP inside an evaluation (optional, not default)

Only consider if:
- one evaluation needs multiple GPUs for wall-clock reasons, and
- the objective budget is large enough to amortize DDP setup cost.

Default: **single-process, single-GPU evaluations**.

---

## Checkpointing + Resume

We need resumability for:
- long searches,
- preemptible GPU nodes,
- iterative refinement.

### What to checkpoint

At minimum, per generation:
- CMA-ES state:
  - `mean`, `sigma`, `cov` (or equivalent internal representation),
  - `rng_state`,
  - `generation`, `best_score`, `best_x`,
  - `population_size`, `param_space_hash`.
- A ledger of completed evaluations:
  - candidate vectors, decoded params, per-seed scores, aggregated score.

### Atomic writes (must)

Use atomic file replace for state/ledger updates:
- write to `*.tmp`, fsync, rename → final path.

---

## Artifacts Layout (Current)

The unified artifacts conventions live in `artifacts/README.md`. CMA-ES runs should follow that structure and
write under:

`artifacts/cmaes/<run_id>/`

Suggested structure:

- `run.json` — immutable run spec (objective, budget, param space, seeds, git hash, environment summary)
- `state/`
  - `cma_state.json` or `cma_state.npz`
  - `ledger.jsonl` (append-only evaluation summaries; do not edit manually in code reviews)
- `eval/`
  - `gen_0000/`
    - `cand_000/`
      - `result.json` (ObjectiveResult)
      - `stdout.log` / `stderr.log`
      - `config.json` (fully resolved config actually used)
      - optional: `checkpoint/` (only when explicitly requested, e.g. best-so-far)
- `tables/`
  - `best.md` (human-readable summary table)
  - `progress.csv` (gen, best, mean, sigma, walltime)

### Telemetry schema (kt8)

For any *training-like* evaluation (nanochat objective, demo objective, proxy runs), prefer emitting a
`summary.json` that follows the minimal telemetry shape in `artifacts/README.md` under:

> “Telemetry Schema (model_guided_research-kt8)” (`schema_version: "mgr.telemetry.v1"`).

You can still keep CMA-ES-specific files (`run.json`, `ledger.jsonl`, `result.json`) — the point is that
downstream tools (dashboards, regressions, CMA-ES analysis) should be able to read a consistent `summary.json`
without needing special-case parsers per objective.

Important:
- Avoid symlinks for portability; copy “best” artifacts into a `best/` directory if needed.
- Always record **dataset identity** (hash/size/split seed) in `run.json` and each `result.json` (bead `wiz`).

---

## Integration Points in This Repo (Implementation Targets)

### nanochat training objective

Likely new implementation components:
- A small objective module inside `nanochat/` (new file is justified: genuinely new functionality).
- Reuse:
  - `nanochat/train.py` logic as a starting point, but refactor into a callable function that returns metrics.
  - `nanochat/loss_eval.py:evaluate_bpb` for a tokenizer-robust metric.
  - `nanochat/checkpoint_manager.py` for optional checkpoint save/load.
  - `nanochat/report.py` for environment/git metadata capture (or a slimmed subset for JSON).

### demo objective

Approach:
- Call demo functions directly (preferred) or use the CLI entrypoints programmatically.
- Use existing `--export-json` artifact structure from `cli.py` as the objective’s telemetry substrate.

---

## Follow-Up Beads (What This Unblocks)

This design doc should unblock:
- `model_guided_research-0hu` — objective validation tests (determinism, toy function, failure modes).
- `model_guided_research-wiz` — dataset snapshot/split pinning to prevent drift across evaluations.
- `model_guided_research-ybp` — demo-target CMA-ES parameter sets/bounds (depends on stable objective patterns).

---

## Implementation Checklist (Next Steps)

1) Add chosen CMA-ES dependency via `uv` (no `pip`).
2) Implement `ParamSpec` + decoder/validator.
3) Implement one objective: nanochat fixed-budget training returning a scalar score + telemetry JSON.
4) Implement the coordinator/worker pool with Rich progress tables.
5) Add resume-safe checkpointing (`cma_state` + ledger).
6) Add validation tests and a tiny toy objective (Rosenbrock) to ensure CMA-ES loop correctness.
