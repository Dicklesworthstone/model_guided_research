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

### Cheap-proxy validation gate

No short-run score may be used as a CMA-ES objective merely because it is
cheaper. Before a search spends its budget, calibrate the proxy on the exact
model, dataset fingerprint, candidate decoder, and seed policy that the search
will use:

1. Sample at least six candidates from a fixed search seed.
2. Evaluate the same candidate configurations and evaluation seeds at the
   proposed proxy budget and the reference budget.
3. Require all of the following before enabling the proxy:
   - the reference scores have detectable spread (`std > 1e-3` for the current
     CE scale),
   - Spearman rank correlation is at least `0.80`,
   - the proxy retains every member of the reference top two, and
   - median proxy wall time is no more than half the median reference wall
     time.
4. Record the candidate scores, data fingerprint, step counts, timings, and
   pass/fail result. A failed gate is a **no-go**, not permission to tune the
   thresholds after seeing the cohort.

The calibration must be repeated when the model shape, dataset, metric,
candidate space, or seed aggregation changes. This is intentionally stricter
than checking loss-curve similarity: identical early loss prefixes prove
reproducibility, but do not prove that the early ranking predicts the final
ranking.

#### CPU calibration result (bead `model_guided_research-2xy`)

The first preregistered calibration tested the existing Phase-1 objective:
mean of the final three training losses. It used six candidates with
`search_seed=17`, `eval_seed=123`, a pinned two-shard dataset fingerprint
`9ed31c98e6496157db24586949e5d15c9e46a5c6253d24dfc817b486fb8dc415`, and
this CPU smoke configuration:

```text
n_layer=2, n_head=2, n_kv_head=2, n_embd=64
sequence_len=64, batch_size=4, warmup_steps=0
proxy=1e9 FLOPs (22 steps), reference=4e9 FLOPs (87 steps)
```

| candidate | proxy score | reference score | proxy seconds | reference seconds |
|---:|---:|---:|---:|---:|
| 0 | 10.507860 | 8.277376 | 18.04 | 16.29 |
| 1 | 10.508038 | 8.276974 | 7.09 | 33.75 |
| 2 | 10.507350 | 8.276485 | 7.64 | 23.26 |
| 3 | 10.508210 | 8.276564 | 12.65 | 41.61 |
| 4 | 10.508458 | 8.277685 | 8.12 | 21.30 |
| 5 | 10.508067 | 8.277201 | 10.02 | 23.04 |

The candidate JSON objects matched exactly across runs, all six evaluations
completed, and every 22-step proxy loss curve exactly matched the corresponding
prefix of its 87-step reference curve (`max_abs_delta = 0`). The proxy was
cheap enough: median duration was `9.07 s` versus `23.15 s` (`39.2%`). It was
not predictive enough:

- reference-score standard deviation: `0.000427` (**fails** the `1e-3` signal
  floor),
- Spearman rank correlation: `0.486` (**fails** `0.80`), and
- proxy top two `{2, 0}` versus reference top two `{2, 3}` (**fails** complete
  retention).

**Verdict: reject the raw short-training-loss proxy.** The failure is not
nondeterminism; the underlying Phase-1 train-loss objective is too flat at this
calibration scale to support candidate ranking. Do not launch more CMA-ES
generations with this proxy. The next calibration must first replace the score
with a fixed validation metric (validation CE or BPB), verify that the
reference cohort has signal, and then apply the same gate above.

#### Validation-CE calibration result (bead `model_guided_research-2c8j`)

The successor harness defaults to `results.val_ce_final`, records
`objective.metric`, `val_interval`, and `val_batches` in `run.json`, restores
them as immutable resume identity, and treats missing/non-finite validation
telemetry as a failed candidate rather than falling back to train loss.

Before evidence, an independent cohort was frozen with `search_seed=23`,
`eval_seed=321`, the same six-candidate/CPU/model coordinates as the first
calibration, and the same dataset fingerprint
`9ed31c98e6496157db24586949e5d15c9e46a5c6253d24dfc817b486fb8dc415`.
Each candidate received exactly one two-batch endpoint validation: step 22 for
the 1e9-FLOP proxy and step 87 for the 4e9-FLOP reference. The complete local
artifacts are retained at `/data/tmp/mgr-cma-valce-OBhfM6`.

| candidate | proxy val CE | reference val CE | proxy seconds | reference seconds |
|---:|---:|---:|---:|---:|
| 0 | 10.494027 | 8.381545 | 9.81 | 30.47 |
| 1 | 10.494459 | 8.376032 | 7.67 | 33.68 |
| 2 | 10.493962 | 8.380633 | 20.99 | 20.75 |
| 3 | 10.494921 | 8.376671 | 8.86 | 46.51 |
| 4 | 10.494877 | 8.376869 | 15.55 | 16.66 |
| 5 | 10.494871 | 8.378277 | 13.16 | 23.76 |

All twelve evaluations completed with finite scores; candidate JSON objects
matched exactly across arms; both run specs recorded the expected dataset
digest; and every summary contained exactly one validation measurement at its
registered endpoint. The frozen gates resolved as follows:

- reference-score sample standard deviation: `0.002272` (**passes** the
  `1e-3` signal floor),
- Spearman rank correlation: `-0.600` (**fails** `0.80`),
- proxy top two `{2, 0}` versus reference top two `{1, 3}`: overlap `0/2`
  (**fails** complete retention), and
- median duration `11.48 s` versus `27.11 s`, ratio `42.3%` (**passes** the
  `50%` cost ceiling).

**Verdict: the validation objective is operational, but reject the 1e9-FLOP
validation proxy.** It is cheap and the 4e9-FLOP reference has detectable
candidate signal, yet the early ranking points in the wrong direction. Per the
preregistered stopping rule, do not tune thresholds, seeds, or cadence on this
cohort and do not launch CMA-ES generations at this rung. Any intermediate
proxy budget must be justified and calibrated as a new pre-evidence cohort.

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
  - exact `rng_state` (the next sampled population must match after resume),
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
  - `optimizer_state.json` (strict non-executable schema; no pickle)
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
