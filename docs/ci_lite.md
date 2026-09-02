# CI-lite — fast local pre-push gate

**Bead:** `model_guided_research-c6v`
**Script:** `scripts/ci_lite.sh` (uv-only, venv-respecting, no pip)

A one-command gate that mirrors the AGENTS.md constraints so you can check work
before `git push` without waiting on a full CI run. No GitHub Actions required.

## Usage

```bash
scripts/ci_lite.sh            # changed files (working tree vs HEAD) — default
scripts/ci_lite.sh --staged   # staged files only (run right before commit)
scripts/ci_lite.sh --all      # whole repo: full ruff + full pytest
scripts/ci_lite.sh --no-tests # lint/format/ubs only (skip pytest)
scripts/ci_lite.sh --help
```

## What it runs

| Stage | Command | Notes |
|---|---|---|
| 1. Lint | `uv run ruff check <files>` | config in `pyproject.toml` (`[tool.ruff.lint]`) |
| 2. Format | `uv run ruff format --check <files>` | non-mutating; fails if a file would reformat |
| 3. Bugs | `ubs --diff` / `--staged` / `.` | skipped with a note if `ubs` not installed (see `docs/ubs_usage.md`) |
| 4. Types | `uv run ty check` (repository-wide, exact extras set) | the same gate CI's typecheck job runs; `--typecheck` runs only this stage |
| 5. Tests | fast subset, or `uv run pytest -q` under `--all` | exits non-zero on first failure |

The **fast test subset** is the deterministic, CPU-cheap core-contract gate
(~1 min, 73 tests): attention goldens, `estimate_flops`, parameterization, and
the hypotheses/theorem registries. The slow suites
(`test_mathematical_properties`, `test_adjudicate`, `test_practical_utility`,
`test_demos`, `test_checkpoint_resume`) run only under `--all`.

## Conventions honored

- **uv only, no pip; venv-respecting** — every stage is `uv run …`.
- **FlexAttention safety** — flex tests self-skip when `torch<2.5` / no CUDA, so
  the gate is green on a CPU-only box (it does not require a GPU).
- **Quality bar** — matches the AGENTS.md "Landing the Plane" gates
  (`ruff check`, tests) plus the UBS golden rule (`ubs --diff` before commit).
- **Shared working tree** — `changed` mode scans the *whole* working-tree diff,
  so in this repo's multi-agent setup it will also surface other agents'
  uncommitted edits. Use `--staged` to scope strictly to what you are about to
  commit.

## Recommended flow

```bash
# while iterating
scripts/ci_lite.sh --no-tests        # fast lint/format/bug pass on your edits
# before committing
git add <your files>
scripts/ci_lite.sh --staged          # lint + bugs scoped to staged files
# before pushing a larger change
scripts/ci_lite.sh                   # changed files + fast tests
# occasionally / before a release
scripts/ci_lite.sh --all             # full ruff + full pytest
```

Note: this is a convenience gate, not a replacement for the full suite or the
research-loop provenance checks (`scripts/e2e_pipeline.py`, `mgr adjudicate`).
</content>
