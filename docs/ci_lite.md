# Running the research checks with DSR

The repository's DSR manifest runs lint, formatting, the locked type environment,
fast and full tests, registry validation, CPU certificates, seven pipeline
scenarios, all 13 seeded demos, the Lean axiom audit and an installed-wheel check. Invoke it from a
clean checkout on a host with Python 3.13, uv and the pinned Lean toolchain.

```bash
DSR_REPOS_FILE="$PWD/dsr.yaml" dsr quality --tool model_guided_research --work-dir "$PWD" --check-timeout 18000
```

DSR writes the receipts, wheel, certificates, and scenario reports under its
quality log directory. Its exit status includes each configured check.

## Smaller checks during development

For a quick pre-push check, configure the existing
[`scripts/ci_lite.sh`](../scripts/ci_lite.sh) as a check in a local DSR profile.
For example, this profile runs the staged-file lint checks, type check and
selected tests from the working checkout.

```yaml
tools:
  mgr-staged:
    checks:
      - bash scripts/ci_lite.sh --staged
```

Set `DSR_REPOS_FILE` to the path where you saved the profile. Then run
`dsr quality --tool mgr-staged --work-dir "$PWD"` from the checkout.

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

The script also accepts `--no-tests` to omit pytest, `--typecheck` for the
repository type check and workflow YAML syntax, and `--all` for all Python
files and the full pytest suite. The canonical DSR profile at the top of
this page also exercises the complete research pipeline and the installed wheel.
