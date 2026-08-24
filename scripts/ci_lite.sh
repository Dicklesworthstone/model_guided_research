#!/usr/bin/env bash
# CI-lite: a fast, local pre-push gate that mirrors the AGENTS.md constraints
# (bead model_guided_research-c6v). uv-only, venv-respecting, no pip.
#
# Modes:
#   scripts/ci_lite.sh            # changed files (working tree vs HEAD) — default
#   scripts/ci_lite.sh --staged   # staged files only (use right before commit)
#   scripts/ci_lite.sh --all      # whole repo: full ruff + full pytest
#   scripts/ci_lite.sh --no-tests # skip pytest (lint/format/ubs/ty still run)
#   scripts/ci_lite.sh --typecheck # full-repo ty gate only
#
# Stages: 1) ruff check  2) ruff format --check  3) ubs (if installed)
#         4) ty  5) fast pytest subset (skipped with --no-tests;
#         full suite with --all)
#
# Exit non-zero on the first failing stage. FlexAttention tests skip themselves
# cleanly when torch<2.5 / no CUDA, so this is safe on a CPU-only box.
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="changed"     # changed | staged | all
RUN_TESTS=1
TYPECHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --staged)   MODE="staged" ;;
    --all)      MODE="all" ;;
    --no-tests) RUN_TESTS=0 ;;
    --typecheck) TYPECHECK_ONLY=1 ;;
    -h|--help)  sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Fast, deterministic, CPU-cheap tests that gate the core contracts. The slow
# suites (test_mathematical_properties, test_adjudicate, test_practical_utility,
# test_demos, test_checkpoint_resume) run only under --all.
FAST_TESTS=(
  tests/test_attention_core_goldens.py
  tests/test_estimate_flops.py
  tests/test_parameterization.py
  tests/test_hypotheses_registry.py
  tests/test_theorem_registry.py
)

# Resolve the Python file set for the lint stages.
py_files() {
  case "$MODE" in
    staged)  git diff --cached --name-only --diff-filter=ACMR -- '*.py' ;;
    changed) git diff --name-only --diff-filter=ACMR -- '*.py';
             git ls-files --others --exclude-standard -- '*.py' ;;
    all)     echo "." ;;
  esac
}

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$1"; }

check_ty() {
  uv run --locked --exact \
    --extra dev --extra server --extra viz --extra tokenizer-train \
    ty check
}

if [[ "$TYPECHECK_ONLY" -eq 1 ]]; then
  step "ty check (repository)"
  check_ty
  exit 0
fi

mapfile -t FILES < <(py_files | sort -u)
if [[ "$MODE" != "all" && ${#FILES[@]} -eq 0 ]]; then
  echo "no changed Python files; nothing to lint."
else
  TARGET=( "${FILES[@]}" )
  [[ "$MODE" == "all" ]] && TARGET=( . )

  step "ruff check (${MODE})"
  uv run ruff check "${TARGET[@]}"

  step "ruff format --check (${MODE})"
  uv run ruff format --check "${TARGET[@]}"

  if command -v ubs >/dev/null 2>&1; then
    step "ubs (${MODE})"
    case "$MODE" in
      staged)  ubs --staged . ;;
      changed) ubs --diff . ;;
      all)     ubs . ;;
    esac
  else
    echo "ubs not installed; skipping (see docs/ubs_usage.md)."
  fi
fi

step "ty check (repository)"
check_ty

if [[ "$RUN_TESTS" -eq 1 ]]; then
  if [[ "$MODE" == "all" ]]; then
    step "pytest (full suite)"
    uv run pytest -q
  else
    step "pytest (fast subset)"
    uv run pytest -q "${FAST_TESTS[@]}"
  fi
fi

printf '\n\033[1;32mci-lite OK (%s)\033[0m\n' "$MODE"
