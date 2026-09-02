# Test package initialization.
#
# Typer forces Rich terminal rendering (ANSI color/style codes) whenever the
# GITHUB_ACTIONS, FORCE_COLOR, or PY_COLORS environment variables are set
# (typer/rich_utils.py: FORCE_TERMINAL). Tests assert on plain substrings of
# CLI output ("--cell cannot be combined ..."), which Rich then splits with
# escape sequences - the CI test job failed on exactly this while every local
# run passed. This package is imported before any test module imports `cli`
# (and therefore typer), so the opt-out lands before typer reads it.
import os

os.environ.setdefault("_TYPER_FORCE_DISABLE_TERMINAL", "1")
