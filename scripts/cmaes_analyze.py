from __future__ import annotations

"""CMA-ES result analysis & parameter sensitivity (bead model_guided_research-0wn).

Reads a completed (or in-progress) CMA-ES run produced by
``scripts/cmaes_phase1.py`` and computes, for the searched hyperparameters:

* **Sensitivity** -- Spearman (rank) and Pearson correlation of each parameter
  with the objective score, ranked by |Spearman|. Log10 params are correlated
  in log space (matching the search encoding).
* **Best candidate(s)** and the score distribution / signal-to-noise, so a
  flat (no-signal) search is called out rather than over-interpreted.
* A **sensitivity bar chart** and a **parameter×parameter correlation
  heatmap** (PNG), plus a machine-readable ``sensitivity.json`` and a
  human-readable ``report.md``.

It walks the canonical ``eval/gen_*/cand_*`` tree (``synaptic_config.json`` +
``seed_*/summary.json``), so it is independent of the ``progress.csv`` schema
and works on both old pilot runs and new multi-seed runs.

Example
-------
    uv run python scripts/cmaes_analyze.py --run-id phase1
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless / CPU-only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402
from scipy import stats  # noqa: E402

console = Console()

SCHEMA_VERSION = "mgr.cmaes.analysis.v1"


def _mean_tail(values: list[float], *, tail: int) -> float:
    if not values:
        return float("inf")
    tail = max(1, int(tail))
    window = values[-tail:]
    return float(sum(window) / len(window))


def _load_param_specs(run_dir: Path) -> list[dict[str, str]]:
    """Param names + kinds from run.json; falls back to config-key union."""
    run_json = run_dir / "run.json"
    if run_json.exists():
        spec = json.loads(run_json.read_text(encoding="utf-8")).get("param_space", {}).get("specs")
        if spec:
            return [{"name": s["name"], "kind": s.get("kind", "linear")} for s in spec]
    # fallback: union of keys across candidate configs (all treated as linear)
    names: set[str] = set()
    for cfg in run_dir.glob("eval/gen_*/cand_*/synaptic_config.json"):
        names |= set(json.loads(cfg.read_text(encoding="utf-8")).keys())
    return [{"name": n, "kind": "linear"} for n in sorted(names)]


def _collect_points(run_dir: Path, *, score_tail: int) -> list[dict[str, Any]]:
    """One record per (gen, cand): decoded params + score (mean over seeds)."""
    points: list[dict[str, Any]] = []
    for cand_dir in sorted(run_dir.glob("eval/gen_*/cand_*")):
        cfg_path = cand_dir / "synaptic_config.json"
        if not cfg_path.exists():
            continue
        decoded = json.loads(cfg_path.read_text(encoding="utf-8"))
        seed_scores: list[float] = []
        for seed_dir in sorted(cand_dir.glob("seed_*")):
            summ = seed_dir / "summary.json"
            if not summ.exists():
                continue
            losses = json.loads(summ.read_text(encoding="utf-8")).get("results", {}).get("losses", [])
            if losses:
                s = _mean_tail([float(x) for x in losses], tail=score_tail)
                if math.isfinite(s):
                    seed_scores.append(s)
        if not seed_scores:
            continue
        parts = cand_dir.name.split("_")
        gen_parts = cand_dir.parent.name.split("_")
        points.append(
            {
                "gen": int(gen_parts[-1]) if gen_parts[-1].isdigit() else -1,
                "cand": int(parts[-1]) if parts[-1].isdigit() else -1,
                "params": decoded,
                "score": float(np.mean(seed_scores)),
                "n_seeds": len(seed_scores),
            }
        )
    return points


def _encode_param(value: float, kind: str) -> float:
    if kind == "log10" and value > 0:
        return math.log10(value)
    return float(value)


def _sensitivity(points: list[dict[str, Any]], specs: list[dict[str, str]], *, enabled: bool) -> list[dict[str, Any]]:
    scores = np.array([p["score"] for p in points], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        name, kind = spec["name"], spec["kind"]
        col = np.array([_encode_param(float(p["params"].get(name, np.nan)), kind) for p in points], dtype=np.float64)
        ok = np.isfinite(col) & np.isfinite(scores)
        n = int(ok.sum())
        spear = pear = float("nan")
        # correlation is meaningless when the objective is flat (noise-level
        # spread), constant, or n<3 -- suppress it rather than report spurious
        # rankings that just fit numerical noise.
        if enabled and n >= 3 and np.std(col[ok]) > 0 and np.std(scores[ok]) > 0:
            spear = float(stats.spearmanr(col[ok], scores[ok]).statistic)
            pear = float(stats.pearsonr(col[ok], scores[ok]).statistic)
        rows.append(
            {
                "name": name,
                "kind": kind,
                "n": n,
                "spearman": None if math.isnan(spear) else spear,
                "pearson": None if math.isnan(pear) else pear,
                "abs_spearman": 0.0 if math.isnan(spear) else abs(spear),
            }
        )
    rows.sort(key=lambda r: r["abs_spearman"], reverse=True)
    return rows


def _param_corr_matrix(points: list[dict[str, Any]], specs: list[dict[str, str]]) -> tuple[list[str], np.ndarray]:
    names = [s["name"] for s in specs]
    mat = np.full((len(names), len(names)), np.nan, dtype=np.float64)
    cols = {
        s["name"]: np.array([_encode_param(float(p["params"].get(s["name"], np.nan)), s["kind"]) for p in points])
        for s in specs
    }
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            a, b = cols[ni], cols[nj]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 3 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                mat[i, j] = float(stats.spearmanr(a[ok], b[ok]).statistic)
    return names, mat


def _plot_sensitivity(rows: list[dict[str, Any]], out_path: Path, run_id: str) -> None:
    named = [(r["name"], r["spearman"]) for r in rows if r["spearman"] is not None]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(rows))))
    if named:
        labels = [n for n, _ in named]
        vals = [v for _, v in named]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Spearman corr( param , score )  (+ = higher param → worse loss)")
        ax.set_xlim(-1, 1)
    else:
        ax.text(0.5, 0.5, "no signal (flat objective)", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title(f"CMA-ES parameter sensitivity — {run_id}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_corr_heatmap(names: list[str], mat: np.ndarray, out_path: Path, run_id: str) -> None:
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(names), 1.0 + 0.6 * len(names)))
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title(f"Param×param Spearman — {run_id}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="CMA-ES result analysis & sensitivity (bead 0wn).")
    parser.add_argument("--run-id", type=str, required=True, help="CMA-ES run id under <artifacts>/cmaes/phase1/.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument(
        "--score-tail", type=int, default=3, help="Mean of last N losses used as score (match the search)."
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=1e-3,
        help="Min score std (in loss units) to treat the search as having usable signal. "
        "Below this, candidate differences are numerical noise and rankings are spurious.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None, help="Output dir (default: <artifacts>/cmaes/analysis/<run-id>)."
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    run_dir = artifacts_dir / "cmaes" / "phase1" / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"No such CMA-ES run: {run_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else (artifacts_dir / "cmaes" / "analysis" / args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_param_specs(run_dir)
    points = _collect_points(run_dir, score_tail=int(args.score_tail))
    if not points:
        raise SystemExit(f"No evaluable candidates found under {run_dir}/eval/")

    scores = np.array([p["score"] for p in points], dtype=np.float64)
    score_std = float(np.std(scores)) if len(scores) > 1 else 0.0
    score_range = float(scores.max() - scores.min()) if len(scores) else 0.0
    has_signal = score_std > float(args.signal_threshold)
    rows = _sensitivity(points, specs, enabled=has_signal)
    names, corr = _param_corr_matrix(points, specs)
    best = min(points, key=lambda p: p["score"])

    _plot_sensitivity(rows, out_dir / "sensitivity.png", args.run_id)
    _plot_corr_heatmap(names, corr, out_dir / "param_corr.png", args.run_id)

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "n_candidates": len(points),
        "score": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "std": score_std,
            "range": score_range,
            "has_signal": has_signal,
        },
        "best": {"gen": best["gen"], "cand": best["cand"], "score": best["score"], "params": best["params"]},
        "sensitivity": rows,
        "param_corr": {"names": names, "matrix": [[None if math.isnan(v) else v for v in r] for r in corr.tolist()]},
        "plots": {"sensitivity": "sensitivity.png", "param_corr": "param_corr.png"},
    }
    (out_dir / "sensitivity.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # report.md
    lines: list[str] = [
        f"# CMA-ES analysis — `{args.run_id}`\n",
        "Bead model_guided_research-0wn. Parameter sensitivity and score distribution "
        f"over `{len(points)}` evaluated candidates.\n",
        "## Score distribution\n",
        f"- min `{scores.min():.5f}` · max `{scores.max():.5f}` · "
        f"mean `{scores.mean():.5f}` · std `{score_std:.2e}` · range `{score_range:.2e}`",
        f"- **{'signal present' if has_signal else 'FLAT — no usable signal'}** "
        f"(std {'>' if has_signal else '≤'} threshold {args.signal_threshold:g})\n",
    ]
    if not has_signal:
        lines.append(
            "> The objective is flat across candidates: the per-config budget is too small to "
            "separate them. Sensitivity below is therefore not meaningful — increase steps/FLOPs "
            "or widen the search before trusting parameter rankings.\n"
        )
    lines.append("## Best candidate\n")
    lines.append(f"- gen `{best['gen']}` cand `{best['cand']}` · score `{best['score']:.5f}`")
    lines.append("\n| param | value |")
    lines.append("|---|---|")
    for s in specs:
        lines.append(f"| `{s['name']}` | `{best['params'].get(s['name']):.6g}` |")
    lines.append("\n## Parameter sensitivity (ranked by |Spearman|)\n")
    lines.append("| param | kind | n | Spearman | Pearson |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        sp = f"{r['spearman']:+.3f}" if r["spearman"] is not None else "—"
        pe = f"{r['pearson']:+.3f}" if r["pearson"] is not None else "—"
        lines.append(f"| `{r['name']}` | {r['kind']} | {r['n']} | {sp} | {pe} |")
    lines.append("\n_Positive Spearman ⇒ larger parameter correlates with **higher** (worse) loss._\n")
    lines.append(
        "## Plots\n- `sensitivity.png` — per-param Spearman bar chart\n- `param_corr.png` — param×param correlation heatmap\n"
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    # console
    table = Table(title=f"CMA-ES sensitivity — {args.run_id}", header_style="bold")
    table.add_column("param")
    table.add_column("Spearman", justify="right")
    table.add_column("Pearson", justify="right")
    for r in rows:
        sp = f"{r['spearman']:+.3f}" if r["spearman"] is not None else "—"
        pe = f"{r['pearson']:+.3f}" if r["pearson"] is not None else "—"
        table.add_row(r["name"], sp, pe)
    console.print(table)
    console.print(
        f"score: min={scores.min():.5f} max={scores.max():.5f} std={score_std:.2e} "
        f"({'signal' if has_signal else 'FLAT'})"
    )
    console.print(f"[bold green]done[/bold green] → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
