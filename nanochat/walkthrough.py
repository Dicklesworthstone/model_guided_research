"""Pedagogical walkthrough mode for nanochat (bead 9f1).

A guided, off-by-default narration that explains the *mathematics* as it runs —
what the Cayley step does, why ultrametric routing is sub-quadratic, what a
tropical margin means — tied to the actual equations and variables, with links
into ``markdown_documentation/``. Off by default and zero-overhead when not
invoked (it is a separate entry point, not a hook in the hot path).

Two explicit modes:

* ``run`` -- narrate a real **nanochat mini-run**: build a tiny model for a
  mechanism, walk the forward pipeline (embedding → norm → RoPE → attention →
  residual → MLP → logits → cross-entropy) with live shapes/values and the
  mechanism's interpretability observable, then take a few training steps and
  explain loss / gradients / the optimizer update.
* ``demo`` -- a conceptual **framework walkthrough**: step through a mechanism's
  core math with small live numeric illustrations (additive coupling is exactly
  invertible, max-plus vs softmax, the strong triangle inequality, …).

```bash
python -m nanochat.walkthrough run  --attention reversible
python -m nanochat.walkthrough demo --topic tropical
```

The narration helpers are reusable: a JAX/torch demo can gate a call on
``walkthrough_enabled()`` (``MGR_WALKTHROUGH=1``) to narrate itself in place.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

DOC_ROOT = "markdown_documentation"


def _console() -> Any:
    from rich.console import Console

    return Console()


def _vec(values: Any, prec: int = 2) -> str:
    """Format a numeric vector with parentheses (square brackets are rich markup)."""
    return "(" + ", ".join(f"{float(v):.{prec}f}" for v in values) + ")"


def walkthrough_enabled(env: dict[str, str] | None = None) -> bool:
    """True when MGR_WALKTHROUGH is set — the env gate for demos to self-narrate."""
    import os

    e = env if env is not None else os.environ
    return str(e.get("MGR_WALKTHROUGH", "")).lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Narration registry                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class MechanismNote:
    """Per-mechanism teaching notes: the idea, the load-bearing equations, the
    interpretability observable, and the doc to read next."""

    key: str
    title: str
    idea: str
    doc: str  # filename under markdown_documentation/
    steps: list[tuple[str, str, str]] = field(default_factory=list)  # (label, equation, why)
    observable: str = ""

    def doc_path(self) -> str:
        return f"{DOC_ROOT}/{self.doc}"


MECHANISM_NOTES: dict[str, MechanismNote] = {
    "standard": MechanismNote(
        key="standard", title="Softmax attention (baseline)",
        idea="Content-based mixing: each token's query scores every key, softmax turns scores into a convex mixture of values.",
        doc="../README.md",
        steps=[
            ("scores", "S = Q Kᵀ / √d_head", "Dot-product similarity; the √d scale keeps logits O(1) so softmax doesn't saturate."),
            ("mask", "S ← S + causal_mask", "Future keys get −∞ so a position can only attend to its past."),
            ("weights", "A = softmax(S)", "Rows are probability distributions over keys (the attention map)."),
            ("mix", "Y = A V", "Each token becomes a convex combination of value vectors."),
        ],
        observable="per-head attention entropy H(A) — low = a selective/induction head, high = a broad/mixing head.",
    ),
    "tropical": MechanismNote(
        key="tropical", title="Tropical (max-plus) attention",
        idea="Replace (+,×) with (max,+): attention becomes a piecewise-linear selection with a certifiable winner.",
        doc="tropical_geometry_and_idempotent_algebra.md",
        steps=[
            ("scores", "S_ij = ⟨q_i, k_j⟩ + b_j", "Additive scores in the max-plus semiring (⊗ = +, ⊕ = max)."),
            ("route", "y_i = argmax_j S_ij", "The 'sum' is a max: each token routes to its single best key (a hard, interpretable route)."),
            ("margin", "γ_i = S_(1) − S_(2)", "Gap between best and runner-up. γ/2 is a robustness certificate: perturbations below it can't flip the route."),
            ("maslov", "max_β(a,b) = (1/β)·log(e^{βa}+e^{βb})", "Maslov dequantization: β→∞ recovers exact max (tropical), finite β is a smooth surrogate."),
        ],
        observable="runner-up margin γ per head — large = confident certifiable route, ~0 = a tie on the PL decision boundary.",
    ),
    "ultrametric": MechanismNote(
        key="ultrametric", title="Ultrametric / p-adic attention",
        idea="Distance obeys the STRONG triangle inequality d(x,z) ≤ max(d(x,y),d(y,z)); routing by longest common prefix is tree-structured and sub-quadratic.",
        doc="ultrametric_worlds_and_p_adic_computation.md",
        steps=[
            ("digits", "x ↦ (d_1,…,d_K)", "Hash each token to K base-p digits (a path in a depth-K p-ary tree)."),
            ("lcp", "ℓ(x,y) = longest common prefix", "Similarity = shared prefix length; nearness is hierarchical, not Euclidean."),
            ("route", "attend within the deepest shared bucket", "O(N log N): bucketed prefix lookup instead of an N×N score matrix."),
            ("ultra", "d(x,z) ≤ max(d(x,y), d(y,z))", "Every triangle is isosceles — the geometry of tries/taxonomies."),
        ],
        observable="bucket occupancy — how tokens distribute across the prefix tree (balanced vs collapsed).",
    ),
    "reversible": MechanismNote(
        key="reversible", title="Reversible / measure-preserving block",
        idea="Additive coupling is exactly invertible, so activations need not be stored — recompute them in the backward pass (O(1) activation memory).",
        doc="reversible_computation_and_measure_preserving_learning.md",
        steps=[
            ("split", "x = [x₁, x₂]", "Split channels in half."),
            ("forward", "y₁ = x₁ + F(x₂);  y₂ = x₂ + G(y₁)", "Each half updates using the other — an invertible coupling."),
            ("inverse", "x₂ = y₂ − G(y₁);  x₁ = y₁ − F(x₂)", "Exact inverse: no activations stored, recompute on demand."),
            ("volume", "det(∂y/∂x) = 1", "Volume-preserving (Liouville): the map is a measure-preserving diffeomorphism."),
        ],
        observable="shadow energy / det≈1 — conservation diagnostics of the (symplectic) flow.",
    ),
    "gauge": MechanismNote(
        key="gauge", title="Matrix-exponential gauge block",
        idea="Parameterize layer maps as exp(A) of structured generators: skew→rotation (orthogonal), symmetric→scaling (SPD) — stable by construction.",
        doc="matrix_exponential_gauge_learning.md",
        steps=[
            ("generator", "A ∈ 𝔤  (Lie algebra)", "Learn the infinitesimal generator, not the matrix directly."),
            ("exp", "U = exp(A)", "The exponential map sends the algebra to the group: skew A ⇒ orthogonal U (‖Ux‖=‖x‖)."),
            ("cayley", "U = (I−A)(I+A)⁻¹", "Cayley transform: a rational exact-orthogonal parameterization (no series truncation)."),
            ("transport", "parallel transport with cumulative gauge field", "Curvature bounds give provable gradient stability."),
        ],
        observable="per-block curvature / rotation angle — geometric health of the transport.",
    ),
    "quaternion": MechanismNote(
        key="quaternion", title="Quaternion (ℍ) attention",
        idea="4-D hypercomplex numbers encode 3-D rotations; the Hamilton product composes them with 4× parameter sharing.",
        doc="octonionic_quaternionic_signal_flow.md",
        steps=[
            ("number", "q = w + xi + yj + zk", "i²=j²=k²=ijk=−1: associative, non-commutative."),
            ("rotor", "y = q ⊗ (k̄ ⊗ v)", "A rotor gate: rotate values by a learned quaternion (norm-preserving)."),
            ("share", "one quaternion = 4 real DOF", "Weight sharing across the 4 components — parameter efficiency."),
        ],
        observable="value-norm preservation ‖y‖≈‖v‖ — rotations don't change magnitude.",
    ),
    "octonion": MechanismNote(
        key="octonion", title="Octonion (𝕆) attention",
        idea="8-D non-associative division algebra; explicit parenthesization makes (ab)c ≠ a(bc) a usable structural prior.",
        doc="octonionic_quaternionic_signal_flow.md",
        steps=[
            ("number", "Cayley–Dickson double of ℍ → 𝕆", "8-D, alternative, NON-associative — the largest normed division algebra (Hurwitz)."),
            ("mix", "y = (q ⊗ k) ⊗ v  vs  q ⊗ (k ⊗ v)", "The two bracketings differ; the model learns which association to use."),
        ],
        observable="associator ‖(ab)c − a(bc)‖ — the non-associativity the layer exploits.",
    ),
    "braid": MechanismNote(
        key="braid", title="Braid / knot-theoretic attention",
        idea="Tokens are strands; attention applies braid-group crossings σᵢ. Topological invariants are robust to deformation.",
        doc="knot_theoretic_programs_and_braid_based_attention.md",
        steps=[
            ("generators", "B_n = ⟨σ₁,…,σ_{n−1}⟩", "Each σᵢ swaps adjacent strands with an over/under crossing."),
            ("artin", "σᵢσⱼ = σⱼσᵢ for |i−j|≥2", "Far-apart crossings commute; adjacent ones satisfy the braid relation σᵢσ_{i+1}σᵢ = σ_{i+1}σᵢσ_{i+1}."),
            ("ybe", "R₁₂R₁₃R₂₃ = R₂₃R₁₃R₁₂", "Yang–Baxter consistency makes the crossing schedule order-independent."),
        ],
        observable="braid charges Q1 (mass defect), Q2 (relation residual) — how well invariants are conserved.",
    ),
    "simplicial": MechanismNote(
        key="simplicial", title="Simplicial / higher-order attention",
        idea="Go beyond pairwise: aggregate over edges (1-simplices) and triangles (2-simplices) for genuine k-way interactions.",
        doc="simplicial_complexes_and_higher_order_attention.md",
        steps=[
            ("complex", "vertices ⊂ edges ⊂ triangles", "Build a simplicial complex over tokens."),
            ("hodge", "Δ_k = ∂ᵀ∂ + ∂∂ᵀ", "Higher Laplacians; Hodge decomposition = gradient ⊕ curl ⊕ harmonic."),
            ("aggregate", "mix 1-hop (edges) and 2-hop (triangles)", "Triangles capture 3-body structure a pairwise map cannot."),
        ],
        observable="Hodge spectrum — the cycle/void structure the layer sees.",
    ),
    "surreal": MechanismNote(
        key="surreal", title="Surreal / transseries scaling",
        idea="Carry magnitude and direction separately on a log scale, so infinitely large and small scales coexist.",
        doc="surreal_numbers_transseries_and_scaling.md",
        steps=[
            ("decompose", "w = exp(s)·v̂,  v̂ = v/‖v‖", "Scale s (log-magnitude) and unit direction v̂."),
            ("dominance", "compare by leading scale first", "Transseries dominance: the biggest scale wins, then the next — exact asymptotics."),
        ],
        observable="scale spectrum s — the dynamic range the representation spans.",
    ),
    "fractal": MechanismNote(
        key="fractal", title="IFS / fractal memory attention",
        idea="Self-similar addressing via an iterated function system: an m-ary tree of contraction maps gives hierarchical, recursive memory.",
        doc="iterated_function_systems_and_fractal_memory.md",
        steps=[
            ("ifs", "H(S) = ⋃ᵢ fᵢ(S)", "Hutchinson operator; the attractor is the fixed point of contractions."),
            ("route", "soft m-ary path (depth d)", "Address memory by a differentiable path through the tree."),
            ("dim", "capacity ~ fractal dimension", "Self-similarity across scales = recursive structure."),
        ],
        observable="path occupancy — which branches of the tree are used.",
    ),
}


PIPELINE_STAGES: list[tuple[str, str, str]] = [
    ("embed", "h = Wₑ[idx]", "Look up a learned vector per token id."),
    ("norm", "ĥ = h / RMS(h)", "RMSNorm: scale to unit root-mean-square (no mean subtraction) for stable scale."),
    ("rope", "q,k ← RoPE(q,k, pos)", "Rotary position embedding: rotate q,k by an angle ∝ position so dot-products encode relative offset."),
    ("attention", "y = Attention(ĥ)", "The mechanism-specific mixing step (see below)."),
    ("residual", "h ← h + y", "Residual add: the block learns a correction to the stream."),
    ("mlp", "h ← h + W₂·relu(W₁·norm(h))²", "Per-token ReLU² MLP (or the tropical max-plus FFN)."),
    ("logits", "z = softcap·tanh(W_lm·norm(h) / softcap)", "Project to vocabulary; tanh-softcap keeps logits bounded."),
    ("loss", "L = cross_entropy(z, targets)", "Negative log-likelihood of the next token."),
]


# --------------------------------------------------------------------------- #
# run mode: live nanochat mini-run                                            #
# --------------------------------------------------------------------------- #


def narrate_run(
    attention_type: str = "standard",
    *,
    device: str = "cpu",
    seed: int = 0,
    steps: int = 3,
    console: Any = None,
) -> dict[str, Any]:
    """Narrate a real nanochat mini-run for ``attention_type``."""
    import torch
    from rich.markup import escape
    from rich.panel import Panel
    from rich.text import Text

    from nanochat.viz import build_probe_model, collect_state, sample_batch

    console = console or _console()
    note = MECHANISM_NOTES.get(attention_type)

    title = note.title if note else f"{attention_type} attention"
    console.rule(f"[bold cyan]Walkthrough · {title}")
    if note:
        console.print(Panel(
            Text.assemble(("Idea: ", "bold"), (note.idea, "")),
            border_style="cyan", title="the big picture",
            subtitle=f"[dim]read more → {note.doc_path()}[/dim]",
        ))

    # 1) Build a tiny model and explain the config.
    model, meta = build_probe_model(
        attention_type, device=device, seed=seed, n_layer=2, n_head=4,
        n_kv_head=4 if attention_type != "reversible" else 2, n_embd=64,
        sequence_len=32, vocab_size=128,
    )
    cfg = meta["config"]
    n_params = sum(p.numel() for p in model.parameters())
    console.print(
        f"[green]model[/green]: {cfg['n_layer']} layers × {cfg['n_head']} heads × d={cfg['n_embd']} "
        f"→ [bold]{n_params:,}[/bold] params  [dim](seed {seed}, device {device})[/dim]\n"
    )

    # 2) Walk the forward pipeline.
    console.print("[bold]Forward pass — the math, stage by stage:[/bold]")
    idx, _ = sample_batch(text=None, batch_size=2, seq_len=32, vocab_size=128, seed=seed, device=device)
    for label, equation, why in PIPELINE_STAGES:
        marker = "  ↳ " if label not in {"attention"} else "  ★ "
        console.print(f"{marker}[bold yellow]{label:9s}[/bold yellow] [cyan]{escape(equation)}[/cyan]")
        console.print(f"      [dim]{escape(why)}[/dim]")
        if label == "attention" and note:
            for slabel, seq, swhy in note.steps:
                console.print(f"        [magenta]{slabel:8s}[/magenta] [cyan]{escape(seq)}[/cyan]  [dim]{escape(swhy)}[/dim]")

    # 3) Live observable from a real forward.
    diag = collect_state(model, idx)
    console.print("\n[bold]Live observable on this batch:[/bold]")
    if diag.has_entropy():
        flat = [v for row in diag.entropy_layer_head for v in row]
        console.print(
            f"  per-head attention entropy ≈ [bold]{sum(flat) / len(flat):.3f}[/bold] nats "
            f"[dim](range {min(flat):.2f}–{max(flat):.2f}; log(32)≈{__import__('math').log(32):.2f} is uniform)[/dim]"
        )
    if diag.has_margins():
        flat = [v for row in diag.margin_layer_head for v in row]
        console.print(f"  tropical runner-up margin γ ≈ [bold]{sum(flat) / len(flat):.4f}[/bold] [dim](larger = more certifiable routes)[/dim]")
    if note and note.observable:
        console.print(f"  [dim]what to watch: {escape(note.observable)}[/dim]")

    # 4) A few training steps — narrate loss / gradient / update.
    console.print("\n[bold]Training — why the loss moves:[/bold]")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    targets = idx.clone()
    losses: list[float] = []
    for s in range(max(1, int(steps))):
        opt.zero_grad(set_to_none=True)
        loss = model(idx, targets=targets)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
        opt.step()
        losses.append(float(loss.detach()))
        console.print(f"  step {s}: L = [bold]{losses[-1]:.4f}[/bold]   ‖∇L‖ = {gnorm:.3f}   [dim](AdamW: θ ← θ − η·m̂/(√v̂+ε))[/dim]")
    trend = "↓ learning" if len(losses) > 1 and losses[-1] < losses[0] else "≈ (more steps needed)"
    console.print(
        f"  [dim]cross-entropy L = −log p(correct token); minimizing it raises the model's probability "
        f"on the true next token. Trend: {trend}.[/dim]"
    )
    console.print(f"\n[bold green]done[/bold green] — read the full theory: [cyan]{note.doc_path() if note else 'README.md'}[/cyan]")
    return {"attention_type": attention_type, "n_params": n_params, "losses": losses}


# --------------------------------------------------------------------------- #
# demo mode: conceptual framework walkthrough with live illustration          #
# --------------------------------------------------------------------------- #


def _illustrate_reversible(console: Any) -> None:
    import torch

    torch.manual_seed(0)
    x1, x2 = torch.randn(4), torch.randn(4)

    def F(t):
        return torch.tanh(t) * 1.3

    def G(t):
        return torch.sin(t) * 0.7

    y1 = x1 + F(x2)
    y2 = x2 + G(y1)
    # exact inverse
    rx2 = y2 - G(y1)
    rx1 = y1 - F(rx2)
    err = float((torch.cat([x1, x2]) - torch.cat([rx1, rx2])).abs().max())
    console.print("  forward:  y₁ = x₁ + F(x₂),  y₂ = x₂ + G(y₁)")
    console.print("  inverse:  x₂ = y₂ − G(y₁),  x₁ = y₁ − F(x₂)")
    console.print(f"  [bold]reconstruction error = {err:.2e}[/bold] [green](exactly invertible → no stored activations)[/green]")


def _illustrate_tropical(console: Any) -> None:
    import torch

    torch.manual_seed(0)
    scores = torch.tensor([2.0, 3.5, 3.2, 1.0])
    tmax, arg = scores.max(0)
    runner = scores.topk(2).values[1]
    soft = torch.softmax(scores, 0)
    console.print(f"  scores = {_vec(scores.tolist())}")
    console.print(f"  tropical 'sum' = max = [bold]{tmax:.2f}[/bold] → route to key {int(arg)} (hard, single winner)")
    console.print(f"  margin γ = best − runner-up = {tmax:.2f} − {runner:.2f} = [bold]{(tmax - runner):.2f}[/bold] [dim](certificate radius γ/2)[/dim]")
    console.print(f"  softmax(scores) = {_vec(soft.tolist(), 3)} [dim](the smooth, classical counterpart)[/dim]")


def _illustrate_ultrametric(console: Any) -> None:
    def lcp(a: str, b: str) -> int:
        n = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                break
            n += 1
        return n

    x, y, z = "11010", "11011", "10000"
    # ultrametric distance = depth - lcp (smaller = closer)
    K = 5
    dxy, dyz, dxz = K - lcp(x, y), K - lcp(y, z), K - lcp(x, z)
    console.print(f"  prefixes: x={x}  y={y}  z={z}")
    console.print(f"  d(x,y)={dxy}  d(y,z)={dyz}  d(x,z)={dxz}")
    ok = dxz <= max(dxy, dyz)
    console.print(
        f"  strong triangle: d(x,z) ≤ max(d(x,y),d(y,z))  →  {dxz} ≤ max({dxy},{dyz})={max(dxy, dyz)}  "
        f"[bold]{'✓ holds' if ok else '✗'}[/bold] [dim](every triangle is isosceles)[/dim]"
    )


def _illustrate_standard(console: Any) -> None:
    import torch

    torch.manual_seed(0)
    q = torch.randn(1, 4)
    k = torch.randn(5, 4)
    s = (q @ k.t()).squeeze(0) / (4 ** 0.5)
    a = torch.softmax(s, 0)
    ent = float(-(a * (a + 1e-12).log()).sum())
    console.print(f"  scores S = qKᵀ/√d = {_vec(s.tolist())}")
    console.print(f"  A = softmax(S) = {_vec(a.tolist(), 3)}")
    console.print(f"  entropy H(A) = [bold]{ent:.3f}[/bold] nats [dim](0 = one-hot/selective, log5≈1.61 = uniform/mixing)[/dim]")


_ILLUSTRATIONS: dict[str, Callable[[Any], None]] = {
    "reversible": _illustrate_reversible,
    "tropical": _illustrate_tropical,
    "ultrametric": _illustrate_ultrametric,
    "standard": _illustrate_standard,
}


def narrate_demo(topic: str = "tropical", *, console: Any = None) -> dict[str, Any]:
    """Conceptual walkthrough of a framework's math, with a live illustration."""
    from rich.markup import escape
    from rich.panel import Panel
    from rich.text import Text

    console = console or _console()
    note = MECHANISM_NOTES.get(topic)
    if note is None:
        console.print(f"[red]unknown topic '{topic}'. Known: {', '.join(sorted(MECHANISM_NOTES))}[/red]")
        return {"topic": topic, "ok": False}

    console.rule(f"[bold magenta]Demo walkthrough · {note.title}")
    console.print(Panel(Text(note.idea), border_style="magenta", subtitle=f"[dim]read more → {note.doc_path()}[/dim]"))
    console.print("[bold]The mathematics, step by step:[/bold]")
    for label, equation, why in note.steps:
        console.print(f"  [magenta]{label:9s}[/magenta] [cyan]{escape(equation)}[/cyan]")
        console.print(f"      [dim]{escape(why)}[/dim]")

    illustrate = _ILLUSTRATIONS.get(topic)
    if illustrate is not None:
        console.print("\n[bold]Live illustration:[/bold]")
        illustrate(console)
    else:
        console.print(f"\n[dim](no numeric illustration for '{topic}'; the equations above + the doc tell the story.)[/dim]")
    console.print(f"\n[bold green]done[/bold green] — full theory: [cyan]{note.doc_path()}[/cyan]")
    return {"topic": topic, "ok": True}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def run_run(args: argparse.Namespace) -> int:
    narrate_run(args.attention, device=args.device, seed=args.seed, steps=args.steps)
    return 0


def run_demo(args: argparse.Namespace) -> int:
    res = narrate_demo(args.topic)
    return 0 if res.get("ok", True) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m nanochat.walkthrough", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="narrate a live nanochat mini-run for a mechanism")
    pr.add_argument("--attention", default="standard", help=f"one of: {', '.join(sorted(MECHANISM_NOTES))}")
    pr.add_argument("--device", default="cpu")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--steps", type=int, default=3)
    pr.set_defaults(func=run_run)

    pd = sub.add_parser("demo", help="conceptual walkthrough of a framework's math")
    pd.add_argument("--topic", default="tropical", help=f"one of: {', '.join(sorted(MECHANISM_NOTES))}")
    pd.set_defaults(func=run_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
