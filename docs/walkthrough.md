# Pedagogical walkthrough mode (`nanochat.walkthrough`)

Bead **9f1**. A guided, **off-by-default** narration that explains the
*mathematics as it runs* — what the Cayley step does, why ultrametric routing is
sub-quadratic, what a tropical margin means — tied to the actual equations and
variables, with links into `markdown_documentation/`. It is a separate entry
point (not a hot-path hook), so a normal run pays nothing.

## Two modes

### `run` — a live nanochat mini-run

Builds a tiny model for a mechanism and walks the **real forward pipeline**
(embedding → RMSNorm → RoPE → attention → residual → MLP → logits →
cross-entropy) with live shapes/values and the mechanism's interpretability
observable, then takes a few AdamW steps and explains loss / gradients / the
update.

```bash
python -m nanochat.walkthrough run --attention reversible   # exactly-invertible coupling
python -m nanochat.walkthrough run --attention tropical     # max-plus routes + margins
python -m nanochat.walkthrough run --attention standard --steps 5
```

It prints the per-stage equations, the mechanism-specific attention math, the
live observable (e.g. attention entropy ≈ 2.19 nats vs uniform log(32)≈3.47),
and the loss trend (`↓ learning`).

### `demo` — a conceptual framework walkthrough (with live illustration)

Steps through a mechanism's core math with a small live numeric demonstration:

```bash
python -m nanochat.walkthrough demo --topic reversible   # reconstruction error ≈ 6e-08 (exact inverse)
python -m nanochat.walkthrough demo --topic tropical     # max route + runner-up margin γ
python -m nanochat.walkthrough demo --topic ultrametric  # strong triangle inequality holds
python -m nanochat.walkthrough demo --topic standard     # softmax map + entropy
```

Mechanisms without a numeric illustration (`gauge`, `quaternion`, `octonion`,
`braid`, `simplicial`, `surreal`, `fractal`) still print their equations + the
doc link.

## In-demo narration (`MGR_WALKTHROUGH=1`)

The JAX demos can narrate themselves in place via the `walkthrough_enabled()`
env gate. The reversible demo is wired:

```bash
MGR_WALKTHROUGH=1 mgr run reversible
# -> prints the "Demo walkthrough · Reversible / measure-preserving block"
#    narration (forward/inverse coupling, det=1, doc link) before the demo runs.
```

The hook is wrapped so narration can never break the demo, and the import/call
cost nothing when `MGR_WALKTHROUGH` is unset.

## Coverage of the 11 frameworks

`MECHANISM_NOTES` carries teaching notes (idea, load-bearing equations,
interpretability observable, doc link) for: standard, tropical, ultrametric,
reversible, gauge, quaternion, octonion, braid, simplicial, surreal, fractal.
Every note links a real file under `markdown_documentation/` (enforced by a
test) so the walkthrough is always one click from the full theory.
</content>
