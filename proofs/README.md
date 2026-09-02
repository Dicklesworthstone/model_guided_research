# Lean Proofs

Machine-checked formalizations for model_guided_research (bead
`model_guided_research-vnl.2`, epic FORMAL). The Lean build IS the test.

## Layout

| path | contents |
|---|---|
| `MGRProofs/RouteStability.lean` | thm-lse-max-sandwich + thm-route-stability (Pillar I / tropical) |
| `MGRProofs/FlatError.lean` | thm-flat-error over a general valued ring (Pillar II / p-adic), plus a bridge from mathlib `AbsoluteValue` and a computable trivial-valuation example |
| `MGRProofs/OrdinalTermination.lean` | thm-ordinal-termination ABSTRACT core (no infinite descent; event-system SPEC for lab.3) - CNF-decrease lemmas deferred, see bead vnl.3 |
| `MGRProofs/SymplecticShear.lean` | thm-kick-kick-symplectic: J^T Omega J = Omega for gradient shears + composition closure |
| `MGRProofs/GromovLCP.lean` | thm-gromov-product-equals-lcp on unit-edge rooted binary trees |
| `AxiomCheck.lean` | axiom-audit script (see below) |
| `lakefile.toml`, `lean-toolchain` | toolchain + dependency pins |
```bash
cd proofs
lake update        # resolve deps; downloads mathlib sources + prebuilt olean
                   # artifacts from mathlib's cloud release cache (Azure blob)
lake build         # compiles this project against them
```

Requires [elan](https://github.com/leanprover-community/lean4/blob/master/docs/elan.md);
the exact toolchain comes from `lean-toolchain` and elan installs it on
demand.

## Pins

| artifact | pin |
|---|---|
| mathlib | tag `v4.34.0-rc2`, commit `85e3a25e006c35636f0e53b0e9296caca2685bc0` |
| Lean toolchain | `leanprover/lean4:v4.34.0-rc2` |

MATHLIB PIN POLICY: the rev is bumped only as a deliberate maintenance event
(its own chore bead: bump, rebuild, fix breakage, document), never as a
drive-by. CI always builds against this pin.

## The no-`sorry` policy

`sorry` only produces a WARNING under `lake` - a green build does NOT mean
the proofs are complete. The real gate is the axiom audit:

```bash
lake env lean AxiomCheck.lean > axioms.txt
! grep -q sorryAx axioms.txt
```

Every headline lemma must depend only on the three standard Lean axioms
(`propext`, `Classical.choice`, `Quot.sound`). CI runs exactly this check
and fails on any `sorryAx`. Current status: the ten headline lemmas of
tranche 1 (bead vnl.2) plus the ABSTRACT ordinal-termination core audit
clean; tranche 2 proper (bead vnl.3: the CNF-decrease lemmas, kick-kick
symplecticity beyond the shear core, Gromov = LCP on general trees) is still
open, so the theorem registry keeps thm-ordinal-termination at
`proved-on-paper`.

## Conventions

- One file per theory pillar.
- Lemma names mirror theorem-registry ids in `hypotheses/theorems.yaml`
  (`lse_max_sandwich` ↔ thm-lse-max-sandwich, etc.).
- Each lemma carries its registry statement in the docstring so drift
  between prose and formal statement is reviewable.
- Executable documentation: concrete numeric instances (`n = 3`, `beta = 2`)
  accompany each pillar as `example`s.

## Build times (recorded per bead vnl.2 close notes)

- Project modules against fetched mathlib artifacts: first elaboration
  ~5 min/module (full-Mathlib import), warm incremental ~8 s.
- Mathlib itself was NOT built from source: `lake update` pulls prebuilt
  artifacts from mathlib's cloud cache. On networks where that cache is
  blocked, a from-source mathlib build (hours) is required before anything
  here type-checks.
