# The Symplectic Transformer: Gradient-Potential Coupling, Shadow Energy, and Normalization-Free Training

**Bead:** u55.5 (EPIC THEORY-II, u55) · **Implementation:** `nanochat/reversible_block_torch.py` (`reversible_mode: symplectic`) · **Status of claims:** §7 separates theorem / theory-backed prediction / empirical bet.

---

## 1. One constraint turns reversible coupling into a symplectic integrator

The additive reversible block on a half-split stream `x = [x1, x2]`,

```
y1 = x1 + F(x2)          y2 = x2 + G(y1)
```

preserves volume for ANY `F, G` (each half-update is a shear: its Jacobian is
unit-triangular). But it conserves nothing else, because arbitrary `F, G` are
not the flow of anything.

Constrain the coupling functions to be **gradients of scalar potentials** and
flip the sign of the second kick:

```
y1 = x1 + ∇φ_F(x2)       y2 = x2 − ∇φ_G(y1)        (kick-kick, corrected sign)
```

**Claim (exact, unconditional).** Each half-update is an *exact symplectic
map* on the phase space `z = (x1, x2)` with canonical form
`Ω = [[0, I], [−I, 0]]`, and so is their composition.

*Proof.* The first update is the time-`1` flow of the Hamiltonian
`K₁(x1, x2) = φ_F(x2)`: Hamilton's equations give `ẋ1 = ∂K₁/∂x2 = ∇φ_F(x2)`,
`ẋ2 = −∂K₁/∂x1 = 0`, whose exact solution is the kick. The second update is
the time-`1` flow of `K₂(x1, x2) = φ_G(x1)`: `ẋ1 = 0`,
`ẋ2 = −∇φ_G(x1)`. Exact Hamiltonian flows are symplectic; compositions of
symplectic maps are symplectic. ∎

Equivalently and concretely: the Jacobian of the first kick is
`[[I, H_F], [0, I]]` with `H_F = ∇²φ_F(x2)`; the shear condition
`J^T Ω J = Ω` reduces to `H_F = H_F^T` — **symmetry of the Hessian is the
entire content of symplecticity here**, and it holds exactly because `F` is a
gradient. An arbitrary `F` (e.g. raw attention output) has a non-symmetric
Jacobian and the block is merely volume-preserving. Validated numerically:
`|J^T Ω J − Ω|_max = 0.0` in fp64 for the gradient kick;
`≈ 0.95` for the same architecture's raw (additive) coupling.

### 1.1 The sign correction (and why it is load-bearing)

The composition above is one step of **symplectic Euler** for the separable Hamiltonian

```
H(x1, x2) = φ_G(x1) + φ_F(x2)
```

(`x1` plays position with potential `φ_G`, `x2` plays momentum with kinetic
energy `φ_F`). The bead description's original form used `+` on both kicks;
that composition is *also* exactly symplectic (shears are symplectic with
either sign) — but it is the splitting integrator of
`H̄(x1, x2) = φ_F(x2) − φ_G(x1)`, whose conserved quantity is a **difference**
of potentials. A difference is non-coercive *by construction*: no confinement
term can bound level sets of `φ_F − φ_G` (the `−(λ/2)‖x1‖²` half pushes the
wrong way). Conserving a non-coercive quantity bounds nothing.

Numerics (256 tied layers, fp64, same nets, kick scale `h`):

| h | conserved-H band, corrected sign | band, original sign | norm growth corrected | norm growth original |
|------|------|------|------|------|
| 0.05 | **0.007** | 10.4 | 6.2× (within level set) | 30× |
| 0.20 | **0.057** | 1.9e4 | 12× peak, bounded | 6.5e4× (divergence) |

The corrected sign is therefore not cosmetic: it is the difference between
"conserves a quantity that bounds activations" and "conserves a quantity that
permits blow-up". The implementation uses the corrected sign; the exact
inverse is `x2 = y2 + ∇φ_G(y1)`, then `x1 = y1 − ∇φ_F(x2)`.

### 1.2 Inverse exactness — a correction to the bead's framing

The kick inverse (negate the kick) is exact to machine epsilon (validated:
2.2e−16 in fp64). The bead called this "even nicer than additive coupling's
fixed-point-free inverse" — that is wrong: standard additive coupling already
has an exact, fixed-point-free inverse (`x2 = y2 − G(y1)`; `x1 = y1 − F(x2)`).
The kick form's advantage is **symplecticity**, not invertibility. The
O(1)-memory property is inherited unchanged.

---

## 2. What symplecticity buys: backward error analysis, honestly scoped

For a small step on a bounded analytic domain, backward error analysis gives
this first-order splitting a modified Hamiltonian whose leading correction is
`O(h)` (Hairer–Lubich–Wanner, *Geometric Numerical Integration*, ch. IX).
The size of the correction for a learned block depends on its potentials and
the region visited. The trainer records the staggered sum `F(p) + G(q′)`.

For quadratic potentials `F(p) = a p²/2` and `G(q) = b q²/2`, substitution of
`q′ = q + h a p` and `p′ = p − h b q′` gives an exact discrete invariant
`H* = (b q² + a p² + h a b q p)/2`. For positive `a,b` it is positive
definite when `h²ab < 4`. At `a=b=h=1` and `(q,p)=(2,0)`, eight tied steps keep
`H*=2` exactly while the staggered sum visits both 2 and 4. The 10% band in
the depth-eight training scenario is an empirical regression threshold for
that seed, model and input. Each logged range covers one training forward.

The theorem is about **iterating the same map**. That gives a claim hierarchy
that the experiment matrix must respect:

1. **Per-layer exact symplecticity + volume preservation** — unconditional,
   any potentials, tied or untied. (Theorem; certified in tests.)
2. **Tied potentials** iterate one map. Small-step estimates also require
   regularity and control of the visited domain. One measured energy band
   was 0.007 over 256 tied layers at h = 0.05.
3. **Slowly-varying untied potentials** — adiabatic regime. One step with
   `φ^{(l)}` followed by one with `φ^{(l+1)}` changes the conserved reference
   by `H^{(l+1)} − H^{(l)}`, so the energy drift across `L` layers is bounded
   by `Σ_l sup_{level set} |H^{(l+1)} − H^{(l)}|` — controlled by
   `‖φ^{(l+1)} − φ^{(l)}‖_∞` on the visited region. Penalizing successive
   potential differences (or initializing tied and letting layers diverge
   slowly) keeps the drift small. This is theory-flavored but with an
   empirical constant.
4. **Independent untied potentials** — volume preservation + per-layer
   symplecticity only; norm behavior is an **empirical bet**, registered as
   such (§7).

## 3. Coercivity: the missing design requirement

Conserved energy bounds activations **only if `H` is coercive** (level sets
bounded: `H → ∞` as `‖z‖ → ∞`). Unconstrained potential networks need not be
coercive. The implementation makes coercivity true *by construction*:

```
φ(x) = φ_net(x) + (λ/2)·‖x‖²,    λ = λ_min + softplus(λ_raw) ≥ λ_min > 0
```

with `φ_net` a **bounded** energy head (per-token `v^T tanh(W u_t)` over the
inner block's output, summed over tokens): `|φ_net(x)| ≤ T·‖v‖₁` always.
Then the explicit activation bound is, along a level set of value `H̃`:

```
(λ_F/2)‖x2‖² ≤ H̃ − φ_G(x1) − φ_F,net(x2) ≤ H̃ + 2T·max(‖v_F‖₁, ‖v_G‖₁)
⟹  ‖x2‖² ≤ (2/λ_min)·(H̃ + 2T·V),   V := max(‖v_F‖₁, ‖v_G‖₁)
```

and symmetrically for `x1`. **The bound is explicit in conserved quantities
and architecture constants** — this is `f(shadow energy, λ_min)` from the
bead, derived. The `λ_min = 0` arm is the registered falsification control: if
unconfined potentials also never drift, coercivity was not the binding
constraint and this account needs revisiting; if they drift, the theory is
doing real work.

Every "drift impossible" phrasing is hereby softened to the theorem shape:
*norm drift is controlled by a conserved (tied) or nearly conserved
(adiabatic) shadow Hamiltonian, under coercivity.*

## 4. The no-norm transformer

LayerNorm/RMSNorm exist to fight activation drift; conservation removes the
cause rather than the symptom. The reversible path in `gpt.py` already runs
**without** internal norms inside the coupling (`f_block(x2)` directly), so
the comparison is clean:

- **Prediction (tied, theory-backed):** symplectic-tied no-norm GPTs train
  stably at depths where unnormalized standard GPTs diverge, because bounded
  activations are a consequence of §2.2 + §3 rather than of normalization.
- **Prediction (untied, empirical):** registered separately. If untied drifts
  but tied does not, that *confirms* the theory (the theorem is about tied
  maps) — it is not a failure of the program.

Freeing the stream from per-token normalization restores **magnitude as an
information channel** — the surreal framework's core claim (scale carries
content; norms destroy it). Two frameworks converge on this point from
independent mathematics.

## 5. Noether channels

Every continuous symmetry of the potentials yields an exactly conserved
charge of the ideal flow, conserved to integrator (shadow) tolerance by the
block. The engineered example: make `φ` **invariant under permutations of
tokens within designated segments** (any per-token energy head summed over the
segment is automatically so). The associated conserved quantity is a multiset
memory: what is in the bag cannot be forgotten by depth, to shadow tolerance.
"Noether regularization" — penalizing symmetry violation of learned
potentials — creates soft conservation laws aligned with task structure; a
regularization family with a physics pedigree. Prototype protocol lives with
the experiments (§7); if the multiset diagnostic task (vdc.1 task 10) has not
landed, a minimal inline generator is used and the duplication noted for C1
to absorb.

## 6. Causal masking: what is conserved, when

`φ` couples all tokens through attention, so the Hamiltonian is token-coupled;
symplecticity holds on the **full token phase space** (a gradient of a scalar
over all tokens is still a gradient — fine, and exactly what the Jacobian
test validates with token mixing on). Two regimes:

- **Teacher-forced forward (training, perplexity evals):** the map applied to
  the full sequence is exactly the kick-kick map above; everything claimed
  (symplecticity per layer, tied shadow conservation, coercive bounds) holds
  as stated. Note `∇φ` is taken through the causal attention, so token `s`
  receives gradient contributions from all `t ≥ s` — the kick is *anti-causal
  in its information flow*. This is intrinsic: a scalar potential over a
  causally-read sequence has a gradient that flows backward in time. For
  next-token training this is a non-issue (the LM head still reads causal
  stream states; the phase-space dynamics are an internal computation), but
  it must be stated.
- **Autoregressive decoding:** past stream values are frozen in the KV cache,
  so the incremental kick on a new token uses `∇φ` restricted to the new
  token's coordinates with the prefix held fixed. The map realized at decode
  time is therefore the *prefix-conditional* kick, which differs from the
  teacher-forced kick (which lets past tokens move under future-token energy
  gradients). What is conserved during generation is the shadow energy **per
  prefix**: each prefix length defines its own (frozen-prefix) Hamiltonian,
  conserved across depth at that step. Cross-step conservation acquires a
  correction term equal to the energy difference between successive prefix
  Hamiltonians evaluated at the new state — measured and logged, not assumed
  away. Consequence for parity tests: teacher-forced and incremental decode
  produce different stream values *by design* in symplectic mode (unlike
  additive mode); decode-parity assertions apply per prefix, not across.

## 7. Registered claims (registry ids, registered before any depth-ladder evidence)

| claim | regime | kind |
|---|---|---|
| `hyp-symplectic-nonorm-depth-tied` | tied, no-norm, depth 16 | theory-backed prediction |
| `hyp-symplectic-nonorm-depth-untied` | untied, no-norm, depth 16 | empirical bet, registered separately |

Headline operationalization (from the bead): symplectic no-norm trains to
within 5% of the normed baseline's loss at depth 16 while standard no-norm
diverges or degrades > 20%. The λ ablation (`λ_min ∈ {0, small, moderate}`)
and the energy/charge telemetry ride along as recorded diagnostics, not
adjudicated observables. Experiment arms (equal params):
`{symplectic-tied no-norm, symplectic-untied no-norm, additive with-norm,
standard with-norm, standard no-norm}` × depth `{8, 16, 32}`.

## 8. Positioning

SympNets and symplectically-integrated network architectures (Jin et al.,
*SympNets*, Neural Networks 2020; Chen et al., symplectic recurrent nets)
learn Hamiltonian *dynamics from data* — the symplectic structure is the
hypothesis class for a physical system. Here the goal is different with the
same machinery: **impose** symplectic structure on a language-model residual
stream to obtain stability theorems (shadow conservation + coercivity ⇒
bounded activations without normalization) and engineered conservation laws
(Noether channels). We are not fitting dynamics; we are buying invariants.

## 9. Implementation map

- `nanochat/reversible_block_torch.py` — `_EnergyHead` (bounded head +
  confinement, learnable `λ` floored at `λ_min`), `SymplecticKick`
  (`∇φ` via `torch.autograd.grad`, `create_graph` during training for the
  autograd-of-autograd training path), `ReversibleBlock(mode=...)` with the
  corrected-sign forward/inverse, shadow-energy/norm telemetry.
- `nanochat/gpt.py` — `GPTConfig.reversible_mode {additive, symplectic}`,
  `reversible_tied`, `reversible_lambda_min`, `reversible_record_energy`;
  tied-block sharing in `GPT.__init__`.
- Tests — fp64 `J^T Ω J = Ω` on the real modules (the load-bearing check),
  energy-drift separation (both directions asserted), exact kick inverse
  round-trip, gradcheck through the kick, coercivity bound check, goldens
  recapture for the new config fields.
- Certify — `reversible.symplectic_jacobian` and
  `reversible.energy_drift_separation` join the B1 named checks.
