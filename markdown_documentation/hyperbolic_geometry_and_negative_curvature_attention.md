# Hyperbolic Geometry & Negative-Curvature Attention — Lorentz Model

*Framework 13: negative curvature as the native geometry of hierarchy.*

## The Core Idea

The Lorentz (hyperboloid) model of hyperbolic space H^n_c at curvature
−1/c lives in R^{n+1}:

    <x, y>_L = −x_t y_t + x_s · y_s,     <x, x>_L = −1/c,   x_t > 0

with hyperbolic distance

    d_H(x, y) = arccosh(−c <x, y>_L) / sqrt(c)

Ball volume grows **exponentially** with radius — the defining property that
lets trees embed with arbitrarily low distortion while Euclidean space of the
same dimension cannot.

## Primitives (demo-validated)

- exp/log maps at the basepoint o = (1/√c, 0), closed-form via cosh/sinh.
- Constraint projection after off-manifold updates: t = sqrt(1/c + |z_s|²).
- Numerics policy: the arccosh argument is clamped to ≥ 1 + 1e-12 (domain
  guard only); fp32 breaks down beyond a MEASURED radius (~30–60 in demo
  units) — recorded in the property table rather than assumed.

## Property Checks (all green in `mgr run hyperbolic`)

1. exp/log round-trip across radii 1e-3 … 60, with true-fp64 reference
   (scoped `jax.enable_x64`) and the measured fp32 break radius.
2. Lorentz constraint maintained through projection (residual at fp32 floor).
3. c → 0 limit: softmax(−d_H/τ) attention converges monotonically to
   distance-based Euclidean attention (3.4e-2 → 6.4e-5 across the sweep).
4. Sarkar-style tree embedding beats equal-budget equal-dimension Euclidean
   fitting on relative distortion.
5. Metric sanity: triangle inequality; radial isometry exact; and the honest
   curvature signature — chordal distance grows like sinh(r) while d_H grows
   linearly, so chord/d_H explodes (measured 1.8 → 186 over r ∈ {2,4,8}).

## Why It Matters

Hierarchy is native here: children placed near parents stay close, while
different subtrees separate exponentially fast. The demo's learning task —
reconstructing leaf embeddings from tree distances — shows the hyperbolic fit
beating an identical-budget Euclidean fit at equal dimension.

## Relation to Other Frameworks

- **ultrametric attention**: ultrametric ≡ tree metric on leaves; the Lorentz
  model realizes such trees continuously — the three-way hierarchical
  comparison set (hyperbolic / ultrametric / fractal) closes with this pair.
- **production path**: mnn.6 ports scoring (−d_H/τ) and tangent-space value
  aggregation into nanochat (`hyperbolic` mechanism) with per-head learnable
  curvature.
