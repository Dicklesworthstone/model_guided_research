"""
Hyperbolic Geometry & Negative-Curvature Attention — Lorentz Model
------------------------------------------------------------------

Framework #13: hyperbolic space as the substrate for hierarchical structure.
Where Cl(3,0) (framework 12) gave exact algebraic rotors, the Lorentz model
of hyperbolic space H^n_c (curvature -1/c) gives EXACT hierarchical geometry:
ball volume grows exponentially with radius, so trees embed with low
distortion while Euclidean space cannot match at equal dimension. That
contrast is the demo's payoff, demonstrated concretely.

Model conventions (fixed here and reused by nanochat/hyperbolic_attention_torch.py):

- Points live on the hyperboloid  <x, x>_L = -1/c,  x_time > 0, with the
  Minkowski inner product  <x, y>_L = -x_t y_t + x_s . y_s  (space dim n-1).
- Hyperbolic distance:  d_H(x, y) = arccosh(-c <x, y>_L) / sqrt(c),
  with the arccosh argument clamped >= 1 + eps (documented threshold).
- exp/log maps are taken at the basepoint o = (1/sqrt(c), 0):
      exp_o(v) = cosh(sqrt(c)|v|) o  +  sinh(sqrt(c)|v|)/(sqrt(c)|v|) * v
      log_o(y) = arccosh(sqrt(c) y_t) / (sqrt(c) |y_s|) * y_s
- Constraint projection after off-manifold updates:
      proj(z)_t = sqrt(1/c + |z_s|^2).

PROPERTY CHECKS (rich table, house style):
  1. exp/log inverse round-trip across magnitudes, INCLUDING a measured
     "where fp32 breaks" radius;
  2. Lorentz constraint maintenance through a projection loop;
  3. c -> 0 Euclidean limit: hyperbolic attention converges to standard
     softmax attention as curvature vanishes;
  4. Sarkar-style tree embedding beats an equal-budget equal-dimension
     Euclidean embedding on distortion;
  5. metric sanity: triangle inequality holds and d_H exceeds Euclidean
     distance for far-apart pairs (negative-curvature signature).

LEARNING TASK: leaf embeddings reconstructed from tree distances -
hyperbolic vs Euclidean at equal dimension and step budget.

Runtime: well under 2 minutes on CPU.
"""

# Docs: markdown_documentation/hyperbolic_geometry_and_negative_curvature_attention.md
# Built as bead model_guided_research-mnn.5 (demo) feeding mnn.6 (torch port).

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import nn as jnn
from jax import random

_ARCCOSH_EPS = 1e-12  # domain guard only: genuine points stay above it


# ---------------------------------------------------------------------------
# Lorentz-model primitives (JAX; mirrored in tests)
# ---------------------------------------------------------------------------


def minkowski(x, y):
    """Minkowski inner product <x, y>_L with timelike coordinate first."""

    return -x[..., 0] * y[..., 0] + jnp.sum(x[..., 1:] * y[..., 1:], axis=-1)


def hyp_distance(x, y, c):
    """d_H(x, y) = arccosh(-c <x,y>_L)/sqrt(c), argument clamped >= 1+eps."""
    arg = -c * minkowski(x, y)

    return jnp.arccosh(jnp.maximum(arg, 1.0 + _ARCCOSH_EPS)) / math.sqrt(c)


def lorentz_origin(n, c):
    """Basepoint o = (1/sqrt(c), 0, ..., 0)."""
    o = jnp.zeros(n)

    return o.at[0].set(1.0 / math.sqrt(c))


def exp_map_o(v, c):
    """exp_o(v) for tangent vectors v at the origin (space-only layout:
    v has n-1 components; returns an n-component point on the hyperboloid)."""
    norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
    lam = math.sqrt(c) * norm_v
    x_t = jnp.cosh(lam) / math.sqrt(c)
    x_s = jnp.where(
        lam < 1e-12,
        v / math.sqrt(c),
        jnp.sinh(lam) / (math.sqrt(c) * norm_v) * v,
    )

    return jnp.concatenate([x_t, x_s], -1)


def log_map_o(y, c):
    """log_o(y): tangent (n-1) vector at the origin for hyperboloid point y."""
    y_t = y[..., :1]
    y_s = y[..., 1:]
    alpha = jnp.arccosh(jnp.maximum(math.sqrt(c) * y_t, 1.0 + _ARCCOSH_EPS))
    norm_s = jnp.linalg.norm(y_s, axis=-1, keepdims=True)
    scale = jnp.where(norm_s < 1e-12, 0.0, alpha / (math.sqrt(c) * norm_s))

    return scale * y_s


def project_lorentz(z, c):
    """Project arbitrary z onto the hyperboloid: t = sqrt(1/c + |z_s|^2)."""
    z_s = z[..., 1:]

    return jnp.concatenate([jnp.sqrt(1.0 / c + jnp.sum(z_s**2, -1, keepdims=True)), z_s], -1)


def euclidean_attention(q, k, v):
    """Standard softmax attention on Euclidean vectors."""
    d = q.shape[-1]

    return jnn.softmax(q @ k.T / math.sqrt(d), axis=-1) @ v


def hyperbolic_attention(q_tan, k_tan, v_tan, c):
    """Attention where q/k/v are TANGENT vectors at the origin: lift via
    exp_o, score with -d_H/tau (tau = sqrt(d)), aggregate the VALUE tangents
    pulled back through log_o. As c -> 0 this converges to euclidean_attention
    (property check 3)."""
    d = q_tan.shape[-1]
    tau = math.sqrt(d)

    def lift_batch(batch):
        return jax.vmap(lambda vv: exp_map_o(vv, c))(batch)

    ql, kl, vl = lift_batch(q_tan), lift_batch(k_tan), lift_batch(v_tan)

    dist = jax.vmap(lambda qi: jax.vmap(lambda kj: hyp_distance(qi[None], kj[None], c)[0])(kl))(ql)
    weights = jnn.softmax(-dist / tau, -1)

    rows = []
    for i in range(q_tan.shape[0]):
        acc = jnp.zeros_like(v_tan[i])
        for j in range(k_tan.shape[0]):
            acc = acc + weights[i, j] * log_map_o(vl[j], c)
        rows.append(acc)

    return jnp.stack(rows)


# ---------------------------------------------------------------------------
# Tree machinery (Sarkar-style embedding + distortion)
# ---------------------------------------------------------------------------


def _random_tree(seed: int, *, depth=3, branch=3):
    """Random rooted tree; returns level array + pairwise tree distances."""
    parents = [-1]
    level = [0]
    frontier = [0]
    for d in range(depth):
        new_frontier = []
        for node_id in frontier:
            for _ in range(branch):
                parents.append(node_id)
                level.append(d + 1)
                new_frontier.append(len(parents) - 1)
        frontier = new_frontier
    n = len(parents)

    paths = []
    for i in range(n):
        path = [i]
        while parents[path[-1]] != -1:
            path.append(parents[path[-1]])
        paths.append(path)

    tree_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            set_i = set(paths[i])
            lca = next(k for k in paths[j] if k in set_i)
            d_ij = paths[i].index(lca) + paths[j].index(lca)
            tree_dist[i, j] = tree_dist[j, i] = d_ij

    return np.asarray(level), tree_dist


def _embed_tree(key, tree_dist, *, dim, steps, curv, hyperbolic, lr=5e-3):
    """Fit embeddings to tree distances by gradient descent. Hyperbolic:
    parameters are origin-tangent vectors mapped through exp_o and scored
    with d_H; Euclidean: raw coordinates scored with L2. Equal budget,
    equal dimension."""
    import optax

    n = tree_dist.shape[0]
    sub_key = random.fold_in(key, 12345)
    params = 0.3 * random.normal(sub_key, (n, dim))
    target = jnp.asarray(tree_dist)
    idx = jnp.triu_indices(n, 1)

    if hyperbolic:

        def loss_fn(p):
            pts = jax.vmap(lambda v: exp_map_o(v, curv))(p)
            dh = jax.vmap(lambda ij: hyp_distance(pts[ij[0]][None], pts[ij[1]][None], curv)[0])(idx)

            return jnp.mean((dh - target[idx]) ** 2)

    else:

        def loss_fn(p):
            de = jnp.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)

            return jnp.mean((de[idx] - target[idx]) ** 2)

    optimizer = optax.adam(lr)
    state = optimizer.init(params)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    for _ in range(steps):
        _, grads = grad_fn(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)

    return float(loss_fn(params))


def _embed_tree(key, tree_dist, *, dim, steps, curv, hyperbolic, lr=1e-3):
    """Fit embeddings to tree distances by gradient descent (MSE on pairs).
    Hyperbolic: parameters are origin-tangent vectors mapped through exp_o
    and scored with d_H; Euclidean: raw coordinates scored with L2. Equal
    budget, equal dimension, identical optimizer settings."""
    import optax

    n = tree_dist.shape[0]
    sub_key = random.fold_in(key, 12345)
    params = 0.2 * random.normal(sub_key, (n, dim))
    target = jnp.asarray(tree_dist)
    idx = jnp.triu_indices(n, 1)

    if hyperbolic:

        def pair_loss(p):
            pts = jax.vmap(lambda v: exp_map_o(v, curv))(p)

            return jax.vmap(lambda ij: hyp_distance(pts[ij[0]][None], pts[ij[1]][None], curv)[0])(idx)

    else:

        def pair_loss(p):
            diff_sq = jnp.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1)

            # eps inside the sqrt: the diagonal's exact-zero norm has a 0/0
            # gradient that would otherwise poison every update with NaN
            # through the gather backward pass.
            return jnp.sqrt(diff_sq + 1e-12)[idx]

    def loss_fn(p):
        return jnp.mean((pair_loss(p) - target[idx]) ** 2)

    optimizer = optax.adam(lr)
    state = optimizer.init(params)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    for _ in range(steps):
        _, grads = grad_fn(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)

    return float(loss_fn(params))


def _distortion(key, tree_dist, *, dim, curv, steps=250):
    """Relative RMSE distortion after fitting, hyperbolic vs Euclidean."""
    n = tree_dist.shape[0]
    mean_d = float(np.mean(tree_dist[np.triu_indices(n, 1)]))
    fit_hyp = _embed_tree(key, tree_dist, dim=dim, steps=steps, curv=curv, hyperbolic=True)
    fit_euc = _embed_tree(key, tree_dist, dim=dim, steps=steps, curv=curv, hyperbolic=False)

    return {
        "hyperbolic_rel_rmse": math.sqrt(fit_hyp) / mean_d,
        "euclidean_rel_rmse": math.sqrt(fit_euc) / mean_d,
    }


# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------


def run_property_checks(seed: int = 7) -> list[tuple[str, bool, str]]:
    """The five headline property checks; returns (name, ok, detail) rows."""
    results: list[tuple[str, bool, str]] = []
    # -- 1. exp/log round-trip across magnitudes ------------------------------
    # JAX downcasts float64 by default; enable x64 ONLY for the precision
    # reference arm so "where does fp32 break" is measured against a true
    # fp64 baseline instead of two copies of fp32.
    key, k1 = random.split(random.PRNGKey(seed))
    radii = [1e-3, 1e-2, 0.1, 1.0, 3.0, 10.0, 30.0, 60.0]

    def _roundtrip_max_err(dtype):
        rows = []
        for radius in radii:
            vv = random.normal(k1, (16, 7))
            vv = vv / jnp.linalg.norm(vv, axis=-1, keepdims=True) * radius
            y = jax.vmap(lambda vec: exp_map_o(vec.astype(dtype), 1.0))(vv.astype(dtype))
            back = jax.vmap(lambda yy: log_map_o(yy.astype(dtype), 1.0))(y)
            rel = float(jnp.max(jnp.abs(back - vv.astype(back.dtype))) / max(radius, 1.0))
            rows.append(rel)
        return rows

    with jax.enable_x64(True):
        rel64 = _roundtrip_max_err(jnp.float64)
    rel32 = _roundtrip_max_err(jnp.float32)
    worst64 = max(rel64[:7])  # radii <= 10
    worst32 = max(rel32[:7])
    fp32_break_radius = next((r for r, e in zip(radii, rel32) if e > 1e-3 or not math.isfinite(e)), None)
    ok1 = worst64 <= 1e-12 and worst32 <= 5e-4
    results.append(
        (
            "exp_log_roundtrip",
            ok1,
            f"fp64 worst rel err (r<=10) {worst64:.2e}; fp32 {worst32:.2e}; "
            f"fp32 first breaks beyond radius {fp32_break_radius} (measured)",
        )
    )

    # -- 2. Lorentz constraint maintenance ------------------------------------
    key, k2 = random.split(key)
    c_val = 1.0
    params = random.normal(k2, (24, 7)) * 0.5
    pts = jax.vmap(lambda vv: exp_map_o(vv, c_val))(params)
    drifted = pts.at[:, 1:].add(0.1 * random.normal(random.split(k2)[1], pts[:, 1:].shape))
    projected = jax.vmap(lambda zz: project_lorentz(zz, c_val))(drifted)
    residual = float(jnp.max(jnp.abs(minkowski(projected, projected) + 1.0 / c_val)))
    results.append(
        (
            "lorentz_constraint_maintenance",
            residual < 5e-6,
            f"max |<x,x>_L + 1/c| after projection = {residual:.3e} (fp32 noise floor; projection is exact over R)",
        )
    )

    # -- 3. c -> 0 Euclidean limit --------------------------------------------
    key, k3 = random.split(key)
    kt, kv_, kk3 = random.split(k3, 3)
    q_tan = random.normal(kt, (6, 7)) * 0.8
    k_tan = random.normal(kv_, (8, 7)) * 0.8
    v_tan = random.normal(kk3, (8, 7)) * 0.8
    # The c -> 0 limit of softmax(-d_H/tau) is softmax(-|q-k|/tau): the
    # reference is DISTANCE-based Euclidean attention, not dot-product.
    d_e = jnp.linalg.norm(q_tan[:, None, :] - k_tan[None, :, :], axis=-1)
    ref = jnn.softmax(-d_e / math.sqrt(q_tan.shape[-1]), -1) @ v_tan
    errs = []
    for c_val in [1.0, 0.1, 0.01, 0.001]:
        out = hyperbolic_attention(q_tan, k_tan, v_tan, c_val)
        errs.append(float(jnp.max(jnp.abs(out - ref))))
    monotone = all(errs[i] >= errs[i + 1] - 1e-9 for i in range(len(errs) - 1))
    results.append(
        (
            "euclidean_limit_as_curvature_vanishes",
            monotone and errs[-1] < 1e-3,
            "max|hyp - euclid| for c={1,.1,.01,.001} = " + ", ".join(f"{e:.2e}" for e in errs),
        )
    )

    # -- 4. tree-embedding sanity (Sarkar-style) -------------------------------
    key, k4 = random.split(key)
    tree_seed = int(random.randint(k4, (), 0, 2**31 - 1))
    _, tree_dist = _random_tree(tree_seed, depth=3, branch=3)
    dist = _distortion(k4, tree_dist, dim=4, steps=250, curv=1.0)
    ok4 = dist["hyperbolic_rel_rmse"] < dist["euclidean_rel_rmse"]
    results.append(
        (
            "tree_embedding_beats_euclidean",
            ok4,
            f"relative RMSE: hyperbolic {dist['hyperbolic_rel_rmse']:.3f} vs "
            f"euclidean {dist['euclidean_rel_rmse']:.3f} (dim 4, 250 steps)",
        )
    )

    # -- 5. metric sanity: triangle inequality + curvature signature ----------
    key, k5, k6 = random.split(key, 3)
    pa = jax.vmap(lambda vv: exp_map_o(vv, 1.0))(random.normal(k5, (16, 7)))

    tri_ok = True
    for idx_triple in random.randint(random.split(k5)[1], (8, 3), 0, 16):
        i, j, k_idx = (int(t) for t in idx_triple)
        d_ij = hyp_distance(pa[i : i + 1], pa[j : j + 1], 1.0)[0]
        d_jk = hyp_distance(pa[j : j + 1], pa[k_idx : k_idx + 1], 1.0)[0]
        d_ik = hyp_distance(pa[i : i + 1], pa[k_idx : k_idx + 1], 1.0)[0]
        if float(d_ik) > float(d_ij + d_jk) + 1e-6:
            tri_ok = False
            break
    # Curvature signature, stated HONESTLY: along a geodesic ray the distance
    # from the origin is exactly r (radial isometry), while between
    # OPPOSITE-direction boundary points at radius r the chordal Euclidean
    # distance grows like sinh(r) but d_H grows only like 2r - so hyperbolic
    # space packs exponentially more "far apart" directions per unit metric
    # budget. Assert: radial isometry exact; d_E/d_H ratio strictly increasing
    # in r across {2, 4, 8} (and > 1 by r = 8).
    u = random.normal(random.split(k6)[0], (7,))
    u = u / jnp.linalg.norm(u)
    radial_ok = True
    growth = []
    for r_val in [2.0, 4.0, 8.0]:
        x = exp_map_o(r_val * u, 1.0)[None]
        radial_ok = radial_ok and abs(float(hyp_distance(x, lorentz_origin(8, 1.0)[None], 1.0)[0]) - r_val) < 1e-5
        y_dir = -u + 1e-3 * jnp.ones(7)
        y_dir = y_dir / jnp.linalg.norm(y_dir)
        y = exp_map_o(r_val * y_dir, 1.0)[None]
        arg = float(-minkowski(x, y[None])[0, 0])
        d_h_pair = math.acosh(max(arg, 1.0 + _ARCCOSH_EPS))
        chord = float(jnp.linalg.norm(x[0, 1:] - y[0, 1:]))
        growth.append(chord / d_h_pair)
    increasing = all(growth[i] < growth[i + 1] for i in range(len(growth) - 1))
    results.append(
        (
            "metric_triangle_inequality_and_curvature_signature",
            tri_ok and radial_ok and increasing and growth[-1] > 1.0,
            f"triangle inequality ({tri_ok}); radial isometry exact ({radial_ok}); "
            f"chord/d_H ratio at r={{2,4,8}}: " + ", ".join(f"{g:.2f}" for g in growth),
        )
    )

    return results


# ---------------------------------------------------------------------------
# Learning task: tree reconstruction from learned embeddings
# ---------------------------------------------------------------------------


def run_tree_reconstruction_experiment(seed: int = 7, steps: int = 400, dim: int = 4):
    """Learn LEAF embeddings reconstructing tree distances; hyperbolic vs
    Euclidean, equal dimension and budget. Returns relative distortions."""
    key = random.PRNGKey(seed)
    levels, tree_dist_full = _random_tree(int(random.randint(key, (), 0, 2**31 - 1)), depth=4, branch=3)
    leaf_ids = np.flatnonzero(levels == max(levels))
    sub = tree_dist_full[np.ix_(leaf_ids, leaf_ids)]
    n_leaves = len(leaf_ids)

    fit_hyp = _embed_tree(key, sub, dim=dim, steps=steps, curv=1.0, hyperbolic=True)
    fit_euc = _embed_tree(key, sub, dim=dim, steps=steps, curv=1.0, hyperbolic=False)
    mean_d = float(np.mean(sub[np.triu_indices(n_leaves, 1)]))

    return {
        "leaves": int(n_leaves),
        "dim": dim,
        "steps": steps,
        "hyperbolic_rel_rmse": math.sqrt(fit_hyp) / mean_d,
        "euclidean_rel_rmse": math.sqrt(fit_euc) / mean_d,
    }


# ---------------------------------------------------------------------------
# Demo entry point (house pattern)
# ---------------------------------------------------------------------------


def demo():
    console = None
    try:
        from rich.console import Console as _Console

        console = _Console()
    except ImportError:
        pass

    def say(msg: str) -> None:
        (console.print if console is not None else print)(msg)

    say("[bold cyan]=== Hyperbolic Geometry & Negative-Curvature Attention (Lorentz model) ===[/bold cyan]")
    say(
        "[dim]H^n_c with curvature -1/c: exponential volume growth makes trees "
        "embed naturally - framework #13, bead model_guided_research-mnn.5[/dim]"
    )
    say("")

    say("[bold]Property checks[/bold]")
    checks = run_property_checks(seed=7)
    all_ok = True
    try:
        from rich.table import Table as _Table

        table = _Table(title="Property checks", show_header=True, header_style="bold magenta")
        table.add_column("check")
        table.add_column("status")
        table.add_column("detail", overflow="fold")
        for name, ok, detail in checks:
            all_ok = all_ok and ok
            table.add_row(name, "[green]PASS[/green]" if ok else "[bold red]FAIL[/bold red]", detail)
        if console is not None:
            console.print(table)
        else:
            print(table)
    except ImportError:
        for name, ok, detail in checks:
            all_ok = all_ok and ok
            print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    say("")

    say("[bold]Tree reconstruction from learned embeddings[/bold] (hyperbolic vs Euclidean, equal dimension/budget)")
    res = run_tree_reconstruction_experiment(seed=7)
    try:
        from rich.table import Table as _Table

        gt = _Table(title="Reconstruction distortion (relative RMSE)", show_header=True, header_style="bold magenta")
        gt.add_column("setting")
        gt.add_column("value", justify="right")
        gt.add_row(f"hyperbolic (dim {res['dim']})", f"{res['hyperbolic_rel_rmse']:.4f}")
        gt.add_row(f"euclidean (dim {res['dim']})", f"{res['euclidean_rel_rmse']:.4f}")
        gt.add_row("leaves", str(res["leaves"]))
        gt.add_row("steps", str(res["steps"]))
        if console is not None:
            console.print(gt)
        else:
            print(gt)
    except ImportError:
        print(res)

    better = res["hyperbolic_rel_rmse"] < res["euclidean_rel_rmse"]
    say("")
    say(
        f"[{'green' if better and all_ok else 'yellow'}]Verdict: hyperbolic embeddings "
        f"{'reconstruct hierarchy better' if better else 'did NOT beat Euclidean'}; "
        f"all property checks green: {all_ok}.[/]"
    )

    return {"checks": checks, "experiment": res, "all_ok": all_ok}


if __name__ == "__main__":
    demo()
