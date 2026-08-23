"""
Clifford Algebra & Geometric Attention — Cl(3,0)
------------------------------------------------

The 12th mathematical framework: geometric (Clifford) algebra as the substrate
for rotation-aware signal flow. Where the octonion demo celebrated the largest
NON-associative division algebra, Cl(3,0) is its disciplined sibling: an
8-dimensional real algebra that is fully ASSOCIATIVE yet still encodes
rotations, orientations, and directed areas natively. The pedagogical contrast
is deliberate and stated in the demo output.

Key ideas embodied here:

- **Programmatic multiplication table:** the 8 basis blades (1, e1, e2, e3,
  e12, e13, e23, e123) are multiplied by a generic blade routine that sorts
  the concatenated index sequence with sign-tracking swaps and contracts
  equal neighbours against the metric diagonal. No hand-typed 8x8 table to
  get wrong; the table the network uses is derived, then verified against
  known identities (e1 e2 e1 = -e2, e12^2 = -1, ...).
- **Dense signed tensor:** the derived table becomes a (8, 8, 8) structure
  constant tensor M; the geometric product of arbitrary multivectors is one
  einsum contraction `gp(a, b)[k] = sum_ij a[i] b[j] M[i, j, k]`. Fully
  differentiable, hardware-friendly.
- **Quaternion subalgebra for free:** the even subalgebra Cl+(3,0) is
  isomorphic to Hamilton's quaternions via (w, x, y, z) |-> w - x e23 -
  y e31 - z e12. The demo checks the geometric product restricted to this
  subalgebra against a direct Hamilton product — the F1 appendix identity,
  now numerical and EXACT.
- **Rotors without matrices:** a rotation by theta in the plane of a unit
  bivector B is R = exp(-theta/2 B); applying it is the sandwich R v ~R.
  Norm preservation is structural, not trained.
- **Equivariance payoff:** a rotor-parameterized regressor trained on
  small-rotation augmentations generalizes to large rotations by
  construction, while an equally-sized MLP degrades — the demo quantifies
  the generalization gap.

Runtime: well under 2 minutes on CPU.
"""

# Docs: markdown_documentation/ (see README section 12 - Clifford algebra &
# geometric attention). Built as bead model_guided_research-mnn.2.

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

# ---------------------------------------------------------------------------
# Basis blades and the programmatic multiplication table
# ---------------------------------------------------------------------------

#: Canonical blade ordering: index -> tuple of generator indices (sorted).
#: Grades: 0:( ) | 1:e1,e2,e3 | 2:e12,e13,e23 | 3:e123.
BLADES: tuple[tuple[int, ...], ...] = (
    (),
    (1,),
    (2,),
    (3,),
    (1, 2),
    (1, 3),
    (2, 3),
    (1, 2, 3),
)

#: Metric diagonal for Cl(3,0): every generator squares to +1.
_METRIC = {1: 1.0, 2: 1.0, 3: 1.0}

_BLADE_INDEX = {blade: i for i, blade in enumerate(BLADES)}


def _blade_product(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], float]:
    """Geometric product of two BASIS blades: returns (canonical blade, sign).

    Derived, not tabulated: concatenate the generator sequences, canonically
    reorder with sign-tracking adjacent swaps (each transposition of distinct
    generators contributes -1), then contract equal neighbours against the
    metric diagonal until stable.
    """
    seq = list(a) + list(b)
    sign = 1.0
    # Phase 1: canonical ordering via bubble passes (odd permutations flip sign).
    moved = True
    while moved:
        moved = False
        for i in range(len(seq) - 1):
            if seq[i] > seq[i + 1]:
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
                sign = -sign
                moved = True
    # Phase 2: contract equal neighbours (sorted => equals are adjacent).
    out: list[int] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == seq[i + 1]:
            sign *= _METRIC[seq[i]]
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out), sign


def _build_structure_tensor() -> np.ndarray:
    """(8, 8, 8) tensor M with a*b = sum_k M[a, b, k] * blade_k."""
    m = np.zeros((8, 8, 8), dtype=np.float64)
    for i, ba in enumerate(BLADES):
        for j, bb in enumerate(BLADES):
            blade, sign = _blade_product(ba, bb)
            m[i, j, _BLADE_INDEX[blade]] = sign
    return m


#: Structure constants as a numpy array (converted to jnp at use sites).
STRUCTURE = _build_structure_tensor()

_STRUCTURE_J = jnp.asarray(STRUCTURE)


def gp(a, b):
    """Geometric product of multivectors (..., 8): one einsum over the
    derived structure tensor. Fully associative — see the demo contrast
    with octonions."""

    return jnp.einsum("...i,...j,ijk->...k", a, b, _STRUCTURE_J)


def reversion(m):
    """Reversion ~M: reverses blade order, sign (-1)^{g(g-1)/2} per grade."""
    grades = jnp.asarray([0, 1, 2, 1, 2, 2, 2, 3])
    signs = jnp.where((grades * (grades - 1) // 2) % 2 == 0, 1.0, -1.0)

    return m * signs


def grade_project(m, grade):
    """Project a multivector onto its grade-`grade` components."""
    grades = jnp.asarray([0, 1, 2, 1, 2, 2, 2, 3])

    return jnp.where(grades == grade, m, 0.0)


def mv_norm(m):
    """Euclidean norm of the 8-component coefficient vector."""

    return jnp.linalg.norm(m)


# ---------------------------------------------------------------------------
# Quaternions inside Cl+(3,0)
# ---------------------------------------------------------------------------

_QUAT_EMBED_IDX = (0, 6, 5, 4)  # w -> 1, x -> e23, y -> e13, z -> e12
_QUAT_EMBED_SIGN = (1.0, -1.0, 1.0, -1.0)  # (w,x,y,z) |-> w - x e23 + y e13 - z e12


def quat_to_clifford(q):
    """Embed Hamilton quaternions (..., 4) into the even subalgebra (..., 8)."""

    parts = [q[..., i] * _QUAT_EMBED_SIGN[i] for i in range(4)]
    zeros = jnp.zeros_like(q[..., 0])
    cols = []
    for blade_idx in range(8):
        if blade_idx in _QUAT_EMBED_IDX:
            k = _QUAT_EMBED_IDX.index(blade_idx)
            cols.append(parts[k])
        else:
            cols.append(zeros)

    return jnp.stack(cols, -1)


def clifford_to_quat(m):
    """Extract (..., 4) Hamilton coefficients from the even subalgebra."""

    comps = [
        m[..., _QUAT_EMBED_IDX[i]] * _QUAT_EMBED_SIGN[i] for i in range(4)
    ]

    return jnp.stack(comps, -1)


def hamilton_qmul(a, b):
    """Direct Hamilton product (..., 4) - the reference the subalgebra must
    reproduce EXACTLY through the Clifford path."""

    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    return jnp.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


# ---------------------------------------------------------------------------
# Rotors
# ---------------------------------------------------------------------------


def rotor_from_bivector(bivector, theta):
    """Rotor R = exp(-theta/2 * B) for a bivector given by its coordinates
    (b12, b13, b23). Pure bivectors square to -|b|^2, so the exponential
    closes exactly: R = cos(|b|*theta/2) - sin(|b|*theta/2)/|b| * B."""
    b = jnp.asarray(bivector)
    norm_b = jnp.linalg.norm(b)
    half_angle = 0.5 * jnp.asarray(theta)
    small = norm_b < 1e-12
    safe_norm = jnp.where(small, 1.0, norm_b)
    scalar = jnp.cos(half_angle * norm_b)
    coeff = jnp.where(small, -half_angle, -jnp.sin(half_angle * norm_b) / safe_norm)

    m = jnp.zeros((8,))
    m = m.at[0].set(scalar)
    m = m.at[4].set(coeff * b[0])
    m = m.at[5].set(coeff * b[1])
    m = m.at[6].set(coeff * b[2])

    return m


def unit_bivector_from_params(params):
    """Map 3 unconstrained parameters to a UNIT bivector (plane selector)."""
    p = jnp.asarray(params)
    n = jnp.linalg.norm(p) + 1e-12
    b = jnp.zeros((3,))
    b = b.at[0].set(p[0] / n)
    b = b.at[1].set(p[1] / n)
    b = b.at[2].set(p[2] / n)

    return b


def apply_rotor(rotor, v):
    """Sandwich product R v ~R for vectors (..., 3)."""
    v_m = jnp.zeros(v.shape[:-1] + (8,)).at[..., 1].set(v[..., 0]).at[..., 2].set(v[..., 1]).at[..., 3].set(v[..., 2])

    out = gp(gp(rotor, v_m), reversion(rotor))

    return out[..., 1:4]


def rotor_compose_matrix(rotor):
    """Matrix of the v |-> R v ~R action on R^3 (columns = images of basis)."""

    cols = []

    for i in range(3):
        e = jnp.zeros((3,)).at[i].set(1.0)
        cols.append(apply_rotor(rotor, e))

    return jnp.stack(cols, -1)


# ---------------------------------------------------------------------------
# Property checks (mirrored in tests/test_mathematical_properties.py)
# ---------------------------------------------------------------------------


def run_property_checks(seed: int = 7) -> list[tuple[str, bool, str]]:
    """Run the five headline property checks; returns (name, ok, detail)."""
    results: list[tuple[str, bool, str]] = []
    key = random.PRNGKey(seed)

    # -- known identities of the derived table -------------------------------
    def _blade_vec(name: str) -> jnp.ndarray:
        idx = _BLADE_INDEX[name]

        return jnp.zeros((8,)).at[idx].set(1.0)

    e1, e2 = _blade_vec((1,)), _blade_vec((2,))
    e1e2e1 = gp(gp(e1, e2), e1)
    id_ok = bool(jnp.allclose(e1e2e1, -e2, atol=0))
    e12_sq = gp(_blade_vec((1, 2)), _blade_vec((1, 2)))
    id_ok = id_ok and bool(jnp.allclose(e12_sq, -jnp.eye(8)[0], atol=0))
    results.append(
        (
            "derived_table_known_identities",
            id_ok,
            "e1 e2 e1 = -e2 and e12^2 = -1 reproduced exactly from the "
            "programmatically built table",
        )
    )

    # -- 1. quaternion subalgebra reduction -----------------------------------
    # The identity is exact over the reals; through an fp32 einsum the
    # reduction order differs from Hamilton's formula, so the check bounds
    # fp32 rounding noise instead of demanding bitwise equality.
    key, k1 = random.split(key)
    qa = random.normal(k1, (64, 4))
    qb = random.normal(random.split(k1)[1], (64, 4))
    ca_, cb_ = quat_to_clifford(qa), quat_to_clifford(qb)
    via_gp = clifford_to_quat(gp(ca_, cb_))
    direct = hamilton_qmul(qa, qb)
    max_diff = float(jnp.max(jnp.abs(via_gp - direct)))
    results.append(
        (
            "quaternion_subalgebra_reduction",
            max_diff <= 1e-6,
            f"max diff via geometric product = {max_diff:.3e}"
            " (Cl+(3,0) even subalgebra, (w,x,y,z)|->w - x e23 + y e13 - z e12;"
            " exact identity in R, fp32 reduction noise numerically)",
        )
    )

    # -- 2. rotor norm preservation ------------------------------------------
    key, k2 = random.split(key)
    planes = random.normal(k2, (16, 3))
    angles = random.uniform(random.split(k2)[1], (16,), minval=-math.pi, maxval=math.pi)
    vecs = random.normal(random.split(k2)[1], (16, 3))
    worst = 0.0
    for plane, ang, v in zip(planes, angles, vecs, strict=True):
        b_unit = unit_bivector_from_params(plane)
        r = rotor_from_bivector(b_unit * jnp.linalg.norm(b_unit), float(ang))
        rv = apply_rotor(r, v[None, :])[0]

        worst = max(worst, float(abs(jnp.linalg.norm(rv) - jnp.linalg.norm(v))))
    results.append(
        (
            "rotor_norm_preservation",
            worst < 1e-5,
            f"worst | ||Rv~R|| - ||v|| | over 16 random rotors = {worst:.3e}",
        )
    )

    # -- 3. grade projections: orthogonality + completeness -------------------
    key, k3 = random.split(key)
    m = random.normal(k3, (32, 8))
    proj = [grade_project(m, g) for g in range(4)]
    completeness = float(jnp.max(jnp.abs(sum(proj) - m)))
    cross = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            dots = jnp.sum(proj[i] * proj[j], -1)
            cross = max(cross, float(jnp.max(jnp.abs(dots))))
    results.append(
        (
            "grade_projection_orthogonal_complete",
            completeness == 0.0 and cross == 0.0,
            f"completeness residual {completeness:.1e}, worst cross-grade inner product {cross:.1e}",
        )
    )

    # -- 4. associativity (contrast with octonions!) ---------------------------
    key, k4 = random.split(key)
    a = random.normal(k4, (64, 8))
    b = random.normal(random.split(k4)[1], (64, 8))
    c = random.normal(random.split(k4)[1], (64, 8))
    lhs = gp(gp(a, b), c)
    rhs = gp(a, gp(b, c))
    assoc = float(jnp.max(jnp.abs(lhs - rhs)))
    results.append(
        (
            "associativity_of_geometric_product",
            assoc < 1e-5,
            f"max |(ab)c - a(bc)| = {assoc:.3e} over 64 random triples"
            " — FULLY associative, unlike the octonion framework",
        )
    )

    # -- 5. rotor composition == rotation composition -------------------------
    key, k5 = random.split(key)
    p1 = random.normal(k5, (3,))
    p2 = random.normal(random.split(k5)[1], (3,))
    th1 = float(random.uniform(random.split(k5)[1], (), minval=0.1, maxval=math.pi))
    th2 = float(random.uniform(random.split(k5)[1], (), minval=0.1, maxval=math.pi))
    r1 = rotor_from_bivector(unit_bivector_from_params(p1), th1)
    r2 = rotor_from_bivector(unit_bivector_from_params(p2), th2)
    m1 = rotor_compose_matrix(r1)
    m2 = rotor_compose_matrix(r2)
    m12 = rotor_compose_matrix(gp(r2, r1))
    gap = float(jnp.max(jnp.abs(m12 - m2 @ m1)))
    results.append(
        (
            "rotor_composition_is_rotation_composition",
            gap < 1e-5,
            f"max |M(R2 R1) - M(R2) M(R1)| = {gap:.3e}",
        )
    )

    return results


def _rotate_about(key, v, axis, angle):
    """Rotate vectors v (N, 3) about a unit axis by angle (Rodrigues)."""

    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    return (
        v * cos_a
        + jnp.cross(axis[None, :], v) * sin_a
        + jnp.outer(v @ axis, axis) * (1.0 - cos_a)
    )


def _make_data(key, *, n_base=32, n_aug=16):
    """Teacher T(v) = rotation about a FIXED hidden axis c by 90 degrees.
    Inputs are augmented by near-identity (train) or large (test) rotations
    about random axes; labels are ALWAYS recomputed as T(input), so every
    pair lies on the teacher's graph. A single rotor can represent T
    exactly; the question is which model class extrapolates there."""
    k_base, k_aug, k_test = random.split(key, 3)
    axis = random.normal(random.fold_in(key, 999), (3,))
    axis = axis / jnp.linalg.norm(axis)

    def teacher(vs):
        return _rotate_about(None, vs, axis, jnp.pi / 2)

    base_v = random.normal(k_base, (n_base, 3))
    base_v = base_v / jnp.linalg.norm(base_v, axis=-1, keepdims=True)

    def augment(k, lo, hi):
        ks = random.split(k, n_aug)
        vs_list = []
        for kk in ks:
            ang = random.uniform(kk, (), minval=lo, maxval=hi)
            ax = random.normal(random.split(kk)[1], (3,))
            ax = ax / jnp.linalg.norm(ax)
            vs_list.append(_rotate_about(None, base_v, ax, ang))
        inputs = jnp.concatenate(vs_list)

        return inputs, teacher(inputs)

    train_v, train_y = augment(k_aug, 0.0, 0.35)  # near-identity rotations
    test_v, test_y = augment(k_test, 0.6 * jnp.pi, jnp.pi)  # far rotations

    return (train_v, train_y), (test_v, test_y), axis


def _rotor_model_loss(params, inputs, targets):
    b_unit = unit_bivector_from_params(params[:3])
    theta = params[3]
    rotor = rotor_from_bivector(b_unit, theta)
    pred = apply_rotor(rotor, inputs)

    return jnp.mean((pred - targets) ** 2)


def _mlp_model_loss(params, inputs, targets):
    hidden = jnp.tanh(inputs @ params["w1"] + params["b1"])
    hidden = jnp.tanh(hidden @ params["w2"] + params["b2"])
    pred = hidden @ params["w3"] + params["b3"]

    return jnp.mean((pred - targets) ** 2)


def run_generalization_experiment(seed: int = 7, steps: int = 600):
    """Train rotor model vs MLP on near-identity rotations; evaluate on far
    rotations. The rotor model gets a few restarts (its 4-parameter landscape
    has local minima); the MLP baseline gets the SAME step budget."""
    import optax

    key = random.PRNGKey(seed)
    (train_v, train_y), (test_v, test_y), _axis = _make_data(key)

    k_rot, k_mlp = random.split(key, 2)
    grad_rot = jax.jit(jax.value_and_grad(_rotor_model_loss))
    grad_mlp = jax.jit(jax.value_and_grad(_mlp_model_loss))

    # Rotor model: 3 plane params + angle; restarts dodge local minima.
    best = (jnp.inf, None)
    for trial in range(4):
        k1, k2 = random.split(random.fold_in(k_rot, trial), 2)
        params_rot = jnp.concatenate(
            [random.normal(k1, (3,)) * 0.5, random.uniform(k2, (), minval=0.3, maxval=3.0)[None]]
        )
        opt_rot = optax.adam(5e-2)
        state_rot = opt_rot.init(params_rot)
        for _ in range(steps):
            _, g_r = grad_rot(params_rot, train_v, train_y)
            updates, state_rot = opt_rot.update(g_r, state_rot)
            params_rot = optax.apply_updates(params_rot, updates)
        final = _rotor_model_loss(params_rot, train_v, train_y)
        if final < best[0]:
            best = (final, params_rot)
    params_rot = best[1]

    # MLP baseline: 3 -> 64 -> 64 -> 3, same step budget.
    k1, k2, k3 = random.split(k_mlp, 3)
    params_mlp = {
        "w1": random.normal(k1, (3, 64)) * 0.3,
        "b1": jnp.zeros(64),
        "w2": random.normal(k2, (64, 64)) * 0.3,
        "b2": jnp.zeros(64),
        "w3": random.normal(k3, (64, 3)) * 0.3,
        "b3": jnp.zeros(3),
    }
    opt_mlp = optax.adam(3e-2)
    state_mlp = opt_mlp.init(params_mlp)
    for _ in range(steps):
        _, g_m = grad_mlp(params_mlp, train_v, train_y)
        updates_m, state_mlp = opt_mlp.update(g_m, state_mlp)
        params_mlp = optax.apply_updates(params_mlp, updates_m)

    b_unit = unit_bivector_from_params(params_rot[:3])
    rotor = rotor_from_bivector(b_unit, params_rot[3])
    pred_rot_train = apply_rotor(rotor, train_v)
    pred_rot_test = apply_rotor(rotor, test_v)
    hidden = jnp.tanh(test_v @ params_mlp["w1"] + params_mlp["b1"])
    hidden = jnp.tanh(hidden @ params_mlp["w2"] + params_mlp["b2"])
    pred_mlp_test = hidden @ params_mlp["w3"] + params_mlp["b3"]
    hidden_tr = jnp.tanh(train_v @ params_mlp["w1"] + params_mlp["b1"])
    hidden_tr = jnp.tanh(hidden_tr @ params_mlp["w2"] + params_mlp["b2"])
    pred_mlp_train = hidden_tr @ params_mlp["w3"] + params_mlp["b3"]

    return {
        "rotor_train_mse": float(jnp.mean((pred_rot_train - train_y) ** 2)),
        "rotor_test_far_mse": float(jnp.mean((pred_rot_test - test_y) ** 2)),
        "mlp_train_mse": float(jnp.mean((pred_mlp_train - train_y) ** 2)),
        "mlp_test_far_mse": float(jnp.mean((pred_mlp_test - test_y) ** 2)),
        "steps": steps,
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
        if console is not None:
            console.print(msg)
        else:
            print(msg)

    say("[bold cyan]=== Clifford Algebra & Geometric Attention (Cl(3,0)) ===[/bold cyan]")
    say(
        "[dim]8 basis blades, programmatic multiplication table, ASSOCIATIVE "
        "(contrast: octonions are not) - bead model_guided_research-mnn.2[/dim]"
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
        console.print(table) if console is not None else print(table)
    except ImportError:
        for name, ok, detail in checks:
            all_ok = all_ok and ok
            print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    say("")

    say("[bold]Rotation-equivariance generalization[/bold] (teacher: 90-degree "
        "rotation about a hidden axis; train on near-identity augmentations)")
    res = run_generalization_experiment(seed=7)
    try:
        from rich.table import Table as _Table

        gt = _Table(title="Generalization (MSE)", show_header=True, header_style="bold magenta")
        gt.add_column("model")
        gt.add_column("train (small rotations)", justify="right")
        gt.add_column("test (large rotations)", justify="right")
        gt.add_column("gap", justify="right")
        rotor_gap = res["rotor_test_far_mse"] - res["rotor_train_mse"]
        mlp_gap = res["mlp_test_far_mse"] - res["mlp_train_mse"]
        gt.add_row(
            "rotor (equivariant)",
            f"{res['rotor_train_mse']:.3e}",
            f"{res['rotor_test_far_mse']:.3e}",
            f"{rotor_gap:+.3e}",
        )
        gt.add_row(
            "MLP baseline",
            f"{res['mlp_train_mse']:.3e}",
            f"{res['mlp_test_far_mse']:.3e}",
            f"{mlp_gap:+.3e}",
        )
        console.print(gt) if console is not None else print(gt)
    except ImportError:
        print(res)

    verdict_rotor = res["rotor_test_far_mse"] < 1e-3
    verdict_gap = res["mlp_test_far_mse"] > 10.0 * max(res["rotor_test_far_mse"], 1e-9)
    say("")
    say(
        f"[{'green' if verdict_rotor and verdict_gap else 'yellow'}]"
        f"Verdict: rotor model {'generalizes' if verdict_rotor else 'did NOT reach low error'}"
        f"; MLP degradation gap {'confirms' if verdict_gap else 'does NOT confirm'} "
        "the equivariance prior.[/]"
    )
    say(f"[bold]All property checks green:[/bold] {all_ok}")

    return {"checks": checks, "experiment": res, "all_ok": all_ok}


if __name__ == "__main__":
    demo()
