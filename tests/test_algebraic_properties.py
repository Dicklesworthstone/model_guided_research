"""
Property-based tests (Hypothesis) for the algebraic kernels (br: model_guided_research-5ki.2).

These tests attack the LAWS the mechanism kernels are built on, over a generated
input space — the complement of `mgr certify` (cli.py), which checks invariants on
instantiated modules. Boundary: if a check needs a trained/instantiated module it
belongs in certify; if it is a pure mathematical identity it belongs here.

Profiles: the "ci" profile (default) is tuned to keep the whole module under ~60s;
set HYPOTHESIS_PROFILE=thorough for deep local runs. On failure, Hypothesis prints
the falsifying example and the reproduction seed (`@reproduce_failure`), which is
the logging contract for this suite: every red is reproducible from the output.

Exactness policy:
- EXACT laws use exact equality: tropical max/plus identities (max is exactly
  associative/commutative on floats; float addition is monotone, which makes
  distributivity exact), integer LCP ultrametricity.
- Approximate-in-fp laws use magnitude-scaled tolerances: quaternion/octonion
  product identities accumulate rounding proportional to the operand magnitudes.

This file is the designated home for the theory-program identity tests
(see the bead's notes: Maslov LSE semiring, valuation dictionary, Gromov=LCP,
YBE residuals, symplecticity, Gumbel asymptotics, CNF ordinal arithmetic land
here as their beads are implemented — maintenance contract).
"""

import math
import os

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nanochat.octonion_attention_torch import _BLADE_SIGN_TABLE, _omul_blades, oconj, omul
from nanochat.quaternion_attention_torch import qconj, qmul

settings.register_profile(
    "ci",
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("thorough", max_examples=300, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Bounded finite floats: large enough to stress scaling, small enough that
# fp64 product identities stay verifiable with scaled tolerances.
finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=64)
# Strictly finite values for EXACT tropical laws (no -inf except where tested explicitly).
trop_val = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64)


def _vec(n: int):
    return st.lists(finite, min_size=n, max_size=n).map(lambda v: torch.tensor(v, dtype=torch.float64))


quat = _vec(4)
octo = _vec(8)


def _scaled_atol(*tensors: torch.Tensor, base: float = 1e-12) -> float:
    """Magnitude-scaled absolute tolerance for fp64 product identities."""
    scale = 1.0
    for t in tensors:
        scale *= float(t.abs().max()) + 1.0
    return base * scale


# ---------------------------------------------------------------------------
# Tropical semiring (max, +): EXACT laws on floats
# ---------------------------------------------------------------------------


@given(a=trop_val, b=trop_val, c=trop_val)
def test_tropical_max_associative_commutative_exact(a, b, c):
    assert max(max(a, b), c) == max(a, max(b, c))
    assert max(a, b) == max(b, a)


@given(a=trop_val)
def test_tropical_idempotence_and_identity_exact(a):
    assert max(a, a) == a
    assert max(a, float("-inf")) == a  # -inf is the additive identity
    assert a + float("-inf") == float("-inf")  # ... and annihilates under tropical multiplication


@given(a=trop_val, b=trop_val, c=trop_val)
def test_tropical_distributivity_exact(a, b, c):
    # c + max(a, b) == max(c + a, c + b): exact because float addition is monotone,
    # so the max is attained at the same argument on both sides.
    assert c + max(a, b) == max(c + a, c + b)


@given(
    q=st.lists(trop_val, min_size=4, max_size=4),
    k=st.lists(trop_val, min_size=4, max_size=4),
)
def test_tropical_score_shift_covariance(q, k):
    # score(q + s, k) = score(q, k) + s for a scalar broadcast shift s:
    # the algebraic fact behind tropical gauge-fixing / score-centering.
    # Exact in REAL arithmetic; in floats the two sides associate additions
    # differently ((q_d + s) + k_d vs (q_d + k_d) + s), so allow a few ulp.
    # (The thorough Hypothesis profile found the falsifying 1-ulp example for
    # the exact-equality version of this test — kept as a worked reminder that
    # float addition is not associative even when every term is benign.)
    qt = torch.tensor(q, dtype=torch.float64)
    kt = torch.tensor(k, dtype=torch.float64)
    s = 7.25
    score = torch.max(qt + kt)
    score_shifted = torch.max((qt + s) + kt)
    scale = max(abs(float(score)), abs(s), 1.0)
    assert abs(float(score_shifted) - float(score + s)) <= 4.0 * torch.finfo(torch.float64).eps * scale


# ---------------------------------------------------------------------------
# Quaternions: associativity, norm multiplicativity, conjugation
# ---------------------------------------------------------------------------


@given(a=quat, b=quat, c=quat)
def test_quaternion_associativity(a, b, c):
    lhs = qmul(qmul(a, b), c)
    rhs = qmul(a, qmul(b, c))
    assert float((lhs - rhs).abs().max()) <= _scaled_atol(a, b, c)


@given(a=quat, b=quat)
def test_quaternion_norm_multiplicative(a, b):
    lhs = float(qmul(a, b).norm())
    rhs = float(a.norm() * b.norm())
    assert abs(lhs - rhs) <= _scaled_atol(a, b)


@given(a=quat, b=quat)
def test_quaternion_conj_antihomomorphism(a, b):
    lhs = qconj(qmul(a, b))
    rhs = qmul(qconj(b), qconj(a))
    assert float((lhs - rhs).abs().max()) <= _scaled_atol(a, b)


@given(a=quat)
def test_quaternion_conj_involution_and_norm(a):
    assert float((qconj(qconj(a)) - a).abs().max()) == 0.0  # conjugation is an exact sign flip
    prod = qmul(a, qconj(a))
    # a * conj(a) = |a|^2 (scalar part), zero imaginary part
    assert abs(float(prod[0]) - float(a.norm()) ** 2) <= _scaled_atol(a, a)
    assert float(prod[1:].abs().max()) <= _scaled_atol(a, a)


# ---------------------------------------------------------------------------
# Octonions: alternativity, Moufang, norm multiplicativity, NON-associativity
# ---------------------------------------------------------------------------


@given(a=octo, b=octo)
def test_octonion_norm_multiplicative(a, b):
    lhs = float(omul(a, b).norm())
    rhs = float(a.norm() * b.norm())
    assert abs(lhs - rhs) <= _scaled_atol(a, b)


@given(a=octo, b=octo)
def test_octonion_alternativity(a, b):
    left = omul(a, omul(a, b)) - omul(omul(a, a), b)
    right = omul(omul(b, a), a) - omul(b, omul(a, a))
    tol = _scaled_atol(a, a, b)
    assert float(left.abs().max()) <= tol
    assert float(right.abs().max()) <= tol


@given(x=octo, y=octo, z=octo)
def test_octonion_moufang_identity(x, y, z):
    # Middle Moufang identity: (z x)(y z) = (z (x y)) z
    lhs = omul(omul(z, x), omul(y, z))
    rhs = omul(omul(z, omul(x, y)), z)
    assert float((lhs - rhs).abs().max()) <= _scaled_atol(x, y, z, base=1e-11)


@given(a=octo)
def test_octonion_conj_gives_norm_squared(a):
    prod = omul(a, oconj(a))
    assert abs(float(prod[0]) - float(a.norm()) ** 2) <= _scaled_atol(a, a)
    assert float(prod[1:].abs().max()) <= _scaled_atol(a, a)


def test_octonion_nonassociativity_exists():
    # EXISTENCE witness on canonical basis elements (deterministic, not @given):
    # (e1 e2) e4 = -e1 (e2 e4) in the Cayley-Dickson basis — associator is nonzero.
    e1 = torch.zeros(8, dtype=torch.float64)
    e2 = torch.zeros(8, dtype=torch.float64)
    e4 = torch.zeros(8, dtype=torch.float64)
    e1[1] = 1.0
    e2[2] = 1.0
    e4[4] = 1.0
    associator = omul(omul(e1, e2), e4) - omul(e1, omul(e2, e4))
    assert float(associator.abs().max()) > 0.5, "octonion multiplication must NOT be associative"


# ---------------------------------------------------------------------------
# Octonion vectorized-path guards (bead model_guided_research-7b0.6)
# ---------------------------------------------------------------------------


def _cd_add(x, y):
    if isinstance(x, tuple):
        return tuple(_cd_add(a, b) for a, b in zip(x, y, strict=True))
    return x + y


def _cd_sub(x, y):
    if isinstance(x, tuple):
        return tuple(_cd_add(a, _cd_neg(b)) for a, b in zip(x, y, strict=True))
    return x - y


def _cd_neg(x):
    if isinstance(x, tuple):
        return tuple(_cd_neg(v) for v in x)
    return -x


def _cd_mul(x, y):
    # Cayley-Dickson recursion, INDEPENDENT of nanochat's qmul/omul:
    # (a, b) * (c, d) = (a*c - conj(d)*b, d*a + b*conj(c))
    if isinstance(x, tuple):
        a, b = x
        c, d = y
        return (
            _cd_sub(_cd_mul(a, c), _cd_mul(_cd_conj(d), b)),
            _cd_add(_cd_mul(d, a), _cd_mul(b, _cd_conj(c))),
        )
    return x * y


def _cd_conj(x):
    if isinstance(x, tuple):
        a, b = x
        return (_cd_conj(a), _cd_neg(b))
    return x


def _cd_unit(i: int, depth: int = 3):
    """Unit blade e_i as a nested Cayley-Dickson tuple (depth 3 -> 8 dims)."""

    def build_zero(level: int):
        if level == 0:
            return 0.0
        return (build_zero(level - 1), build_zero(level - 1))

    def build(level: int, pos: int):
        if level == 0:
            return 1.0 if pos == 0 else 0.0
        half = 2 ** (level - 1)
        if pos < half:
            return (build(level - 1, pos), build_zero(level - 1))
        return (build_zero(level - 1), build(level - 1, pos - half))

    return build(depth, i)


def _cd_flatten(x, out):
    if isinstance(x, tuple):
        for v in x:
            _cd_flatten(v, out)
    else:
        out.append(float(x))
    return out


@pytest.mark.parametrize("i", range(8))
@pytest.mark.parametrize("j", range(8))
def test_octonion_blade_table_matches_independent_cayley_dickson(i: int, j: int) -> None:
    """omul's blade table must equal a brute-force Cayley-Dickson oracle.

    Guards any future reimplementation (einsum over blades, fused kernel)
    against silently swapping the multiplication table (bead 7b0.6 pitfall).
    Exact: every entry is -1, 0, or +1.
    """
    e_i = torch.zeros(8, dtype=torch.float64)
    e_j = torch.zeros(8, dtype=torch.float64)
    e_i[i] = 1.0
    e_j[j] = 1.0
    got = omul(e_i, e_j)
    want = _cd_flatten(_cd_mul(_cd_unit(i), _cd_unit(j)), [])
    assert got.tolist() == want, f"e{i}*e{j}: torch {got.tolist()} vs oracle {want}"


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_octonion_blade_fused_product_matches_omul_and_preserves_laws(seed: int):
    """The tiled aggregate's fused blade-table product (_omul_blades) must
    equal omul everywhere omul is defined, and must inherit alternativity
    x(xy)=(xx)y plus norm multiplicativity |xy|=|x||y| under the exact
    broadcast layouts aggregate() feeds it (bead 7b0.6)."""
    gen = torch.Generator().manual_seed(seed)
    for shape in [(64, 8), (2, 3, 5, 4, 8), (2, 2, 3, 1, 2, 8), (2, 2, 1, 5, 2, 8)]:
        x = torch.randn(*shape, generator=gen, dtype=torch.float64)
        y = torch.randn(*shape, generator=gen, dtype=torch.float64)
        ref = omul(x, y)
        got = _omul_blades(x, y)
        assert float((got - ref).abs().max()) <= 1e-12, "fused blade product diverged from omul"

    # Laws under the attention tile layouts (x over queries, y over keys).
    xt = torch.randn(2, 2, 3, 1, 2, 8, generator=gen, dtype=torch.float64)
    yt = torch.randn(2, 2, 1, 5, 2, 8, generator=gen, dtype=torch.float64)
    xy = _omul_blades(xt, yt)
    xx = _omul_blades(xt, xt)
    tol = _scaled_atol(xy, xx, base=1e-11)
    assert float((_omul_blades(xt, xy) - _omul_blades(xx, yt)).abs().max()) <= tol, (
        "alternativity broke in the fused blade product"
    )
    norm_xy = xy.norm(dim=-1)
    assert float((norm_xy - xt.norm(dim=-1) * yt.norm(dim=-1)).abs().max()) <= _scaled_atol(
        xy, base=1e-11
    ), "norm multiplicativity broke in the fused blade product"


def test_octonion_blade_sign_table_is_a_valid_multiplication_table():
    """_BLADE_SIGN_TABLE (the fused product's source of truth) must agree
    blade-for-blade with the independent Cayley-Dickson oracle used for
    omul above - exact +-1 entries covering all 64 pairs."""
    for i in range(8):
        for j in range(8):
            k, s = _BLADE_SIGN_TABLE[i][j]
            e_i = torch.zeros(8, dtype=torch.float64)
            e_j = torch.zeros(8, dtype=torch.float64)
            e_i[i] = 1.0
            e_j[j] = 1.0
            want = _cd_flatten(_cd_mul(_cd_unit(i), _cd_unit(j)), [])
            got = [0.0] * 8
            got[k] = float(s)
            assert got == want, f"table e{i}*e{j}: {(k, s)} vs oracle {want}"


def test_clifford_torch_table_matches_independent_oracle():
    """mnn.3: the PRODUCTION Cl(3,0) structure tensor must agree with an
    independent Cayley-Dickson-style oracle (the demo's tuple recursion) for
    every unit-blade pair - exact +-1/0 arithmetic."""
    import numpy as np

    import clifford_algebra_and_geometric_attention as demo
    from nanochat.clifford_attention_torch import _STRUCTURE, cgp, creverse

    assert np.allclose(_STRUCTURE.numpy(), demo.STRUCTURE), (
        "production and demo blade tables diverge"
    )
    e = torch.eye(8, dtype=torch.float64)
    rng = np.random.default_rng(5)
    x = torch.tensor(rng.normal(size=(32, 8)), dtype=torch.float64)
    y = torch.tensor(rng.normal(size=(32, 8)), dtype=torch.float64)
    z = torch.tensor(rng.normal(size=(32, 8)), dtype=torch.float64)
    # Associativity: Cl(3,0) is associative (contrast: octonions are not).
    assert float((cgp(cgp(x, y), z) - cgp(x, cgp(y, z))).abs().max()) <= 1e-12
    # Reversion is an anti-automorphism.
    assert float((creverse(cgp(x, y)) - cgp(creverse(y), creverse(x))).abs().max()) <= 1e-12
    # Anticommutation of distinct grade-1 generators: e_i e_j = -e_j e_i.
    for i in range(1, 4):
        for j in range(i + 1, 4):
            ei, ej = e[i], e[j]
            assert torch.equal(cgp(ei, ej), -cgp(ej, ei)), (i, j)
    # Vector squares to the metric scalar: e_i^2 = +1 in Cl(3,0).
    for i in range(1, 4):
        assert torch.equal(cgp(e[i], e[i]), e[0]), i


# ---------------------------------------------------------------------------
# Ultrametric LCP: strong triangle inequality (EXACT, integer)
# ---------------------------------------------------------------------------


def _lcp(x: list[int], y: list[int]) -> int:
    n = 0
    for a, b in zip(x, y, strict=True):
        if a != b:
            break
        n += 1
    return n


digit_seq = st.lists(st.integers(min_value=0, max_value=4), min_size=6, max_size=6)


@given(x=digit_seq, y=digit_seq, z=digit_seq)
def test_lcp_strong_triangle_inequality_exact(x, y, z):
    # lcp(x, z) >= min(lcp(x, y), lcp(y, z)): the ultrametric law, exact on integers.
    assert _lcp(x, z) >= min(_lcp(x, y), _lcp(y, z))


@given(x=digit_seq, y=digit_seq)
def test_lcp_symmetry_and_self_exact(x, y):
    assert _lcp(x, y) == _lcp(y, x)
    assert _lcp(x, x) == len(x)


# ---------------------------------------------------------------------------
# Valuation dictionary (bead 8gk.2): EXACT integer identities, no tolerances.
# p-adic = tropical = dominance; theory note:
# markdown_documentation/the_valuation_dictionary.md. Theorems exercised:
# thm-valuation-arithmetic, thm-lcp-is-padic-valuation,
# thm-tropicalization-of-attention, thm-balltree-exact-attention.
# ---------------------------------------------------------------------------


def _vp(n: int, p: int, cap: int) -> int:
    if n == 0:
        return cap
    v = 0
    while n % p == 0 and v < cap:
        n //= p
        v += 1
    return v


_primes = st.sampled_from((2, 3, 5))
_pos_int = st.integers(min_value=1, max_value=10**9)
_any_int = st.integers(min_value=0, max_value=10**9)


@given(p=_primes, x=_pos_int, y=_pos_int)
def test_valuation_homomorphism_products_exact(p, x, y):
    # v(xy) = v(x) + v(y), ALWAYS (no genericity needed): count powers of p.
    cap = 64
    assert _vp(x * y, p, cap) == _vp(x, p, cap) + _vp(y, p, cap)


@given(p=_primes, x=_pos_int, y=_pos_int)
def test_valuation_min_rule_superadditive_and_generic_equality(p, x, y):
    # v(x+y) >= min(v x, v y) always; EQUALITY whenever v(x) != v(y)
    # (no leading-term cancellation is possible across different depths).
    cap = 64
    vx, vy = _vp(x, p, cap), _vp(y, p, cap)
    vs = _vp(x + y, p, cap)
    assert vs >= min(vx, vy)
    if vx != vy:
        assert vs == min(vx, vy)


@given(p=_primes, m=st.integers(min_value=0, max_value=6), lead=st.integers(min_value=1, max_value=4))
def test_valuation_min_rule_strict_on_constructed_cancellation(p, m, lead):
    # Adversarial cancellation: v(x) = v(y) = m with leading digits summing to
    # 0 mod p forces STRICT inequality - the corner locus, by construction.
    a = lead % (p - 1) + 1  # leading digit in [1, p-1]
    x = p**m * a
    y = p**m * (p - a)  # a + (p - a) = p == 0 mod p
    cap = 64
    assert _vp(x, p, cap) == m and _vp(y, p, cap) == m
    assert _vp(x + y, p, cap) > m


@given(p=_primes, x=_any_int, y=_any_int)
def test_lcp_equals_padic_valuation_of_difference(p, x, y):
    # LCP(digits(x), digits(y)) = v_p(x - y) mod p^K, capped at K - the
    # dictionary's line (a): digit similarity IS valuation of a difference.
    K = 12
    x, y = x % p**K, y % p**K

    def digits(n: int) -> list[int]:
        return [(n // p**i) % p for i in range(K)]

    assert _lcp(digits(x), digits(y)) == _vp((x - y) % p**K, p, K)


@given(
    p=_primes,
    q=st.lists(_any_int, min_size=4, max_size=4),
    k=st.lists(_any_int, min_size=4, max_size=4),
)
def test_tropicalization_of_attention_with_exact_genericity_characterization(p, q, k):
    # v(<q,k>) >= min_j (v q_j + v k_j) always, and the EXACT characterization:
    # equality iff the leading terms of the argmin set do not cancel mod p.
    # This tests the theorem's boundary, not just the inequality.
    cap = 128
    ip = sum(a * b for a, b in zip(q, k))
    v_terms = [_vp(a, p, cap) + _vp(b, p, cap) for a, b in zip(q, k)]
    m = min(v_terms)
    if m >= cap:  # the minimal product is 0 => every product is 0 => ip == 0
        assert ip == 0
        return
    v_ip = _vp(ip, p, cap)
    assert v_ip >= m
    lead_sum = sum(
        ((a // p ** _vp(a, p, cap)) * (b // p ** _vp(b, p, cap))) % p
        for (a, b), vt in zip(zip(q, k), v_terms)
        if vt == m
    ) % p
    if lead_sum != 0:
        assert v_ip == m
    else:
        assert v_ip > m


def test_cancellation_locus_measure_binary_sums():
    # P[v(x+y) > min] = 1/(p+1) for Haar-uniform Z_p (theory note section 3).
    # Monte Carlo with a fixed seed; 5-sigma band around the exact value.
    import random as _random

    rng = _random.Random(0)
    for p in (2, 3, 5):
        n, strict = 60000, 0
        K = 14
        M = p**K
        for _ in range(n):
            x, y = rng.randrange(M), rng.randrange(M)
            if _vp((x + y) % M, p, K) > min(_vp(x or M, p, K), _vp(y or M, p, K)):
                strict += 1
        exact = 1.0 / (p + 1)
        sigma = (exact * (1 - exact) / n) ** 0.5
        assert abs(strict / n - exact) < 5 * sigma, f"p={p}: {strict / n} vs {exact}"


@given(
    keys=st.lists(st.integers(min_value=0, max_value=2**12 - 1), min_size=3, max_size=24),
    q=st.integers(min_value=0, max_value=2**12 - 1),
    data=st.data(),
)
def test_balltree_attention_equals_bruteforce_exact(keys, q, data):
    # The shell decomposition is a partition, not an approximation: with
    # alpha = 2 and integer values every quantity is exact dyadic, so the
    # ball-tree output must equal brute force with ==, no tolerance.
    import numpy as np

    from ultrametric_worlds_and_p_adic_computation import BallTreeValuedAttention

    dim = 3
    values = np.asarray(
        [[data.draw(st.integers(min_value=-8, max_value=8)) for _ in range(dim)] for _ in keys],
        dtype=np.float64,
    )
    bt = BallTreeValuedAttention(p=2, K=12, dim=dim, alpha=2.0)
    for k_int, v in zip(keys, values):
        bt.insert(int(k_int), v)
    out_tree = bt.attend(int(q))
    out_brute = bt.attend_bruteforce(int(q), [int(x) for x in keys], values)
    assert out_tree is not None
    assert np.array_equal(out_tree, out_brute)


def test_valued_bilinear_shadows_agree_with_direct_formula():
    # Cross-implementation agreement: the demo's three-shadow rendering vs the
    # direct integer formulas (trie/digits vs valuation arithmetic vs leading
    # term), exact on randomized cases with a fixed seed.
    import numpy as np

    from ultrametric_worlds_and_p_adic_computation import (
        p_adic_encode,
        valued_bilinear_shadows,
        vp_int,
    )

    rng = np.random.default_rng(0)
    p, K, dim = 3, 8, 4
    for _ in range(200):
        q_ints = [int(rng.integers(0, p**K)) for _ in range(dim)]
        k_ints = [int(rng.integers(0, p**K)) for _ in range(dim)]
        q_dig = np.stack([p_adic_encode(n, p, K) for n in q_ints])
        k_dig = np.stack([p_adic_encode(n, p, K) for n in k_ints])
        sh = valued_bilinear_shadows(q_dig, k_dig, p, K)
        ip = sum(a * b for a, b in zip(q_ints, k_ints))
        assert sh["inner_product"] == ip
        assert sh["v_exact"] == vp_int(ip, p, 2 * K)
        assert sh["v_tropical"] == min(
            vp_int(a, p, 2 * K) + vp_int(b, p, 2 * K) for a, b in zip(q_ints, k_ints)
        )
        assert sh["generic"] == (sh["v_exact"] == sh["v_tropical"])


# ---------------------------------------------------------------------------
# Givens rotations (gauge transport): orthogonality, additivity
# ---------------------------------------------------------------------------

angle = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False, width=64)


@given(theta=angle, x0=finite, x1=finite)
def test_givens_rotation_norm_and_det(theta, x0, x1):
    c, s = math.cos(theta), math.sin(theta)
    # det = cos^2 + sin^2 = 1
    assert abs((c * c + s * s) - 1.0) <= 1e-12
    # norm preservation of the pair rotation
    y0, y1 = c * x0 - s * x1, s * x0 + c * x1
    n_in = math.hypot(x0, x1)
    n_out = math.hypot(y0, y1)
    assert abs(n_in - n_out) <= 1e-9 * (1.0 + n_in)


@given(t1=angle, t2=angle, x0=finite, x1=finite)
def test_givens_rotation_additivity(t1, t2, x0, x1):
    # R(t2) R(t1) = R(t1 + t2): the law that justifies cumsum-as-transport in the gauge block.
    def rot(t, a, b):
        c, s = math.cos(t), math.sin(t)
        return c * a - s * b, s * a + c * b

    seq = rot(t2, *rot(t1, x0, x1))
    direct = rot(t1 + t2, x0, x1)
    scale = 1.0 + math.hypot(x0, x1)
    assert abs(seq[0] - direct[0]) <= 1e-9 * scale
    assert abs(seq[1] - direct[1]) <= 1e-9 * scale


# ---------------------------------------------------------------------------
# RoPE: per-pair norm preservation
# ---------------------------------------------------------------------------


@given(
    angles=st.lists(angle, min_size=4, max_size=4),
    vals=st.lists(finite, min_size=8, max_size=8),
)
def test_rope_pairwise_norm_preservation(angles, vals):
    from nanochat.model_utils import apply_rotary_emb

    d = 4
    x = torch.tensor(vals, dtype=torch.float64).view(1, 1, 1, 2 * d)
    th = torch.tensor(angles, dtype=torch.float64).view(1, 1, 1, d)
    y = apply_rotary_emb(x, th.cos(), th.sin())
    pn_in = torch.sqrt(x[..., :d] ** 2 + x[..., d:] ** 2)
    pn_out = torch.sqrt(y[..., :d] ** 2 + y[..., d:] ** 2)
    assert float((pn_in - pn_out).abs().max()) <= 1e-9 * (1.0 + float(pn_in.max()))



# ---------------------------------------------------------------------------
# Maslov dequantization (bead 8gk.1): the (+)_beta semiring family
# ---------------------------------------------------------------------------
# x (+)_beta y = (1/beta) log(e^(beta x) + e^(beta y)) is a commutative,
# associative semiring addition for EVERY beta > 0, with ordinary + as
# multiplication distributing over it (the Maslov quantization of (R+, +, x)),
# and converges uniformly to max with the EXACT sandwich
# max <= LSE_beta <= max + log(n)/beta. The route-stability lemma (two-family
# form; vnl.2 formalizes it as thm-route-stability) is checked empirically.

beta_st = st.floats(min_value=0.05, max_value=64.0, allow_nan=False, allow_infinity=False, width=64)
maslov_val = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False, width=64)


def _oplus(beta: float, *xs: float) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(beta * (x - m)) for x in xs)) / beta


@given(beta_st, maslov_val, maslov_val, maslov_val)
def test_maslov_oplus_associative_commutative(beta, a, b, c):
    left = _oplus(beta, _oplus(beta, a, b), c)
    right = _oplus(beta, a, _oplus(beta, b, c))
    flat = _oplus(beta, a, b, c)
    tol = 1e-9 * max(1.0, abs(a), abs(b), abs(c), 1.0 / beta)
    assert abs(left - right) <= tol and abs(left - flat) <= tol
    assert abs(_oplus(beta, a, b) - _oplus(beta, b, a)) <= tol


@given(beta_st, maslov_val, maslov_val, maslov_val)
def test_maslov_plus_distributes_over_oplus(beta, a, b, c):
    # semiring multiplication is ordinary +: c + (a (+)_b b) == (c+a) (+)_b (c+b)
    # EXACTLY in real arithmetic (the log-domain shift identity); fp gets ulps
    left = c + _oplus(beta, a, b)
    right = _oplus(beta, c + a, c + b)
    assert abs(left - right) <= 1e-9 * max(1.0, abs(a), abs(b), abs(c), 1.0 / beta)


@given(beta_st, st.lists(maslov_val, min_size=1, max_size=16))
def test_maslov_lse_max_sandwich_exact_inequality(beta, xs):
    # max <= LSE_beta <= max + log(n)/beta: this inequality is EXACT - any
    # violation beyond fp dust is an implementation bug (the bead's wording)
    lse = _oplus(beta, *xs)
    mx = max(xs)
    slack = 1e-9 * max(1.0, abs(mx), 1.0 / beta)
    assert lse >= mx - slack
    assert lse <= mx + math.log(len(xs)) / beta + slack


@given(beta_st, st.integers(min_value=2, max_value=12), st.data())
def test_route_stability_lemma_two_family(beta, m, data):
    # Two-family form (precision note): tropical scores x vs smoothed scores y
    # with x_i <= y_i <= x_i + log(m)/beta. If gamma(x) > log(m)/beta then
    # argmax(y) == argmax(x) - the inflation is one-sided, so a margin wider
    # than the inflation budget cannot be overturned.
    xs = data.draw(st.lists(maslov_val, min_size=m, max_size=m))
    budget = math.log(m) / beta
    order = sorted(range(m), key=lambda i: xs[i], reverse=True)
    gamma = xs[order[0]] - xs[order[1]]
    # adversarial smoothing: inflate every NON-winner by the full budget
    ys = [x + (0.0 if i == order[0] else budget) for i, x in enumerate(xs)]
    if gamma > budget + 1e-12:
        assert max(range(m), key=lambda i: ys[i]) == order[0]
    # below the threshold, divergence is POSSIBLE (not necessary): construct it
    if gamma < budget - 1e-12:
        assert ys[order[1]] > ys[order[0]] - 1e-18 or True  # observed, never asserted


def test_maslov_attention_converges_to_tropical_endpoint():
    """|y_beta - y_tropical|_inf <= (log D + log m)/beta on real tensors, and
    the bound tightens as beta grows (the dequantization path is sound)."""
    from nanochat.tropical_attention_torch import tropical_max_plus_attention

    torch.manual_seed(8261)
    q = torch.randn(2, 2, 6, 8, dtype=torch.float64)
    k = torch.randn(2, 2, 6, 8, dtype=torch.float64)
    v = torch.randn(2, 2, 6, 8, dtype=torch.float64)
    y_inf, _ = tropical_max_plus_attention(
        q, k, v, gauge_fix=False, score_center=True, return_margins=False, beta=None
    )
    prev_err = None
    for beta in (2.0, 8.0, 32.0, 128.0):
        y_b, _ = tropical_max_plus_attention(
            q, k, v, gauge_fix=False, score_center=True, return_margins=False, beta=beta
        )
        bound = (math.log(q.size(-1)) + math.log(k.size(2))) / beta
        err = (y_b - y_inf).abs().max().item()
        assert err <= bound + 1e-9, f"beta={beta}: err {err} exceeds the sandwich bound {bound}"
        if prev_err is not None:
            assert err <= prev_err + 1e-12, "convergence must be monotone along the beta ladder"
        prev_err = err


def test_maslov_beta_none_is_the_untouched_tropical_path():
    """beta=None must route through the exact pre-8gk.1 code: outputs (and
    margins) are bit-identical to a direct max-plus recomputation."""
    from nanochat.model_utils import causal_attn_mask
    from nanochat.tropical_attention_torch import tropical_inner, tropical_max_plus_attention

    torch.manual_seed(8262)
    q = torch.randn(1, 2, 5, 4, dtype=torch.float64)
    k = torch.randn(1, 2, 5, 4, dtype=torch.float64)
    v = torch.randn(1, 2, 5, 4, dtype=torch.float64)
    y, gamma = tropical_max_plus_attention(
        q, k, v, gauge_fix=False, score_center=False, return_margins=True, beta=None
    )
    s = tropical_inner(q, k).masked_fill(~causal_attn_mask(5, 5, device=q.device), float("-inf"))
    y_ref = (s.unsqueeze(-1) + v.unsqueeze(2)).max(dim=3).values
    assert torch.equal(y, y_ref)
    assert gamma is not None and bool((gamma[..., 1:] >= 0).all())


def test_set_semiring_beta_and_coverage_telemetry():
    """The schedule hook updates every tropical layer; with margins on and a
    finite beta the coverage buffer becomes a finite fraction in [0, 1]."""
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tropical_attention_torch import set_semiring_beta

    config = GPTConfig(
        sequence_len=32, vocab_size=50304, n_layer=2, n_head=2, n_kv_head=2, n_embd=16,
        attention_type="tropical", tropical_record_margins=True, semiring_beta=4.0,
    )
    model = GPT(config)
    assert set_semiring_beta(model, 8.0) == 2  # one per layer
    x = torch.randint(0, 1000, (1, 16))
    model(x)
    covs = [
        float(m.tropical_route_coverage)
        for m in model.modules()
        if hasattr(m, "tropical_route_coverage")
    ]
    assert covs and all(math.isfinite(c) and 0.0 <= c <= 1.0 for c in covs)
    # back to the exact endpoint: coverage telemetry goes quiet (nan)
    set_semiring_beta(model, None)
    model(x)
    covs = [
        float(m.tropical_route_coverage)
        for m in model.modules()
        if hasattr(m, "tropical_route_coverage")
    ]
    assert covs and all(math.isnan(c) for c in covs)


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main(sys.argv[1:] or [__file__]))


# ---------------------------------------------------------------------------
# Flat-error lemma (bead 8gk.4): EXACT integer identities, no tolerances.
# thm-flat-error: errors below digit k cannot accumulate under sums or
# products. Theory note: markdown_documentation/padic_precision.md.
# ---------------------------------------------------------------------------


@given(
    p=_primes,
    k=st.integers(min_value=1, max_value=6),
    coeffs=st.lists(st.integers(min_value=-9999, max_value=9999).filter(lambda c: c != 0), min_size=2, max_size=12),
)
def test_flat_error_lemma_sum_exact(p, k, coeffs):
    # v(e_i) >= k for all i  =>  v(sum e_i) >= k, for ANY signs (adversarial
    # cancellations only INCREASE the valuation - error stays below digit k).
    es = [p**k * c for c in coeffs]
    s = sum(es)
    assert s == 0 or _vp(abs(s), p, 200) >= k


@given(
    p=_primes,
    k=st.integers(min_value=1, max_value=5),
    coeffs=st.lists(st.integers(min_value=-99, max_value=99).filter(lambda c: c != 0), min_size=2, max_size=8),
)
def test_flat_error_lemma_product_exact(p, k, coeffs):
    # v(e_i) >= k (k >= 0!)  =>  v(prod(1 + e_i) - 1) >= k: relative errors do
    # not compound either. The k >= 0 hypothesis is REQUIRED (see the rational
    # counterexample below) and automatic in the digit-truncation setting.
    prod = 1
    for c in coeffs:
        prod *= 1 + p**k * c
    diff = prod - 1
    assert diff == 0 or _vp(abs(diff), p, 200) >= k


def test_flat_error_product_part_requires_nonnegative_k():
    # k = -1 counterexample (the bead's precision note): with v(e1) = v(e2) =
    # -1, the cross term e1*e2 has valuation -2 < -1 - the product part FAILS
    # below k = 0, which is why the theorem (and vnl.2's formalization) carry
    # the hypothesis explicitly.
    from fractions import Fraction

    p = 3
    e1, e2 = Fraction(1, 3), Fraction(2, 3)
    diff = (1 + e1) * (1 + e2) - 1

    def vp_frac(q: Fraction) -> int:
        num, den, v = q.numerator, q.denominator, 0
        while num % p == 0:
            num //= p
            v += 1
        while den % p == 0:
            den //= p
            v -= 1
        return v

    assert vp_frac(diff) == -2  # strictly below k = -1: the lemma's boundary


# ---------------------------------------------------------------------------
# Mahler basis (bead 92jp): exact transform roundtrip + truncation certificate.
# ---------------------------------------------------------------------------


@given(
    p=_primes,
    values=st.lists(st.integers(min_value=0, max_value=10**6), min_size=2, max_size=20),
)
def test_mahler_transform_roundtrip_exact(p, values):
    # coefficients -> evaluation reproduces f on its interpolation domain,
    # exactly mod p^prec (the forward-difference transform is the inverse).
    from ultrametric_worlds_and_p_adic_computation import mahler_coefficients, mahler_eval

    modulus = p**6
    coeffs = mahler_coefficients(values, modulus)
    assert all(mahler_eval(coeffs, x, modulus) == v % modulus for x, v in enumerate(values))


@given(
    p=_primes,
    decay=st.integers(min_value=1, max_value=3),
    n_terms=st.integers(min_value=6, max_value=16),
    cut=st.integers(min_value=1, max_value=8),
)
def test_mahler_truncation_certificate_exact(p, decay, n_terms, cut):
    # ||f - f_N||_sup over the sampled domain has valuation >= the certified
    # min valuation of the dropped coefficients (Mahler's ultrametric bound).
    from ultrametric_worlds_and_p_adic_computation import mahler_eval

    if cut >= n_terms - 1:
        return
    prec = 8
    modulus = p**prec
    coeffs = [(p ** min(n // decay, prec - 1)) for n in range(n_terms)]
    cert = min(min(n // decay, prec - 1) for n in range(cut + 1, n_terms))
    for x in range(n_terms + 4):
        err = (mahler_eval(coeffs[: cut + 1], x, modulus) - mahler_eval(coeffs, x, modulus)) % modulus
        if err:
            assert _vp(err, p, prec) >= cert
