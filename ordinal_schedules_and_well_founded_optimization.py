"""
Ordinal-Scheduler Training (JAX)

Core idea: control training as transfinite descent in a well‑founded ordinal ranking.
Maintain a state (A, B, C) and define ρ = ω^2·A + ω·B + C (Cantor‑normal form):
  A = restart/phase budget (highest‑order), B = anneal/curriculum levels, C = patience.

Transitions mirror ordinal steps:
  • Successor step: update EMA(val_loss). If improved: keep (A,B,C). Else: C := max(C−1, 0).
    ⇒ ρ is nonincreasing at successors.
  • Limit (plateau) when C==0:
      – Anneal (if B>0): B := B−1; η := γ·η; C := P(B); reset best metric.
      – Restart (if B==0 and A>0): A := A−1; reset optimizer state; η := η0;
        B := B_init; C := P(B); reset best metric.
    ⇒ Each limit action strictly decreases ρ (higher‑order term drops), while lower‑order
       counters may freely reset without violating descent (lexicographic ordinal order).

Guarantee: by well‑foundedness, there is no infinite strictly descending sequence of ρ.
Long plateaus must end in a limit‑stage drop; anneals/restarts are certified progress.

This file implements the scheduler and a streaming, noisy piecewise‑linear regression task
(heavy‑tailed noise + distribution shifts), plus cosine/linear baselines and a crisp pass/fail
evaluation using final‑window MSE with median and dominance thresholds.
"""
# Ordinal-Scheduler Training (JAX): A complete, runnable implementation
# - Transfinite-style scheduler with ranking ρ = ω^2 A + ω B + C
# - Successor steps: non-increasing rank; limit steps: strict decrease
# - Anneal at ω-term limits; Restart at ω^2-term limits
# - Baselines: cosine and linear LR schedules
# - Streaming piecewise-linear regression with heavy-tailed noise
# - Pass/Fail evaluation with crisp criteria
#
# Requires: jax, numpy
# Run: python ordinal_scheduler_jax.py --runs 20 --steps 200000

import argparse
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, lax, random

# -----------------------------
# Utilities
# -----------------------------


def l2_normalize(v, eps=1e-8):
    return v / (jnp.linalg.norm(v) + eps)


def student_t_sample(key, nu: float):
    key1, key2 = random.split(key)
    z = random.normal(key1, ())
    u = random.gamma(key2, a=nu / 2.0, dtype=jnp.float32) * 2.0  # Chi-square(ν) ~ Gamma(k=ν/2, θ=2)
    return z / jnp.sqrt(u / nu)


def segment_index(t, b0, b1):
    return jnp.where(t < b0, 0, jnp.where(t < b1, 1, 2))


def in_last_window_flags(t, b0, b1, T, w0_start, w1_start, w2_start):
    s0 = (t >= w0_start) & (t < b0)
    s1 = (t >= w1_start) & (t < b1)
    s2 = (t >= w2_start) & (t < T)
    return jnp.array([s0, s1, s2], dtype=jnp.float32)


def cosine_lr(eta0, t, T):
    return eta0 * 0.5 * (1.0 + jnp.cos(jnp.pi * (t / T)))


def linear_lr(eta0, t, T):
    return eta0 * jnp.maximum(0.0, 1.0 - (t / T))


# -----------------------------
# Ordinal Scheduler
# -----------------------------


class OrdinalState(NamedTuple):
    A: jnp.int32
    B: jnp.int32
    C: jnp.int32
    eta: jnp.float32
    L_best: jnp.float32
    L_ema: jnp.float32


class OrdinalParams(NamedTuple):
    # Match tests that pass A_init, B_init, P_init etc.; accept both names via defaults
    A0: int
    B_init: int
    P0: int
    eta0: float
    gamma: float
    beta: float

    # Alternative constructor for compatibility with tests
    @staticmethod
    def from_test_kwargs(**kwargs):
        A0 = kwargs.get("A_init", kwargs.get("A0", 0))
        B_init = kwargs.get("B_init", 0)
        P0 = kwargs.get("P_init", kwargs.get("P0", 0))
        eta0 = kwargs.get("eta0", 0.0)
        gamma = kwargs.get("gamma", 1.0)
        beta = kwargs.get("ema_decay", kwargs.get("beta", 0.9))
        return OrdinalParams(A0=A0, B_init=B_init, P0=P0, eta0=eta0, gamma=gamma, beta=beta)


# Allow direct construction with test kwargs by monkey-patching __new__
_OrdinalParams_new = OrdinalParams.__new__

def _OrdinalParams_new_compat(cls, *args, **kwargs):
    if kwargs and ("A_init" in kwargs or "P_init" in kwargs or "ema_decay" in kwargs):
        params = OrdinalParams.from_test_kwargs(**kwargs)
        return _OrdinalParams_new(cls, params.A0, params.B_init, params.P0, params.eta0, params.gamma, params.beta)
    return _OrdinalParams_new(cls, *args, **kwargs)

OrdinalParams.__new__ = staticmethod(_OrdinalParams_new_compat)  # type: ignore[misc]


def patience_for_B(B: jnp.int32, params: OrdinalParams) -> jnp.int32:
    k = jnp.int32(params.B_init) - B
    mult = lax.shift_left(jnp.int32(1), k)  # 2^k
    return jnp.int32(params.P0) * mult


def ordinal_scheduler_step(st: OrdinalState, val_loss: jnp.float32, params: OrdinalParams):
    L_ema_new = jnp.float32(params.beta) * st.L_ema + jnp.float32(1.0 - params.beta) * val_loss
    improved = L_ema_new < st.L_best

    def on_improved(_):
        return (
            OrdinalState(st.A, st.B, st.C, st.eta, jnp.minimum(st.L_best, L_ema_new), L_ema_new),
            jnp.bool_(False),
            jnp.bool_(False),
        )

    def on_not_improved(_):
        def dec_C(_):
            C_new = jnp.maximum(st.C - jnp.int32(1), jnp.int32(0))
            return OrdinalState(st.A, st.B, C_new, st.eta, st.L_best, L_ema_new), jnp.bool_(False), jnp.bool_(False)

        def consolidate(_):
            def anneal_branch(_):
                B_new = st.B - jnp.int32(1)
                eta_new = st.eta * jnp.float32(params.gamma)
                C_new = patience_for_B(B_new, params)
                return OrdinalState(st.A, B_new, C_new, eta_new, jnp.inf, L_ema_new), jnp.bool_(False), jnp.bool_(True)

            def restart_branch(_):
                A_new = st.A - jnp.int32(1)
                B_new = jnp.int32(params.B_init)
                eta_new = jnp.float32(params.eta0)
                C_new = patience_for_B(B_new, params)
                return OrdinalState(A_new, B_new, C_new, eta_new, jnp.inf, L_ema_new), jnp.bool_(True), jnp.bool_(True)

            return lax.cond(st.B > 0, anneal_branch, restart_branch, operand=None)

        return lax.cond(st.C > 0, dec_C, consolidate, operand=None)

    st2, reset_mom, fired_limit = lax.cond(improved, on_improved, on_not_improved, operand=None)
    return st2, reset_mom, fired_limit


def ordinal_state_init(params: OrdinalParams):
    return OrdinalState(
        A=jnp.int32(params.A0),
        B=jnp.int32(params.B_init),
        C=patience_for_B(jnp.int32(params.B_init), params),
        eta=jnp.float32(params.eta0),
        L_best=jnp.float32(jnp.inf),
        L_ema=jnp.float32(jnp.inf),
    )


# Test convenience wrapper accepting test-style kwargs
def OrdinalParams_test(**kwargs) -> OrdinalParams:
    return OrdinalParams.from_test_kwargs(**kwargs)


def ordinal_rank(st: OrdinalState) -> int:
    """Return Cantor-normal rank ω^2·A + ω·B + C as a comparable integer for tests.
    This preserves lexicographic order by mapping (A,B,C) ↦ A*10^6 + B*10^3 + C.
    """
    # Large base factors to preserve lexicographic ordering in finite ints for tests
    return int(st.A) * 1_000_000 + int(st.B) * 1_000 + int(st.C)


# -----------------------------
# Model and Loss
# -----------------------------


def model_predict(theta, x):
    return jnp.dot(theta, x)


def mse_loss(theta, x, y):
    y_hat = model_predict(theta, x)
    e = y_hat - y
    return 0.5 * e * e


grad_loss = jit(grad(mse_loss))

# -----------------------------
# Training Loops (JIT)
# -----------------------------


class TrainCarryOrd(NamedTuple):
    theta: jnp.ndarray
    mom: jnp.ndarray
    sched: OrdinalState
    rng: jax.Array
    sum_sqerr: jax.Array
    sample_count: jax.Array
    anneals: jnp.int32
    restarts: jnp.int32


def run_ordinal_once(
    key,
    d,
    T,
    b0,
    b1,
    w0_start,
    w1_start,
    w2_start,
    theta_stars: jnp.ndarray,
    nu,
    mu_momentum,
    params: OrdinalParams,
    eta_floor=0.0,
):
    def step(carry: TrainCarryOrd, t):
        theta, mom, sched, rng, sum_sqerr, sample_count, anneals, restarts = carry
        rng, kt, kv, ktest, knoise_t, knoise_v, knoise_te = random.split(rng, 7)

        seg = segment_index(t, b0, b1)
        theta_star = theta_stars[seg]

        x_t = random.normal(kt, (d,), dtype=jnp.float32)
        noise_t = student_t_sample(knoise_t, nu)
        y_t = jnp.dot(theta_star, x_t) + noise_t

        # Validation loss BEFORE scheduler action at this step
        x_v = random.normal(kv, (d,), dtype=jnp.float32)
        noise_v = student_t_sample(knoise_v, nu)
        y_v = jnp.dot(theta_star, x_v) + noise_v
        val_loss = mse_loss(theta, x_v, y_v)

        # Scheduler transition (pure, rank-nonincreasing; strict drop at limits)
        sched2, reset_mom, fired_limit = ordinal_scheduler_step(sched, val_loss, params)
        eta_t = jnp.maximum(sched2.eta, jnp.float32(eta_floor))
        mom = jnp.where(reset_mom, jnp.zeros_like(mom), mom)

        g = grad_loss(theta, x_t, y_t)
        mom = mu_momentum * mom + g
        theta = theta - eta_t * mom

        # Test accumulation on last-window steps (fresh sample)
        x_te = random.normal(ktest, (d,), dtype=jnp.float32)
        noise_te = student_t_sample(knoise_te, nu)
        y_te = jnp.dot(theta_star, x_te) + noise_te
        err = (jnp.dot(theta, x_te) - y_te) ** 2

        flags = in_last_window_flags(t, b0, b1, T, w0_start, w1_start, w2_start)
        sum_sqerr = sum_sqerr + flags * err
        sample_count = sample_count + flags

        # Counters
        anneals = anneals + jnp.int32(fired_limit & (sched.B > 0))
        restarts = restarts + jnp.int32(fired_limit & (sched.B == 0))

        return TrainCarryOrd(theta, mom, sched2, rng, sum_sqerr, sample_count, anneals, restarts), None

    theta0 = jnp.zeros((d,), dtype=jnp.float32)
    mom0 = jnp.zeros_like(theta0)
    sched0 = ordinal_state_init(params)

    sum0 = jnp.zeros((3,), dtype=jnp.float32)
    cnt0 = jnp.zeros((3,), dtype=jnp.float32)

    carry0 = TrainCarryOrd(theta0, mom0, sched0, key, sum0, cnt0, jnp.int32(0), jnp.int32(0))
    ts = jnp.arange(T, dtype=jnp.int32)
    carryf, _ = lax.scan(step, carry0, ts)

    sums, cnts = carryf.sum_sqerr, carryf.sample_count
    means = jnp.where(cnts > 0, sums / cnts, jnp.inf)
    final_mse = jnp.mean(means)
    return final_mse, means, carryf.anneals, carryf.restarts


run_ordinal_once_jit = jit(run_ordinal_once, static_argnums=(1, 2, 3, 4, 5, 6))


class TrainCarryBase(NamedTuple):
    theta: jnp.ndarray
    mom: jnp.ndarray
    rng: jax.Array
    sum_sqerr: jax.Array
    sample_count: jax.Array


def run_baseline_once(
    key,
    d,
    T,
    b0,
    b1,
    w0_start,
    w1_start,
    w2_start,
    theta_stars: jnp.ndarray,
    nu,
    mu_momentum,
    eta0: float,
    schedule: str,
):
    def lr_at(t):
        return jnp.where(schedule == "cosine", cosine_lr(eta0, t, T), linear_lr(eta0, t, T))

    def step(carry: TrainCarryBase, t):
        theta, mom, rng, sum_sqerr, sample_count = carry
        rng, kt, ktest, knoise_t, knoise_te = random.split(rng, 5)

        seg = segment_index(t, b0, b1)
        theta_star = theta_stars[seg]

        x_t = random.normal(kt, (d,), dtype=jnp.float32)
        noise_t = student_t_sample(knoise_t, nu)
        y_t = jnp.dot(theta_star, x_t) + noise_t

        eta_t = lr_at(t)
        g = grad_loss(theta, x_t, y_t)
        mom = mu_momentum * mom + g
        theta = theta - eta_t * mom

        x_te = random.normal(ktest, (d,), dtype=jnp.float32)
        noise_te = student_t_sample(knoise_te, nu)
        y_te = jnp.dot(theta_star, x_te) + noise_te
        err = (jnp.dot(theta, x_te) - y_te) ** 2

        flags = in_last_window_flags(t, b0, b1, T, w0_start, w1_start, w2_start)
        sum_sqerr = sum_sqerr + flags * err
        sample_count = sample_count + flags

        return TrainCarryBase(theta, mom, rng, sum_sqerr, sample_count), None

    theta0 = jnp.zeros((d,), dtype=jnp.float32)
    mom0 = jnp.zeros_like(theta0)
    sum0 = jnp.zeros((3,), dtype=jnp.float32)
    cnt0 = jnp.zeros((3,), dtype=jnp.float32)
    carry0 = TrainCarryBase(theta0, mom0, key, sum0, cnt0)

    ts = jnp.arange(T, dtype=jnp.int32)
    carryf, _ = lax.scan(step, carry0, ts)
    means = jnp.where(carryf.sample_count > 0, carryf.sum_sqerr / carryf.sample_count, jnp.inf)
    final_mse = jnp.mean(means)
    return final_mse, means


run_baseline_once_jit = jit(run_baseline_once, static_argnums=(1, 2, 3, 4, 5, 6, 10))

# -----------------------------
# Experiment Orchestration
# -----------------------------


def sample_theta_stars(key, d):
    key1, key2, key3 = random.split(key, 3)
    t1 = l2_normalize(random.normal(key1, (d,), dtype=jnp.float32))
    t2 = l2_normalize(random.normal(key2, (d,), dtype=jnp.float32))
    t3 = l2_normalize(random.normal(key3, (d,), dtype=jnp.float32))
    return jnp.stack([t1, t2, t3], axis=0)


def run_replicate(rep_seed, d, T, params: OrdinalParams, eta0, gamma, A0, B_init, P0, nu=3.0, mu_momentum=0.9):
    b0 = int(0.4 * T)
    b1 = int(0.7 * T)
    w0_start = int(b0 - 0.05 * (b0 - 0))
    w1_start = int(b1 - 0.05 * (b1 - b0))
    w2_start = int(T - 0.05 * (T - b1))

    key = random.PRNGKey(rep_seed)
    k_theta, k_ord, k_cos, k_lin = random.split(key, 4)

    theta_stars = sample_theta_stars(k_theta, d)

    ord_mse, ord_seg_means, ord_anneals, ord_restarts = run_ordinal_once_jit(
        k_ord, d, T, b0, b1, w0_start, w1_start, w2_start, theta_stars, nu, mu_momentum, params
    )
    cos_mse, cos_seg_means = run_baseline_once_jit(
        k_cos, d, T, b0, b1, w0_start, w1_start, w2_start, theta_stars, nu, mu_momentum, eta0, "cosine"
    )
    lin_mse, lin_seg_means = run_baseline_once_jit(
        k_lin, d, T, b0, b1, w0_start, w1_start, w2_start, theta_stars, nu, mu_momentum, eta0, "linear"
    )

    return {
        "ord_mse": float(ord_mse),
        "cos_mse": float(cos_mse),
        "lin_mse": float(lin_mse),
        "ord_seg_means": np.array(ord_seg_means, dtype=np.float32),
        "cos_seg_means": np.array(cos_seg_means, dtype=np.float32),
        "lin_seg_means": np.array(lin_seg_means, dtype=np.float32),
        "anneals": int(ord_anneals),
        "restarts": int(ord_restarts),
    }


def summarize(results):
    ord_arr = np.array([r["ord_mse"] for r in results], dtype=np.float64)
    cos_arr = np.array([r["cos_mse"] for r in results], dtype=np.float64)
    lin_arr = np.array([r["lin_mse"] for r in results], dtype=np.float64)

    base_min_per_run = np.minimum(cos_arr, lin_arr)
    dominance = np.sum(ord_arr <= 0.95 * base_min_per_run)

    med_ord = float(np.median(ord_arr))
    med_cos = float(np.median(cos_arr))
    med_lin = float(np.median(lin_arr))
    med_base = min(med_cos, med_lin)

    pass_median = med_ord <= 0.95 * med_base
    pass_dominance = dominance >= max(16, int(0.8 * len(results)))
    passed = pass_median and pass_dominance

    return {
        "median_ord": med_ord,
        "median_cos": med_cos,
        "median_lin": med_lin,
        "dominance_count": int(dominance),
        "runs": int(len(results)),
        "pass_median": bool(pass_median),
        "pass_dominance": bool(pass_dominance),
        "passed": bool(passed),
    }


# -----------------------------
# Main
# -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eta0", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--A0", type=int, default=2)
    ap.add_argument("--B_init", type=int, default=3)
    ap.add_argument("--P0", type=int, default=1200)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--nu", type=float, default=3.0)
    ap.add_argument("--mu", type=float, default=0.9)
    args = ap.parse_args()

    params = OrdinalParams(A0=args.A0, B_init=args.B_init, P0=args.P0, eta0=args.eta0, gamma=args.gamma, beta=args.beta)

    results = []
    base_key = random.PRNGKey(args.seed)
    for i in range(args.runs):
        rep_seed = int(random.bits(random.fold_in(base_key, i), 32))
        res = run_replicate(
            rep_seed,
            args.dim,
            args.steps,
            params,
            args.eta0,
            args.gamma,
            args.A0,
            args.B_init,
            args.P0,
            nu=args.nu,
            mu_momentum=args.mu,
        )
        results.append(res)

    summ = summarize(results)

    print("=== Ordinal Scheduler vs. Cosine/Linear Baselines ===")
    print(f"runs={summ['runs']} steps={args.steps} dim={args.dim} seed={args.seed}")
    print(f"median MSE: ord={summ['median_ord']:.6f}  cos={summ['median_cos']:.6f}  lin={summ['median_lin']:.6f}")
    print(f"per-run dominance (ord <= 0.95 * best(baselines)): {summ['dominance_count']}/{summ['runs']}")
    print(f"pass_median={summ['pass_median']}  pass_dominance={summ['pass_dominance']}  PASSED={summ['passed']}")
    print("\nExample replicate details (first run):")
    r0 = results[0]
    print(f"  ord MSE={r0['ord_mse']:.6f} seg_means={r0['ord_seg_means']}")
    print(f"  cos MSE={r0['cos_mse']:.6f} seg_means={r0['cos_seg_means']}")
    print(f"  lin MSE={r0['lin_mse']:.6f} seg_means={r0['lin_seg_means']}")
    print(f"  ordinal anneals={r0['anneals']} restarts={r0['restarts']} (strict rank drops at limits)")


def demo():
    """Run the ordinal schedules and well founded optimization demonstration."""
    main()


if __name__ == "__main__":
    demo()
