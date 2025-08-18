r"""Simplicial Mixer (JAX): incidence‑only message passing with an exactly conserved scalar invariant.

Core ideas:
• States are cochains on oriented k‑simplices. Signed boundary matrices D\_k encode faces; A\_k := |D\_k| encodes incidence multiplicities.
• A layer alternates down‑lifts (k→k−1) and up‑lifts (k→k+1). Feature updates lie in im(D\_k) and im(D\_{k+1}^T), so their next boundary/coboundary vanishes by D∘D=0. Odd nonlinearities preserve orientation sign‑equivariance.
• A single scalar “mass” channel m\_k is transferred strictly along incidences with zero‑sum flows (via A\_k), making the global sum Σ\_k 1ᵀ m\_k exactly invariant. No same‑dimension mixing; no feature‑similarity attention—only words in {D\_k, D\_k^T}.
• Training encourages cross‑dimension consistency with a Stokes‑style boundary loss:
L\_bdry = Σ\_k || D\_k h\_k − h\_{k−1} ||² + Σ\_k || D\_{k+1}^T h\_k − h\_{k+1} ||².
• Complexity is O(#incidences · d) using sparse SpMV; code includes a synthetic task where labels depend on D\_2 x\_2 (pairwise models have no signal) and a sanity suite that checks mass conservation and homological consistency at every sub‑update.
"""

from functools import partial

import jax
import jax.numpy as jnp

try:
    from jax.experimental.sparse import BCOO

    HAVE_SPARSE = True
except Exception:
    HAVE_SPARSE = False


def _to_dense(M):
    return M.todense() if HAVE_SPARSE and isinstance(M, BCOO) else M


def _make_sparse(x):
    x = jnp.array(x)
    return BCOO.fromdense(x) if HAVE_SPARSE else x


def _spmv(M, v):
    return M @ v if HAVE_SPARSE and isinstance(M, BCOO) else jnp.matmul(M, v)


def _spmm(M, X):
    return M @ X if HAVE_SPARSE and isinstance(M, BCOO) else jnp.matmul(M, X)


def _spt(M):
    return M.T if HAVE_SPARSE and isinstance(M, BCOO) else M.T


def _spabs(M):
    if HAVE_SPARSE and isinstance(M, BCOO):
        return BCOO((jnp.abs(M.data), M.indices), shape=M.shape)
    return jnp.abs(M)


def build_complete_complex_K2(p):
    edges = [(int(i), int(j)) for i in range(p) for j in range(i + 1, p)]
    e2idx = {e: i for i, e in enumerate(edges)}
    n0, n1 = p, len(edges)
    D1 = jnp.zeros((n0, n1), dtype=jnp.float32)
    D1 = D1.at[[u for (u, _) in edges], jnp.arange(n1)].set(-1.0)
    D1 = D1.at[[v for (_, v) in edges], jnp.arange(n1)].set(1.0)
    tris = [(int(i), int(j), int(k)) for i in range(p) for j in range(i + 1, p) for k in range(j + 1, p)]
    n2 = len(tris)
    D2 = jnp.zeros((n1, n2), dtype=jnp.float32)
    for t, (i, j, k) in enumerate(tris):
        e_jk = e2idx[(j, k)]
        e_ik = e2idx[(i, k)]
        e_ij = e2idx[(i, j)]
        D2 = D2.at[e_jk, t].set(1.0)
        D2 = D2.at[e_ik, t].set(-1.0)
        D2 = D2.at[e_ij, t].set(1.0)
    D = [None, _make_sparse(D1), _make_sparse(D2)]
    A = [None, _spabs(D[1]), _spabs(D[2])]
    dims = [n0, n1, n2]
    return {"K": 2, "D": D, "A": A, "dims": dims, "edges": edges, "tris": tris}


def build_flag_complex_from_adjacency(A):
    """Build the 2-skeleton (vertices, edges, triangles) of the flag/clique complex of a graph.

    A is an n x n symmetric 0/1 adjacency with zero diagonal.
    Edges exist where A[u,v]=1 (u<v). Triangles are 3-cliques with orientation (i,j,k), i<j<k.
    Returns the same dict format as build_complete_complex_K2.
    """
    A = jnp.array(A)
    n0 = int(A.shape[0])
    edges = []
    for i in range(n0):
        for j in range(i + 1, n0):
            if int(A[i, j]) == 1:
                edges.append((i, j))
    e2idx = {e: idx for idx, e in enumerate(edges)}
    n1 = len(edges)
    D1 = jnp.zeros((n0, n1), dtype=jnp.float32)
    if n1 > 0:
        D1 = D1.at[[u for (u, _) in edges], jnp.arange(n1)].set(-1.0)
        D1 = D1.at[[v for (_, v) in edges], jnp.arange(n1)].set(1.0)
    tris = []
    for i in range(n0):
        for j in range(i + 1, n0):
            if int(A[i, j]) != 1:
                continue
            for k in range(j + 1, n0):
                if int(A[i, k]) == 1 and int(A[j, k]) == 1:
                    tris.append((i, j, k))
    n2 = len(tris)
    D2 = jnp.zeros((n1, n2), dtype=jnp.float32)
    for t, (i, j, k) in enumerate(tris):
        e_jk = e2idx[(j, k)]
        e_ik = e2idx[(i, k)]
        e_ij = e2idx[(i, j)]
        D2 = D2.at[e_jk, t].set(1.0)
        D2 = D2.at[e_ik, t].set(-1.0)
        D2 = D2.at[e_ij, t].set(1.0)
    D = [None, _make_sparse(D1), _make_sparse(D2)]
    Aabs = [None, _spabs(D[1]), _spabs(D[2])]
    dims = [n0, n1, n2]
    return {"K": 2, "D": D, "A": Aabs, "dims": dims, "edges": edges, "tris": tris}


def build_er_graph_flag_complex(p, edge_prob, key):
    """Sample an undirected Erdos–Renyi graph G(n=p, p=edge_prob) and build its flag complex."""
    keyA, = jax.random.split(key, 1)
    mask = jax.random.bernoulli(keyA, edge_prob, (p, p))
    A = jnp.triu(mask.astype(jnp.int32), k=1)
    A = A + A.T
    return build_flag_complex_from_adjacency(A)

def cycle_indicator_on_edges(p, cycle_vertices, edges=None):
    """Create a cycle indicator vector on edges.

    Args:
        p: Number of vertices (used if edges is None)
        cycle_vertices: List of vertices forming the cycle
        edges: Optional list of edges in the complex. If None, assumes complete graph.

    Returns:
        Indicator vector r with same dimension as number of edges
    """
    if edges is None:
        # Default to complete graph edges
        edges = [(int(i), int(j)) for i in range(p) for j in range(i + 1, p)]

    e2idx = {e: i for i, e in enumerate(edges)}
    r = jnp.zeros((len(edges),), dtype=jnp.float32)

    for i in range(len(cycle_vertices)):
        u = int(cycle_vertices[i])
        v = int(cycle_vertices[(i + 1) % len(cycle_vertices)])
        
        # Determine edge key and sign based on cycle direction
        if u < v:
            edge_key = (u, v)
            sign = 1.0
        else:
            edge_key = (v, u)
            sign = -1.0

        # Only add if edge exists in the complex
        if edge_key in e2idx:
            r = r.at[e2idx[edge_key]].add(sign)

    return r


def glorot(key, m, n):
    k1, k2 = jax.random.split(key)
    lim = jnp.sqrt(6.0 / (m + n))
    return jax.random.uniform(k1, (m, n), minval=-lim, maxval=lim)


def init_params(key, K, d):
    keys = jax.random.split(key, 8 * K + 8)
    Wd = [None] + [glorot(keys[2 * k], d, d) for k in range(1, K + 1)]
    Ud = [None] + [glorot(keys[2 * k + 1], d, d) for k in range(1, K + 1)]
    Wu = [glorot(keys[2 * K + 2 * k], d, d) for k in range(0, K)]
    Uu = [glorot(keys[2 * K + 2 * k + 1], d, d) for k in range(0, K)]
    wgd = [None] + [glorot(keys[4 * K + 2 * k], d, 1) for k in range(1, K + 1)]
    wgu = [glorot(keys[4 * K + 2 * k + 1], d, 1) for k in range(0, K)]
    bgd = [None] + [jnp.zeros((1,)) for _ in range(1, K + 1)]
    bgu = [jnp.zeros((1,)) for _ in range(0, K)]
    w_cls = glorot(keys[-2], d, 1)
    return {"Wd": Wd, "Ud": Ud, "Wu": Wu, "Uu": Uu, "wgd": wgd, "wgu": wgu, "bgd": bgd, "bgu": bgu, "w_cls": w_cls}


def tanh(x):
    return jnp.tanh(x)


def sigm(x):
    return 1 / (1 + jnp.exp(-x))


def step_down(params, complex_, h, m):
    K, D, A = complex_["K"], complex_["D"], complex_["A"]
    dh = [jnp.zeros_like(h[k]) for k in range(K + 1)]
    dm = [jnp.zeros_like(m[k]) for k in range(K + 1)]
    for k in range(1, K + 1):
        Dk, Ak = D[k], A[k]
        gate = sigm(h[k] @ params["wgd"][k] + params["bgd"][k])[:, 0]
        flow = gate * m[k]
        dm[k] = dm[k] - flow
        dm[k - 1] = dm[k - 1] + _spmv(Ak, flow / (k + 1.0))
        dh[k - 1] = dh[k - 1] + _spmm(Dk, h[k] @ params["Wd"][k])
        dh[k] = dh[k] - _spmm(_spt(Dk), h[k - 1] @ params["Ud"][k])
    h_new = [tanh(h[k] + dh[k]) for k in range(K + 1)]
    m_new = [m[k] + dm[k] for k in range(K + 1)]
    return h_new, m_new, dh, dm


def step_up(params, complex_, h, m):
    K, D, A = complex_["K"], complex_["D"], complex_["A"]
    dh = [jnp.zeros_like(h[k]) for k in range(K + 1)]
    dm = [jnp.zeros_like(m[k]) for k in range(K + 1)]
    for k in range(0, K):
        Dkp1, Akp1 = D[k + 1], A[k + 1]
        gate = sigm(h[k + 1] @ params["wgu"][k] + params["bgu"][k])[:, 0]
        flow = gate * m[k + 1]
        dm[k] = dm[k] + _spmv(Akp1, flow / (k + 2.0))
        dm[k + 1] = dm[k + 1] - flow
        dh[k] = dh[k] + _spmm(Dkp1, h[k + 1] @ params["Wu"][k])
        dh[k + 1] = dh[k + 1] - _spmm(_spt(Dkp1), h[k] @ params["Uu"][k])
    h_new = [tanh(h[k] + dh[k]) for k in range(K + 1)]
    m_new = [m[k] + dm[k] for k in range(K + 1)]
    return h_new, m_new, dh, dm


def layer(params, complex_, h, m):
    h1, m1, dh_d, dm_d = step_down(params, complex_, h, m)
    h2, m2, dh_u, dm_u = step_up(params, complex_, h1, m1)
    return h2, m2, (dh_d, dh_u), (dm_d, dm_u)


def boundary_inconsistency(h, complex_):
    K, D = complex_["K"], complex_["D"]
    s = 0.0
    for k in range(1, K + 1):
        s = s + jnp.sum((_spmm(D[k], h[k]) - h[k - 1]) ** 2)
    for k in range(0, K):
        s = s + jnp.sum((_spmm(_spt(D[k + 1]), h[k]) - h[k + 1]) ** 2)
    return s


def total_mass(m):
    return sum([jnp.sum(mk) for mk in m])


def init_state(complex_, d, xK=None, mass_top=1.0):
    K, n = complex_["K"], complex_["dims"]
    h = [jnp.zeros((n[k], d), dtype=jnp.float32) for k in range(K + 1)]
    m = [jnp.zeros((n[k],), dtype=jnp.float32) for k in range(K + 1)]
    if xK is not None:
        h[K] = h[K].at[:, 0].set(xK)
    m[K] = m[K].at[:].set(mass_top)
    return h, m


def forward_L(params, complex_, h0, m0, L):
    def body(_, carry):
        h, m = carry
        h, m, _, _ = layer(params, complex_, h, m)
        return h, m

    return jax.lax.fori_loop(0, L, body, (h0, m0))


def readout_edge(params, h, r):
    y = h[1] @ params["w_cls"]
    return jnp.dot(r, y[:, 0])


def bce_logits(logit, y):
    return jnp.maximum(logit, 0) - logit * y + jnp.log1p(jnp.exp(-jnp.abs(logit)))


def model_loss_single(params, complex_, L, xK, y, r, lam_bdry):
    h0, m0 = init_state(complex_, d=params["w_cls"].shape[0], xK=xK, mass_top=1.0)
    hL, mL = forward_L(params, complex_, h0, m0, L)
    s = readout_edge(params, hL, r)
    ce = bce_logits(s, y)
    bd = boundary_inconsistency(hL, complex_)
    return ce + lam_bdry * bd, (s, bd, total_mass(mL))


def model_loss_batch(params, complex_, L, X, Y, r, lam_bdry):
    f = partial(model_loss_single, complex_=complex_, L=L, r=r, lam_bdry=lam_bdry)
    (losses, (logits, bds, masses)) = jax.vmap(f, in_axes=(None, None, None, 0, 0, None, None))(
        params, complex_, L, X, Y, r, lam_bdry
    )
    return jnp.mean(losses), {"logits": logits, "bd": bds, "mass": masses}


def sgd_update(params, grads, lr):
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


@jax.jit
def train_step(params, complex_, L, X, Y, r, lam_bdry, lr):
    def loss_fn(p):
        loss_val, aux = model_loss_batch(p, complex_, L, X, Y, r, lam_bdry)
        return loss_val, aux

    (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    new_params = sgd_update(params, grads, lr)
    logits = aux["logits"]
    preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
    acc = jnp.mean((preds == Y).astype(jnp.float32))
    bd = jnp.mean(aux["bd"])
    mass = jnp.mean(aux["mass"])
    return new_params, loss_value, acc, bd, mass


def generate_dataset(key, complex_, num, r=None, unbalanced=False):
    K = complex_["K"]
    assert K == 2
    D2 = _to_dense(complex_["D"][2])
    n2 = complex_["dims"][2]

    # If r is not provided, create a default one
    if r is None:
        # Default to a cycle on all vertices
        p = complex_["dims"][0]
        edges = complex_.get("edges", None)
        r = cycle_indicator_on_edges(p, list(range(p)), edges=edges)

    def sample(k):
        k1, k2 = jax.random.split(k)
        sel = jax.random.bernoulli(k1, 0.5, (n2,)).astype(jnp.float32)
        sgn = jax.random.choice(k2, jnp.array([-1.0, 1.0]), (n2,))
        x2 = sel * sgn
        y = ((D2 @ x2) @ r) != 0.0
        if unbalanced:
            y = jnp.where(jax.random.bernoulli(k1, 0.7), 0.0, y.astype(jnp.float32))
        return x2, y.astype(jnp.float32)

    keys = jax.random.split(key, num)
    X, Y = jax.vmap(sample)(keys)
    return X, Y


def sanity_suite(params, complex_, d):
    key = jax.random.PRNGKey(0)
    K = complex_["K"]
    n = complex_["dims"]
    xK = jax.random.normal(key, (n[K],))
    h0, m0 = init_state(complex_, d=d, xK=xK, mass_top=1.0)
    M0 = total_mass(m0)
    h1, m1, dh_d, dm_d = step_down(params, complex_, h0, m0)
    M1 = total_mass(m1)
    h2, m2, dh_u, dm_u = step_up(params, complex_, h1, m1)
    M2 = total_mass(m2)
    D = complex_["D"]

    def norm(x):
        return jnp.sqrt(jnp.sum(x**2))

    eps = 1e-8
    hn = []
    for k in range(1, K + 1):
        if k - 1 >= 1:
            hn.append(norm(_spmm(D[k - 1], dh_d[k - 1])) / (1 + norm(dh_d[k - 1]) + eps))
        else:
            hn.append(0.0)
        if k + 1 <= K:
            hn.append(norm(_spmm(_spt(D[k + 1]), dh_d[k])) / (1 + norm(dh_d[k]) + eps))
        else:
            hn.append(0.0)
    for k in range(0, K):
        if k >= 1:
            hn.append(norm(_spmm(D[k], dh_u[k])) / (1 + norm(dh_u[k]) + eps))
        else:
            hn.append(0.0)
        if k + 2 <= K:
            hn.append(norm(_spmm(_spt(D[k + 2]), dh_u[k + 1])) / (1 + norm(dh_u[k + 1]) + eps))
        else:
            hn.append(0.0)
    bd0 = boundary_inconsistency(h0, complex_)
    bd2 = boundary_inconsistency(h2, complex_)
    return {
        "mass0": float(M0),
        "mass1": float(M1),
        "mass2": float(M2),
        "mass_conservation_down": float(jnp.abs(M1 - M0)),
        "mass_conservation_up": float(jnp.abs(M2 - M1)),
        "homology_norms": [float(v) for v in hn],
        "boundary_loss_before": float(bd0),
        "boundary_loss_after": float(bd2),
    }


def main():
    key = jax.random.PRNGKey(42)
    p = 6
    complex_ = build_complete_complex_K2(p)
    r = cycle_indicator_on_edges(p, list(range(p)), edges=complex_["edges"])
    K = complex_["K"]
    d = 16
    L = 3
    params = init_params(key, K, d)
    key_data = jax.random.PRNGKey(7)
    X_train, Y_train = generate_dataset(key_data, complex_, num=1024, r=r)
    X_test, Y_test = generate_dataset(jax.random.PRNGKey(8), complex_, num=256, r=r)
    lr = 1e-2
    lam_bdry = 1e-4
    batch_size = 64

    def batches(X, Y, bs):
        n = X.shape[0]
        for i in range(0, n, bs):
            yield X[i : i + bs], Y[i : i + bs]

    for epoch in range(10):
        perm = jax.random.permutation(jax.random.fold_in(key, epoch), X_train.shape[0])
        X_train = X_train[perm]
        Y_train = Y_train[perm]
        losses = []
        accs = []
        for Xb, Yb in batches(X_train, Y_train, batch_size):
            params, loss, acc, bd, mass = train_step(params, complex_, L, Xb, Yb, r, lam_bdry, lr)
            losses.append(loss)
            accs.append(acc)
        loss_epoch = float(jnp.mean(jnp.array(losses)))
        acc_epoch = float(jnp.mean(jnp.array(accs)))
        print(
            f"epoch {epoch:02d} | loss {loss_epoch:.4f} | acc {acc_epoch:.3f} | mean_boundary {float(bd):.6f} | mean_mass {float(mass):.3f}"
        )

    @jax.jit
    def eval_batch(X, Y):
        loss_val, aux = model_loss_batch(params, complex_, L, X, Y, r, lam_bdry=0.0)
        preds = (jax.nn.sigmoid(aux["logits"]) > 0.5).astype(jnp.float32)
        acc = jnp.mean((preds == Y).astype(jnp.float32))
        return loss_val, acc

    test_loss, test_acc = eval_batch(X_test, Y_test)
    print(f"test_loss {float(test_loss):.4f} | test_acc {float(test_acc):.3f}")
    s = sanity_suite(params, complex_, d=d)
    print("sanity:", s)


def demo():
    """Run the simplicial complexes and higher order attention demonstration."""
    main()


if __name__ == "__main__":
    demo()
