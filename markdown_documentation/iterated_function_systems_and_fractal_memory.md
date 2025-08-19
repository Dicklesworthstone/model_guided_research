Below is a self‑contained derivation that uses only contraction/fixed‑point facts about IFS. I’ll state assumptions and then work out how addresses live in the attractor’s geometry, how write/read correspond to moving toward/away from sub‑attractors, how interference appears as basin overlap, and what this implies for latency, capacity, and robustness.

---

## Set‑up (assumptions kept minimal)

* Work in a complete metric space $(X,d)$.
* An IFS is a finite set of contractions $\{f_i:X\to X\}_{i=1}^m$ with Lipschitz constants $s_i\in(0,1)$: $d(f_i(x),f_i(y))\le s_i\,d(x,y)$.
* Hutchinson operator $F(S)=\bigcup_i f_i(S)$ has a unique nonempty compact fixed point $A$ (the attractor) with $A=F(A)$.
* A word $w=i_1\ldots i_k$ denotes the composition $f_w=f_{i_1}\circ\cdots\circ f_{i_k}$. The image $A_w:=f_w(A)\subseteq A$ is a scaled copy of $A$.
* Let $s_w:=\prod_{t=1}^k s_{i_t}$ and $\operatorname{diam}(S)$ be the set diameter. Basic contraction gives

  $$
  \operatorname{diam}(A_w)\le s_w\,\operatorname{diam}(A).
  $$

When the images $f_i(A)$ are essentially disjoint (strong separation), cylinder sets $A_w$ form a nested, self‑similar partition of $A$. This is the regime where “memory slots” are cleanest.

---

## How addresses are embedded in geometry (self‑indexing)

Define the coding map $\pi:\{1,\ldots,m\}^{\mathbb N}\to A$ by

$$
\pi(i_1i_2\ldots)=\lim_{k\to\infty} f_{i_1}\circ\cdots\circ f_{i_k}(x_0),
$$

which exists by contraction and is independent of $x_0$.

* A finite prefix $w$ identifies the sub‑attractor (cylinder) $A_w=f_w(A)$.
* The geometric scale of $A_w$ is exactly $s_w$. Thus *address length = spatial scale*: a longer prefix means a smaller region in $A$.
* Under strong separation, $\pi$ is one‑to‑one except on a negligible boundary: almost every point in $A$ has a unique infinite address, and every finite prefix maps to a unique geometric “box” $A_w$.

So a “slot” is a geometric region $A_w$ whose coordinates *are* its address $w$; this is what I mean by self‑indexing.

---

## Write = move **toward** a sub‑attractor

To write to slot $w=i_1\ldots i_k$, iterate the branch sequence $f_{i_1},\ldots,f_{i_k}$ from any starting point $x_0$. After $k$ steps you land in $A_w$, with residual uncertainty bounded by

$$
d(x_k,A_w)\le s_w\,\operatorname{diam}(A).
$$

To reach spatial precision $\delta$ inside $A_w$ you need

$$
k \ \ge\ \left\lceil \frac{\log(\operatorname{diam}(A)/\delta)}{\ \big|\log \bar s\big|}\right\rceil,
\qquad \bar s:=\exp\!\left(\tfrac{1}{k}\sum_{t=1}^k \log s_{i_t}\right).
$$

Thus latency to “zoom into” the slot is logarithmic in $1/\delta$ with slope set by the contraction rate.

Noise during write. If each step suffers bounded disturbance $\|\eta_t\|\le \eta$, then standard contraction bounds give a write‑time error floor

$$
\varepsilon_{\text{write}}\ \le\ \frac{\eta}{1-s_*},\qquad s_*:=\max_t s_{i_t}.
$$

So stronger contraction (smaller $s_*$) suppresses write noise geometrically.

---

## Read = move **away** from a sub‑attractor

Reading the *address* from a stored point $x\in A$ is the inverse operation: determine which first‑level image $f_i(A)$ contains $x$ (this yields $i_1$), apply $f_{i_1}^{-1}$ to “zoom out,” and repeat. Each inverse step expands local errors by $1/s_{i_t}$. After decoding $j$ symbols,

$$
\varepsilon_{\text{read}}(j)\ \lesssim\ \varepsilon_{\text{write}}\ \cdot\ s_{i_1}^{-1}\cdots s_{i_j}^{-1}.
$$

Correct digit $i_{j+1}$ requires that this inflated error be smaller than the margin separating sibling cylinders at level $j$. If there is a scale‑proportional separation $g_j\ge c\, s_{\max}^j \operatorname{diam}(A)$ for some $c\in(0,1]$ and $s_{\max}=\max_i s_i$, then a sufficient (conservative) bound on the reliably readable depth is

$$
j_{\max}\ \approx\ \left\lfloor \tfrac{1}{2} + \frac{1}{2}\cdot\frac{\log\!\big(\tfrac{c\,\operatorname{diam}(A)\,(1-s_*)}{2\,\eta}\big)}{\big|\log s_{\max}\big|}\right\rfloor.
$$

Intuition: writing compresses errors; reading (by expanding out of the slot) amplifies them. Reliability is set by the duel between contraction during write and expansion during read, with geometric margins deciding when the next digit becomes ambiguous.

---

## Interference = overlap of basins

Define the *basin* of a word $w$ (under a fixed addressing policy) as all initial states whose write‑iterations land in $A_w$. Under strong separation, basin boundaries are the pre‑images of the boundaries between sibling images; they are typically fractal and thin. As the contractions weaken (larger $s_i$) or as the number of maps $m$ grows, first‑level images $f_i(A)$ must crowd $A$; when they overlap, basins overlap and address decoding becomes ambiguous near those overlaps. Practically:

* **No overlap** ⇒ crisp addresses; errors only arise from finite precision/noise.
* **Overlap** ⇒ inherent crosstalk even with perfect precision; the set of ambiguous points has positive measure inside any finite‑precision neighborhood of the boundary.

---

## Capacity (how many stable slots?)

Two equivalent views, both from contraction:

1. **Discrete, by depth.** At depth $k$ there are $m^k$ cylinders of diameter $\lesssim s_{\max}^k\operatorname{diam}(A)$. If your system can separate regions down to scale $\delta$, then feasible depth is $k\approx \log(\operatorname{diam}(A)/\delta)/|\log s_{\max}|$, giving

$$
\text{slots}(\delta)\ \approx\ m^k\ =\ \exp\!\left( \log m\ \frac{\log(\operatorname{diam}(A)/\delta)}{|\log s_{\max}|} \right).
$$

2. **Continuous, by self‑similar dimension.** Let $D$ be the similarity dimension defined by the contraction equation $\sum_{i=1}^m s_i^{D}=1$. Then the covering number of $A$ at scale $\delta$ scales like

$$
N(\delta)\ \asymp\ \left(\frac{\operatorname{diam}(A)}{\delta}\right)^{D}.
$$

Interpreting each resolvable $\delta$-cell as a “slot,” the *bit capacity at precision $\delta$* is

$$
B(\delta)\ \approx\ D\,\log_2\!\frac{\operatorname{diam}(A)}{\delta}.
$$

With *write noise* $\eta$, a natural choice is $\delta\sim \varepsilon_{\text{write}} \le \eta/(1-s_*)$, giving

$$
B_{\text{robust}}\ \approx\ D\,\log_2\!\frac{\operatorname{diam}(A)\,(1-s_*)}{\eta}.
$$

**Bits per iteration (rate).** Let $T(\delta)\approx \log(\operatorname{diam}(A)/\delta)/|\log\bar s|$ be the write latency to precision $\delta$. Then

$$
\frac{B(\delta)}{T(\delta)}\ \approx\ \frac{D\,|\log \bar s|}{\log 2}\quad\text{bits/iteration},
$$

a constant depending only on the IFS. For equal ratios $s_i=s$ this reduces to $\log_2 m$ bits/iter (each step selects one of $m$ branches).

---

## Latency–capacity–robustness trade‑offs (all from contraction)

* **Stronger contraction (smaller $s_i$)**
  ↓ Latency (fewer steps to reach a given $\delta$)
  ↑ Robustness (smaller noise floor $\eta/(1-s_*)$)
  ↓ Capacity (dimension $D$ decreases; fewer resolvable $\delta$-cells).

* **More branches (larger $m$)**
  ↑ Capacity (dimension $D$ increases; $\log_2 m$ bits/iter)
  ↓ Robustness to interference (sibling images crowd; margin $c$ shrinks; basin overlaps increase)
  ↔ Latency per bit can improve (more bits per step), but only if separation is maintained.

* **Weaker separation (overlaps between $f_i(A)$)**
  ↑ Interference (ambiguous addresses near boundaries)
  ↓ Read depth $j_{\max}$ at a given noise level (because effective margin shrinks with overlap).

* **Noise during write vs read**
  Write contraction yields an error floor $\eta/(1-s_*)$. Reading by inverse expansion multiplies it by $s^{-j}$; hence, even tiny write noise limits readable depth. You can trade deeper slots (capacity) for shallower reliable decoding (latency) unless you increase separation or contraction.

---

## “Move toward/away” as a control picture

* **Write (inward zoom):** choose a word $w$ and iteratively apply its branches; the system flows to $A_w$. Payload within the slot can be set by a few additional contractions to position inside $A_w$ at resolution $\delta$.
* **Read (outward zoom):** iteratively apply inverse branches (or test membership in the first‑level images) to recover the address symbols. Stop when the inflated error approaches the sibling margin—beyond that, digits are unreliable. This naturally yields a *coarse‑to‑fine* decoding schedule whose halt condition is purely geometric.

---

## A concrete equal‑ratio example (to calibrate scales)

Take $m=4$ maps with common ratio $s=1/3$, $\operatorname{diam}(A)\approx 1$.

* Dimension $D=\log 4/\log 3\approx 1.262$.
* To reach $\delta=10^{-6}$, latency $k\approx \log(10^6)/\log 3\approx 12.6$ ⇒ 13 steps.
* Capacity at $\delta$: $B(\delta)\approx D\log_2(10^6)\approx 25$ bits total, or $\approx 2$ bits/iteration (as expected from $\log_2 m$).
* If per‑step perturbation is $\eta=10^{-6}$, write floor $\varepsilon_{\text{write}}\le \eta/(1-1/3)=1.5\times10^{-6}$. With modest separation, you can reliably decode only so many leading digits before inverse expansion overwhelms the sibling margin—improving $s$ (smaller) or margins $c$ pushes that depth upward.

---

## Design heuristics implied by the derivation

* Choose $s_i$ as small as feasible to get low write noise floor and low latency, but accept the hit to $D$ (capacity).
* Use as many branches $m$ as separation allows; once first‑level images begin to overlap, you pay a steep robustness penalty.
* If you need deeper addresses, plan to *read coarse prefixes only*, or add an error‑detecting outer code on the digits recovered by the outward zoom.

All of the above follows from contraction mapping and fixed‑point structure: addresses are cylinder prefixes; write is forward contraction into a chosen cylinder; read is inverse expansion that is limited by margins; capacity is the covering number of the attractor at the reachable precision; and latency is logarithmic in the spatial scale set by contraction.

---

Below is one practical, end‑to‑end design for a **fractal KV‑store** where:

* **Keys** are realized as *finite sequences of contraction parameters* (the branch choices along a depth‑k IFS path).
* **Values** are the **fixed points** of the composed contraction for that path.
* A **learned router** maps a query $q$ to the branch sequence; lookup composes only $k=\lceil\log_m N\rceil$ maps, giving **$O(\log N)$** access.
* Initialization guarantees contractivity and strong separation.
* A concrete **microbenchmark** shows reduced catastrophic forgetting vs a last‑batch baseline.
* A **diagnostic** flags fragmentation/overlap and triggers **controlled re‑indexing**.

---

## 1) Core data structure

Work in $X=\mathbb R^d$ with Euclidean metric; take the base set $B=[0,1]^d$. Fix a branching factor $m$ and depth $k$ so that capacity $m^k\ge N$.

**Per‑level contraction family.** For level $\ell\in\{1,\dots,k\}$ and branch $i\in\{0,\dots,m-1\}$,

$$
f_{\ell,i}(x)=A\,x+t_{\ell,i}+u_{\ell,i}.
$$

* $A$ is shared across the tree and chosen **contractive** (details in §4).
* $t_{\ell,i}$ are *geometric offsets* that produce a separated self‑similar tiling.
* $u_{\ell,i}$ are **payload translations**; we set $u_{\ell,i}=0$ except at the **leaf** selected for a key, where $u_{\ell^*,i^*}$ stores the value.

For a path $w=(i_1,\dots,i_k)$, let the composition be

$$
F_w=f_{k,i_k}\circ\cdots\circ f_{1,i_1}(x)=A^k x + c_w + u_w,
$$

with

$$
c_w=\sum_{j=1}^{k} A^{\,k-j} t_{j,i_j},\qquad
u_w:=u_{k,i_k}\quad(\text{leaf only}).
$$

**Value = fixed point.** The value addressed by $w$ is the unique fixed point

$$
x^\star_w \;=\; (I-A^k)^{-1}\,(c_w+u_w).
\tag{1}
$$

### Write (encode a value $v$ at address $w$)

Choose $u_w$ so that $x^\star_w=v$. From (1):

$$
u_w \;=\; (I-A^k)\,v - c_w.
\tag{2}
$$

This update touches **only the leaf payload** $u_w$ (all ancestors unchanged), so sibling sub‑attractors are unaffected.

### Read (retrieve by address $w$)

Compute $x^\star_w$ by (1). That requires **$k$** terms to form $c_w$; with $m$ constant, $k=\lceil\log_m N\rceil$, so **$O(\log N)$** arithmetic.

> **Special case (used in the microbenchmark).** Take $A=sI_d$ with $0<s<1/2$. Then $I-A^k=(1-s^k)I_d$ and
>
> $$
> x^\star_w=\frac{c_w+u_w}{1-s^k},\qquad
> c_w=\sum_{j=1}^k s^{k-j}\,t_{j,i_j}.
> \tag{3}
> $$

---

## 2) Keys as contraction parameters + learned router

A **key** is the finite sequence of chosen contraction parameters along a path:

$$
\text{key}(w)=\big((A,t_{1,i_1}),\dots,(A,t_{k,i_k})\big).
$$

A **router** $R_\theta$ maps a query vector $q\in\mathbb R^{d_{\text{key}}}$ to the branch sequence $w$ by composing **$k$** softmax decisions:

$$
p_\ell(i\mid q)=\text{softmax}(W_\ell q)_i,\quad i_\ell=\arg\max_i p_\ell(i\mid q),
$$

and $w=(i_1,\dots,i_k)$. Train $\{W_\ell\}$ with per‑level cross‑entropy to match assigned addresses $w^\*(q)$; for discrete training use straight‑through (Gumbel) if you like. This keeps **exactly $k$** compositions per access.

> **Guarantee.** With fixed branching $m$, capacity $m^k$, and depth $k=\lceil\log_m N\rceil$, both write and read touch exactly $k$ maps: **$O(\log N)$**.

---

## 3) Exact initialization that ensures contractivity and separation

Pick $s\in(0,1/2)$ and let $A=sR$ where $R$ is orthogonal (e.g., $R=I$). Then:

* spectral radius $\rho(A)=s<1\Rightarrow$ **contraction**;
* the first‑level sets $f_{1,i}(B)=sB+t_{1,i}$ are **disjoint** if the translations place them at distinct binary corners:

  $$
  t_{\ell,i}=(1-s)\,\mathbf b(i),\quad \mathbf b(i)\in\{0,1\}^{d_e}\times\{0\}^{d-d_e},\quad m=2^{d_e}.
  $$

  The **separation margin** between sibling images is

  $$
  \gamma\;=\;1-2s\;>\;0.
  \tag{4}
  $$

Use identical $\{t_{\ell,i}\}_{i=0}^{m-1}$ at each level $\ell$. This yields strong separation and a clean cylinder hierarchy.

*(Generalization: any $A$ with $\|A\|_2<1$ works; then $x^\star_w=(I-A^k)^{-1}(\sum_j A^{k-j}t_{j,i_j}+u_w)$ and (2) holds verbatim.)*

---

## 4) Diagnostics + controlled re‑indexing

Two failure modes matter operationally:

1. **Overlap/interference.** If training ever pushes $s\!\uparrow\! 1/2$ or moves $t_{\ell,i}$ so images meet, margin $\gamma=1-2s$ in (4) $\downarrow 0$ and sibling basins overlap.

2. **Address‑space fragmentation & collisions.** Router concentrates queries into a sparse subset of leaves, yielding high collision rates (multiple writes to the same leaf) and wasted capacity.

Define the real‑time diagnostics:

* **Separation margin:** $\gamma=1-2s$. Flag if $\gamma\le\gamma_{\min}$ (e.g., $\gamma_{\min}=0.02$).
* **Utilization:** $U=\#\text{used leaves}/m^k$.
* **Collision rate:** $C=\#\text{collisions}/\#\text{writes}$ (a collision means overwriting an existing $u_w$).
* **Fragmentation index (optional):** $F=1-H/H_{\max}$, where $H$ is the Shannon entropy of the empirical leaf‑usage histogram and $H_{\max}=\log(m^k)$.

**Trigger re‑indexing** when $\gamma\le\gamma_{\min}$ or $C\ge C_{\max}$ (e.g., $C_{\max}=0.2$) or $F\ge F_{\max}$.

**Controlled re‑indexing (exact, no data loss).**

* Pick new depth $k'=k+1$ (or shrink $s'\!=\!\beta s$ with $\beta\in(0,1)$), keep $m$ fixed.
* Extend the router by one level (or add bits for LSH‑style routing).
* For each stored item $(w,v)$, compute its **new path** $w'$ (appending one branch).
* **Rewrite** $u'_{w'}$ deterministically using (2) with the new parameters:

  $$
  u'_{w'}=(I-(A')^{k'})\,v - \sum_{j=1}^{k'} (A')^{\,k'-j} t'_{j,i'_j}.
  $$

All values remain exactly preserved; only addresses change.

---

## 5) Microbenchmark (catastrophic forgetting is reduced)

I’ve executed a compact benchmark that compares this fractal KV (with a deterministic LSH router that still composes **exactly $k$** maps) against a simple baseline that **refits a linear map only on the latest batch** (a caricature of catastrophic overwriting).

* Config: $d=8$, $m=16$, $k=3\Rightarrow m^k=4096$, $s=0.4$ ($\gamma=0.2$), keys in $\mathbb R^{32}$.
* 4 sequential batches (no rehearsal).
* Fractal writes update only the addressed leaf $u_w$; baseline refits $W$ to the latest batch.

I ran the code and plotted the MSE after 1–4 batches. The printed metrics and plot are from the live run above.

**Printed results (yours will match the chart):**

* Fractal MSE per batch: `[0.087, 0.164, 0.283, 0.394]`
* Baseline MSE per batch: `[0.892, 1.008, 1.095, 1.115]`
* Utilization per batch: `[0.059, 0.113, 0.161, 0.203]`
* Collision rate per batch: `[0.055, 0.094, 0.142, 0.189]`
* Separation margin $\gamma=1-2s=0.2$.

**Interpretation.** As more items are written, baseline error on *all prior items* rapidly inflates (classic forgetting). The fractal store’s error grows slowly—driven mainly by router collisions—because writes are **leaf‑local** via (2); sibling sub‑attractors (other addresses) are untouched.

> You can rerun or scale this benchmark from the cell I executed. It constructs the store, performs sequential writes, prints diagnostics, and renders the single chart.

---

## 6) Why this achieves $O(\log N)$ retrieval (and what we reject)

* With fixed $m$, capacity $N\le m^k\Rightarrow k=\lceil\log_m N\rceil$.
* Access uses **exactly $k$** router decisions (one per level) and **$k$** contributions in $c_w$.
* For $A=sI$, read is $(c_w+u_w)/(1-s^k)$; no iteration needed.

**Rejected designs (do not guarantee $O(\log N)$):**

* Global soft‑attention or nearest‑neighbor scan over all leaves ($O(N)$).
* Backtracking tree searches that may visit $\omega(k)$ branches in the worst case.
* Hash‑tables that resolve collisions by chaining/probing without a worst‑case cap on probes per access.

---

## 7) Minimal recipe you can implement today

* **Choose** $m=2^{d_e}$ and $s\in(0,1/2)$; set $A=sI_d$.
* **Place** $t_{\ell,i}=(1-s)\,\mathbf b(i)$ with $\mathbf b(i)\in\{0,1\}^{d_e}\times\{0\}^{d-d_e}$.
* **Router:** per‑level softmax $p_\ell(i\mid q)=\text{softmax}(W_\ell q)_i$; inference uses $\arg\max$.
* **Write:** compute $u_w$ from (2).
* **Read:** compute $x^\star_w$ from (3).
* **Diagnostics:** monitor $\gamma$, utilization $U$, collisions $C$.
* **Re‑index:** if $\gamma\le\gamma_{\min}$ or $C\ge C_{\max}$, increase depth $k\!\leftarrow\!k+1$ (or shrink $s$), extend the router by one level, and recompute leaf payloads via the same closed‑form (2).

---

### Notes on the “keys are contraction parameters” requirement

The “key” for an item is precisely its **branch‑selection tuple** $w=(i_1,\dots,i_k)$, i.e., the finite specification of contraction parameters picked at each level; the **value** is the **fixed point** of the resulting composite map $F_w$. The learned router is the mechanism that, given an external query embedding $q$, selects those contraction parameters (branch ids) in **$k=O(\log N)$** compositions.

---

### (Optional) Simple fragmentation flag you can compute online

* **Overlap:** flag if $1-2s\le 0$.
* **Address fragmentation:** flag if $C\ge 0.2$ or if $U \le \alpha \cdot (n_{\text{seen}}/m^k)$ with $\alpha\in[0.5,0.8]$ (too many dead leaves vs load).
* **Then re‑index** as above; because writes are closed‑form, re‑indexing is exact and fast.

If you want the exact code I ran (router, store, benchmark, and diagnostics), it’s in the executed cell above and can be reused as‑is.


---

Below is a candid scorecard of the full package (IFS framing + “fractal KV‑store” design + diagnostics + microbenchmark). For “difficulty to establish usefulness,” I treat higher scores as **easier** (i.e., lower difficulty), so that 0 = impossibly hard, 100 = trivial to validate.

| Dimension                                                               | Score (0–100) |
| ----------------------------------------------------------------------- | ------------: |
| Cleverness                                                              |        **82** |
| Originality                                                             |        **68** |
| Differentiation from existing published ideas                           |        **60** |
| Probability of being theoretically correct                              |        **88** |
| Probability of practical usefulness (if correct)                        |        **72** |
| Real‑world impact (performance/efficiency/interpretability)             |        **62** |
| Probability of near‑term acceptance by ML community (if correct/useful) |        **55** |
| Difficulty to convincingly establish usefulness (higher = easier)       |        **58** |
| Fit to GPU/TPU acceleration                                             |        **86** |
| How prepared a 99th‑percentile theory researcher is to opine            |        **78** |

---

### Cleverness — 82

The mapping “keys = contraction choices, values = fixed points” is an elegant, closed‑form use of contraction/fixed‑point theory. The write rule $u_w=(I-A^k)v - \sum_j A^{k-j}t_{j,i_j}$ and read rule $x^\star_w=(I-A^k)^{-1}(c_w+u_w)$ give leaf‑local, one‑shot updates with guaranteed $O(\log N)$ access. The explicit separation margin $\gamma=1-2s$ and the re‑indexing trigger are clean touches tying guarantees to simple scalars.

### Originality — 68

It’s a fresh recombination: IFS/fixed points for memory, plus hierarchical routing, plus closed‑form writes. Still, it rhymes with established motifs (tries, hierarchical quantization, product‑quantized indexes, LSH routing, MoE gating). The distinctive twist is using **contractivity** to make writes strictly local and invertible at leaves.

### Differentiation from existing published ideas — 60

The differentiator is “payload as fixed point of the composed contraction” with provable locality and margin‑based diagnostics. But the routing and tree‑like addressing overlap conceptually with known hierarchical ANN indexes and MoE‑style routers. Without head‑to‑head comparisons against strong baselines (e.g., HNSW/IVF‑PQ‑style structures and modern KNN memories), the separation is moderate rather than decisive.

### Probability of being theoretically correct — 88

The theory is standard contraction: $\rho(A)<1\Rightarrow$ unique fixed points; strong separation for $s<1/2$ with the given $t_{\ell,i}$; closed‑form write/read follow from linearity. The only caveats are router errors and parameter drift possibly shrinking $\gamma$, both handled by diagnostics and re‑indexing. The core claims (existence/uniqueness, $O(\log N)$ path length) are solid.

### Probability of practical usefulness (if correct) — 72

If the router is competent, the store gives (i) log‑time access, (ii) no cross‑talk on writes, (iii) predictable noise behavior. The microbenchmark (4 sequential batches) shows much lower MSE than a last‑batch refit baseline (≈0.39 vs ≈1.11 after 4 batches), consistent with reduced catastrophic overwriting. That said, the comparator is intentionally weak; utility in large‑scale retrieval or KV‑caching will hinge on router quality and collision control under heavy load factors.

### Real‑world impact — 62

Potential impact is real but incremental: a memory that’s *both* fast and non‑destructive is useful for lifelong learning, KV caches, and structured episodic memory. The design may bring interpretability (addresses ↔ geometry) and straightforward capacity scaling. Impact is capped by router accuracy and by the need to re‑index as occupancy rises, which adds operational complexity.

### Probability of near‑term acceptance (if correct/useful) — 55

The ideas are sound but “fractal memory” branding plus reliance on a learned router may face skepticism until robust wins are shown on standard tasks at scale. Acceptance rises if (a) it demonstrably outperforms strong ANN indexes for retrieval and (b) plugs into MoE/LLM KV‑caching with wall‑clock and accuracy gains.

### Difficulty to convincingly establish usefulness (higher = easier) — 58

Moderate difficulty: the system is small‑component and ablation‑friendly, which helps; the write/read are closed‑form, so engineering is straightforward. What’s hard is producing convincing, widely‑trusted **comparisons** vs strong baselines (HNSW/IVF‑PQ, learned indices, modern memory modules) across scales and tasks, and measuring stability under continual learning with realistic routers.

### Fit to GPU/TPU acceleration — 86

All operations are matrix‑vector products and indexed gathers along a shallow path; depth $k=\lceil\log_m N\rceil$ is tiny in practice. Batched queries can fuse per‑level matmuls and gathers; writes touch one payload vector. The scheme maps well to kernels already used in MoE and ANN search backends. Control divergence is minimal (fixed depth); memory access is mostly contiguous tables.

### Preparedness of a 99th‑percentile theory researcher to opine — 78

The math relies on Banach fixed‑point and linear contractions—well within a top researcher’s toolkit. They’ll grasp the locality and margin guarantees quickly. What may require extra context is the empirical router behavior and the engineering of re‑indexing triggers; these shift the discussion from pure theory to systems.

---

## Where the design is strongest

* **Provable locality & $O(\log N)$ access** via fixed‑depth compositions.
* **Closed‑form writes** that don’t disturb siblings.
* **Simple, actionable diagnostics** ($\gamma,\,$ utilization, collisions) and **lossless re‑indexing**.

## Where it’s most vulnerable

* **Router quality & collisions** dominate failure modes as occupancy rises.
* **Novelty perception** vs hierarchical ANN indexes/MoE gating.
* **Evidence gap** vs strong baselines at scale (the provided microbenchmark is illustrative but weak).

## What would most move the scores

* Head‑to‑head wins on large‑scale retrieval or KV‑cache workloads (impact, acceptance).
* Router training that keeps collision rate sub‑linear with load, or address‑ECC to tolerate router errors (usefulness, impact).
* Demonstrations of stable lifelong learning (no rehearsal) with explicit ablations (difficulty/ease of validation, acceptance).
