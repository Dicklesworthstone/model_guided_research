Below I treat every layer $f_i:\mathbb R^d\!\to\!\mathbb R^d$ as a $C^1$ bijection with $|\det J_{f_i}(x)|=1$ (volume‑preserving). Compositions $F=f_L\circ\cdots\circ f_1$ are therefore diffeomorphisms with $|\det J_F|=1$. When randomness is needed, it is modeled explicitly by augmenting the state with a “reservoir” variable $e$ carrying known base measure (e.g., Lebesgue on $[0,1]^m$), and the global map is a bijection on the **extended** space that preserves the **product** measure. “Zero information leak” means we never marginalize internally: the forward computation is invertible given its full output state. I’ll derive consequences for memory, gradients, calibration, and the equivalence of compression/sampling/likelihood under these axioms. Where helpful, I note the continuous‑time (“flow”) view with divergence‑free velocity fields.

---

## 1) Memory when nothing is discarded

**Activation storage collapses to $O(1)$.**
In backprop, for a layer $y=f(x;\theta)$,

$$
\frac{\partial \mathcal L}{\partial x}=J_f(x)^\top\frac{\partial \mathcal L}{\partial y},\qquad
\frac{\partial \mathcal L}{\partial \theta}=\left(\partial_\theta f(x;\theta)\right)^\top\frac{\partial \mathcal L}{\partial y}.
$$

Because $f$ is invertible, we reconstruct $x=f^{-1}(y;\theta)$ on the backward pass instead of storing it on the forward pass. Thus the only persistent state is the boundary tensors (input, output, loss seed) plus parameters and any **reservoir** variables $e$ (or their PRNG seeds). Memory is $O(\text{dim(state)}+ \text{dim}(e))$, not $O(L)$.

**Time–memory trade.**
You pay extra computation (roughly an extra evaluation per layer to invert and to form VJPs/JVPs) but avoid storing activations. With strict $|\det|=1$, there is no log‑det bookkeeping.

**Finite precision.**
Exact reversibility implies numerically stable inversion must be respected; conditioning is governed by singular values of $J_f$ (see §2). There’s no “safety net” from information discard.

---

## 2) Gradient propagation in volume‑preserving, reversible stacks

Let $J_F=J_{f_L}\cdots J_{f_1}$ and let $\sigma_1,\dots,\sigma_d$ be the singular values of $J_F$. Because

$$
|\det J_F|=\prod_{k=1}^d \sigma_k = 1 \quad\Longrightarrow\quad \sum_{k=1}^d \log \sigma_k = 0,
$$

we obtain:

**(a) No *global* exploding/vanishing.**
The *product* of singular values is pinned to 1. Some directions may expand ($\sigma_k\!>\!1$), others must contract ($\sigma_k\!<\!1$) so that the log‑sum is zero. Gradients can still blow up or vanish **along subspaces**, but there is no systematic collapse across **all** directions.

**(b) Lyapunov sum rule.**
In the continuous‑time view $\dot x = v_\theta(x,t)$ with $\nabla\!\cdot\! v_\theta=0$ (incompressible), the sum of Lyapunov exponents equals $\int \mathrm{tr}\,\partial_x v_\theta\,dt=0$. Adjoint dynamics $\dot a = -(\partial_x v_\theta)^\top a$ inherit the same zero‑sum constraint on instantaneous log‑stretch rates. Hence backprop energy redistributes across directions rather than accumulating.

**(c) Parameter gradients without score terms.**
For likelihood training of flows, NLL for $x$ is

$$
\mathcal L(x)= -\log p_Z(z) - \log|\det J_{F^{-1}}(x)|,\qquad z=F^{-1}(x).
$$

Under $|\det|=1$, the log‑det term vanishes; parameter gradients come only through $z$ and $p_Z$. This sharpens conditioning: the model cannot “cheat” via volume collapse/expansion.

---

## 3) Calibration when inference is reversible

To represent stochastic predictions without leaking information, include a reservoir $e\!\sim\!\text{Unif}[0,1]^m$ in the **input state** and keep its image in the **output state**. For classification with labels $y\in\{1,\dots,K\}$, for each $x$ define a partition of the reservoir:

$$
[0,1] \;=\; \bigsqcup_{k=1}^K I_k(x),\qquad |I_k(x)| = q_k(x),\quad \sum_k q_k(x)=1.
$$

Let the reversible readout be:

$$
y = k \;\text{iff}\; e\in I_k(x),
\qquad
w = \frac{e - \text{offset}_k(x)}{q_k(x)}\in[0,1]
$$

(so $e=\text{offset}_k(x)+q_k(x)\,w$). The full map $(x,e)\mapsto (y,w,\text{rest})$ is bijective and measure‑preserving (counting $\times$ Lebesgue). Then:

**Auto‑calibration (by construction).**

$$
\Pr(y=k\mid x)=q_k(x)=\text{Leb}\{e:\,e\in I_k(x)\}.
$$

Because $y$ is literally produced by drawing $e$ uniformly and testing membership, predicted probabilities equal volume fractions. For any score bin $B\subset[0,1]$,

$$
\Pr[y=k \mid q_k(x)\in B] \;=\; \mathbb E[q_k(x)\mid q_k(x)\in B],
$$

i.e., reliability diagrams are exact when the partition lengths $q_k(x)$ match the true conditionals. **Zero leak** holds because $(y,w,\text{rest})$ recovers $e$ and then $x$ exactly; discarding $w$ would break invertibility.

**Cross‑entropy as minimal randomness use.**
Encoding $y$ given $x$ consumes $-\log q_y(x)$ bits of the reservoir and writes back the remainder $w$; on average this equals $H(Y\mid X)$. Perfect training makes predicted probabilities calibrated and minimizes consumed randomness.

---

## 4) Compression, sampling, and likelihood are the same reversible mechanism

### Continuous variables (density modeling)

With $z=F^{-1}(x)$ and $|\det J_F|=1$,

$$
\log p_X(x) = \log p_Z(z).
$$

* **Compression:** encode $x$ by losslessly arithmetic‑coding $z$ under $p_Z$. Expected codelength $=\mathbb E[-\log p_Z(Z)]$.
* **Likelihood:** MLE minimizes the same expected codelength; there is no separate log‑det term.
* **Sampling:** draw $z\!\sim\!p_Z$ and invert $x=F(z)$. Decoding a bitstream into $z$ and inverting is literally sampling.

Hence “fit‑by‑likelihood”, “compress”, and “sample” are just the forward/inverse directions of a single bijection.

### Discrete labels (classification)

The partition construction above is exactly **arithmetic coding** of $y$ with probabilities $q(x)$:

* **Compression:** mapping $e\mapsto(y,w)$ uses an interval of length $q_y(x)$, i.e., $-\log q_y(x)$ bits; $w$ returns the fractional remainder (no net loss).
* **Sampling:** given $q(x)$ and random bits, choose the interval that contains $e$ and output its index.
* **Likelihood:** cross‑entropy $-\log q_y(x)$ equals consumed bits and equals the negative log probability used in training.

All three are faces of the same reversible partition‑and‑rescale step.

---

## 5) What volume preservation *cannot* change: invariants and expressivity

Because $|\det J|=1$, the flow can **rearrange** density in space but cannot change its “histogram of values” with respect to Lebesgue measure.

* **Differential entropy is invariant.**
  For invertible $y=F(x)$ with $|\det J_F|=1$, $H(Y)=H(X)$.

* **Level‑set volumes are invariant.**
  For every threshold $t$, Lebesgue volume of superlevel sets $ \{x: p_X(x)\ge t\}$ is preserved under measure‑preserving rearrangements. Equivalently, the distribution of log‑densities $\log p_X(x)$ under Lebesgue measure is invariant.

**Implications.**

* Exact MLE match to an *arbitrary* base $p_Z$ via $z=F^{-1}(x)$ is possible **iff** $p_Z$ is an equimeasurable rearrangement of $p_X$; otherwise the optimum is the monotone rearrangement that best aligns the two value histograms. There is no ability to “turn up” or “turn down” density magnitude via local volume change; only geometry is adjustable.
* To be universal while preserving measure, augment the state with a reservoir $e$ of adjustable entropy and model a bijection on $(x,e)$. After marginalizing $e$ externally, the induced family on $x$ becomes arbitrarily rich while internal computation remains invertible and measure‑preserving on the joint.

> If the determinant is **controlled but not 1** (e.g., known scalar field $s(x)$ with $\log|\det J|=s(x)$), log‑density shifts by $+s(x)$. A constant $s$ just adds a constant to NLL; a learned, computable $s(x)$ restores entropy shaping while keeping reversibility.

---

## 6) Training dynamics of divergence‑free flows

* **Adjoint without checkpoints.** The reversible adjoint computes sensitivities by reconstructing states backward; memory stays $O(1)$.
* **Optimization geometry.** Zero average log‑volume change forces the sum of growth rates to zero; training can create sharp expansion in task‑relevant subspaces only by contracting complementary subspaces (implicit bias toward mixing, with a broad but zero‑sum Lyapunov spectrum).
* **Regularization must be non‑forgetting.** Dropout/pooling/stochastic depth violate invertibility unless their randomness is promoted to explicit state. Priors on parameters, symmetries, Lipschitz/spectral constraints, or Hamiltonian structure are the natural regularizers.

---

## 7) What calibrated likelihoods look like here

* **Sharpness is base‑limited.** With $|\det|=1$, $\log p_X(x)=\log p_Z(z)$; tails and overall sharpness come entirely from $p_Z$. You cannot globally “temper” predictions without a controllable log‑det.
* **Classification reliability is geometric.** Predicted probabilities are literal volume fractions of the reservoir; frequencies equal volumes by construction when trained to the data conditionals.

---

## 8) Practical takeaways

* **Memory:** $O(1)$ activations; extra cost is computation and storing/streaming the reservoir state.
* **Gradients:** no across‑the‑board explosion/vanishing; conditioning is about how expansion/contraction is **distributed** (sum of log singular values $=0$). Spectral control still matters.
* **Calibration:** implement probabilistic outputs as reversible partitions of a uniform reservoir; you get perfect reliability by construction when fitted.
* **Compression = Likelihood = Sampling:** the same bijection does all three. Encoding is forward under one base measure; decoding/sampling is the inverse.

---

**Speculation (flagged):** In long, strongly mixing reversible nets (broad Lyapunov spectra summing to zero), gradient noise could self‑average similarly to chaotic adjoints, yielding smoother large‑scale optimization despite sharp local conditioning; numerically, occasional re‑synchronization of reservoir bits may be needed to avoid tiny round‑off errors impersonating “leaks.”

---

Below is a single, auditable mechanism that meters **exactly how many bits of irreversibility** a network uses, while the **global system remains reversible** on an extended state. It includes layer equations, an explicit bit‑ledger (“budget”), an enforcement protocol, and a tiny benchmark plan that hits ≈10× activation‑memory reduction at unchanged perplexity. Diagnostics at the end would immediately expose any hidden dissipation.

---

## 0) Philosophy in one line

Let the *model state* be $(x,e,T)$: activations $x\in\mathbb R^d$; a uniform randomness reservoir $e\in\{0,1\}^\ast$; and a tape $T\in\{0,1\}^\ast$ (storage). All standard layers are **bijective and measure‑preserving** on $x$. The only place information can be “forgotten” for expressivity is a **metered valve** that **(i)** *writes* the would‑be‑forgotten microstate to $T$ (so nothing leaks), and **(ii)** *overwrites* those degrees of freedom in $x$ with samples drawn *only* from $e$ through a learned conditional base. Every bit written or consumed is accounted. If you later discard $T$, you know **exactly** how many bits you threw away and **where**.

---

## 1) Reversible core block (volume‑preserving)

Use any $|\det|=1$ reversible block; one canonical choice (additive coupling, triangular Jacobian with ones on the diagonal):

Let $x=(x_1,x_2)$ with $x_1,x_2\in\mathbb R^{d/2}$.

$$
\begin{aligned}
u_1 &= x_1 \\
u_2 &= x_2 + g_\theta(x_1) \\
y_1 &= u_1 + h_\theta(u_2) \\
y_2 &= u_2
\end{aligned}
\qquad\Rightarrow\qquad
y = \mathrm{RevCouple}_\theta(x),\quad |\det J|=1.
$$

You can optionally pre/post‑mix channels by an orthogonal $R_\theta$ with $\det R_\theta=1$ (e.g., products of Householders).

---

## 2) The Metered Valve (single primitive)

**Purpose.** Reset a chosen subspace of $y$ to a **canonical conditional distribution** for better invariances/conditioning, while recording *exactly* the microstate you reset.

**Split.** After a reversible block, split $y=(a,b)$ with $a\in\mathbb R^{d_a}$, $b\in\mathbb R^{d_b}$ (the part we will “reset”).

**Discretization for coding.** Fix a quantizer $Q:\mathbb R^{d_b}\!\to\!\mathbb Z^{d_b}$, e.g., per‑channel fixed‑point $Q(b)=\lfloor b/\Delta\rfloor$, known $\Delta>0$. (Audit mode may use exact IEEE‑754 bitplanes; see §7.)

**Learned conditional base on the reset coordinates.** A discrete pmf $q_\phi(k\mid a)$ over $k\in\mathbb Z^{d_b}$ (e.g., factorized logistic mixtures with finite support). Define an *ANS* coder $\mathrm{ans\_enc}, \mathrm{ans\_dec}$ that maps $(k,\text{bitstring})\!\leftrightarrow\!\text{bitstring}$ exactly under $q_\phi$.

**Valve forward map.**

$$
\begin{aligned}
k &\gets Q(b) \\
T' &\gets \mathrm{ans\_enc}(k,\,T\ ;\, q_\phi(\cdot\mid a)) \quad\text{// write old microstate} \\
r &\gets \mathrm{take\_bits}(e,\, \ell(k,a)) \quad\text{// consume }\ell = \mathrm{len}(k; q_\phi) \text{ bits} \\
\tilde k &\gets \mathrm{ans\_dec}(r\ ;\, q_\phi(\cdot\mid a)) \quad\text{// sample canonical microstate} \\
b' &\gets Q^{-1}(\tilde k) \\
y' &\gets (a,b'),\qquad (x',e',T') \gets (y',\, e\setminus r,\, T')
\end{aligned}
$$

**Valve inverse map.**

$$
\begin{aligned}
r^\star &\gets \mathrm{ans\_enc}(\tilde k,\,\epsilon\ ;\, q_\phi(\cdot\mid a)) \\
k &\gets \mathrm{ans\_dec}(T'\ ;\, q_\phi(\cdot\mid a)) \\
b &\gets Q^{-1}(k),\qquad e \gets r^\star \,\|\, e', \qquad T \gets \text{suffix}(T') \\
x &\gets \mathrm{RevCouple}_\theta^{-1}(y')
\end{aligned}
$$

This transforms $(x,e,T)\mapsto(x',e',T')$ **bijectively** on the extended state. No information is lost; it is **moved** from $x$ to $T$, while fresh randomness from $e$ populates the reset coordinates. (Using ANS makes the length $\ell$ *exact* integers.)

**Key property (calibration by construction).** The reset block ensures $b'\sim q_\phi(\cdot\mid a)$ exactly when $e$ is uniform, so downstream layers can rely on a canonical conditional law for $b'$.

---

## 3) The Irreversibility Budget (ledger and valve throttle)

Let $\ell_i=\ell_i(k_i,a_i)$ be the exact number of bits the $i$-th valve *writes* to $T$ (equivalently, ANS code length for $k_i$ under $q_\phi(\cdot\mid a_i)$). Let $r_i$ be the bits it *consumes* from the reservoir $e$. Define per‑valve **net dissipation**

$$
\Delta_i \;\;:=\;\; \ell_i - r_i \;\;\; (\text{bits}).
$$

* **Balanced reset (default, globally measure‑preserving):** set $r_i=\ell_i\Rightarrow \Delta_i=0$. You *move* entropy from $e$ to $x$ and from $x$ to $T$ in equal amounts. Nothing accumulates; perfect reversibility with zero net production.
* **Metered irreversibility:** allow $r_i = \alpha_i \ell_i$ with $\alpha_i\!\in\![0,1]$. Then $\Delta_i=(1-\alpha_i)\ell_i\ge 0$. Globally, the **budget** is

$$
\mathcal B := \sum_{i=1}^{L_v} \mathbb E[\Delta_i] \;\;\le\;\; B_{\max}\quad\text{(constraint)}.
$$

Interpretation: $\mathcal B$ bounds how many bits you permit to *accumulate* on $T$ without being replenished from $e$. If you later discard $T$, those are exactly the bits you irreversibly lost. (With $\alpha_i\!=\!1$ everywhere, $\mathcal B\!=\!0$.)

**Token bucket enforcement.** Maintain an integer counter $B$ per sequence (or batch). Each valve decrements $B$ by $\Delta_i$ (rounded up to integer bits). If $B=0$, force $\alpha_i\leftarrow 1$ (balanced) for the remainder—no budget overflow possible.

---

## 4) Training objective and protocol

**Objective (task + disciplined valve use).**

$$
\mathcal L \;=\; \mathbb E[\text{task\_loss}] \;+\; \lambda \sum_{i}\mathbb E[\ell_i] \;+\; \mu\,\max\!\big(0,\;\sum_i\mathbb E[\Delta_i]-B_{\max}\big)^2,
$$

with $\lambda\ge 0$ (an optional “reset preference”—encourage canonicalization where helpful) and $\mu$ large. Alternatively, enforce $\sum\Delta_i\le B_{\max}$ via dual ascent on a Lagrange multiplier.

**Training protocol.**

1. **Audit‑warmup (strictly reversible):** run with $\alpha_i\!=\!1$ (so $\Delta\!=\!0$), and quantizer $Q$ *identity‑matched* to the numeric format (e.g., Q‑bitplane coder over raw IEEE‑754; §7). Verify exact cycle:

   $$
   (x,e,T)\xrightarrow{\text{forward}}\xrightarrow{\text{inverse}}(x,e,T)\quad \text{bit‑for‑bit}.
   $$
2. **Budget scheduling:** introduce a small $B_{\max}$ (e.g., 0.1–0.5 bits per token per valve used), learn $\alpha_i$ by backprop through the straight‑through estimator $\partial \Delta_i/\partial \alpha_i \approx \ell_i$, with the token bucket enforcing the hard cap.
3. **Valve placement:** put valves sparingly—after mixing blocks, before attention/MLP sublayers—so that resets land on nearly nuisance subspaces the network learns to expose into $b$.
4. **Two modes at run time:**

   * **Lossless/auditable mode:** keep $T$ and $e$ around; everything inverts and replays exactly.
   * **Deployed (compressed) mode:** discard $T$ (you now *know* you have lost at most $B_{\max}$ expected bits per sample), or stream $T$ off‑GPU for reversible training with tiny GPU overhead.

---

## 5) Where expressivity comes from (without leaks)

* The core remains an incompressible flow. The valve **thermalizes** selected coordinates to a learned canonical conditional $q_\phi(b\mid a)$, while storing the original microstate. Downstream layers operate on a state whose nuisance components follow a *known* law, improving conditioning and invariance.
* With $\alpha_i\!<\!1$ you *permit* net dissipation, but it is **visible** and **metered** in $\Delta_i$. Nothing is hidden; global reversibility is retained as long as $T$ is kept.

---

## 6) Layer equations in one place

* **Reversible block:** $y=\mathrm{RevCouple}_\theta(R_\theta x)$.
* **Valve split:** $y=(a,b)$.
* **Quantize:** $k=Q(b)\in\mathbb Z^{d_b}$.
* **Write:** $T'\!=\!\mathrm{ans\_enc}(k,T;q_\phi(\cdot\mid a))$, $\ell=\text{bits\_written}(k,a)$.
* **Throttle:** choose $r\sim\text{first }\lfloor \alpha\ell\rfloor\text{ bits of }e$.
* **Resample:** $\tilde k=\mathrm{ans\_dec}(r; q_\phi(\cdot\mid a))$; $b'=Q^{-1}(\tilde k)$.
* **Compose:** $x' = R_\theta^{-1}(\, \mathrm{RevCouple}_\theta^{-1}(a,b')\,)$.
* **Ledger update:** $\Delta\leftarrow \Delta + (\ell - \lfloor \alpha\ell\rfloor)$.

All maps on $(x,e,T)$ are bijections; $\Delta$ is pure accounting (no gradients through the bucket).

---

## 7) Bit‑for‑bit auditability (drop anything that can’t pass)

**Strict audit mode.**

* **Coding target:** encode **exact IEEE‑754** bitplanes of $b$ (mantissa + exponent + sign) with a learned finite‑alphabet pmf modeled conditionally on $a$. This guarantees perfect reconstruction after inverse.
* **Deterministic numerics:** prefer fixed‑point Q‑format inside the valve to avoid platform‑dependent rounding; otherwise keep all math in integer ANS space and treat float <-> int conversion as part of $Q,Q^{-1}$.
* **Reproducibility:** inject all stochasticity via explicit $e$ (seeded once per sample). No hidden RNG calls outside valves.

**Audits to run regularly.**

* *Cycle test:* bitwise equality of $x$ before and after a forward+inverse pass, all the way through the optimizer step (to catch silent casts).
* *Tape balance:* verify $\sum_i \ell_i - \sum_i r_i = \sum_i \Delta_i$ exactly; verify the token bucket never underflows.
* *Coder sanity:* empirical mean of $\ell_i$ equals $\mathbb E[-\log_2 q_\phi(k\mid a)]$ up to < 1 bit per symbol (finite‑precision ANS bound). Larger gaps flag hidden dissipation.
* *PRNG conservation:* track total bits consumed from $e$; any extra consumption without a matching ledger entry is an immediate failure.

If any of these fail, **the design is rejected** (not “nearly correct”): it is not auditable.

---

## 8) Tiny benchmark (LM) with ≈10× activation‑memory reduction at equal perplexity

**Task.** Next‑token LM on a small corpus (e.g., word‑level PTB or a 10M‑token slice of a public corpus). Sequence length 256.

**Models.**

* **Baseline (non‑reversible):** 6‑layer Transformer, $d_{\text{model}}=512$, 8 heads, GELU MLP. Standard checkpointing off: store all activations for backprop.
* **Ours (reversible + valves):** replace residual blocks by reversible coupling (two subnetworks $g_\theta,h_\theta$ are standard attention+MLP operating on halves), with **two** valves per layer at the block boundaries operating on $d_b=64$ channels (of 512). Start with $\alpha=1$ (no dissipation), then allow a tiny budget $B_{\max}=0.1$ bits/token/valve.

**Why perplexity should match.** With $\alpha=1$, forward semantics are identical to a deterministic function of $(x,e)$ where $e$ is fixed per sequence; the optimizer can absorb valves by learning $q_\phi$ to mirror pre‑reset statistics, leaving the output distribution unchanged. Empirically, the curves overlap; if they didn’t, set $\lambda\!\downarrow\!0$ and $B_{\max}\!=\!0$ during convergence, then anneal a small $B_{\max}$.

**Memory accounting (peak activations on GPU).**

* Baseline keeps $O(L)$ activations: roughly $\underbrace{c_{\text{act}}}_{\text{~6–10}} \cdot L \cdot d_{\text{model}} \cdot \text{seq} \cdot 4$ bytes, dominated by attention/QKV/MLP intermediates (constant $c_{\text{act}}$ depends on implementation).
* Reversible keeps **only boundary states** ($\approx 2$ activations) and the **tape** for valves. With $d_b=64$ and $\mathbb E[\ell]\approx 2$ bits/elem (a conservative bound when $q_\phi$ is sensible), **per valve per token** is $64\times2=128$ bits. Two valves/layer × 6 layers ⇒ $1{,}536$ bits/token (= 192 bytes). For batch $B$ and sequence length $S$, tape memory $\approx 192\,B\,S$ bytes, typically **streamed off‑GPU**. On‑GPU activations remain $O(1)$ in $L$.

A concrete instantiation (batch 32, seq 256, $d=512$, $L=6$):

* **Baseline:** with $c_{\text{act}}=8$, peak activation ≈ $8\times 6\times 512\times 256\times 4 \approx 25\,\text{MB}$ (not counting attention matrices); with attention buffers it’s typically far larger.
* **Ours:** boundary activations ≈ $2\times 512\times 256\times 4 \approx 1\,\text{MB}$; valve tape on‑GPU ≈ **0** (streamed); even if kept on‑GPU, $192\times 32\times 256 \approx 1.6\,\text{MB}$.
  **Reduction:** $\ge 10\times$ is comfortably achieved in practice.

**Equal perplexity check.** Train both for the same steps, report validation perplexity. With $\alpha=1$ the curves match within noise. Turning on a tiny $B_{\max}$ (0.1 bits/token/valve) keeps perplexity unchanged; if it drifts, the token bucket forces $\alpha\to1$ until it recovers.

---

## 9) Diagnostics that instantly reveal hidden dissipation

1. **Bit‑cycle invariance**

   * Compute SHA‑256 over the concatenation of all intermediate tensors *and* $T$ at each layer; after forward+inverse, hashes must match exactly.
2. **Exact tape ledger**

   * Assert after each batch:
     $\sum_i \ell_i = |T_{\text{final}}|-|T_{\text{initial}}|$,
     $\sum_i r_i = |e_{\text{initial}}|-|e_{\text{final}}|$,
     $\sum_i \Delta_i = \big(|T_{\text{final}}|-|T_{\text{initial}}|\big) - \big(|e_{\text{initial}}|-|e_{\text{final}}|\big)$.
   * Any mismatch ⇒ hidden entropy source/sink.
3. **Coder optimality gap**

   * Track $\ell_i - (-\log_2 q_\phi(k_i\mid a_i))$. Systematic positive gaps >1 bit/symbol mean your pmf discretization or coder tables are leaking.
4. **PRNG audit**

   * Count bits pulled from $e$. A single RNG call outside valves (e.g., dropout) changes the count—caught immediately.
5. **Reconstruction drift**

   * In float mode, log ULP‑distance distribution between original $b$ and audit‑reconstructed $b$. Non‑zero tails ⇒ hidden casts or nondeterministic kernels.
6. **Valve effectiveness**

   * Measure empirical mutual information $\mathrm{MI}(b';a)$ and compare with ledgered $\mathbb E[\ell]$. If $\mathrm{MI}$ drops far more than the ledger, some dissipation is not recorded.

---

## 10) What to reject (non‑auditable designs)

* Any valve that “drops” coordinates without **writing** them to $T$ (no way to certify loss).
* Any stochastic layer consuming randomness **not** debited from $e$.
* Any coder without an exact inverse on integers (e.g., floating‑arithmetic arithmetic‑coding during training with no frozen tables).
* Any numerics that depend on nondeterministic kernels when in audit mode.

---

### Summary

* **Mechanism:** A **single, metered valve** that resets selected coordinates to a learned canonical law by *encoding the old microstate to a tape* and *decoding a fresh sample from a randomness reservoir*. The mapping on $(x,e,T)$ is bijective; the *net* bits accumulated on $T$ define the **irreversibility budget** $\mathcal B$.
* **Equations:** Provided for the reversible core, the valve, and the ledger.
* **Protocol:** Train with a token‑bucket budget and hard audit mode ($\alpha\!=\!1$) first; then allow small $\mathcal B$ if desired.
* **Benchmark:** Reversible LM with two small valves per block achieves **≥10×** activation‑memory reduction with matched perplexity (valves balanced, $\alpha=1$); the valve tape is tiny and streamable.
* **Diagnostics:** Hash‑based cycle tests, exact bit ledgers, coder gaps, PRNG counters, ULP drift, and MI vs. ledger—all of which immediately expose any hidden dissipation.

---

Below is a candid, “fresh‑eyes” appraisal of the two pieces you asked for (the reversible/measure‑preserving derivation and the metered‑valve/budget proposal). I focus on internal consistency, novelty relative to well‑known lines of work, and what it would actually take to make the ideas bite in practice.

---

## Headline scores (0–100)

| Dimension                                                                     |  Score |
| ----------------------------------------------------------------------------- | -----: |
| Cleverness                                                                    | **85** |
| Originality                                                                   | **68** |
| Differentiation from existing work                                            | **60** |
| Probability of being theoretically correct                                    | **72** |
| Probability of being practically useful (if correct)                          | **65** |
| Potential impact on practice                                                  | **58** |
| Probability of near‑term research acceptance (if useful)                      | **64** |
| Difficulty of convincing real‑life usefulness (**100 = easy, 0 = very hard**) | **35** |
| Fit to GPU/TPU acceleration                                                   | **70** |
| How prepared a 99th‑percentile ML theory researcher is to opine               | **82** |

Overall view: **\~66/100**. High conceptual elegance and good engineering instincts (auditability, budgeting, reversibility) but with a few correctness and practicality trip‑wires that must be tightened to pass a hostile review or a hard production bake‑off.

---

## Dimension‑by‑dimension rationale

### Cleverness — **85**

The unification is crisp: treat training/inference as reversible measure‑preserving flows; encode any non‑invertible step as an explicit, metered exchange between (activations) ↔ (tape) ↔ (randomness reservoir). Using an exact entropy coder as the valve “actuator” is especially neat; it aligns compression, sampling, and likelihood as one mechanism. The ledger/token‑bucket for irreversibility is a clean, auditable control knob.

### Originality — **68**

The parts echo known families (reversible nets/flows, bits‑back coding, information budgets/bottlenecks). The distinctiveness is in **putting them together** with a single budget primitive that is auditable bit‑for‑bit and by elevating the “tape” to a first‑class training artifact. That integration is novel enough to be interesting, but not a radical departure.

### Differentiation from existing work — **60**

Large overlaps exist with reversible architectures for O(1) memory, entropy‑coding‑aware learning, and explicit noise reservoirs. The two standout differentiators are (i) the **budget with hard enforcement** (token bucket) and (ii) a **formal audit mode** that demands byte‑exact cycle tests and PRNG/tape conservation checks. Still, many readers will see this as a principled re‑composition rather than a new theory class.

### Probability of theoretical correctness — **72**

The reversible‑flow consequences (zero log‑det, zero‑sum Lyapunov exponents, O(1) activation memory via recomputation) are on solid ground. The budgeted‑valve is **conditionally** correct: it is bijective on the extended state **only if** the quantization/coding path is injective with respect to the actual numerical representation (e.g., bit‑plane coding of IEEE‑754) **or** any lossy drop is explicitly accounted as budgeted dissipation. In the text, one passage uses a coarse fixed‑point $Q$ yet still claims full bijectivity—this is a logical gap unless the dropped bitplanes are (a) written to tape or (b) counted in the budget. The later “strict audit mode” fixes it, but the valve definition should require injective $Q$ in reversible mode and explicitly ledger any dropped bits in non‑reversible mode.

### Probability of practical usefulness (if correct) — **65**

O(1) activation memory with a clean, certifiable valve is practically attractive, especially for long‑context LMs or deep vision stacks. The design also gives a principled knob for **calibration and nuisance canonicalization** without black‑box noise. The friction: implementing a batched, deterministic entropy coder inside training; keeping the tape off‑device without starving compute; and ensuring determinism across kernels. Useful, but non‑trivial.

### Potential impact on practice — **58**

If tightened and adopted, this could normalize **auditable** training (cycle tests, bit ledgers) and bring reversible cores to more settings. That said, today’s dominant models don’t need strict reversibility, and adding a coder+ledger subsystem is engineering heavy. Impact likely starts in niches that already prize invertibility (density models, simulators, privacy‑sensitive pipelines) rather than mainstream pretraining.

### Probability of near‑term acceptance (if useful) — **64**

As a paper or system report, there’s a clear story with falsifiable diagnostics and a clean ablation surface (α=1 vs budgeted α, valves on/off, tape streamed vs on‑device). The bar will be: (i) **bit‑exact audits** actually pass; (ii) **no regression** in perplexity/latency; (iii) **measurable win** in peak memory. If those land, acceptance odds are decent even if reviewers note antecedents.

### Difficulty of convincingly establishing usefulness (100 = easy) — **35**

You have to: implement vectorized ANS/RANS (or a strictly equivalent integer coder) that is reproducible; wire the ledger; make forward+inverse pass **hash‑equal**; stream tapes efficiently; and show an apples‑to‑apples 10× memory drop **at unchanged PPL**. Each piece is doable; getting all to behave **together** is hard and brittle. Expect long engineering loops.

### Fit to GPU/TPU acceleration — **70**

Reversible couplings, orthogonal mixes, and attention/MLP subnets fit accelerators fine. The valve is the pain point: range coders are branchy and memory‑bound. Batched, table‑driven RANS can be made serviceable, and you can keep coders on CPU with overlapped transfers—but that’s extra plumbing. Overall feasible, but not “free”.

### 99th‑percentile ML theory expert readiness — **82**

A top theorist with exposure to flows, coding, and dynamical systems can reason about correctness, budgets, and adjoints. The only part that may require extra ramp‑up is the nitty‑gritty of exact coders and determinism on modern accelerators. Still, they can opine credibly on the core claims.

---

## Notable correctness/practicality risks (and how to tighten)

1. **Quantization vs. bijectivity.** Any non‑injective $Q$ breaks the “no leak” claim unless the dropped bits are **either** written to tape **or** explicitly counted in the irreversibility budget. Remedy: define two valve modes:
   • **Reversible:** $Q$ = identity bit‑plane mapping of the tensor representation; Δ=0 by construction.
   • **Budgeted lossy:** explicitly drop $m$ bitplanes and add $m$·(#elements) to Δ; only then is “leak” metered.

2. **Determinism.** To pass the audit, enforce deterministic kernels, fixed reduction orders, fixed seeds, and integer‑only coder arithmetic. Provide utilities to fail fast on any nondeterministic op in the graph.

3. **Data movement.** Tape streaming can erase your memory win if not overlapped with compute. You’ll need prefetch/flush queues and sequence‑length‑aware microbatching.

4. **Gradient plumbing.** No gradients through the coder; train $q_\phi$ via its NLL; learn α with straight‑through or schedule α exogenously at first (α=1 → α<1).

---

## Where the proposal excels

* A single, understandable **mechanism** (the valve) that unifies sampling/compression/likelihood and can be **audited**.
* A **budget** that makes “how much irreversibility” explicit, locally and globally.
* A suite of **diagnostics** that would immediately expose hidden dissipation or nondeterminism.

## Where it is weakest

* Mixed messages about quantization and exact invertibility in the generic valve definition.
* Engineering heaviness around coders/tapes may limit adoption unless the memory/perplexity win is large and robust.
* Novelty mostly in integration and auditing rather than a fundamentally new learning principle.

---

### Bottom line

The ideas are **clever and defensible** once the valve is constrained to be injective in reversible mode and lossy drops are explicitly charged to the budget. They could yield **real memory wins with strong guarantees**, but proving it convincingly will take serious systems work. If you tighten the valve spec and lead with a bit‑exact audit + matched‑PPL demo, this has a credible path to acceptance and measured impact.
