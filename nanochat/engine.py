"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import ast
import operator as _op
import signal
from collections import deque
from contextlib import contextmanager, nullcontext

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.torch_imports import F, torch


# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    prev_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    prev_remaining = signal.alarm(int(duration))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)
        if prev_remaining:
            signal.alarm(prev_remaining)


_SAFE_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
}

_SAFE_UNARY_OPS = {
    ast.UAdd: _op.pos,
    ast.USub: _op.neg,
}


def _eval_safe_node(node):
    if isinstance(node, ast.Expression):
        return _eval_safe_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str)):
            return node.value
        raise ValueError("Unsupported constant type")
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
        return _SAFE_BIN_OPS[type(node.op)](_eval_safe_node(node.left), _eval_safe_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return _SAFE_UNARY_OPS[type(node.op)](_eval_safe_node(node.operand))
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Constant)
            and isinstance(func.value.value, str)
            and func.attr == "count"
            and not node.keywords
            and len(node.args) == 1
        ):
            target = func.value.value
            needle = _eval_safe_node(node.args[0])
            return target.count(needle if isinstance(needle, str) else str(needle))
    raise ValueError("Unsupported expression")


def _safe_eval_expression(expr: str, max_time: int = 3, *, max_chars: int = 4096):
    expr = expr.strip()
    if not expr or len(expr) > max_chars:
        return None
    try:
        with timeout(max_time, expr):
            tree = ast.parse(expr, mode="eval")
            return _eval_safe_node(tree)
    except Exception:
        return None


def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")
    return _safe_eval_expression(expr)


# -----------------------------------------------------------------------------
class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    Note that the .pos advances automatically after the last layer of the Transformer inserts.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer.
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0  # current position in time in the cache
        # Optional per-layer cached state for attention variants that need more than K/V.
        # - Simplicial attention caches the 1-hop outputs so 2-hop diffusion stays KV-cache consistent.
        self.simplicial_y1_cache = None

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        """
        Prefill given another KV cache. Optionally expand along batch dim.
        This is used when we do batch 1 prefill and then want to generate
        multiple samples in parallel from there.
        """
        # 1) validate the shapes
        if self.kv_cache is not None:
            raise RuntimeError("Cannot prefill a non-empty KV cache")
        if other.kv_cache is None:
            raise RuntimeError("Cannot prefill with a None KV cache")
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            # ix 0: num_layers, 1: k/v, 2: batch_size, 3: num_heads, 4: seq_len, 5: head_dim
            if ix in [0, 1, 3, 5]:
                # num_layers, k/v, num_heads, head_dim must match
                if dim1 != dim2:
                    raise ValueError(f"Dim {ix} mismatch: {dim1} != {dim2}")
            elif ix == 2:
                # batch_size can be expanded
                if not (dim1 == dim2 or dim2 == 1):
                    raise ValueError(f"Batch dim mismatch: {dim1} != {dim2}")
            elif ix == 4:
                # seq_len: self must be longer than other
                if dim1 < dim2:
                    raise ValueError(f"Seq len mismatch: {dim1} < {dim2}")
        # 2) initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        other_kv = other.kv_cache[:, :, :, :, : other.pos, :]
        if other_kv.size(2) == self.kv_shape[2]:
            self.kv_cache[:, :, :, :, : other.pos, :] = other_kv
        elif other_kv.size(2) == 1:
            expanded = other_kv.expand(other_kv.size(0), other_kv.size(1), self.kv_shape[2], *other_kv.shape[3:])
            self.kv_cache[:, :, :, :, : other.pos, :] = expanded
        else:
            raise ValueError(f"Cannot expand KV cache batch dim {other_kv.size(2)} -> {self.kv_shape[2]}")
        # 4) update the pos
        self.pos = other.pos
        # 5) Copy any extra per-cache state (e.g., synaptic presyn state)
        if hasattr(other, "presyn_state"):
            other_state = other.presyn_state
            if other_state is None:
                self.presyn_state = None
            else:
                if not isinstance(other_state, list):
                    raise TypeError(
                        "KVCache.prefill expected other.presyn_state to be a list[dict|None] "
                        "(per-layer synaptic state)."
                    )

                target_B = self.kv_shape[2]

                def _expand_batch(t: torch.Tensor) -> torch.Tensor:
                    if t.size(0) == target_B:
                        return t.clone()
                    if t.size(0) == 1:
                        return t.expand(target_B, *t.shape[1:]).clone()
                    raise ValueError(f"Cannot expand presyn_state batch dim {t.size(0)} -> {target_B}")

                def _copy_presyn_state(state: dict[str, object]) -> dict[str, object]:
                    st: dict[str, object] = {}
                    for key in ("rrp", "res", "c", "sn", "cl", "amp", "en"):
                        val = state.get(key)
                        if isinstance(val, torch.Tensor):
                            st[key] = _expand_batch(val).to(device=device)
                    delay = state.get("delay")
                    if isinstance(delay, list):
                        st["delay"] = [_expand_batch(d).to(device=device) for d in delay if isinstance(d, torch.Tensor)]
                    buf = state.get("BUF")
                    if isinstance(buf, torch.Tensor):
                        st["BUF"] = _expand_batch(buf).to(device=device)
                    else:
                        rrp = st.get("rrp")
                        if isinstance(rrp, torch.Tensor):
                            st["BUF"] = torch.zeros_like(rrp)

                    # Common alias keys used by synaptic codepaths / fused kernels.
                    if "rrp" in st:
                        st["RRP"] = st["rrp"]
                    if "res" in st:
                        st["RES"] = st["res"]
                    if "c" in st:
                        st["C"] = st["c"]
                    if "sn" in st:
                        st["PR"] = st["sn"]
                    if "cl" in st:
                        st["CL"] = st["cl"]
                    if "en" in st:
                        st["E"] = st["en"]

                    return st

                self.presyn_state = [
                    (_copy_presyn_state(layer_state) if isinstance(layer_state, dict) else None)
                    for layer_state in other_state
                ]

        if hasattr(other, "simplicial_y1_cache"):
            other_cache = other.simplicial_y1_cache
            if other_cache is None:
                self.simplicial_y1_cache = None
            elif not isinstance(other_cache, torch.Tensor):
                raise TypeError("KVCache.prefill expected other.simplicial_y1_cache to be a torch.Tensor | None")
            else:
                if other_cache.ndim != 5:
                    raise ValueError(
                        "KVCache.prefill expected other.simplicial_y1_cache to have shape (num_layers, B, H, T, D)"
                    )
                if other_cache.size(0) != self.kv_shape[0]:
                    raise ValueError(
                        f"num_layers mismatch for simplicial_y1_cache: {other_cache.size(0)} != {self.kv_shape[0]}"
                    )
                if other.pos > other_cache.size(3):
                    raise ValueError(
                        f"Expected simplicial_y1_cache time dim >= other.pos ({other.pos}), got {other_cache.size(3)}"
                    )

                target_B = self.kv_shape[2]
                target_T = self.kv_cache.size(4)

                if other_cache.size(1) == target_B:
                    expanded = other_cache
                elif other_cache.size(1) == 1:
                    expanded = other_cache.expand(other_cache.size(0), target_B, *other_cache.shape[2:]).clone()
                else:
                    raise ValueError(f"Cannot expand simplicial_y1_cache batch dim {other_cache.size(1)} -> {target_B}")

                self.simplicial_y1_cache = torch.zeros(
                    (self.kv_shape[0], target_B, expanded.size(2), target_T, expanded.size(4)),
                    dtype=expanded.dtype,
                    device=device,
                )
                self.simplicial_y1_cache[:, :, :, : other.pos, :] = expanded[:, :, :, : other.pos, :].to(device=device)

    def ensure_simplicial_y1_cache(
        self,
        *,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if self.simplicial_y1_cache is not None:
            return
        if self.kv_cache is None:
            raise RuntimeError("ensure_simplicial_y1_cache requires kv_cache to be initialized")
        seq_len = self.kv_cache.size(4)
        self.simplicial_y1_cache = torch.zeros(
            (self.kv_shape[0], self.kv_shape[2], num_heads, seq_len, head_dim),
            dtype=dtype,
            device=device,
        )

    def insert_simplicial_y1(self, layer_idx: int, t0: int, y1: torch.Tensor) -> torch.Tensor:
        if self.simplicial_y1_cache is None:
            raise RuntimeError("simplicial_y1_cache is not initialized")
        if y1.ndim != 4:
            raise ValueError("insert_simplicial_y1 expects y1 of shape (B, H, T, D)")

        B, H, T_add, D = y1.shape
        if B != self.simplicial_y1_cache.size(1):
            raise ValueError(f"Batch mismatch for simplicial_y1_cache: {B} != {self.simplicial_y1_cache.size(1)}")
        if H != self.simplicial_y1_cache.size(2) or D != self.simplicial_y1_cache.size(4):
            raise ValueError(
                "Head/dim mismatch for simplicial_y1_cache: "
                f"y1={(B, H, T_add, D)} vs cache={tuple(self.simplicial_y1_cache.shape)}"
            )

        t1 = t0 + T_add
        if t1 > self.simplicial_y1_cache.size(3):
            raise ValueError(f"simplicial_y1_cache too small for write: t1={t1} > {self.simplicial_y1_cache.size(3)}")
        self.simplicial_y1_cache[layer_idx, :, :, t0:t1, :] = y1
        return self.simplicial_y1_cache[layer_idx, :, :, :t1, :]

    def insert_kv(self, layer_idx, k, v):
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # Insert new keys/values to the cache and return the full cache so far
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.size(4):
            old_t = self.kv_cache.size(4)
            t_needed = t1 + 1024  # as much as we need plus buffer of 1024
            t_needed = (t_needed + 1023) & ~1023  # then round up to the nearest multiple of 1024
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - old_t
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape
            if self.simplicial_y1_cache is not None:
                extra_shape = list(self.simplicial_y1_cache.shape)
                extra_shape[3] = t_needed - old_t
                extra = torch.zeros(
                    extra_shape,
                    dtype=self.simplicial_y1_cache.dtype,
                    device=self.simplicial_y1_cache.device,
                )
                self.simplicial_y1_cache = torch.cat([self.simplicial_y1_cache, extra], dim=3).contiguous()
        # Insert k, v into the cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        # Increment pos after the last layer of the Transformer processes
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


# -----------------------------------------------------------------------------


class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []  # Current token sequence for this row
        self.forced_tokens = deque()  # Queue of tokens to force inject
        self.in_python_block = False  # Whether we are inside a python block
        self.python_expr_tokens = []  # Tokens of the current python expression
        self.completed = False  # Whether this row has completed generation


class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer  # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        if not (isinstance(tokens, list) and tokens and isinstance(tokens[0], int)):
            raise TypeError("expecting list of ints for tokens")
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        def get_special_optional(s: str) -> int | None:
            try:
                return self.tokenizer.encode_special(s)
            except Exception:
                return None

        python_start = get_special_optional("<|python_start|>")
        python_end = get_special_optional("<|python_end|>")
        output_start = get_special_optional("<|output_start|>")
        output_end = get_special_optional("<|output_end|>")
        assistant_end = get_special_optional("<|assistant_end|>")  # if sampled, ends row
        if any(tok is None for tok in (python_start, python_end, output_start, output_end)):
            python_start = None
            python_end = None
            output_start = None
            output_end = None
        bos = self.tokenizer.get_bos_token_id()  # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        result = self.model.forward(ids, kv_cache=kv_cache_prefill)
        # Handle both GPT (returns logits) and GPTSynaptic (returns (logits, None))
        if isinstance(result, tuple):
            logits, _ = result
        else:
            logits = result
        logits = logits[:, -1, :]
        if num_samples > 1:
            logits = logits.expand(num_samples, -1)
        next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
        sampled_tokens = next_ids[:, 0].tolist()

        # 2) Replicate the KV cache for each sample/row
        # Ensure KV cache can prefill even if the prompt exceeds the configured sequence length.
        kv_length_hint = len(tokens)
        if max_tokens is not None:
            kv_length_hint += max_tokens
        kv_length_hint = max(kv_length_hint, self.model.config.sequence_len)
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill  # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        first_iteration = True
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Get sampled tokens - either from prefill or from forward pass
            if first_iteration:
                # Use the tokens we already sampled from prefill (one per row).
                first_iteration = False
            else:
                # Forward the model and get the next token for each row
                result = self.model.forward(ids, kv_cache=kv_cache_decode)  # (B, T, vocab_size) or (logits, None)
                # Handle both GPT (returns logits) and GPTSynaptic (returns (logits, None))
                if isinstance(result, tuple):
                    logits, _ = result
                else:
                    logits = result
                logits = logits[:, -1, :]  # (B, vocab_size) at last time step
                next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = []  # contains the next token id along each row
            token_masks = []  # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0  # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1)  # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if (assistant_end is not None and next_token == assistant_end) or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if python_start is not None and next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif python_end is not None and next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None and output_start is not None and output_end is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1
            # Prepare ids for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        try:
            assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        except Exception:
            assistant_end = None
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if (assistant_end is not None and token == assistant_end) or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time

    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type=device_type)
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    )

    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    if device_type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    with autocast_ctx:
        for token in stream:
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs)  # note: runs in fp32
    if device_type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with autocast_ctx:
        for token_column, token_masks in stream:
            token = token_column[0]  # only print out the first row
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
