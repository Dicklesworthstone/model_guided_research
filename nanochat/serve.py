"""
Inference Server for NanoChat (JAX)
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ModuleNotFoundError as e:
    raise ImportError(
        "nanochat.serve requires optional server dependencies. Install them with `uv sync --extra server`."
    ) from e

import jax
import jax.numpy as jnp
import numpy as np
from rich.console import Console

# Import local modules
from nanochat.jax_checkpoint import JaxCheckpointError, JaxServingCheckpoint, load_serving_checkpoint

console = Console()
_UI_PATH = Path(__file__).with_name("ui.html")
CHECKPOINT_ENV = "NANOCHAT_JAX_CHECKPOINT_DIR"

_inference_state: JaxServingCheckpoint | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = Field(default=0.8, ge=0.0)
    top_k: int = Field(default=50, ge=1)
    max_tokens: int = Field(default=512, ge=1, le=4096)


def load_model(checkpoint_dir: str | Path | None = None) -> JaxServingCheckpoint:
    """Load the configured, validated checkpoint without publishing partial state."""

    configured_path = checkpoint_dir
    if configured_path is None:
        configured_path = os.environ.get(CHECKPOINT_ENV)
        if configured_path is None or not configured_path.strip():
            raise JaxCheckpointError(
                f"no JAX serving checkpoint configured; set {CHECKPOINT_ENV} to a checkpoint directory"
            )

    console.print("[bold cyan]Loading JAX serving checkpoint:[/bold cyan]", str(configured_path))
    state = load_serving_checkpoint(configured_path)
    console.print(
        "[bold green]JAX checkpoint ready[/bold green]",
        f"({state.config.n_layer} layers, {state.config.n_embd} dimensions, {state.config.vocab_size} tokens)",
    )
    return state


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _inference_state

    loaded_state = load_model()
    _inference_state = loaded_state
    try:
        yield
    finally:
        _inference_state = None


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def get_ui() -> str:
    return _UI_PATH.read_text(encoding="utf-8")


@app.get("/health")
async def health():
    state = _inference_state
    if state is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "backend": "jax"})
    return {
        "status": "ready",
        "backend": "jax",
        "checkpoint": str(state.checkpoint_dir),
        "step": state.step,
        "architecture": "nanochat.gpt_jax.GPT",
    }


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    state = _inference_state
    if state is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    # Tokenize
    # We use the tokenizer's render_conversation
    # But wait, render_conversation returns (ids, mask). We just need ids.
    # And we need to append the assistant start token for completion.
    # The tokenizer has render_for_completion but that expects the last message to be assistant (empty?)
    # Let's just use encode directly for simplicity or adapt.

    # Simple prompt construction for now:
    # Join messages with newlines
    prompt = ""
    for m in request.messages:
        prompt += f"{m.role}: {m.content}\n"
    prompt += "assistant: "

    encoded_ids = state.tokenizer.encode(prompt)
    input_ids = jnp.array([encoded_ids], dtype=jnp.int32)  # [1, T]

    # Truncate if too long
    if input_ids.shape[1] > state.config.sequence_len:
        input_ids = input_ids[:, -state.config.sequence_len :]

    async def generate_stream():
        current_ids = input_ids

        for _ in range(request.max_tokens):
            # Forward pass
            # We need to handle context length.
            # If current_ids > sequence_len, we crop.
            if current_ids.shape[1] > state.config.sequence_len:
                cond_ids = current_ids[:, -state.config.sequence_len :]
            else:
                cond_ids = current_ids

            raw_logits: object = state.model.apply(state.variables, cond_ids, train=False)
            if not isinstance(raw_logits, jax.Array) or raw_logits.ndim != 3:
                raise RuntimeError("GPT inference must return a rank-3 JAX logits array")
            logits = raw_logits
            tokenizer_vocab_size = int(state.tokenizer.get_vocab_size())
            next_token_logits = logits[0, -1, :tokenizer_vocab_size]

            # Sampling
            # Temperature
            if request.temperature > 0:
                next_token_logits = next_token_logits / request.temperature
                # Top-k
                k = min(int(request.top_k), int(next_token_logits.shape[-1]))
                top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, k)
                # We need to sample from these.
                # Convert to numpy for sampling (easier)
                probs = jax.nn.softmax(top_k_logits)
                probs = np.array(probs)
                indices = np.array(top_k_indices)

                next_token_idx = int(np.random.choice(indices, p=probs))
            else:
                next_token_idx = int(np.argmax(next_token_logits))

            # Decode
            token_str = state.tokenizer.decode([next_token_idx])

            # Yield
            yield f"data: {json.dumps({'token': token_str})}\n\n"

            # Update ids
            next_token_id = jnp.array([[next_token_idx]], dtype=jnp.int32)
            current_ids = jnp.concatenate([current_ids, next_token_id], axis=1)

            # Stop if EOS (if we had one)
            # if next_token_idx == tokenizer.eos_token_id:
            #     break

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("NANOCHAT_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("NANOCHAT_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
