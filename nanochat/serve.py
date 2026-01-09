"""
Inference Server for NanoChat (JAX)
"""

import os
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ModuleNotFoundError as e:
    raise ImportError(
        "nanochat.serve requires optional server dependencies. "
        "Install them with `uv sync --extra server`."
    ) from e

import jax
import jax.numpy as jnp
import numpy as np
from rich.console import Console

# Import local modules
from nanochat.gpt_jax import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

console = Console()
_UI_PATH = Path(__file__).with_name("ui.html")

# Global state
model_state = None
tokenizer = None
config = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = Field(default=0.8, ge=0.0)
    top_k: int = Field(default=50, ge=1)
    max_tokens: int = Field(default=512, ge=1, le=4096)


def load_model() -> None:
    global model_state, tokenizer, config
    
    console.print("[bold cyan]Loading model...[/bold cyan]")
    config = GPTConfig()
    # Use the same config as training
    config.n_layer = 4
    config.n_head = 4
    config.n_kv_head = 4
    config.n_embd = 128
    config.sequence_len = 256
    
    # Initialize model
    model = GPT(config)
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, config.sequence_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input, train=False)['params']
    
    model_state = {
        "params": params,
        "apply_fn": model.apply
    }
    
    console.print("[bold cyan]Loading tokenizer...[/bold cyan]")
    tokenizer = get_tokenizer()
    console.print("[bold green]Model and tokenizer loaded.[/bold green]")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def get_ui() -> str:
    return _UI_PATH.read_text(encoding="utf-8")

@app.get("/health")
async def health():
    return {"status": "ok", "backend": "jax"}

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global model_state, tokenizer, config
    
    if not model_state:
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
    
    input_ids = tokenizer.encode(prompt)
    input_ids = jnp.array([input_ids], dtype=jnp.int32) # [1, T]
    
    # Truncate if too long
    if input_ids.shape[1] > config.sequence_len:
        input_ids = input_ids[:, -config.sequence_len:]
    
    async def generate_stream():
        current_ids = input_ids
        
        for _ in range(request.max_tokens):
            # Forward pass
            # We need to handle context length. 
            # If current_ids > sequence_len, we crop.
            if current_ids.shape[1] > config.sequence_len:
                cond_ids = current_ids[:, -config.sequence_len:]
            else:
                cond_ids = current_ids
                
            logits = model_state["apply_fn"]({'params': model_state["params"]}, cond_ids, train=False)
            next_token_logits = logits[0, -1, :]
            
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
            token_str = tokenizer.decode([next_token_idx])
            
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
