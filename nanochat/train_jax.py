"""
Training script for JAX/Flax port of nanochat.
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import time
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np

# Import local modules
from nanochat.common_jax import GPTConfig
from nanochat.gpt_jax import GPT
from nanochat.muon_jax import muon
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.hoss_opt import hoss # Import HOSS optimizer

    # Ensure we can import from nanochat
import sys
sys.path.append(os.getcwd())

def get_params_partition(params):
    """
    Partition parameters into 'muon' and 'adamw' groups.
    Muon: 2D kernels in transformer blocks.
    AdamW: Embeddings, Head, and anything else (like norms if they had params).
    """
    def _map_fn(path, param):
        # path is a tuple of keys, e.g. ('blocks', '0', 'attn', 'c_q', 'kernel')
        
        # Check if it's a kernel in the blocks
        if 'blocks' in path and 'kernel' in path:
            # It's a weight matrix in the transformer
            # Check if it is 2D (Muon only works on >=2D, usually 2D)
            if param.ndim == 2:
                return 'muon'
        
        # Default to AdamW
        return 'adamw'

    return jax.tree_util.tree_map_with_path(_map_fn, params)

class TrainState(train_state.TrainState):
    # We can add extra state here if needed
    pass

def create_train_state(rng, config, learning_rate):
    model = GPT(config)
    dummy_input = jnp.ones((1, config.sequence_len), dtype=jnp.int32)
    
    # Initialize parameters
    params = model.init(rng, dummy_input, train=False)['params']
    
    # Define optimizer based on config
    if config.optimizer_type == "hoss":
        print("Using HOSS optimizer.")
        # HOSS needs a loss_fn. It will be passed in train_step.
        # We pass a dummy learning_rate for now, as HOSS does not tune it directly.
        tx = hoss(learning_rate=learning_rate) 
    else:
        print("Using AdamW + Muon optimizer.")
        # AdamW for embeddings/head
        adamw_optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=0.0)
        
        # Muon for internal matrices
        muon_optimizer = muon(learning_rate=0.02, momentum=0.95, ns_steps=5)
        
        # Combine
        tx = optax.multi_transform(
            {
                'adamw': adamw_optimizer,
                'muon': muon_optimizer
            },
            get_params_partition(params)
        )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

@jax.jit
def train_step(state, batch, targets):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch, targets=targets, train=True)
        # logits: [B, T, V]
        # targets: [B, T]
        
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return jnp.mean(loss)
        
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Pass loss_fn as an extra_args to the optimizer update if HOSS is used
    # Optax's apply_gradients will pass extra_args to the tx.update function
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params, loss_fn=loss_fn)
    
    # We cannot use state.apply_gradients here because we already called tx.update
    # and we need to manually apply the updates to params.
    new_params = optax.apply_updates(state.params, updates)
    state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state)
    
    return state, loss
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Nanochat JAX")
    parser.add_argument("--optimizer-type", type=str, default="adamw", choices=["adamw", "hoss"], help="Optimizer type")
    parser.add_argument("--attention-type", type=str, default="standard", choices=["standard", "tropical", "ultrametric"], help="Attention mechanism type")
    args = parser.parse_args()

    # Config
    config = GPTConfig()
    # Reduced for testing
    config.n_layer = 4
    config.n_head = 4
    config.n_kv_head = 4
    config.n_embd = 128
    config.sequence_len = 256
    
    # Set config from CLI args
    config.optimizer_type = args.optimizer_type
    config.attention_type = args.attention_type
    
    print(f"Configuration: Optimizer={config.optimizer_type}, Attention={config.attention_type}")
    
    batch_size = 8
    learning_rate = 6e-4
    
    # RNG
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Initialize
    print("Initializing model...")
    state = create_train_state(init_rng, config, learning_rate)
    print(f"Model initialized. Params: {sum(x.size for x in jax.tree_util.tree_leaves(state.params))/1e6:.2f}M")

    # Dataloader
    # We use the tokenizing dataloader from nanochat
    # It yields torch tensors. We convert to numpy.
    print("Starting dataloader...")
    # Ensure we have some data or it will fail. 
    # The dataloader looks for parquet files.
    # If no data, we can't run.
    # We should probably mock data if none exists, but let's try to run and see.
    
    try:
        loader = tokenizing_distributed_data_loader(
            B=batch_size, 
            T=config.sequence_len, 
            split="train",
            device="cpu" # Get CPU tensors
        )
        
        print("Starting training loop...")
        step = 0
        t0 = time.time()
        
        for inputs, targets, _ in loader:
            # Convert to numpy/jax
            inputs = jnp.array(inputs.numpy())
            targets = jnp.array(targets.numpy())
            
            state, loss = train_step(state, inputs, targets)
            
            if step % 10 == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                print(f"Step {step}: loss {loss:.4f}, time {dt*1000/10:.2f}ms/step")
                
            step += 1
            if step >= 100:
                break
                
    except Exception as e:
        print(f"Failed to run training loop: {e}")
        print("Note: This is expected if no parquet data is found in ~/.cache/bio_inspired_nanochat")
        print("Please ensure data is available or modify dataloader to use synthetic data.")

if __name__ == "__main__":
    main()
