
import os
import time
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nanochat.common import get_dist_info, compute_init, compute_cleanup
from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader

from nanochat.ordinal_scheduler import OrdinalLRScheduler

def train(args):
    # Init distributed mode if necessary
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type="cuda")
    
    # Config
    config = GPTConfig()
    config.n_layer = 4
    config.n_head = 4
    config.n_kv_head = 4
    config.n_embd = 128
    config.sequence_len = 256
    config.optimizer_type = args.optimizer_type
    config.attention_type = args.attention_type
    
    # Model
    model = GPT(config)
    model.to(device)
    model.init_weights()
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module
    else:
        raw_model = model

    # Optimizer
    optimizers = raw_model.setup_optimizers(
        unembedding_lr=args.learning_rate, 
        embedding_lr=args.learning_rate, 
        matrix_lr=args.learning_rate
    )
    
    # Scheduler
    schedulers = []
    if args.scheduler_type == "ordinal":
        # We attach an ordinal scheduler to each optimizer
        # Note: Ordinal scheduler updates LR based on loss.
        for opt in optimizers:
            schedulers.append(OrdinalLRScheduler(opt, eta_init=args.learning_rate))
    
    # Dataloader
    loader = tokenizing_distributed_data_loader(
        B=args.batch_size,
        T=config.sequence_len,
        split="train",
        device="cuda"
    )
    
    print(f"Starting training on {device}...")
    
    # Training loop
    step = 0
    t0 = time.time()
    
    is_hoss = (args.optimizer_type == "hoss")

    for inputs, targets in loader: # loader yields 2 items
        
        for opt in optimizers:
            opt.zero_grad()
            
        # Define closure for HOSS
        def closure():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if is_hoss:
                # HOSS needs create_graph=True to compute HVP
                loss.backward(create_graph=True)
            else:
                loss.backward()
            return loss

        if is_hoss:
            # Step with closure
            loss = optimizers[0].step(closure) 
        else:
            # Standard AdamW/Muon
            # Forward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            
            for opt in optimizers:
                opt.step()
                
        # Scheduler Step
        if args.scheduler_type == "ordinal":
            # Step schedulers with current loss
            for sched in schedulers:
                sched.step(loss.item())
            
        # Logging
        if step % 1 == 0 and ddp_rank == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"Step {step}: loss {loss.item():.4f}, time {dt*1000:.2f}ms/step")
            
        step += 1
        if step >= 20:
            break
            
    compute_cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--optimizer-type", type=str, default="adamw") 
    parser.add_argument("--attention-type", type=str, default="standard")
    parser.add_argument("--scheduler-type", type=str, default="none")
    args = parser.parse_args()
    
    train(args)
