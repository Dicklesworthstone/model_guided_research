"""
Utilities for saving and loading model/optim/state checkpoints.

Checkpoint directory contract (bead rz8.1) — consumers (eval harness C2,
distillation teachers C6, scaling rungs E1, sampling D5) depend on this layout:

    <checkpoint_dir>/
        model_<step:06d>.pt           # raw (uncompiled) model state_dict, clean keys
        optim_<step:06d>_rank<r>.pt   # per-rank training state:
                                      #   {"optimizers": [state_dict, ...],   # in setup_optimizers order
                                      #    "schedulers": [state_dict, ...],   # OrdinalLRScheduler counters
                                      #    "dataloader": {"batches_consumed": int,
                                      #                    "pq_idx": int, "rg_idx": int},
                                      #    "rng": {"torch_cpu": Tensor,
                                      #             "cuda": list[Tensor] | None,
                                      #             "numpy": tuple, "python": tuple}}
        meta_<step:06d>.json          # {"step": last completed step (0-based),
                                      #  "model_config": asdict(GPTConfig), "model_type": str,
                                      #  "optimizer_type": str, "scheduler_type": str,
                                      #  "seed": int, "world_size": int,
                                      #  "batches_consumed": int,
                                      #  "budget": {"max_steps", "target_flops", "flops_per_step_est",
                                      #              "flops_consumed_est"},
                                      #  "lineage": {"run_id": str, "parent_run_ids": [str, ...]}}

`step` is the 0-based index of the last COMPLETED optimizer step; a resumed
run continues at step+1. The optimizer list layout deliberately accommodates
future optimizers with exotic state (e.g. HOSS when D3 lands): anything that
implements torch's Optimizer state_dict protocol slots in unchanged.
"""

import glob
import json
import logging
import os
import re

from nanochat.common import get_base_dir, setup_default_logging
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.torch_imports import torch

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)


def log0(message):
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(message)


def _load_synaptic_model_types():
    """Load optional synaptic model support only when a checkpoint needs it."""
    try:
        from nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
    except Exception as exc:
        raise ImportError("gpt_synaptic not found but synapses=True in metadata") from exc
    return GPTSynaptic, GPTSynapticConfig


def _atomic_torch_save(obj, path):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    """Write a checkpoint with crash-safe ordering: every file lands via
    tmp-write + atomic rename, and meta_<step>.json is written LAST - its
    existence is the COMMIT POINT. find_last_step/resume discover checkpoints
    through the meta file, so a SIGKILL mid-save can never expose a truncated,
    half-written checkpoint as resumable (the rz8.8 e2e resume scenario hit
    exactly that: a torn optim_*.pt picked up by --resume-from latest).
    NOTE: under multi-rank DDP, non-zero ranks' optimizer shards can still be
    in flight when rank 0 commits the meta; a cross-rank barrier before the
    meta write is the future-proof fix when sharded training lands for real.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Optimizer state is sharded across ranks, so each rank saves its own -
    # and it must land BEFORE the meta commit point below.
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        _atomic_torch_save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")
    if rank == 0:
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        _atomic_torch_save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Ensure meta_data exists and mark synaptic models
        meta_data = meta_data or {}
        # Check if model_data contains synaptic-specific keys (heuristic detection)
        # This is a fallback; ideally the caller should set synapses=True in meta_data
        if "synapses" not in meta_data:
            # Check for synaptic-specific buffer names in state dict
            synaptic_keys = [
                k
                for k in model_data.keys()
                if any(x in k for x in ["pre.", "post.", "H_fast", "U_buf", "V_buf", "gate_m"])
            ]
            if synaptic_keys:
                meta_data["synapses"] = True
        # Save the metadata dict as json - LAST, atomically: the commit point
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        os.replace(tmp_meta, meta_path)
        logger.info(f"Saved metadata to: {meta_path}")


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device, weights_only=True)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        # weights_only loading works because train.py serializes the RNG lanes
        # into weights-only-safe types (tensors/ints/lists) before saving.
        optimizer_data = torch.load(optimizer_path, map_location=device, weights_only=True)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def checkpoint_file_paths(checkpoint_dir, step, rank=0):
    """The (model, optim, meta) file triple for a step — single source of naming truth."""
    return (
        os.path.join(checkpoint_dir, f"model_{step:06d}.pt"),
        os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt"),
        os.path.join(checkpoint_dir, f"meta_{step:06d}.json"),
    )


def prune_checkpoints(checkpoint_dir, steps_saved_this_run, *, keep, rank=0):
    """Retention: keep the newest `keep` checkpoints among `steps_saved_this_run`.

    Deletes ONLY files this run itself created (the caller passes the list of
    steps it saved) — never user files or checkpoints from other runs sharing
    the directory. Returns the list of pruned steps; mutates steps_saved_this_run.
    """
    if keep <= 0:
        return []
    pruned: list[int] = []
    while len(steps_saved_this_run) > keep:
        oldest = steps_saved_this_run.pop(0)
        for path in checkpoint_file_paths(checkpoint_dir, oldest, rank=rank):
            # Model/meta exist only on rank 0; optim shards on every rank.
            if os.path.exists(path):
                os.remove(path)
        pruned.append(oldest)
        logger.info(f"Pruned checkpoint step {oldest} from {checkpoint_dir} (keep={keep})")
    return pruned


def verify_checkpoint_roundtrip(checkpoint_dir, step, device, *, model_data, optimizer_data, rank=0):
    """Integrity tripwire (--checkpoint-verify): reload what was just saved and
    compare every tensor bitwise. Returns a list of mismatch descriptions
    (empty = verified). Cheap insurance against flaky storage on long runs."""

    def tensors_equal(a, b) -> bool:
        return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a.cpu(), b.cpu()))

    mismatches: list[str] = []
    reloaded_model, reloaded_optim, _meta = load_checkpoint(
        checkpoint_dir, step, device, load_optimizer=optimizer_data is not None, rank=rank
    )
    if rank == 0:
        for key, tensor in model_data.items():
            other = reloaded_model.get(key)
            if other is None or not tensors_equal(tensor, other):
                mismatches.append(f"model[{key}]")
    if optimizer_data is not None and reloaded_optim is not None:

        def walk(a, b, path):
            if isinstance(a, torch.Tensor):
                if not (isinstance(b, torch.Tensor) and tensors_equal(a, b)):
                    mismatches.append(path)
            elif isinstance(a, dict):
                for k in a:
                    walk(a[k], b.get(k) if isinstance(b, dict) else None, f"{path}[{k!r}]")
            elif isinstance(a, (list, tuple)):
                if not isinstance(b, (list, tuple)) or len(a) != len(b):
                    mismatches.append(path)
                else:
                    for i, (x, y) in enumerate(zip(a, b)):
                        walk(x, y, f"{path}[{i}]")

        walk(optimizer_data, reloaded_optim, "optim")
    return mismatches


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    if phase not in ["train", "eval"]:
        raise ValueError(f"Invalid phase: {phase}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    # annealed checkpoints (bead y2h9): the recorded config carries the
    # schedule's b0, but the weights were saved at the live beta recorded
    # beside them - construct at that value or an annealed-to-32 tropical
    # model silently runs at beta=1
    if "semiring_beta_live" in meta_data:
        model_config_kwargs = {**model_config_kwargs, "semiring_beta": meta_data["semiring_beta_live"]}
    log0(f"Building model with config: {model_config_kwargs}")

    # Check if this is a synaptic model
    if meta_data.get("synapses", False):
        synaptic_model_type, synaptic_config_type = _load_synaptic_model_types()
        from nanochat.synaptic import SynapticConfig

        syn_cfg = SynapticConfig()  # Use defaults; could load from meta_data if saved
        model_config = synaptic_config_type(
            sequence_len=model_config_kwargs["sequence_len"],
            vocab_size=model_config_kwargs["vocab_size"],
            n_layer=model_config_kwargs["n_layer"],
            n_head=model_config_kwargs["n_head"],
            n_kv_head=model_config_kwargs.get("n_kv_head", model_config_kwargs["n_head"]),
            n_embd=model_config_kwargs["n_embd"],
            syn_cfg=syn_cfg,
        )
        with torch.device("meta"):
            model = synaptic_model_type(model_config)
    else:
        model_config = GPTConfig(**model_config_kwargs)
        with torch.device("meta"):
            model = GPT(model_config)

    # Load the model state
    model.to_empty(device=device)
    # init_weights is what (re)builds the NON-persistent buffers (rotary
    # cos/sin, diagnostic buffers) after to_empty; the parameter draws it also
    # makes are discarded by the strict load below. Fork the RNG around it so
    # that loading a checkpoint never advances the caller's global stream -
    # a sampler that seeds once and then loads two checkpoints of different
    # sizes must see the same subsequent draws.
    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=fork_devices):
        model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.train(mode=False)
    else:
        model.train(mode=True)
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    if tokenizer.get_vocab_size() != model_config_kwargs["vocab_size"]:
        raise ValueError("Tokenizer vocab size does not match model config")
    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Discover through the META files, not model_*.pt: meta_<step>.json is
    # written LAST and atomically (see save_checkpoint), so its existence is
    # the COMMIT POINT. Globbing model_*.pt here would expose a checkpoint
    # whose optimizer shard/meta never landed after a SIGKILL mid-save —
    # resume would then die on the missing meta instead of falling back to
    # the last committed step. load_checkpoint() requires the meta anyway.
    metas = glob.glob(os.path.join(checkpoint_dir, "meta_*.json"))
    if not metas:
        raise FileNotFoundError(f"No committed checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(m).split("_")[-1].split(".")[0] for m in metas))
    return last_step


# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure


def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data


def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
