# Synaptic FlexAttention correctness (non-flex vs flex)

- Run ID: `20251217_235653`
- Generated: 2025-12-17 23:56:53 EST
- Overall: PASS

## Command

```bash
uv run python scripts/verify_flex_correctness.py --model synaptic --device cuda --dtype bfloat16 --compile --batch-size 1 --sequence-len 64 --vocab-size 1024 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 128
```

## Notes

- This script compares SDPA (reference) vs FlexAttention for standard GPT across:
  - full forward
  - KV-cache last-token decode
  - optional KV chunk decode

See `summary.json` for full details.
