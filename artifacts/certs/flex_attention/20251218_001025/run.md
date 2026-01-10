# FlexAttention correctness (SDPA vs Flex)

- Run ID: `20251218_001025`
- Generated: 2025-12-18 00:10:25 EST
- Overall: PASS

## Command

```bash
uv run python scripts/verify_flex_correctness.py --model standard --device cuda --dtype bfloat16 --compile --batch-size 1 --sequence-len 64 --vocab-size 1024 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 128 --atol 2e-2
```

## Notes

- This script compares SDPA (reference) vs FlexAttention for standard GPT across:
  - full forward
  - KV-cache last-token decode
  - optional KV chunk decode

See `summary.json` for full details.
