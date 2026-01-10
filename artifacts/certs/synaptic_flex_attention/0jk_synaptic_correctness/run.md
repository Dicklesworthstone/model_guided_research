# Synaptic FlexAttention correctness (non-flex vs flex)

- Run ID: `0jk_synaptic_correctness`
- Generated: 2025-12-17 22:42:29 EST
- Overall: FAIL

## Command

```bash
uv run python scripts/verify_flex_correctness.py --model synaptic --device cuda --dtype float32 --run-id 0jk_synaptic_correctness
```

## Notes

- This script compares SDPA (reference) vs FlexAttention for standard GPT across:
  - full forward
  - KV-cache last-token decode
  - optional KV chunk decode

See `summary.json` for full details.
