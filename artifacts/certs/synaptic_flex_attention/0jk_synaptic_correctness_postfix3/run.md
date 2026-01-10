# Synaptic FlexAttention correctness (non-flex vs flex)

- Run ID: `0jk_synaptic_correctness_postfix3`
- Generated: 2025-12-17 23:12:38 EST
- Overall: PASS

## Command

```bash
uv run python scripts/verify_flex_correctness.py --model synaptic --device cuda --dtype float32 --run-id 0jk_synaptic_correctness_postfix3
```

## Notes

- This script compares SDPA (reference) vs FlexAttention for standard GPT across:
  - full forward
  - KV-cache last-token decode
  - optional KV chunk decode

See `summary.json` for full details.
