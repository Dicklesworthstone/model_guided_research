# FlexAttention correctness (SDPA vs Flex)

- Run ID: `0jk_standard_correctness_postfix2`
- Generated: 2025-12-17 23:06:55 EST
- Overall: PASS

## Command

```bash
uv run python scripts/verify_flex_correctness.py --model standard --device cuda --dtype float32 --run-id 0jk_standard_correctness_postfix2 --suite
```

## Notes

- This script compares SDPA (reference) vs FlexAttention for standard GPT across:
  - full forward
  - KV-cache last-token decode
  - optional KV chunk decode

See `summary.json` for full details.
