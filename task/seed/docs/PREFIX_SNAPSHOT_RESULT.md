# Qwen4 exact prefix snapshot result

Validated on Qwen3.8-Flash-Next Apple8 on Apple M2 with CTX=262144, QWEN4_CACHE=32, Metal attention/QSA enabled, BNNS BF16 enabled, and Metal GDN disabled.

Real-model gate (`prefix_snapshot_gate`, prefix `1 2 3 4 5`, suffix `7 9 11`, 8 generated tokens):

- snapshot payload: 112.87 MiB
- snapshot creation: 110.90 ms
- cold prefix replay: 34,531.01 ms
- warmed prefix replay: 15,952.24 ms
- restore: 135.44 ms
- restore / warmed replay: 0.0085x
- replayed state: bit-exact
- restored state: bit-exact
- generated tokens: exact
- generated logits: bit-exact
- final causal state: bit-exact

Generated continuation:

`[6693, 220, 2423, 4766, 26297, 4466, 16638, 515]`

The gate requires restore cost to remain below 25% of warmed replay cost. This run passed by a wide margin.

This result validates the low-level snapshot/restore primitive only. It does not add an LRU, radix tree, disk persistence, or cache policy.
