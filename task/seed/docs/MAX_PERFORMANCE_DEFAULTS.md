# Qwen4 max-performance defaults

Normal Qwen4 `.coli` execution uses the fastest configuration that has already passed real-model A/B validation. Explicit environment values always override these defaults.

Default fast policy:

- `QWEN_GDN_SINGLE_COPY=1`
- `QWEN_ATTN_METAL=1`
- `QWEN_QSA_INDEX_METAL=1`
- `QWEN_APPLE8_DIRECT=1`
- `QWEN_APPLE8_OVERLAP=1`
- `QWEN_SHARED_IO_OVERLAP=1`
- `QWEN_PREFIX_CACHE=1`
- `QWEN_PREFIX_CACHE_WRITE=1`
- Apple Silicon: `QWEN_BNNS_BF16=1`
- Apple Silicon: `QWEN_GDN_METAL=0`

Set any existing boolean knob to `0` to opt out. Rejected or unvalidated experimental paths are not enabled by this policy.
