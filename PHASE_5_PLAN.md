# Phase 5 — Asymmetric K/V quantization

Atlas ships asymmetric variants (`fp8k_turbo3v`, `fp8k_turbo4v`,
`bf16k_turbo3v`, etc.) where K stays at high precision and V uses
turbo. The rationale: K is more sensitive to quantization noise than
V in attention math (a small K error rotates the score vector and
spreads attention mass, which propagates everywhere; a V error just
shifts the output by `score * Δv`, bounded by softmax weights ≤ 1).

**This pattern doesn't apply directly to DSV4 MLA.**

## Why MLA breaks the standard asymmetric pattern

In a vanilla transformer's KV cache, K and V are computed during
prefill (`K = W_K @ x`, `V = W_V @ x`) and stored *independently*.
You can quantize the K cache with one dtype and the V cache with
another — they are physically separate buffers.

In MLA (DSV4's attention), there is ONE stored entry per token: the
post-RoPE compressed_kv latent (`c_kv = W_DKV @ x; c_kv_rope = RoPE(c_kv)`).
At attention time, K and V are different *projections* of that
single stored entry:

```
K = W_UK @ c_kv_nope || c_kv_rope    (head split, n_nope + n_rot dims)
V = W_UV @ c_kv_nope                  (head split, n_nope dims)
```

There is no separate K cache or V cache to quantize differently —
both are derived from the same packed turbo-stored row at use time.

## What CAN be asymmetric on MLA

Several axes remain open:

### (A) RoPE-vs-latent asymmetry — **already shipped**

`c_kv_rope` (the n_rot=64 RoPE tail) stays as raw floats; `c_kv_nope`
(the n_nope=448 latent body) gets the Lloyd-Max codebook. This is
the existing turbo3/4/2 layout. The RoPE tail is small (64 × 4 = 256
B/row) and dynamics-sensitive (the rotary frequencies amplify any
quant error), so leaving it as float is the right call. Already in
all three turbo variants.

### (B) Codebook-asymmetry per projection

W_UK and W_UV project the same stored latent into different K/V
representations. We *could* design the storage so that the dequant
on the W_UK path uses one codebook and the W_UV path uses another.
Since both happen at attention time from the same packed bytes, the
codebooks would need to be applied per-projection — meaning two
different dequant kernels per attention call.

Why: K is more sensitive to noise (per the standard argument). So
maybe the K dequant could use a finer codebook (turbo4 levels) and
the V dequant a coarser one (turbo3 or turbo2).

But the stored bytes are *one packed representation*. Different
codebooks at read time mean the bytes were quantized to neither
codebook optimally. Would need joint optimization: pack each row
once such that BOTH codebooks decode to acceptable approximations.
That's a research-grade open problem (joint codebook design), not a
straight port.

**Effort**: research + multi-day, payoff uncertain. Probably not
worth it.

### (C) Per-layer codebook selection

Some layers might tolerate more quantization noise than others. The
Phase 3a comp_cache work hinted at this: ratio-0 layers (no
compressor) might warrant higher-precision storage; deep layers
(compressor outputs) might be noisier and warrant coarser codebooks.

Implementation: per-layer `--kv-cache` dtype selection, e.g.,
`--kv-cache-layers fp8:0-15,turbo3:16-47,turbo2:48-59`. Each layer
allocs its raw cache at the per-layer row stride.

**Effort**: moderate (CLI parser + per-layer alloc + per-layer
dispatch lookup). Quality validation needs a sensitivity sweep
across layers, multi-hour of bench time.

### (D) K-dot precision vs V-acc precision

We could keep the storage as turbo3 (3-bit) but do the K-dot in
higher precision (e.g., dequant to fp16 + simdgroup_matrix
intermediate) and the V-acc in lower precision (turbo3 inline). That
biases compute precision toward the K-dot path which the argument
above says benefits more.

This is mostly a kernel structure decision, not a storage one. Worth
prototyping when the simdgroup_matrix path (Phase 6.6) lands.

## Recommendation

**Defer Phase 5 indefinitely.** The MLA-native form of asymmetric
quantization is (D) — kernel-precision asymmetry, not storage. Best
attacked as a 6.7 follow-up after the simdgroup_matrix V-acc work
matures.

If a user specifically asks for "K is more precise than V", the
practical answer today is: use `--kv-cache turbo4` (the smallest
quantization MSE we ship). Both K and V dequants benefit from the
4-bit codebook. Not "asymmetric" per se, but the same quality goal.

## What atlas's fp8k_turbo3v means for ds4

The atlas asymmetric variants apply to non-MLA models (head_dim=128,
separate K/V cache). For those, fp8k+turbo3v makes immediate sense.
DSV4 is the only MLA model atlas targets, and atlas's MLA support
appears to also use the symmetric storage pattern (one compressed_kv
per token), so there's no upstream precedent for MLA asymmetry to
port from.

If a future DSV5 (or similar MLA-with-extras model) adds separate K
and V projection paths back to the cache, the standard asymmetric
pattern would apply. Watch for that.
