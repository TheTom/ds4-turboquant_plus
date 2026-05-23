# turbo3 KV cache roadmap

`--kv-cache turbo3` is the TurboQuant+ KV cache port from
[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant).
See the prior-art chain at the top of `dsv4_turbo3_kv_quantize_row_inplace_cpu`
in `ds4.c` and the byte-layout comment in `ds4.h` (the `DS4_KV_TURBO3` enum).

This doc tracks what's done, what's deferred, and why.

## Phase 1 — quality simulation (shipped, commits `d8cf423`, `c759d7d`)

In-place float-sim round trip on the latent KV row before storage.  Same
storage layout as fp8 (one float per element); only the per-element
quantization error differs.  Lets the engine validate the turbo3 algorithm
end-to-end without changing any cache layout.

  * `--logprob-vectors` passes on the live model
  * Generation output is byte-identical to fp8 on the smoke prompts
  * `ds4-bench` 2K/8K/14K/16K throughput within 1% of fp8

Files: `ds4.c` (`dsv4_turbo3_kv_quantize_row_inplace_cpu`), `ds4_cuda.cu`
(`turbo3_kv_quantize_kernel`), `ds4.h` (`ds4_kv_dtype` enum).

## Phase 2a — packed-byte cache storage (this commit set)

The GPU `layer_raw_cache` and `mtp_raw_cache` are sized and written as
packed turbo3 bytes (not floats) when the active dtype is turbo3.

Byte layout per row (head_dim=512, n_rot=64, GROUP_SIZE=64):
```
bytes  0..167  : 7 groups of 64 elements -> 24 packed bytes each = 168 B
bytes 168..174 : 7 FP8 E4M3 matched-norm scales (one per group)
bytes 175..430 : 64 raw little-endian floats (RoPE tail, untouched)
total            431 bytes/row vs 2048 bytes/row for fp8 -> 4.75x smaller
```

Read path: before each attention dispatch the per-graph
`raw_cache_dequant_scratch` (raw_cap * DS4_N_HEAD_DIM floats, allocated only
when dtype=turbo3) is filled by the dequant-to-scratch kernel.  The existing
12 attention kernels read the scratch unchanged.

This is **Option B** from the kernel-inventory memo
(`/tmp/ds4_phase2_kernel_inventory.md`).  It captures:

  * The full cache memory footprint shrink (4.75x on the SWA ring).
  * The disk-cache footprint shrink (Phase 2c — see below).

It does NOT capture:

  * The per-attention-call V-load bandwidth reduction.  The dequant-to-
    scratch kernel reads 9x less bytes vs the float-sim cache, but the
    attention kernel then reads the full float scratch — net memory
    traffic shrinks ~4-5x for the dequant pass but stays the same for
    the attention pass.  On DSV4 IQ2XXS at decode T=1 this matters less
    than expected because the model is expert-weight-bandwidth-bound
    (80 GB of MoE weights moved per token) not KV-cache-bandwidth-bound.

Test gates:
  * `make cuda-spark`, `make cpu`, `make ds4_test` -- all clean
  * `DS4_TEST_KV_DTYPE=turbo3 ./ds4_test --logprob-vectors` -- OK on
    all 4 live vectors
  * `./ds4 --kv-cache turbo3 -p "..." -n 16 --temp 0 --nothink` -> output
    byte-identical to fp8 on the smoke prompts

## Phase 2b — inline dequant in 12 attention kernels (deferred)

This is the bandwidth-win follow-up.  Each attention kernel inline-dequants
the packed cache bytes per-row (via `turbo3_dequant_group64_device` in
`ds4_cuda.cu`) instead of reading a pre-dequanted float scratch.  Net per
attention call: 4-5x less memory traffic on V-load, ~3-4x more compute per
element (offset by the per-block scaled-centroid hoist from
`/tmp/ds4_phase2_llamacpp_patterns.md` §2.3 -- `sc[8] = centroid[c] * scale`
hoisted out of the inner loop).

### Why deferred (no scope creep)

The kernel-inventory memo (`/tmp/ds4_phase2_kernel_inventory.md`)
catalogues the 8 attention kernels that need rewriting:

  1. `attention_prefill_raw_kernel`              (Wave 1 EASY)
  2. `attention_prefill_mixed_kernel`            (Wave 1 EASY)
  3. `attention_decode_mixed_kernel`             (Wave 1 EASY)
  4. `attention_indexed_mixed_kernel`            (Wave 1 EASY)
  5. `attention_indexed_mixed_heads8_online`     (Wave 2 MEDIUM)
  6. `attention_static_mixed_heads8_online`      (Wave 2 MEDIUM)
  7. `attention_decode_mixed_heads8_online`      (Wave 2 MEDIUM)
  8. `attention_indexed_mixed_heads8_rb4_kernel` (Wave 3 HARD)

Each is mechanical but careful: the inner V-load loop changes from
`acc += scores[r] * raw_kv[r*head_dim + d]` to one of two patterns:

  * Scalar (Wave 1): pre-stage the row into per-block shmem scratch
    once via the dequant primitive, then walk the dot loop unchanged.
  * float4-vectorized (Wave 2 + 3): keep the float4 inner loop but
    populate `kv_shared[off]` from a per-block dequant scratch instead
    of from the raw_kv pointer.

Estimated effort: 1-2 days of focused kernel surgery + test validation.

### When it makes sense

The bandwidth crossover where Phase 2b's V-load shrink moves the gen_tps
needle on this model is at ctx 128K+ (where the SWA ring is fully active
and the model isn't expert-bound for the cache-read portion of each
attention call).  Below that, Phase 2a's cache memory + disk savings
are the user-visible win and Phase 2b is performance-neutral.

For models with smaller MoE weights or dense models, the crossover is
much lower — the inline-dequant work matters earlier.

## Phase 2c — packed-byte disk cache (deferred, separate commit)

Currently sessions saved with `--kv-cache turbo3` write the v1 disk format
(floats), by dequanting the cache to scratch first.  This lets v1 files
load on any dtype but doesn't capture the disk shrink.

Phase 2c bumps the disk header to v2 with a `kv_dtype` field per-stream
(per llama-cpp-turboquant's `llama-kv-cache.cpp:298` per-K-per-V dtype
discipline) and writes packed bytes directly.  v1 readers continue to load
v1 files; v2 readers validate that the saved dtype matches the active
session dtype.

Phase 2c shrinks the on-disk session payload by 4.75x on the SWA ring rows
(matching the in-memory shrink).  Session save/restore latency drops
proportionally.

## Phase 2d — compressor pool + indexer pool packing (out of scope)

The compressor (`layer_attn_comp_cache`) and indexer (`layer_index_comp_cache`)
pools stay as floats / F16 across all of Phase 2.  They consume softmax-
weighted accumulations of raw KV rows; packing them requires either
storing rotated values (and rotating the consumers) or accepting an extra
iWHT pass on every compressor decode.

The compressor pool is potentially larger than the raw SWA ring at long
context (4-32 GB at 256K ctx) so Phase 2d would be the biggest absolute
shrink.  Tracking as a separate research milestone.

## Prior art chain

  1. Google TurboQuant — arXiv:2504.19874 (ICLR 2026).  The canonical
     Randomized Hadamard rotation + Lloyd-Max codebook design.
  2. `TheTom/turboquant_plus` — Tom Turney's research umbrella for the
     beyond-Google TQ+ work (matched-norm L2, asymmetric K/V, InnerQ,
     sparse V, layer-aware V).
  3. `TheTom/llama-cpp-turboquant` — engine reference for the per-block
     centroid hoist (`fattn-vec.cuh:478-512`), shmem Q*centroid LUT
     (`fattn-vec.cuh:140-326`), and seed=42 sign tables.

ds4's port follows (1) and (3) closely.  The deferred Phase 2b inline-
dequant work directly transcribes (3)'s per-block centroid hoist into ds4's
attention kernels.
