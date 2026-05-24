# Phase 7 — comp_cache turbo3 compression: implementation plan

Companion design doc for the foundation commit `719c5c2` (Phase 7).
The foundation (`--comp-cache` CLI flag, `comp_dtype` field, CUDA pack
/ dequant entry points, engine_open guard) is in place. This doc maps
out the alloc/store/read plumbing that crashed in the original
attempt (`39f15ae`, reverted `256362a`) so it can be re-landed
incrementally without repeating the breakage.

## Why this is risky

The previous attempt wired ~10 ds4.c sites in a single commit. The
crash was at `compressor_prefill_tensor` returning `false` on the
first turbo3 comp layer: the compressor pipeline expected float input
but the comp pool had been re-allocated as packed bytes. The
mid-pipeline read interpreted the packed bytes as floats → garbage →
predicate fail → false return.

The fix is to make the alloc / store / read changes **lock-step**, not
spread across commits.

## Sub-phases (incremental staging)

### Phase 7.1 — Alloc the packed pool alongside the existing float pool

Add a new field `g->layer_attn_comp_cache_packed[il]` (per layer) of
type `ds4_gpu_tensor *`. Allocated only when
`g_ds4_comp_dtype == DS4_KV_TURBO3` AND
`g_ds4_kv_dtype == DS4_KV_TURBO3` (foundation guard already rejects
turbo3 on Metal so this lands on the CUDA backend only).

Size: `g->layer_comp_cap[il] * ds4_comp_row_bytes(DS4_N_HEAD_DIM, DS4_KV_TURBO3)`
= `comp_cap * 200` bytes per layer at `DS4_N_HEAD_DIM=512`.

Comp_cap is per-layer (different ratios per layer). Total at ctx=32k
across 60 layers: ~96 MiB packed vs ~1 GiB float.

**Behavior change**: zero — the packed pool is allocated but never
written or read. Useful only as a startup sanity check that the
allocator can satisfy the request.

Touch sites (ds4.c):
* `metal_graph_create_arena_for_ctx` (line ~9687): alloc condition
* Engine teardown: free condition
* Struct definition: add `layer_attn_comp_cache_packed`

### Phase 7.2 — Wire the compressor pack write path

When comp_dtype=turbo3, after the compressor writes float into
`g->compressor_pool`, immediately pack-and-store into the per-layer
packed pool.

Where: `compressor_prefill_tensor` (the function that wraps the
compressor projection + write to `layer_attn_comp_cache[il]`).

Pattern (mirror of raw cache Phase 2):
```c
if (g_ds4_comp_dtype == DS4_KV_TURBO3) {
    rc = ds4_gpu_dsv4_turbo3_comp_pack_tensor(
            float_source, g->layer_attn_comp_cache_packed[il],
            n_rows, DS4_N_HEAD_DIM, 0 /* n_rot */,
            ds4_comp_row_bytes(DS4_N_HEAD_DIM, DS4_KV_TURBO3));
}
/* Also still write the float pool so reads work in 7.2 — drop the
 * float write in 7.4 once all reads are dequant-from-packed. */
```

**Why keep dual-write in 7.2**: lets us A/B compare a turbo3-read
path against the existing float-read path within the same session.
Saves about 100 MiB of cache validation effort vs single-write.

### Phase 7.3 — Wire the attention dequant read path

When comp_dtype=turbo3, attention reads of `comp_cache` go through
the existing `ds4_gpu_dsv4_turbo3_kv_dequant_to_scratch_tensor`
helper (already in place via Phase 7 foundation `ds4_cuda.cu`
additions). Output: float scratch of size
`g->comp_cap * DS4_N_HEAD_DIM * sizeof(float)`.

Touch sites (read sites that consume `g->layer_attn_comp_cache[il]`):
1. `attention_decode_mixed_batch_heads_tensor` and indexed siblings
2. The decode-only `attention_decode_heads_tensor` sibling
3. The prefill-chunk indexed/static heads8 sites

Pattern (mirror of raw cache):
```c
ds4_gpu_tensor *comp_attn = (g_ds4_comp_dtype == DS4_KV_TURBO3)
        ? dequant_comp_to_scratch(g->layer_attn_comp_cache_packed[il],
                                  g->comp_cache_dequant_scratch,
                                  g->comp_cap, DS4_N_HEAD_DIM)
        : g->layer_attn_comp_cache[il];
```

Need a new `g->comp_cache_dequant_scratch` tensor (mirror of
`g->raw_cache_dequant_scratch` from Phase 2). Allocated once at
graph creation, reused across layers.

### Phase 7.4 — Drop the float dual-write

Once 7.3 is validated end-to-end, the float `layer_attn_comp_cache[il]`
pool can be removed (or kept as fp8 when comp_dtype=fp8). This is
the actual memory win: at ctx=32k, ~864 MiB freed.

### Phase 7.5 — PPL gate

TQ+'s Lloyd-Max codebook assumes near-Gaussian per-group input. This
holds for MLA latents (the raw cache) because compressed_kv ≈ a
projection of layer activations, which empirically Gaussianize after
WHT. **It is uncertain whether compressor outputs are
Gaussian enough** — the compressor is a learned projection, not
guaranteed to preserve Gaussianity.

Validate via `ds4-bench --ppl-prompt FILE --comp-cache turbo3` on a
held-out text. Accept threshold: ppl Δ vs fp8 < 2% (similar bar to
turbo3 raw cache).

If ppl regression is unacceptable, fallback options:
* Per-layer calibration constant (apply a learned scale before quant)
* fp8 comp + turbo3 raw (already supported by the `--comp-cache`
  independence from `--kv-cache`)
* Train a comp-specific Lloyd-Max codebook (would require new offline
  calibration tooling)

### Phase 7.6 — Metal port

Mirror the CUDA packed pack/dequant kernels to MSL. Same pattern as
Phase 2b Wave M0/M1/M3 work (the raw cache port). Existing `metal/
dsv4_turbo3.metal` already has the per-row dequant primitive
(`turbo3_dequant_group64`) — comp rows just use it with n_rot=0.

Estimated effort: 1-2 days following the Phase 2b template.

## What we DON'T know

* Memory contention impact: packed comp_cache is 10x smaller, which
  is great for L2 residency, but the dequant-to-scratch path adds a
  64-byte-aligned scratch buffer per attention call. Net bandwidth
  win at long context is plausible but unmeasured.
* Quality: TQ+ on compressor outputs is the open question above.
* Per-layer ratio interaction: layers with `ratio=0` (no compressor)
  have `layer_attn_comp_cache[il] = NULL`. The dispatch must
  fall through cleanly (the previous attempt crashed when
  `view_dispatch` returned NULL for these layers).

## What's safe to land autonomously

Nothing past the foundation (`719c5c2`). Each of the sub-phases above
needs interactive debug cycles: the previous attempt's crash was in
a code path that exercises only after the compressor runs, which
means full model startup, prefill, and decode on a real prompt to
reproduce. That's not autonomous-loop-friendly.
