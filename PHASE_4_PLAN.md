# Phase 4 — CUDA port plan for turbo4 + turbo2

The new dtypes shipped in Phase 3 / 3.1 / 3.2 are Metal-only this
round. Phase 3.3 (`481b8df`) added CUDA linker stubs + engine-open
guard so `make cuda` builds and users get a clean error if they pick
`--kv-cache turbo4|turbo2 --backend cuda`. **Phase 4 ships the real
CUDA implementations** so the dtypes work on Spark too.

## Atlas reference kernels (already in repo)

```
atlas/kernels/gb10/common/reshape_and_cache_turbo.cu
   — turbo2/3/4 codebooks, signs, WHT primitives
atlas/kernels/gb10/common/paged_decode_attn_turbo2_128.cu
atlas/kernels/gb10/common/paged_decode_attn_turbo4_128.cu
atlas/kernels/gb10/common/paged_decode_attn_turbo4_512.cu
   — decode attention with inline turbo dequant (the Wave-M3-on-Mac
     equivalent for CUDA, already exists)
atlas/kernels/gb10/common/inferspark_prefill_paged_turbo2.cu
atlas/kernels/gb10/common/inferspark_prefill_paged_turbo4*.cu
   — prefill attention with inline turbo dequant
```

Atlas uses GROUP_SIZE=16 and WHT256 in the canonical reference. **ds4
uses GROUP_SIZE=64 and WHT64** to match the existing turbo3 layout. So
the port must re-derive the codebook unpack/pack for the 64-element
group cadence, not copy atlas's verbatim. Codebook/bounds/MAX
constants ARE identical and can be lifted directly.

## Pattern to follow: Phase 2b turbo3 (Spark, commits `694fe91..b2ac445`)

The turbo3 CUDA port is the canonical template:

* `ds4_cuda.cu` adds `__device__ __constant__` arrays for TURBO3_CODEBOOK,
  TURBO3_BOUNDS, TURBO_SIGNS{1,2}_64, TURBO3_MAX
* `turbo3_dequant_group64_device` — inline per-thread dequant (called
  from every attention kernel's K-dot and V-acc paths)
* `turbo3_kv_pack_kernel`, `turbo3_kv_dequant_to_scratch_kernel`,
  `turbo3_kv_pack_batch_kernel` — the byte-pack pool kernels
* `attention_decode_mixed_turbo3_kernel`, `attention_prefill_raw_turbo3_kernel`,
  `attention_decode_mixed_heads8_online_turbo3_kernel`,
  `attention_indexed_mixed_heads8_online_turbo3_kernel` — the four
  attention kernels with inline dequant
* `ds4_gpu_*_turbo3_*_tensor` C wrappers that dispatch the kernels
* All sites read the same `ds4_kv_row_bytes(head_dim, n_rot, DS4_KV_TURBO3)`
  for ring stride math

## Phase 4 sub-phases

### Phase 4.1 — turbo4 CUDA primitives

Add to `ds4_cuda.cu`:

```cuda
__device__ __constant__ float TURBO4_CODEBOOK_D[16] = {...};   // copy from ds4.c
__device__ __constant__ float TURBO4_BOUNDS_D[15] = {...};
#define TURBO4_MAX_D 2.7326f
#define TURBO4_DATA_BYTES_PER_GROUP 32u

__device__ __forceinline__ void turbo4_dequant_group64_device(
        float out64[64],
        const unsigned char *row_base,
        uint32_t group_idx,
        uint32_t n_nope,
        int signs_on) {
    // 32 B data + 1 FP8 scale per group, 4 bits per element, 2 nibbles per byte
    // ... (mirror of Metal kernel in metal/dsv4_turbo3.metal)
}

__device__ __forceinline__ unsigned char turbo4_quantize_index_device(float v) {
    // linear scan over 15 bounds → idx [0..15] (or atlas-style binary tree)
}
```

### Phase 4.2 — turbo4 byte-pack pool kernels

* `turbo4_kv_pack_kernel` (single-token pack, grid <<<n_tok, 64>>>)
* `turbo4_kv_pack_batch_kernel` (ring-aware batch)
* `turbo4_kv_dequant_to_scratch_kernel` (for Phase 2a-style fallback)

These mirror their turbo3 siblings 1:1, only the codebook width changes.

### Phase 4.3 — turbo4 attention kernels

* `attention_decode_mixed_turbo4_kernel` (siblings of Phase 2b Wave
  1.2 + 1.4 turbo3 kernel — same V-acc tile pattern, same K-dot
  inline dequant, only the codebook+pack-width changes)
* `attention_prefill_raw_turbo4_kernel`
* `attention_decode_mixed_heads8_online_turbo4_kernel`
* `attention_indexed_mixed_heads8_online_turbo4_kernel`

The h8 / online softmax pattern from Metal Phase 6 is not strictly
needed on CUDA — the existing CUDA Wave 1.2 + 1.4 turbo3 path
achieves fp8 parity on GB10 (Blackwell tensor cores don't help fp8
decode at B=1; everything is scalar). Just mirror that pattern for
turbo4.

### Phase 4.4 — C wrappers + engine-open guard

* `ds4_gpu_dsv4_turbo4_kv_quantize_tensor`
* `ds4_gpu_dsv4_turbo4_kv_pack_tensor`
* `ds4_gpu_dsv4_turbo4_kv_dequant_to_scratch_tensor`
* `ds4_gpu_dsv4_turbo4_kv_pack_batch_tensor`
* `ds4_gpu_attention_{decode_heads,decode_mixed_batch,indexed_mixed_batch,prefill_raw}_turbo4_heads_tensor`

These replace the linker stubs from Phase 3.3 (`481b8df`).

Drop the `--kv-cache turbo4 + --backend cuda` reject in
`ds4_engine_open` (still in place from Phase 3.3).

### Phase 4.5 — turbo2 mirror

Phase 4.1-4.4 again with the 4-level / 2-bit codebook. Same pattern.

### Phase 4.6 — Validate on Spark

* `make cuda-spark` clean build
* `./ds4 --cuda --kv-cache turbo4 ...` coherent output, decode parity
  with fp8 (turbo3 achieved this on Spark per project memory note)
* `./ds4-bench --cuda --kv-cache turbo4 --ppl-prompt ...` ppl Δ
  matches Metal numbers (turbo4 within fp8 noise, turbo2 +~3%)

## Estimated effort

~3 days focused work on a developer with Spark access. The turbo3
port (Phase 2b) took similar effort across several sessions.
turbo4/2 are simpler in some ways (Metal port already debugged the
pack/unpack), harder in others (more attention-kernel variants need
copying because CUDA path has 4 kernels vs Metal's 1).

## What needs Spark hardware

* Bench validation (decode tps measurement is the load-bearing
  acceptance criterion)
* `cuda-spark` compile (needs CUDA toolkit + sm_120)
* Sanity testing on the actual DSV4 model (the Mac model is rsynced
  from Spark per project memory note)

## What can be done autonomously without Spark

* Write the CUDA kernels (compile-only check via `make cuda-generic`
  or atlas's `Cargo build`)
* Cross-check codebook / bounds constants against atlas + Metal
* Write the C wrapper signatures (mirror Phase 3.3 stub signatures
  but with real implementations)

So Phase 4.1-4.3 + 4.5 (CUDA kernel work) is autonomous-friendly
**up to compilation**. 4.4 + 4.6 (engine wiring + bench) need Spark.
