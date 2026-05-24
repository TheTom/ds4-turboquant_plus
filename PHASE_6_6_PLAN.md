# Phase 6.6 — simdgroup_matrix V-acc rewrite

Closes the remaining decode gap to fp8 at long context (~15% at
n_raw=256). The h8 Flash kernel (Phase 6 / 6.1 / 6.2 / 6.3 / 6.4) is
scalar-K-dot + scalar-V-acc + online-softmax; Apple's fp8 attention
uses `simdgroup_matrix` tensor units for the same matmul shape.

## Why it's multi-day

The clean way to close the gap is to keep the **accumulator** in
`simdgroup_matrix` form across chunks. That fights the per-head
online-softmax M/L state because:

* In the current scalar h8 design, each simdgroup handles 1 head.
  M, L, acc[16] live in per-simdgroup registers; scaling acc by `ms =
  exp(M - M_new)` is per-thread elementwise (cheap, correct).
* With `simdgroup_matrix<half, 8, 8>`, each simdgroup naturally
  handles one **output column tile** spanning all 8 heads. The
  accumulator becomes 8x8 matrix where each ROW corresponds to a
  different head. Per-row scaling of a simdgroup_matrix isn't a
  primitive operation — Metal's `simdgroup_multiply_accumulate`
  takes a full 8x8 left operand.
* Workaround: store per-head `ms[h]` in a small threadgroup buffer,
  construct an 8x8 diagonal `ms_diag` matrix, simdgroup_multiply
  acc_mat by ms_diag to scale rows. That's 64 FMAs to do 8 useful
  multiplies — 1/8 utilization, but the rest of the V-acc gains
  may pay it back. Needs measurement.

## Algorithm sketch (head-batched, simdgroup_matrix V-acc)

```
HEADS_PER_TG = 8, TILE_C = 16 (or 24, mem budget permitting)
COL_TILES = head_dim / 8 = 64  (DSV4 head_dim=512)

// Per simdgroup: assigned COL_TILES/NSG = 8 column tiles
simdgroup_float8x8 acc_mat[8];  // 8 col tiles per simdgroup
for (int i = 0; i < 8; i++) acc_mat[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

// Per-head state in threadgroup memory (8 floats each):
threadgroup float M_per_head[8];
threadgroup float L_per_head[8];
// Initialize: M_per_head[h] = -inf, L_per_head[h] = 0

for (uint r_base = 0; r_base < raw_count; r_base += TILE_C) {
    // 1. Cooperatively dequant tile into kv_tile_h (already in current kernel)

    // 2. K-dot: compute scores[8 heads, TILE_C rows]
    //    Each simdgroup handles 1 head's K-dot via simd_sum (as today)
    //    Write scores[h, j] to threadgroup scores buffer (8 × TILE_C floats)

    // 3. Per-head online softmax stats update:
    //    For h in 0..7: compute new M_per_head[h], L_per_head[h], ms[h]
    //    Each simdgroup updates its assigned head; results to threadgroup

    // 4. Apply ms[h] scaling to acc_mat:
    //    construct ms_diag[8,8] from M_per_head ms values
    //    For each col-tile assigned to this simdgroup:
    //        simdgroup_multiply(acc_mat[i], ms_diag, acc_mat[i])  // scales each row by ms[h]

    // 5. V-acc accumulate:
    //    For each inner iter k = 0..TILE_C/8:
    //        Load scores tile A[8, 8] = exp(scores[h, k:k+8] - M_per_head[h])
    //          (via simdgroup_load from threadgroup buffer)
    //        For each col-tile c assigned to this simdgroup:
    //            Load V tile B[8, 8] = kv_tile_h[k:k+8, c*8:c*8+8]
    //            simdgroup_multiply_accumulate(acc_mat[c], A, B, acc_mat[c])
}

// Final divide by L_per_head[h] and store to output
//   simdgroup_store each acc_mat to output, with per-row divide by L
//   (might need cross-simdgroup reduce of L if multiple simdgroups touch same head)
```

## Sub-phases

### Phase 6.6.1 — Add simdgroup_matrix V-acc as alternate code path

Add new kernel `kernel_dsv4_attention_decode_h8sg_turbo3_f32`. Don't
replace the existing h8 kernel; gate the new one behind
`DS4_METAL_TURBO3_H8SG=1` env. Lets us A/B against the proven scalar h8.

### Phase 6.6.2 — Validate correctness vs h8 scalar

Run `--ppl-prompt` on both. simdgroup_matrix path should match scalar
h8 within FP noise. If PPL diverges significantly, the per-head
ms_diag scaling has a bug.

### Phase 6.6.3 — Bench long context

Decode at n_raw=256 is the target workload. Current h8 scalar:
~29-30 t/s. fp8: ~36 t/s. Target: 33-34 t/s (close half the gap).

### Phase 6.6.4 — If 6.6.3 wins, replace h8 default

If simdgroup_matrix h8 wins by >5% with PPL parity, make it the new
default. Otherwise keep scalar h8 as the proven path and ship 6.6.x
as an opt-in env.

## What we DON'T know

* **ms_diag scaling efficiency**: 1/8 utilization for 64 FMAs may
  amortize across many chunks. Or may dominate. Empirical only.
* **Apple9 simdgroup_matrix throughput**: documented but real-world
  achievable depends on memory pressure, tensor unit contention.
* **Cross-simdgroup reduce of L_per_head**: if multiple simdgroups
  contribute to the same head's L, need a reduce. May serialize.

## Why not autonomous

The risk of shipping a broken kernel that produces NaN or garbage is
high. PPL validation cycle is multi-minute per iteration. Estimated
3-5 days of focused debugging to get 6.6.1 + 6.6.2 right.

Alternative if user prioritizes the long-context gap: bench-validate
the FP8 simdgroup_matrix path more carefully and consider whether the
existing flash_attn_ext_vec kernel can be adapted directly (would
need a deq_k_t4 / deq_v_t4 macro for inline turbo3 dequant; that
kernel has all the simdgroup_matrix scaffolding already in place).
The "adapt existing Flash kernel" path is probably safer than
"write new one from scratch with simdgroup_matrix".
