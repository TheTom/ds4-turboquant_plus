# turbo3 KV cache A/B bench

`gb10_fp8.csv` and `gb10_turbo3.csv` capture an A/B sweep run on the GB10
(ASUS Ascent GX10, 128 GB unified memory) with the IQ2XXS DeepSeek-V4-Flash
checkpoint at `/home/pidtom/models/ds4-model/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.

Reproduce one cell:

```
./ds4-bench [--kv-cache turbo3] -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 16384 --step-incr 6144 --gen-tokens 64 \
  --csv /tmp/bench_$DTYPE.csv
```

Same machine, same prompt, no other GPU clients.  Both runs done back-to-back
on a fresh model load — model weights cached after the first ~20s warmup, so
the numbers below reflect steady-state inference throughput.

| ctx | prefill_tps fp8 → turbo3 | gen_tps fp8 → turbo3 |
|-----|--------------------------|----------------------|
|  2K | 399.48 → 399.04 (-0.11%) | 13.71 → 13.62 (-0.66%) |
|  8K | 398.73 → 396.36 (-0.59%) | 13.59 → 13.49 (-0.74%) |
| 14K | 383.73 → 381.41 (-0.60%) | 13.41 → 13.33 (-0.60%) |
| 16K | 373.74 → 372.68 (-0.28%) | 13.45 → 13.34 (-0.82%) |

The turbo3 kernel costs ~0.5-0.8% throughput vs fp8 across the sweep — within
bench noise.  No quality loss observed on the smoke prompts ("The capital of
France is" -> "Paris.", "Write the Python code to compute the factorial of n
recursively." -> identical 32-token completion).  `--logprob-vectors` with
`DS4_TEST_KV_DTYPE=turbo3` passes; the same test on the fp8 default also passes
the long-context vectors but trips a pre-existing argmax flip on
`short_code_completion` step 1 that is reproducible on a clean checkout of
this branch's parent commit (`9ae1eeb`) — i.e. unrelated to this work.

Notes:

  - The throughput difference is small because ds4's KV cache is a quality
    simulation, not a compressed-bytes store.  The expensive part of both
    paths is identical (one 64-element group walk per non-RoPE chunk).  Turbo3
    pays an extra WHT butterfly + matched-norm reduction per group; FP8 pays
    one amax reduction + 64 E4M3 round-trips per group.  The two cancel out
    within bench noise on this model.
  - The real bandwidth win lives in a future Metal port that stores the
    rotated 3-bit Lloyd-Max indices + per-group FP8 scale instead of full
    floats.  At DS4_N_HEAD_DIM=512 / DS4_N_ROT=64 / 64-element groups, a
    proper packed-byte layout would store the 448-element latent as
    448·3/8 + 7 FP8 scale bytes = 175 bytes/row vs 1792 bytes/row today — a
    10x reduction on the dominant KV-cache memory footprint.  The CUDA path
    here proves the math is correct; the Metal port lays packed bytes.
