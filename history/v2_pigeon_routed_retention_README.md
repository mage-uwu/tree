# Routed Retention + LUT BitLinear — a single pure attention layer (no MLP)

## What was built

**(2) Routed retention.** This is a RetNet-style mixer whose memory is split across the leaves of a learned hyperplane tree (depth 4, so 16 leaves per head). Keys write only to their leaf; queries read only from their leaf. With the `local` option, each head also keeps one fast-decay state that every token writes to:

```
parallel   O = (Q~ K~ᵀ ⊙ (D_local + D_leaf ⊙ M)) V        M[n,m] = [leaf(q_n) == leaf(k_m)] = Pq Pkᵀ
chunkwise  intra-chunk parallel + cross-chunk per-leaf states (RetNet chunk recurrence, per leaf)
recurrent  S_local ← γ S_local + k~ᵀv ;  S_leaf(k) ← γ_leaf^Δt S_leaf(k) + k~ᵀv   (lazy decay, O(1))
           o = q~ S_local + γ_leaf^Δt' q~ S_leaf(q)
```

- Q~ and K~ are RoPE-rotated by absolute position. Routing uses the un-rotated q/k, so it's by content, not position.
- Router: forward is hard (one leaf); backward uses soft leaf probabilities (products of sigmoids along the path) via STE. Aux losses push for balanced leaf use and a routing margin.
- With depth 0 it reduces exactly to RetNet retention (+RoPE).

**Composition with softmax / BitNet attention.** Every head is either `S` (causal softmax, RoPE, KV cache) or `R` (routed retention). All heads share the same BitNet b1.58 BitLinear q/k/v/g/o (absmean ternary weights, absmax int8 activations), the same per-head subLN, and the same output gate. So one layer can mix them freely, as `SRRR` does here, and a BitNet attention checkpoint's projections drop in unchanged.

**(1) LUT BitLinear (T-MAC-style), AVX2, bit-exact.** Weights are grouped 3 at a time, giving 27 patterns, which fold into 14 sign-symmetric table entries plus a sign bit. That's one byte per 3 weights (2.67 bits/weight). Per input vector, the kernel builds int16 tables split into lo/hi byte planes for +T and −T. Each `pshufb` then looks up 32 output rows at once, and the sign bit doubles as `pshufb`'s zeroing bit, so no blend is needed. Accumulation is int16, flushed to int32 every 64 groups. There are no multiplies.

## Exactness

| Check | Result |
|---|---|
| parallel vs chunkwise vs recurrent (torch fp64, all head mixes, hard routing) | ≤ 3.7e-15 |
| same, trained models, fp32, 512 tokens | ≤ 1.1e-5 max \|Δlogit\|, 100% argmax agreement |
| C engine vs torch recurrent | ≤ 1.1e-5, identical loss to 4 decimals |
| LUT vs AVX2 sign-dot vs naive int8 dot (int32 outputs) | 0 mismatches (≈360k outputs + all model matrices) |
| C logits across the 3 projection backends | bit-identical |

## Quality: 1 layer, d=128, 4 heads, no MLP, token-shift on k/v, single seed

TinyShakespeare char-level, val cross-entropy, 1500 steps, ctx 128:

| Heads | Val loss |
|---|---|
| SSSS softmax | **1.850** |
| RRRR RetNet (1 leaf) | 1.971 |
| RRRR routed, 16 leaves | 2.187 |
| RRRR routed + local | 1.978 |
| SRRR routed (1 softmax + 3 routed) | 2.062 |
| SRRR routed + local | **1.935** |

MQAR associative recall, accuracy (trained at 32 pairs, tested at 48 and 64 to stress the fixed-size memory):

| Heads | 32 pairs | 48 | 64 |
|---|---|---|---|
| SSSS softmax | 1.000 | 0.999 | 0.991 |
| RRRR RetNet | 0.994 | 0.921 | 0.805 |
| RRRR routed | 1.000 | 0.991 | 0.945 |
| RRRR routed + local | 1.000 | **0.995** | **0.962** |
| SRRR routed | 1.000 | 0.990 | 0.939 |
| SRRR routed + local | 1.000 | 0.993 | 0.946 |

## Speed: C, 1 thread, d=128, per token as context grows

| Model | ctx ≤128 | ≤1k | ≤4k | ≤8k |
|---|---|---|---|---|
| softmax SSSS | 17 µs | 68 µs | 290 µs | 544 µs |
| RetNet RRRR | 10 µs | 10 µs | 10 µs | 10 µs |
| routed + local RRRR | 14 µs | 12 µs | 13 µs | 13 µs |
| hybrid SRRR + local | 15 µs | 27 µs | 72 µs | 143 µs |

LUT kernel vs a properly vectorized AVX2 ternary dot (`sign_epi8` + `maddubs`), 1 thread:

| Matrix | AVX2 sign-dot | LUT | Speedup |
|---|---|---|---|
| 128×128 | 0.6 µs | 0.5 µs | 1.15× |
| 768×768 | 18.0 µs | 9.5 µs | 1.89× |
| 2048×2048 | 218 µs | 77 µs | 2.84× |
| 4096×4096 | 868 µs | 406 µs | 2.14× |
| 11008×4096 | 2413 µs | 1097 µs | 2.20× |

At d=128 the projections are a small part of each token's cost, so the LUT barely moves end-to-end numbers there. It matters at real widths.

## Read

1. **The duality holds exactly.** Routing enters only as a mask factor M = Pq·Pkᵀ, so the parallel, chunkwise, and recurrent forms stay one function, and the lazy decay keeps 16 leaves at O(1) per token.
2. **Routing buys memory.** At 64 pairs, routed + local recovers 96% recall vs RetNet's 80%, which closes most of the gap to softmax (99%) at constant per-token cost.
3. **Pure routing hurts language modeling** (2.187). Partitioning cuts the recent context out of each query's view. The local state fixes this: routed + local matches RetNet on LM (1.978 vs 1.971) and beats it on recall.
4. **Softmax heads still win LM at one layer.** The best mix was 1 softmax + 3 routed + local (1.935), but its KV cache brings back linear cost growth with context.
5. **Caveats:** single seed, tiny scale, short training. Differences of about 0.01 on Shakespeare are within noise.

## Next

- Try a larger leaf count (64–256) with top-2 reads, since recall should keep scaling.
- Use a learned per-leaf decay, or data-dependent gating (GLA / Gated DeltaNet style), which still keeps the duality.
- Put the MLP back (the FFF trees from v1) and scale up to a BitNet b1.58 checkpoint: keep its attention projections and swap 3 of 4 heads per layer to routed + local.
- Build an AVX-512 LUT path (64 rows per `pshufb`) and nibble-pack indices to 1.67 bits/weight.

## Files

`attn_model.py` model + three forms · `train_attn.py` training (`--task shakespeare|mqar --kinds --rdepth --local`) · `test_forms.py` fp64 equality · `export_attn.py` → `.cha2` · `attn.c` C engine (`check`, `dump`, `bench`, `gen`) · `lut.h` LUT + AVX2 kernels · `lutbench.c` · `parity_attn.py` · `runs/` checkpoints, exports, logs

```
gcc -O3 -march=native -o attn attn.c -lm
./attn runs/shakespeare_routedlocal.cha2 check
./attn runs/shakespeare_routedlocal.cha2 bench 8192 2
./attn runs/shakespeare_routedlocal.cha2 gen 300 0.8 3 "ROMEO:"
```
