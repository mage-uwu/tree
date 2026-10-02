# Chimera — train as a net, infer as a tree + RNN

Prototype at TinyShakespeare scale (char-level, ~0.85M params, 4 layers, d=128).
Everything trained and benchmarked on 2 CPU cores, no GPU.

## What it is

| Part | Train (PyTorch, parallel/dense) | Infer (C, CPU) |
|---|---|---|
| Mixer | Retention, parallel form `(QKᵀ ⊙ D)V` | Retention, recurrent form, fixed 4×32×32 state per layer, no KV cache |
| FFN | FFF tree: all nodes computed, hard-route STE (fwd = `a>0`, bwd = sigmoid grad) | Walk one root→leaf path per tree |
| Weights | BitNet b1.58 BitLinear: absmean ternary weights, absmax int8 activations, STE | int8 × {-1,0,+1} → int32, no weight multiplies |
| Depth | Matryoshka: random depth truncation on 50% of steps | `max_depth` knob = early exit / min path |
| Aux losses | margin (pushes route logits away from 0) + balance (each node ~50/50) | stable routes under int8 rounding |

## BitNet b1.58 compatibility (established now)

- `BitLinear` follows b1.58: per-tensor absmean weight scale, `round(w/s).clamp(-1,1)`, per-token absmax int8 activations, pre-RMSNorm, no bias, `[out, in]` layout.
- Embedding and LM head stay full precision, as in BitNet.
- FFF node matrices `w_in`, `w_out` use the same ternary quantizer, so a BitNet checkpoint's attention/projection weights drop in directly. Only the FFN changes.
- Export (`export.py`) packs ternary as 2-bit codes `q+1 ∈ {0,1,2}`, 4 per byte, plus an fp32 scale. That's the same encoding family as bitnet.cpp's I2_S, so moving to an upstream kernel layout is a repack, not a re-quantization.

## Results

**Quality** (val cross-entropy, nats/char, 2000 steps each, same retention stack):

| FFN | Neurons active / token / layer | Val loss |
|---|---|---|
| none (tree model at depth 0) | 0 | 2.151 |
| FFF 2 trees × depth 8 (255 nodes each) | 16 | 2.118 |
| FFF 16 trees × depth 5 ("forest") | 80 | 2.108 |
| Dense ternary FFN, hidden 510 | 510 | **2.021** |

Early exit (forest): depth 2 → 2.131, 3 → 2.122, 4 → 2.115, 5 → 2.108. It degrades smoothly, as intended.

**Parity** (512 val tokens): C engine vs PyTorch loss within ~0.002; argmax agreement 99.2–99.8%. Parallel vs recurrent is exact in fp64 (1.8e-15). Residual fp32 differences are single-position route or int8 rounding flips.

**Speed, end-to-end C, 1 thread** (whole model, per token):

| Model | µs/token | tok/s |
|---|---|---|
| dense ternary | 138 | 7.2k |
| FFF 2×8 | 55 | 18.2k (2.5×) |
| forest 16×5 | 70 | 14.2k (2.0×) |

At this size the FFN is already cheap, so retention and the LM head dominate (Amdahl).

**Speed, single FFN layer** (`microbench`, random ternary weights, naive kernels, 1 thread):

| d | nodes | depth | dense | tree path | speedup | weight bytes touched |
|---|---|---|---|---|---|---|
| 128 | 255 | 8 | 15.5 µs | 1.2 µs | 13× | 16 KB vs 0.5 KB |
| 768 | 4095 | 12 | 908 µs | 7.4 µs | 123× | 1.5 MB vs 4.5 KB |
| 1024 | 4095 | 12 | 1200 µs | 9.8 µs | 122× | 2 MB vs 6 KB |
| 4096 | 16383 | 14 | 40.7 ms | 45 µs | ~900× | 32 MB vs 28 KB |

The dense kernel here is naive single-thread, so real multithreaded/SIMD kernels would close some of that gap. The bytes-touched column is the part that doesn't go away.

## Honest read

1. **The speed side works.** The tree path is tiny and lives in L1, and the gap grows with width.
2. **The quality side is the open problem.** At this scale the tree recovers only about a quarter to a third of the dense FFN's gain (0.03–0.04 of 0.13 nats). Too few neurons fire per token.
3. **Amdahl shows up immediately.** Once the FFN is cheap, the retention state update (H·hd² per layer) and the head are the bill.

## Next steps (in order)

1. **Close the quality gap:** distill from the dense model (KL on logits); try leaf blocks (each leaf a small ternary FFN of width 8–32, as in the original FFF paper); try more trees at shallow depth; train longer, since trees seemed to converge more slowly.
2. **Packed SIMD kernels:** keep weights 2-bit in memory and add AVX2 `maddubs`-style kernels (I2_S / TL-style lookup) for the projections.
3. **Scale up:** take a BitNet b1.58 checkpoint (or a 1.58-bit BERT), keep its BitLinear attention/projections, swap FFNs for FFF, and distill. The format above is already set up for this.

## Files

- `model.py` — BitLinear, Retention (parallel + recurrent), FFF with STE, Chimera
- `train.py` — training (`--ffn fff|dense --trees --depth --matryoshka`)
- `export.py` — checkpoint → `.chim` (2-bit packed ternary)
- `chimera.c` — C inference: `gen`, `bench`, `dump` modes
- `parity.py` — C vs PyTorch parallel vs PyTorch recurrent
- `microbench.c` — single-layer dense vs tree
- `*.pt`, `*.chim`, `log_*.txt` — trained models and logs

```
gcc -O3 -march=native -o chimera chimera.c -lm
./chimera forest.chim gen 500 5 0.8 7 "ROMEO:"
./chimera forest.chim bench 3000 3      # early exit at depth 3
```
