# Tree attention: convert a trained BitNet attention layer into a boosted-tree ensemble

**Goal:** take a normally trained BitNet b1.58 softmax attention layer and make it run inference as a boosted tree, with minimal retraining. All BitNet weights stay frozen.

## Method

1. **Train a normal BitNet attention layer.** Ternary q/k/v/o BitLinear (absmean weights, int8 absmax activations), RoPE, causal softmax, and `attn_sub_norm` before o_proj. Pure attention, no MLP.
2. **Fit boosted trees with no training.** From about 25k calibration tokens, fit S oblique trees of depth D per head on the **rotated keys**, as **residual boosting**: tree s splits the residual left after trees 1..s-1, and its leaves store mean vectors. Splits use the top principal direction, with the threshold at the widest gap near the median (a max-margin, balanced split). Key trees are fit in the query metric (`(k−k̂)ᵀ Σ_q (k−k̂)`), and that transform is folded back into the split weights and leaves, so inference is unchanged. Values can be encoded the same way.
3. **Optional minimal retraining:** distill **only the leaf values** for 300 steps (KL to the frozen base model, no labels).

**Inference form:**

```
per key (write):  r = rope(k); for s: leaf_s = tree_s(r); r -= c_s[leaf_s]   ->  S bytes in the cache
per query:        T[s][l] = q · c_s[l] / √hd                                  (S·L small dots, once)
per key (read):   score = Σ_s T[s][leaf_s]      <- a boosted tree ensemble whose leaf values are set by the query
values:           P[s][l] += p_j for each value code;  o = Σ P[s][l]·c^v_s[l]
```

The training form, attention with k̂ = Σ_s c_s[leaf_s], is the same function: C matches PyTorch to ≤1e-5 (median 5e-7) with 100% argmax agreement. Three robustness fixes were needed to get there, all applied identically in fitting, PyTorch, and C: max-margin thresholds, residual snapping below 1e-5·|x|, and degenerate pass-through nodes.

## Results

**TinyShakespeare** (1 layer, d=128, 4 heads, hd=32). Base val loss **2.041**; with attention removed, **3.729**.

| Trees × depth | Bits/key | Score error | Keys only: zero-shot | Keys only: distilled | Keys + values: zero-shot | Keys + values: distilled | Cache bytes/key/head (K+V) |
|---|---|---|---|---|---|---|---|
| 4×6 | 24 | 34% | 2.427 | 2.311 | 2.644 | 2.333 | 8 |
| 8×6 | 48 | 19% | 2.168 | 2.142 | 2.401 | 2.200 | 16 |
| 16×6 | 96 | 8% | **2.060** | 2.067 | 2.281 | **2.136** | 32 |
| 16×8 | 128 | 4.5% | **2.046** | 2.053 | 2.275 | 2.138 | 32 |

For reference, an fp16 K+V cache is 128 bytes/key/head at hd=32 and 512 at hd=128.

**MQAR associative recall** (2-layer pure attention; the second layer is calibrated on the converted first layer's outputs). Base accuracy at 32/48/64 pairs: 1.000 / 0.9998 / 0.995. With attention removed: ~0.015.

| Config | Zero-shot @32/48/64 | Distilled @64 |
|---|---|---|
| keys only, 8×6 | 1.000 / 0.999 / 0.989 | 0.990 |
| keys only, 16×6 | 1.000 / 1.000 / 0.988 | 0.989 |
| keys + values, 8×6 (16 B/key) | 1.000 / 0.999 / **0.989** | 0.989 |
| keys + values, 16×6 | 1.000 / 1.000 / 0.988 | 0.989 |
| keys + values, 16×8 | 0.956 / 0.936 / 0.887 | 0.871 (value trees with 256 leaves overfit the calibration set) |

**Speed, one head, one query against N cached keys** (`attnbench`, fp32 SIMD baseline with -ffast-math, 1 thread):

| hd, trees | N=512 | 2k | 8k | 32k | Cache vs fp16 |
|---|---|---|---|---|---|
| hd=32, 16×6 | 0.21× | 0.35× | 0.58× | 0.51× | 4× smaller |
| hd=128, 16×6 | 0.31× | 1.04× | 1.87× | 2.43× | 16× smaller |
| hd=128, 8×6 | 0.65× | 1.82× | **3.15×** | **4.96×** | **32× smaller** |

At hd=32, a SIMD dot product of 32 floats is cheaper than 16 scattered lookups, so the tree loses. It wins at real head sizes (hd=128) once the context is long enough that per-key cost dominates the per-query table build.

## Read

1. **Keys convert almost for free.** At 96–128 bits per key, a training-free conversion keeps 98.7–99.7% of what attention contributes to loss, and recall stays at 98.8–99.0%. Every key is scored as a boosted tree ensemble, with no dot product per key.
2. **Values are harder at 1 layer.** In a single-layer char model, values depend only on the current character, which makes the fit brittle. Leaf-only distillation recovers most of the gap: 2.28 → 2.14 at 32 bytes/key, against a base of 2.04. On MQAR's 2-layer model, keys + values at 16 bytes/key loses only 0.6 points of recall.
3. **Distillation helps a lot at low bit budgets** (24–48 bits) and is neutral to slightly negative at high budgets with this learning rate. Keep the zero-shot fit when it's already good.
4. **Where it pays:** long context with large head dimension. The cache is 16–32× smaller and attention reads are 2–5× faster per head at 8k–32k keys.
5. **Caveats:** tiny models, single seed. Evaluation lengths beyond the 128 training context show RoPE extrapolation loss for both base and converted models.

## Next

- Convert a real BitNet b1.58 checkpoint (hd=128, multiple layers) layer by layer with the same calibrate-then-distill recipe, and measure perplexity against bits per key.
- Use AVX2 gathers or `pshufb` for the per-key lookups (with S ≤ 16 and L=16 the tables fit in registers), plus nibble-packed codes.
- Tune trees per head (heads differ a lot in how compressible they are) and add an outlier fallback that keeps a few keys in fp16.

## Files

`tree_attn.py` (BitNet attention + BoostedTrees, fit, convert) · `train_base.py` · `convert.py` (grid: zero-shot + leaf distillation) · `export_tree.py` → `.tre` · `treeattn.c` (C engine: standard KV cache or byte-code tree cache) · `parity_tree.py` · `attnbench.c` · `data.py` · `runs/` (bases, converted models, logs, JSON results)

```
gcc -O3 -march=native -o treeattn treeattn.c -lm
./treeattn runs/tree_shakespeare_16x6.tre gen 300 0.8 3 "ROMEO:"
./treeattn runs/tree_shakespeare_16x6.tre bench 8192
gcc -O3 -march=native -ffast-math -o attnbench attnbench.c -lm && ./attnbench 128 8 6
```

---

## Full-model trial on TinyShakespeare (4 layers, attention + ternary MLP)

A real BitNet-style transformer: d=128, 4 heads, 4 layers, MLP hidden 4d, all linears ternary b1.58, 2500 steps, ctx 128. Every attention layer was converted, layer by layer, with each layer's trees fitted on the outputs of the already-converted layers below it. All BitNet weights stayed frozen.

Base val loss **1.565**; with attention removed (MLPs kept), **3.569**.

| Trees × depth | Bits/key | Keys only: zero-shot | Keys only: distilled | Keys + values: zero-shot | Keys + values: distilled |
|---|---|---|---|---|---|
| 8×6 | 48 | 1.665 | 1.650 | 1.778 | 1.715 |
| 16×6 | 96 | **1.579** | 1.580 | 1.639 | **1.615** |
| 16×8 | 128 | **1.573** | 1.572 | – | – |

- Keys only at 96–128 bits, with no retraining: +0.008 to +0.014 nats over base, which keeps over 99% of what attention contributes.
- Keys + values at 32 bytes/key/head, after 300 steps of leaf-only distillation: +0.05 nats.
- C engine parity on the 4-layer converted models: loss within 0.002 of PyTorch, 99–100% argmax agreement. The base model shows the same small int8-rounding drift.
- Speed at this size (hd=32) is worse than the standard path. The per-query leaf tables (S·L·hd per head per layer) dominate at short context, and a 32-float SIMD dot is cheaper than 16 lookups. The per-key benchmark above shows where it turns: hd=128 with thousands of cached keys.
- Samples from all three models read as Shakespeare-like for the first ~128 characters and then degrade together. That's the base model's 128-token training context (RoPE extrapolation), not the conversion.

Files: `runs/full_shakespeare.{pt,tre}` (base), `runs/full_tree_k_16x6.*`, `runs/full_tree_kv_16x6.*`, `runs/full_k.json`, `runs/full_kv.json`.

---

## Tree-MLP: converting the BitNet MLP (frozen weights, closed-form fit)

**Requirement:** 5–20× the runtime speed of the normal MLP, with the least possible post-training adaptation.

**Form.** A GELU/ReLU MLP is nearly piecewise linear, so its natural tree form is a *model tree*:

```
leaf = tree(u)                      D oblique splits on the normed input (int8 split vectors, int8 activations)
out  = c[leaf] + Q[leaf] (P[leaf] u)     leaf constant + rank-r linear map (int8, per-row scales)
```

**Fit.** There is no gradient training and the BitNet weights are untouched. Using activations from the training text: each split is the input direction that best predicts the top principal component of the MLP output in that node, with a max-margin threshold near the median; leaf constants are means; leaf maps are ridge plus reduced-rank regression. Layers are fitted in order, each on the outputs of the already-converted layers below. Optional minimal adaptation: 300 steps training only the leaf tables against the frozen base (KL), about 3.5 minutes on 2 CPU cores.

### What didn't work
| Idea | Result at ~8–14× | Why |
|---|---|---|
| Tree picks k exact neurons per leaf | 1.81–1.92 | Activations aren't sparse. Even a perfect per-token top-64 of 512 gives 1.695, covering only 45% of the output. |
| Global low-rank linear + leaf constants | 2.05–2.10 | The MLP isn't globally linear. |
| Leaf constants only (1024–4096 leaves) | 1.95–2.01 at ~100× | Flat leaves are too crude, though remarkably cheap. |
| Higher rank (64) or 4096 leaves | worse | Too few sample tokens per leaf; it overfits. |

### Quality (4-layer BitNet transformer, TinyShakespeare; base **1.565**, MLP removed 3.277)
| Tree | Closed-form only | + 300-step leaf tuning | + int8 leaf maps and router |
|---|---|---|---|
| 1024 leaves, rank 16 | 1.656 | 1.627 | **1.627** |
| 256 leaves, rank 16 | 1.720 | 1.637 | **1.638** |

The MLPs are worth 1.71 nats in this model, so the converted versions keep about 96% of that. Calibration data matters a lot: the 1024-leaf closed-form fit went 1.677 → 1.656 going from 262k to 1M sample tokens.

### Runtime
In the full C engine, on real text (2048-token context, attention running between MLP calls), per MLP call:

| MLP | µs/call | Speedup |
|---|---|---|
| dense ternary (LUT kernel, exact erf GELU) | 12.2–13.4 | 1× |
| tree, 1024 leaves, rank 16, int8 | 2.30 | **5.3–5.8×** |
| tree, 256 leaves, rank 16, int8 | 1.74–2.10 | **5.8–7.7×** |

C matches PyTorch on both tree models (loss within 0.0002, 100% argmax agreement).

In isolation (`mlpbench`, random weights, runtime only; leaves visited at random = worst-case cache behavior):

| Width | Tree | vs dense with table GELU | vs dense with exact erf | Warm cache |
|---|---|---|---|---|
| d=128, hidden 512 | 256 leaves, r=16 | 6.1× | 9.6× | 10× |
| d=128, hidden 512 | 1024 leaves, r=16 | 2.8× | 5.0× | 9.5× |
| d=768, hidden 3072 | 1024 leaves, r=32 | 12× | 16× | 24× |
| d=2560, hidden 6912 (BitNet 2B size) | 1024 leaves, r=64 | 17× | 15× | 41× |

### The price: storage
The tree trades memory for speed. Int8 leaf tables are 1.0 MB/layer (256 leaves) or 4.2 MB/layer (1024 leaves) here, against 0.03 MB for the ternary MLP they replace. At BitNet 2B widths with rank 64 that would be about 335 MB per layer against 8.9 MB. Only one leaf (a few KB to ~330 KB) is touched per token, which is why it's fast, but total model size grows a lot. The rank and leaf count needed at real scale are untested; the large-width rows above measure speed only, not quality.

Files: `tree_mlp.py`, `convert_mlp.py`, `quant_eval.py`, `oracle.py`, `mlpbench.c`, `runs/mlp_grid*.log`, `runs/full_mlptree_8_0_16_0_q8.tre`.

---

## All-tree stack: tree attention + tree MLP on the 4-layer BitNet model

Every attention layer's keys are scored by boosted trees and every MLP is a model tree. BitNet weights stay frozen. Base val loss **1.565**.

### How to combine them (this mattered)
| Procedure | Val loss |
|---|---|
| Joint layer-by-layer fit (each tree block fitted on the outputs of the tree blocks below), closed-form | 1.866 |
| … with tree-MLPs fitted to the base model's residual stream instead | 1.855 |
| … + 300 steps joint leaf-only tuning | 1.789–1.791 |
| **Convert each half separately against the base model, then plug together** | **1.667** |
| … + 300 steps joint leaf-only tuning | 1.669 (no gain) |

Separately converted halves compose almost additively (attention +0.015, MLP +0.072, together +0.102). Fitting them jointly was much worse: tree-MLPs fitted on tree-attention outputs generalize worse, and leaf tuning couldn't recover it.

### Pairings (plugged, int8 MLP leaves)
| Attention | MLP | Val loss | Notes |
|---|---|---|---|
| keys 16×6 (96 bits/key) | 256 leaves, rank 16 | **1.669** | |
| keys 16×6 | 1024 leaves, rank 16 | **1.654** | best all-tree |
| keys + values 16×6 (32 B/key) | 1024 leaves, rank 16 | 1.757 | value trees compound badly with tree-MLPs |
| keys + values 16×6 | 256 leaves, rank 16 | 1.766 | |

C engine parity on the all-tree models: identical loss to 4 decimals, max |Δlogit| ≤ 3e-3, 100% argmax agreement.

### Speed (C, 1 thread, per token)
| Model | ctx ≤128 | ≤512 | ≤1024 | ≤2048 | MLP per call |
|---|---|---|---|---|---|
| base (dense attention + dense MLP) | 103 µs | 230 | 426 | 805 | 14.7 µs |
| tree MLP only (256 leaves) | **57 µs** | **173** | **373** | **778** | 1.74 µs |
| all-tree: keys + tree MLP | 476 | 592 | 711 | 1095 | 2.0 µs |
| all-tree: keys + values + tree MLP (1024) | 816 | 910 | 1290 | 1436 | 3.1 µs |

At this model's 32-dim heads, tree attention is slower than a SIMD dot (the per-query leaf tables dominate), so the all-tree stack is slower end-to-end even though its MLP is 7× faster. **The fastest build here is tree MLP + standard attention** (1.8× per token at short context, where the MLP is a large share of the work). Tree attention pays off only at larger heads and long contexts (see the hd=128 per-key benchmark above).

Files: `convert_both.py` (joint fit, kept for the record), `plug.py` (separate-then-combine), `runs/alltree_plug_*.{log,json}`, `runs/alltree_plug_k_m8.tre`.

---

## Fast tree attention (fast-scan): a genuinely CPU-friendly version

The first tree attention was slower than standard attention. This rewrite fixes that; it now wins once the context passes ~1–2k keys, on this model's small 32-dim heads.

### What changed
| Piece | Before | Now |
|---|---|---|
| Tree shape | 16 trees × 64 leaves | 24–32 trees × **16 leaves** (4 bits each, 12–16 bytes per key) |
| Per-key scoring | 16 scalar table lookups | **`pshufb` over nibble codes: 32 keys per instruction**, two trees per code byte, 16-bit table entries as two byte planes (near-exact) |
| Per-query table | 16×64×32 scalar multiply-adds | 24×16×32 from int8 leaf values, 4 trees at a time in registers |
| Key encoding | 96 chained scalar dots | all 15 nodes of a tree in one small int8 matvec, or **axis-aligned splits: 4 compares, no dots** |
| Tree tables | fp32, ~150 KB/head | int8 with per-row scales, ~25 KB/head (no quality change) |
| Weights | scalar `expf` per key | AVX2 exp, 8 keys at a time; keys more than 16 nats below the best are skipped |
| Values | axpy into memory per key | accumulated in registers |
| Caches | fp32 K/V | fp16 K/V |

The standard path got the same shared upgrades (fp16 caches, vector exp, register value sum, cached RoPE), so the comparison is fair.

### Quality (4-layer BitNet transformer, keys only, no retraining; base 1.5646)
| Trees | Bytes/key | Val loss |
|---|---|---|
| oblique 24×4 | 12 | 1.5795 (int8: 1.5788) |
| oblique 32×4 | 16 | 1.5704 (int8: 1.5704) |
| axis 24×4 | 12 | 1.6007 |
| axis 32×4 | 16 | 1.5848 |
| axis 40×4 | 20 | 1.5773 |
| axis 48×4 | 24 | 1.5748 |

Axis-aligned trees need about 40 trees to match 24 oblique ones. Through the C engine, the fast path and the float reference path give the same validation loss within 0.003 (e.g. 1.5190 vs 1.5170), and the reference path matches PyTorch.

### Speed
Scoring stage alone (`scanbench`, one head, one query vs N keys, vs an AVX2+FMA float dot):

| Head dim, trees | N=512 | 2k | 8k | 32k | Key cache vs fp16 |
|---|---|---|---|---|---|
| 32, 24×4 | 1.1× | 2.4× | 3.5× | 5.7× | 5× smaller |
| 128, 32×4 | 1.0× | 3.0× | 10× | 15× | 16× smaller |

Whole model, per token (C, 1 thread):

| Model | ctx ≤128 | ≤512 | ≤1024 | ≤2048 | ≤4096 | ≤8192 |
|---|---|---|---|---|---|---|
| base (standard attention, dense MLP) | 67 µs | 96 | 152 | 243 | 432 | 828 |
| tree attention (oblique 24×4) | 116 | 132 | ~170 | 238 | 366 | 573 |
| tree attention (axis 24×4) | 100 | 117 | 153 | 217 | 326 | 566 |
| tree MLP + standard attention | **30** | **53** | **104** | 210 | 395 | 771 |
| all-tree (oblique 24×4 + tree MLP), loss 1.668 | 103 | 98 | 124 | **184** | **315** | **531** |

Time breakdown per head at ~1k keys of real text: standard = scoring 46%, values 45%. Tree = scoring 11%, key encode 27%, table 13%, values 41%.

### Read
1. **Scoring is no longer the bottleneck.** It went from 3.6 µs to 0.9 µs per head at 1k keys. What remains is a fixed cost per token (encode + table, ~2.5 µs per head) and the value read, which trees on keys don't touch.
2. **Crossover is ~1–2k keys at 32-dim heads.** Below that the fixed cost loses; above it tree attention pulls ahead (1.45× whole-model at 8k). At 128-dim heads the scoring gap is far larger (10–15×), but that's measured in isolation, not on a trained model.
3. **Values are the next wall.** Nearly every key's value gets read (92–97%), because this model's attention isn't peaked enough for score pruning to skip much. Speeding that up needs compressed values, which hurt quality earlier.
4. **Best builds:** short context → tree MLP + standard attention (2.2× at ≤128). Long context → all-tree (1.56× at 8k).

Files: `treeattn.c` (fast-scan path; `EXACT=1` selects the float reference path, `TAU=` sets the pruning margin, `ATTSTAT=1` prints the breakdown), `scanbench.c`, `quant_trees.py`, `evalc.py`, `fastcheck.py`, `tree_attn.py` (`axis=True` trees).
