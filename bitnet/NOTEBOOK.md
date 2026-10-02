# Tree-BitNet at scale: microsoft/bitnet-b1.58-2B-4T (lab notebook)

## Summary (2026-10-02)
| piece | verdict | quality (WikiText-2, base 13.92) | speed |
|---|---|---|---|
| tree MLP (leaf tables, shared subspace) | **fails** — the MLP is high-rank at d=2560 | 68 MB/layer recovers 1/3 of the KL of deleting the MLP | – |
| key trees *replacing* keys | **fails at 30 layers** | 32 B/key: ppl 16.09 (+16%) | – |
| key trees *selecting* keys + exact rescoring | works | at floor, reads 56% of KV (tau 8); KL 0.0046 at 27% (tau 6); 0.017 at 10% (tau 4) | not in engine yet (long-context win) |
| sparse *exact* MLP (exact gate, per-token energy 0.99) | works, lossless | KL at floor, 1916 / 6912 neurons | 1.19× / 1.10× decode (1 / 4 threads) |
| tree output layer (128 vocab trees, 8192 exact) | works | +1.0% ppl, 99.96% top-1 | 1.51× / 1.21× decode |
| **head + sparse MLP in bitnet.cpp** | | **+1.0% ppl (engine)** | **1.97× / 1.68× / 1.44× decode at 1 / 2 / 4 threads** |
| all three (PyTorch) | | 14.06 (+1.0%), KL 0.017 | |

Also found: stock bitnet.cpp HEAD runs 2B-4T with SiLU instead of relu² (PPL 91 → 13.1 fixed).

Code: `bitnet/` (harness `common.py`, key trees `keytrees.py`, tree MLP `treemlp.py`, phases `phase*.py`).
Compute: RunPod RTX 3090 (community, $0.22/h), one-shot jobs via `pod_boot.sh` + `jobs/*.sh`, results in `runs/`.

## Model facts (from the real modeling code, transformers 5.18)
- FFN is **gated**: `down(ffn_sub_norm(relu(gate u)^2 * up u))` — 3 projections, 53M MACs/layer (not 35M).
  Because of `ffn_sub_norm` the MLP output depends only on the *direction* of its input.
- Attention: q/k/v → RoPE → softmax → `attn_sub_norm` → o_proj. GQA 20 q / 5 kv heads, head dim 128.
- BitLinear (online quant): weight absmean ternary per tensor, activation per-token absmax int8, compute in bf16.
- Hooks: RoPE is a module-global function called at runtime, so wrapping `modeling_bitnet.apply_rotary_pos_emb`
  plus a pre-hook recording the layer gives post-RoPE q/k without reimplementing attention. Identity key
  replacement gives KL exactly 0 (checked every run).

## Phase 1 — baseline (gate: passed)
Windows of 2048 tokens, BOS prepended to each window, loss over the 2047 predicted tokens.

| data | tokens | ppl |
|---|---|---|
| WikiText-2 test (all) | 288,627 | **13.92** |
| UltraChat-200k test_sft (96 windows) | 196,512 | **4.48** |

Greedy chat generation is coherent. No published WikiText-2 number for this checkpoint was found
(tech report gives benchmark accuracies, not ppl); 13.9 is in the expected range for a 2B / 4T-token model.

**Attention peakedness** (queries with ≥1024 keys, 4 wiki windows): fraction of keys within TAU nats of the max
score, and the attention mass they hold:

| layer | sink mass | TAU=4 read / mass | TAU=8 | TAU=12 | TAU=16 |
|---|---|---|---|---|---|
| 0 | 0.165 | 5.9% / 0.853 | 35% / 0.991 | 65% / 0.9998 | 76% / 1.0 |
| 7 | 0.045 | 2.8% / 0.800 | 29% / 0.980 | 78% / 0.9997 | 97% |
| 15 | 0.029 | 3.3% / 0.830 | 32% / 0.982 | 83% / 0.9998 | 99% |
| 22 | 0.038 | 1.7% / 0.819 | 25% / 0.973 | 83% / 0.9997 | 99% |
| 29 | 0.033 | 11% / 0.682 | 71% / 0.984 | 97% / 0.9999 | 100% |

→ Default TAU=16 skips almost nothing (as on the tiny model). TAU≈10–12 would skip ~20%, TAU=8 ~70% of
value reads at ~2% mass loss; needs a quality measurement.

## Phase 2 — keys, one layer
First run invalid: **every S from 16 to 96 gave identical results**. Cause: the "widest gap in the 45–55% band
must exceed tol" test. With 262k calibration keys the widest gap is tiny, so after ~16 trees every node was
marked degenerate (the handoff's "degenerate node" trap, in a second place). Fixed: degeneracy is decided by
the node's spread only; the gap only has to exceed float resolution. Runs now log `degenerate_nodes`.

Still valid from that run: keeping the BOS key exact (`exact_sink`) helps a lot — layer 15 KL 0.050→0.032,
layer 28 0.020→0.014, layer 2 0.323→0.289. Early layers are far more sensitive (layer 2 KL 0.32 at the same
setting vs 0.05 at layer 15).

**Fixed run** (`runs/phase2.jsonl`; calib 128 windows × 2048 = 262k keys per KV head; eval 64 wiki + 24 chat
windows; base on these slices: wiki 13.578, chat 4.305; 0 degenerate nodes everywhere):

| layer | trees | B/key/kv-head | sink exact | score rel err | wiki ppl | wiki KL | chat ppl | chat KL |
|---|---|---|---|---|---|---|---|---|
| 15 | 16 | 8 | yes | 0.0896 | 13.640 | 0.0155 | 4.340 | 0.0136 |
| 15 | 32 | 16 | yes | 0.0539 | 13.624 | 0.0087 | 4.323 | 0.0077 |
| 15 | 48 | 24 | yes | 0.0353 | 13.603 | 0.0056 | 4.319 | 0.0049 |
| 15 | 64 | 32 | yes | 0.0234 | 13.598 | 0.0040 | 4.313 | 0.0035 |
| 15 | 96 | 48 | yes | 0.0107 | 13.585 | 0.0026 | 4.309 | 0.0022 |
| 15 | 64 | 32 | no | 0.0234 | 13.598 | 0.0041 | 4.310 | 0.0034 |
| 2 | 16 | 8 | yes | 0.0510 | 14.221 | 0.0699 | 4.278 | 0.0435 |
| 2 | 32 | 16 | yes | 0.0296 | 13.792 | 0.0310 | 4.292 | 0.0201 |
| 2 | 48 | 24 | yes | 0.0190 | 13.684 | 0.0187 | 4.304 | 0.0124 |
| 2 | 64 | 32 | yes | 0.0126 | 13.620 | 0.0107 | 4.301 | 0.0073 |
| 2 | 96 | 48 | yes | 0.0057 | 13.586 | 0.0049 | 4.298 | 0.0033 |

Gate 2: **passed** (monotone in bits, small at ≥32 B/key). Observations:
- fp16 keys are 256 B/key/kv-head; 32 B is 8× smaller, 48 B is 5.3×.
- Layer 2 has *lower* score error than layer 15 but ~2.5× the KL: early layers are more sensitive. Budget
  bits per layer (more trees early) rather than uniformly.
- With the split fix, keeping the sink exact no longer matters at layer 15.
- Chat ppl sometimes dips *below* base with KL > 0 (e.g. 4.278 vs 4.305): ppl on 49k tokens is noisy;
  KL to the base model is the metric to trust.
- If per-layer KLs add up (they did almost linearly on the tiny model), all 30 layers at 32 B/key would cost
  KL ≈ 0.15–0.2. Plain replacement looks expensive at scale → test **select + rescore** (tree codes only
  choose candidate keys; exact keys score them; only their values are read).

## Phase 2b — attention: TAU value pruning, select + rescore (`runs/phase2b.jsonl`)
Custom attention registered with transformers' AttentionInterface (`common.tree_attention`, fp32). **Floor**:
this exact fp32 path vs stock bf16 sdpa already differs by KL 0.0026 (all layers) / 0.0014 (one layer) — the
model's own numerical noise level. Compare against the floor, not 0. Eval: 32 wiki + 16 chat windows.

**Value pruning on exact scores, all 30 layers** (drop keys whose exact score is < max − tau_v; queries with
≥1024 keys):

| tau_v | 4 | 6 | 8 | 10 | 12 | none (floor) |
|---|---|---|---|---|---|---|
| keys/values read | 4.1% | 14% | 35% | 63% | 83% | 100% |
| wiki KL | 0.0274 | 0.0060 | 0.0029 | 0.0026 | 0.0026 | 0.0026 |
| chat KL | 0.0174 | 0.0042 | 0.0022 | 0.0019 | 0.0019 | 0.0018 |

→ Open question 3: **yes**. TAU=8 skips ~65% of value reads for +0.0003 KL; layer 29 is the least peaked.

**Select + rescore, one layer** (trees choose keys within tau_sel of the max tree score + sink + 64 most
recent; exact keys score them; only their values are read):

| layer 15 | wiki KL | read |
|---|---|---|
| floor (same fp32 path) | 0.00144 | 100% |
| replace, 16 trees (8 B/key) | 0.01540 | 100% |
| select, 16 trees, tau_sel=4 | **0.00181** | **9.4%** |
| select, 16 trees, tau_sel=8 | 0.00145 | 63% |
| replace, 32 trees (16 B/key) | 0.00864 | 100% |
| select, 32 trees, tau_sel=4 | **0.00180** | **7.9%** |

→ Using the trees to *choose* keys rather than *replace* them is ~40× better in excess KL at equal bits and
reads <10% of keys and values. Exact keys must still be stored (int8 would do) but are only read for the
selected ~10%. The recent window matters: pruning on exact scores at tau 4 without it costs KL 0.027.

## Phase 4 — tree MLP, one layer (layer 15), shared-subspace form (`runs/phase4.jsonl`) — **gate FAILED**
`out = c[leaf] + Q_g M[leaf] P_g (u − μ)`, router in z-space, per-leaf ridge shrunk to the global map;
calibration 1M tokens (512 windows), shared fit on 98k tokens. Eval 32 wiki + 16 chat windows
(base wiki 14.021, chat 4.272). Config = input subspace : r_in : r_out : depth.

| config | leaves | MB/layer int8 | out rel err | wiki ppl | wiki KL | chat KL |
|---|---|---|---|---|---|---|
| MLP removed (mean output) | – | – | 1.000 | 15.330 | 0.0585 | 0.0457 |
| jac:64:64:0 (global linear) | 1 | 0.3 | 0.884 | 14.582 | 0.0472 | 0.0484 |
| jac:64:0:10 (leaf constants) | 1024 | 2.7 | 0.962 | 15.096 | 0.0529 | 0.0453 |
| jac:64:64:8 | 256 | 2.0 | 0.856 | 14.454 | 0.0449 | 0.0476 |
| jac:64:64:10 | 1024 | 6.9 | 0.854 | 14.422 | 0.0445 | 0.0469 |
| pca:64:64:10 | 1024 | 6.9 | 0.849 | 14.300 | 0.0449 | 0.0461 |
| jac:128:128:10 | 1024 | 19.3 | 0.817 | 14.303 | 0.0417 | 0.0444 |
| jac:256:64:10 | 1024 | 19.5 | 0.831 | 14.288 | 0.0429 | 0.0460 |
| jac:256:256:10 | 1024 | 68.0 | 0.778 | 14.299 | 0.0384 | 0.0416 |

(ternary MLP = 3·6912·2560·1.58 bit ≈ 10.5 MB/layer; int8 variants identical to fp32 within noise.)

Why: the MLP at d=2560 is **high-rank**. Top-256 directions hold only 50% of the input variance, 50% of the
output variance and 37% of E[JᵀJ] (Jacobian energy); top-64: 28% / 26% / 18%. No low-rank or piecewise
low-rank form at an acceptable size gets near it — even 68 MB/layer recovers only 22% of the variance and a
third of the KL. Leaf constants alone explain 4%. The tiny model (d=128) hid this: its whole space was rank 128.

**But the relu² intermediate is very sparse in energy** (per-token oracle, layer 15, 8k tokens):

| exact neurons kept per token (of 6912) | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| share of ‖h‖² | 0.793 | 0.950 | 0.987 | 0.999 |
| MLP output rel err | 0.199 | 0.048 | 0.012 | 0.0006 |

41% of gate pre-activations are ≤ 0 (exactly zero neurons). → Pivot: keep the frozen ternary weights and use
a cheap selector (tree / low-rank gate predictor) to choose which ~15% of neurons to compute exactly
(phase 4b). This keeps the owner's constraint even more strictly (no fitted output tables at all).

## Phase 4b — sparse *exact* MLP: which neurons to compute (`runs/phase4b*.jsonl`)
Frozen ternary weights; per token only k of 6912 intermediate neurons are computed (gate/up rows, down
columns); the mask is applied before `ffn_sub_norm`. Layer 15, eval 32 wiki + 16 chat windows. Same code path
with all neurons: KL **0.000** (bit-exact). wiki KL:

| selector | MLP MACs @k=1024 | k=512 | k=1024 | k=1536 | k=2048 |
|---|---|---|---|---|---|
| oracle top-|h| (upper bound) | 7.9M | 0.0034 | 0.0019 | 0.0016 | 0.0015 |
| exact gate → top relu(g) | 22.9M | 0.0059 | 0.0030 | 0.0020 | 0.0016 |
| exact gate → top relu(g)²·w (w = E|up|·‖down col‖) | 22.9M | 0.0059 | 0.0029 | 0.0020 | 0.0016 |
| low-rank 128 gate predictor (corr 0.85) | 9.1M | 0.0139 | 0.0083 | 0.0056 | 0.0042 |
| low-rank 256 gate predictor (corr 0.89) | 10.3M | 0.0104 | 0.0058 | 0.0039 | 0.0029 |
| tree leaf sets, depth 8 | 7.9M | 0.0296 | 0.0222 | 0.0169 | 0.0133 |
| tree D8 (k/2) ∪ low-rank 256 (k/2) | 10.3M | 0.0152 | 0.0094 | 0.0066 | – |
| dense MLP | 53.1M | | | | |

- **Noise floor of the model itself**: the oracle at k=2048 has output rel err 0.0009 yet KL 0.0015. Any
  non-bit-exact perturbation of one layer costs ~0.0015 KL because BitNet's per-token int8 activation rounding
  amplifies it (the fp32-vs-bf16 attention floor is the same size). Differences below ~0.001 are not signal.
- Tree leaf sets fail: the active neuron set varies too much per token for a leaf to hold it (rel err 0.32 at
  k=2048). Low-rank predictors are mediocre (the gate is high-rank, like everything else in this MLP).
- **Exact gate + top-k** is the working selector: k=1536 → KL 0.0020 (≈ floor + 0.0005) with 2.1× fewer MLP MACs;
  k=1024 → 0.0030 at 2.3×. Skipping only the neurons with gate ≤ 0 is lossless (41% of up/down work).

## Engine baseline — stock bitnet.cpp on this machine
**Bug found: stock bitnet.cpp HEAD (0b341e5, llama.cpp submodule isHuangXin/release-bitnet-embedding-0.6b-270m)
runs BitNet-b1.58-2B-4T with SiLU instead of relu² in the FFN** (`src/models/bitnet.cpp` passes `LLM_FFN_SILU`
for every bitnet arch). WikiText-2 test, `llama-perplexity -c 2048 --chunks 6`: **PPL 91.2 stock → 13.14 with
the one-line fix** (`engine/bitnet_relu2_fix.patch`). All speed numbers below use the fixed build (the activation
costs the same either way: 6.72 → 6.50 tok/s at 1 thread is noise).
Intel Xeon (Sapphire Rapids class, AVX-512 VNNI), 4 vCPU. Official `BitNet-b1.58-2B-4T` I2_S gguf.
`llama-bench -p 0 -n 64` (fixed build): **6.50 tok/s (1 thread), 21.4 tok/s (4 threads)**; with context depth
(4 threads): 16.2 tok/s @ 2048, **7.9 tok/s @ 8192** → at 8k, attention is ~80 of 126 ms/token (~63%).

Bytes per decoded token: ternary layers ≈ 0.52 GB (MLP ≈ 0.40 GB), **tied output layer 0.66 GB (f16,
128256 × 2560)**. Timed in ggml (`engine/headbench.c`): output layer f16 = **60.6 ms of 149 ms/token on 1 thread
(41%), 16.7 of 46 ms on 4 threads (36%)**; q8_0 49.3 / 13.7 ms; q4_0 52.6 / 8.9 ms.

→ Amdahl at short context: output layer ~40%, MLPs ~45%, attention projections ~14%, attention scores small.
The output layer is the same maximum-inner-product problem as key scoring, so it is a natural tree target:
cluster the vocabulary, score centroids, compute exact logits only for candidate clusters (`phase_vocab.py`).

## Tree output layer — select + rescore over the vocabulary (`phase_vocab.py`, `runs/phase_vocab_*.jsonl`)
Same machinery as the key trees, with the 128256 token embeddings as the "keys" and the final hidden states as
the queries (metric = hidden-state second moments). Every token gets S 4-bit tree codes (S/2 bytes); per decoded
token the trees score all 128k tokens by table lookup, the top-N are rescored exactly, the rest keep their tree
score (so the softmax is defined everywhere and ppl/KL are measurable). 32k wiki + 32k chat positions; exact head
ppl on these positions: wiki 14.126, chat 4.271.

First try, k-means clusters (C = 512–2048, top-m clusters exact): **failed** — k-means on these embeddings is
degenerate (median cluster 4–5 tokens, largest 17–28k), best top-1 agreement 0.85 / 0.92.

| trees (B/token) | N exact | top-1 agree wiki / chat | mass covered | KL wiki / chat | ppl wiki / chat | MACs |
|---|---|---|---|---|---|---|
| 16 (8) | 1024 | 0.877 / 0.805 | 0.84 / 0.79 | 1.57 / 3.16 | 70.8 / 97.5 | 3.3M |
| 32 (16) | 1024 | 0.943 / 0.907 | 0.90 / 0.89 | 0.78 / 1.52 | 32.2 / 20.1 | 3.9M |
| 64 (32) | 1024 | 0.978 / 0.967 | 0.94 / 0.95 | 0.34 / 0.56 | 20.5 / 7.6 | 5.2M |
| 64 (32) | 4096 | 0.9958 / 0.9940 | 0.983 / 0.990 | 0.092 / 0.119 | 15.51 / 4.85 | 13.1M |
| 64 (32) | 8192 | 0.9987 / 0.9978 | 0.992 / 0.995 | 0.035 / 0.049 | 14.58 / 4.48 | 23.6M |
| 128 (64) | 2048 | 0.9979 / 0.9970 | 0.981 / 0.991 | 0.060 / 0.066 | 15.04 / 4.62 | 10.5M |
| 128 (64) | 4096 | 0.9992 / 0.9990 | 0.989 / 0.996 | 0.028 / 0.028 | 14.49 / 4.41 | 15.7M |
| **128 (64)** | **8192** | **0.9997 / 0.9995** | 0.994 / 0.998 | **0.012 / 0.014** | **14.25 / 4.34** | 26.2M |
| dense | – | 1 | 1 | 0 | 14.126 / 4.271 | 328.3M |

- With 128 trees (8 MB of codes for the whole vocabulary vs 656 MB f16) and 8192 exact candidates, greedy decoding
  changes about 1 token in 3000; the remaining KL comes from the approximate tail, not from the top tokens.
- CPU speed (`engine/vocabbench.c`, 1 thread, random data): tables 1.1 ms + fast-scan 1.2 ms + top-N 0.3 ms +
  exact f16 rescoring 4.5 ms = **7.1 ms vs 60.6 ms for ggml's f16 head (8.5×)**; N=4096: 4.8 ms (12.8×).

## Engine microbenchmarks (this machine, 1 thread)
- Sparse exact MLP (`engine/sparsemlp.c`, bitnet.cpp's I2_S packing, same AVX2 kernel style for both paths,
  software prefetch for gathered rows; noisy VM, ranges over runs): dense MLP 0.95–1.2 ms; exact gate alone
  0.30–0.39 ms; sparse k=1024 **1.8–2.2× faster**, k=1536 **1.6–1.75×**. Gathered rows cost more per row than
  streamed ones; the dense gate is ~55% of the sparse path, so gate-based selection caps the MLP at ~3×.

## Phase 3 — keys, all 30 layers (sequential calibration; full WikiText-2 test + 64 chat windows; base wiki 13.92)
Floor for the custom fp32 attention path on all layers: KL ≈ 0.0026.

| mode | trees (B/key/kv-head) | wiki ppl | wiki KL | chat KL | keys+values read |
|---|---|---|---|---|---|
| replace (k̂ scores) | 64 (32) | **16.09** | **0.174** | 0.180 | 100% |
| select+rescore, tau_sel=8 | 32 (16) | 13.90 | **0.0028** | 0.0021 | **56%** |
| select+rescore, tau_sel=4 | 32 (16) | 13.81 | 0.0173 | 0.0089 | 9.9% |
| select+rescore, tau_sel=4 | 16 (8) | 13.98 | 0.0386 | 0.0167 | 12.4% |

- **Replacing keys with tree reconstructions fails at scale** (+16% ppl even at 32 B/key): per-layer errors add up.
- **Select + rescore at tau_sel=8 is at the noise floor** while reading 56% of the KV cache; tau_sel=4 reads ~10%
  for KL 0.017. Per-layer tau (looser early layers) is the obvious next refinement.

## Phase 5 — sparse exact MLP, all 30 layers (`runs/pod_yg2e_phase5.jsonl`; MLP-path floor = 0)
| selector | mean neurons / 6912 | MLP MACs | wiki ppl | wiki KL | chat KL |
|---|---|---|---|---|---|
| per-token energy coverage 0.995 of relu(g)² | 2085 | 28.4M | 13.92 | 0.0026 | 0.0020 |
| energy 0.99 | 1916 | 27.5M | 13.93 | 0.0028 | 0.0021 |
| energy 0.98 | 1709 | 26.4M | 13.94 | 0.0035 | 0.0029 |
| fixed top-2048 by relu(g) | 2048 | 28.2M | 13.73 | 0.0186 | 0.0149 |
| fixed top-1536 | 1536 | 25.6M | 13.72 | 0.0369 | 0.0288 |
| dense | 6912 | 53.1M | 13.92 | 0 | 0 |

**Adaptive per-token k is far better than fixed k** (energy 0.98 with 1709 neurons on average: KL 0.0035 vs fixed
2048: 0.0186) — some tokens need many neurons, most need few. Energy 0.99 is at the noise floor.

## Phase 6 — all pieces together (`runs/phase6.jsonl`; attention: select+rescore 32 trees tau_sel=8;
MLP: energy 0.99; head: 128 vocab trees, N=8192; each converted against the base model, then plugged)
| combo | wiki ppl | wiki KL | wiki top-1 | chat KL | notes |
|---|---|---|---|---|---|
| attention | 13.90 | 0.0028 | 0.971 | 0.0021 | 56% of KV read |
| MLP | 13.93 | 0.0028 | 0.971 | 0.0021 | 1916 neurons |
| head | 14.08 | 0.0142 | **0.9996** | 0.0157 | |
| attention + MLP | 13.92 | 0.0031 | 0.970 | 0.0023 | errors do not stack beyond the floor |
| **attention + MLP + head** | **14.06 (+1.0%)** | 0.0171 | 0.969 | 0.0198 | rerun (`runs/phase6b.jsonl`); KL ≈ head + floor |
| attention tau_sel=6 | 13.85 | 0.0046 | 0.963 | 0.0029 | **27% of KV read** |
| attention tau 6 + MLP 0.98 + head | 14.02 (+0.7%) | 0.0193 | 0.958 | 0.0184 | 27% KV, 1711 neurons |

## Phase 7 — inside bitnet.cpp (end to end, this machine)
`engine/tree-bitnet.{h,cpp}` + `engine/bitnet_tree.patch` (llama.cpp submodule of bitnet.cpp 0b341e5; the patch also
contains the relu² fix). Decode-path ggml custom ops, enabled by env vars:
- sparse exact MLP: stock dense gate matmul → `op_hsel` (per-token selection by relu(g)² energy, up rows of the
  selected neurons, relu²·up) → stock RMSNorm·w → `op_down` (transposed down rows, built lazily at first use).
- tree output layer: `op_tables` (S·16 leaf scores) → `op_scan` (pshufb fast-scan of 128k tokens, 16-bit tables
  as two byte planes) → `op_head` (top-N by histogram, exact f16 rows for candidates, tree score elsewhere).

Correctness: with all active neurons the layer-0 MLP output equals stock to 1e-7 relative (`llama-eval-callback`);
ppl differences of ~0.2% between equivalent paths are the model's own rounding chaos (stock itself gives 10.630
batched vs 10.611 token-by-token on chunk 1).

`llama-perplexity -c 2048 --chunks 3` (WikiText-2 test, TREE_ALL=1): stock **13.717** | tree head N=8192 **13.859
(+1.0%)** | N=4096 14.127 (+3.0%) — same as the PyTorch numbers (+0.9% / +2.6%).

`llama-bench -p 0 -n 64` decode tok/s (short context):

| threads | stock | tree head N=8192 | tree head N=4096 | sparse MLP 0.99 only | head N=8192 + sparse MLP 0.99 |
|---|---|---|---|---|---|
| 1 | 6.60 | **9.79 (1.48×)** | 10.44 (1.58×) | 7.35 (1.11×) | **11.81 (1.79×)** |
| 4 | 22.83 | 27.64 (1.21×) | 27.95 (1.22×) | 21.85 (0.96×) | 27.07 (1.19×) |

Mean selected MLP neurons on llama-bench's tokens: 2070 (frac 0.99), 1850 (0.98), 1497 (0.95); 2888 have g>0.
- The tree head is the big end-to-end win (it removes ~40% of single-thread decode work).
- The sparse MLP helps single-threaded but not at 4 threads: decode there is bandwidth-bound and gathered rows
  (scattered 640 B reads, 32 B slices per thread in `op_down`) use bandwidth badly. Fixes to try: neuron-major
  partition with a reduction op, row reordering by co-activation so selected rows are contiguous.
- Attention select+rescore is not in the engine yet (it pays off at long context: stock drops to 7.9 tok/s at 8k).

**Update — neuron-partitioned down op** (each thread accumulates full transposed rows for its share of the selected
neurons into a private slot; a second op sums 16 slots). Sparse MLP alone, frac 0.99: 1 thread 6.54 → 7.77 (1.19×),
4 threads 22.45 → 24.75 (1.10×, was 0.96×). Combined (`llama-bench -p 0 -n 64 -r 3`):

| threads | stock | head N=8192 | **head 8192 + MLP 0.99** | head 4096 + MLP 0.98 |
|---|---|---|---|---|
| 1 | 6.53 | 9.83 (1.51×) | **12.87 (1.97×)** | 14.20 (2.17×) |
| 2 | 12.35 | 17.03 (1.38×) | **20.80 (1.68×)** | 23.19 (1.88×) |
| 4 | 21.80 | 26.38 (1.21×) | **31.46 (1.44×)** | 35.03 (1.61×) |

Engine ppl (3 chunks, stock 13.717): head 8192 + MLP 0.99 = **13.856 (+1.0%)** (same as head alone — the sparse MLP
costs nothing measurable); head 4096 + MLP 0.98 = 14.209 (+3.6%).

## Salvaging the tree MLP: trees that *route*, not trees that *approximate* (`phase_treegate.py`)
Lesson from keys and vocab: trees fail when they replace a high-rank function, and work when they select items
that are then computed exactly. The MLP has the same structure: relu² leaves ~1900 of 6912 neurons doing the work
per token, and finding them is a max-inner-product search over the gate rows (g_i = w_i·x). So: encode the 6912
ternary gate rows with boosted trees (input metric), score all neurons by table lookup, take the top C,
compute gate/up/down exactly for those (energy rule 0.99 among candidates). MACs are per token per layer
(dense MLP 53.1M; exact-gate sparse MLP = F·d + 2k·d).

**Tree scorer, per layer** (eval 32 wiki + 16 chat windows; `runs/phase_treegate.jsonl`):

| layer | scorer | C | energy captured | MLP MACs | wiki KL |
|---|---|---|---|---|---|
| 15 | exact gate | all | 1.000 | 32.9M | 0.0014 |
| 15 | 64 trees (gate corr 0.87) | 3072 | 0.935 | 22.0M | 0.0021 |
| 15 | 32 trees | 3072 | 0.911 | 20.1M | 0.0031 |
| 2 | exact gate | all | 1.000 | 22.2M | 0.0023 |
| 2 | 64 trees | 3072 | **0.772** | 12.7M | **0.053** |
| 28 | exact gate | all | 1.000 | 26.0M | 0.0005 |
| 28 | 64 trees | 3072 | 0.971 | 17.7M | 0.0015 |
| all | 32 trees, C=2048 | | 0.853 | 12.7M | **0.678 (ppl 26.1)** |

→ Tree-only scoring **fails across all layers**: in early layers the trees capture only ~70% of the gate energy
however many trees are used.

**Partial-input exact sum** (score = exact ternary sum over the m largest-|x| input dims, per token;
`runs/phase_partial.jsonl`) beats the trees everywhere:

| layer | m | C | energy captured | MLP MACs | wiki KL |
|---|---|---|---|---|---|
| 15 | 256 | 3072 | 0.924 | 20.8M | 0.0020 |
| 15 | 512 | 2048 | 0.879 | 17.7M | 0.0022 |
| 15 | 1024 | 3072 | 0.981 | 28.0M | 0.0015 |
| 2 | 256 | 2048 | 0.948 | 10.4M | 0.0041 |
| 2 | 512 | 3072 | 0.981 | 15.3M | 0.0028 |

Why: the MLP input has **token-specific outlier dimensions** that dominate the gate. A static tree code is fitted to
the average input metric and cannot follow them; a per-token partial sum can. Next: hybrid = exact partial sum over
the outlier dims + trees fitted in the outlier-removed metric for the rest.

**Hybrid (exact partial sum on outlier dims + residual trees) and all-layer results** (`runs/phase_hybrid.jsonl`;
all-layer = full WikiText-2 test + 64 chat windows, base 13.92):

| scope | scorer | m | trees | C | energy captured | MLP MACs | wiki KL | wiki ppl |
|---|---|---|---|---|---|---|---|---|
| layer 15 | hybrid | 256 | 32 | 2048 | 0.869 | 17.1M | 0.0025 | |
| layer 15 | partial | 512 | – | 2048 | 0.879 | 17.7M | 0.0022 | |
| layer 2 | hybrid | 256 | 32 | 2048 | 0.940 | 11.6M | 0.0047 | |
| layer 2 | partial | 256 | – | 2048 | 0.948 | 10.4M | 0.0041 | |
| all | exact gate (phase 5) | – | – | – | 1 | 27.5M | **0.0028** | 13.93 |
| all | partial | 256 | – | 3072 | 0.957 | 17.6M | 0.026 | 14.04 |
| all | partial | 512 | – | 2048 | 0.944 | 15.9M | 0.038 | 13.92 |
| all | hybrid | 256 | 32 | 2048 | 0.933 | 15.3M | 0.055 | 13.95 |
| all | hybrid | 128 | 32 | 2048 | 0.912 | 14.0M | 0.092 | 14.38 |
| all | trees only | – | 32 | 2048 | 0.853 | 12.7M | 0.678 | 26.05 |

**Verdict on salvaging the tree MLP:**
1. Fitted leaf maps: dead (high rank). Tree-routed neuron selection: works per layer in mid/late layers, but **tree
   scoring fails in early layers** and across the whole model (KL 0.68).
2. Residual trees on top of an exact outlier partial sum add nothing over spending the same MACs on more exact
   dims (all-layer: hybrid 0.055 vs partial 0.038 KL at ~15.5M MACs).
3. Root cause: trees encode the *items* in an average query metric. That works when queries are well behaved
   (RoPE'd attention queries, the normed final hidden state). The MLP input has **token-specific outlier
   dimensions** that dominate the gate, so any static code is least accurate exactly where each token needs it.
4. What does work for the MLP: exact gate (lossless at energy 0.99, 1.9× fewer MACs, in the engine), or the
   per-token outlier partial sum as a cheaper, lossy selector (3.0× fewer MLP MACs at KL 0.026, +0.9% ppl).

### Can a change of basis or a per-regime metric make tree scoring work? (`runs/phase_spectral*.jsonl`)
Energy of relu(g)² captured by the top-3072 tree-scored candidates (64 trees unless noted), eval 32 wiki + 16 chat:

| scorer | layer 2 energy / KL | layer 15 energy / KL |
|---|---|---|
| plain trees (input metric) | 0.773 / 0.052 | 0.935 / 0.0021 |
| trees after a random orthogonal rotation | 0.773 / 0.053 | – |
| 4 regimes (k-means on the |x| outlier profile), one code each | 0.760 / 0.048 | 0.933 / 0.0020 |
| 8 regimes | 0.802 / 0.048 | 0.932 / 0.0020 |
| 16 regimes, 32 trees | 0.796 / 0.048 | 0.909 / 0.0027 |
| (exact partial sum, 256 dims, C=2048, for reference) | 0.948 / 0.0041 | 0.829 / 0.0029 |

- **Rotation does nothing**, as predicted: oblique (PC-split) trees and inner products are rotation-equivariant, so
  a Hadamard / spectral / eigen basis change leaves the scores unchanged. Rotations help axis-aligned methods
  (quantization grids), not these trees.
- **Per-regime metrics barely help** (layer 2: 0.77 → 0.80 energy, KL 0.052 → 0.048) at K× storage.
- **Root cause, refined:** it is not the query metric, it is that the gate rows are incompressible where it matters.
  For a token, g_i is dominated by a few outlier input dims j*, i.e. by the row's own ternary entries W[i, j*].
  Across the 6912 rows those entries are essentially independent ±1/0, so no shared code (tree, low-rank,
  mixture) of 32–64 B/row can reproduce them; only reading the actual entries can (that is why the partial
  sum works). Keys and vocab embeddings, by contrast, are highly structured, so a short code captures them.
