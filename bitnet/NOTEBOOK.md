# Tree-BitNet at scale: microsoft/bitnet-b1.58-2B-4T (lab notebook)

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
