# STATUS (2026-10-02) — read this first
The scale-up was done; full record in `bitnet/NOTEBOOK.md` (summary table at the top), code in `bitnet/`.
- Harness: `bitnet/common.py` hooks the real HF model (RoPE wrapper, MLP hooks, custom `tree` attention, KL eval).
- What survived at 2B scale: **select+rescore attention** (trees choose keys, exact keys score them),
  **sparse exact MLP** (exact gate, per-token relu(g)² energy coverage), **tree output layer** (vocab codes,
  exact rescoring of top-N). What failed: tree MLP with fitted leaf maps (MLP is high-rank), key *replacement*.
- Engine: `bitnet/engine/` — ops for bitnet.cpp (`tree-bitnet.cpp`, `bitnet_tree.patch` against bitnet.cpp
  0b341e5's llama.cpp submodule; also fixes stock's SiLU-instead-of-relu² bug). Head + sparse MLP: 1.97× decode
  (1 thread), 1.44× (4 threads) at +1.0% ppl. Vocab tree file: `bitnet/export_vocab_trees.py` (GPU, ~1 min).
- GPU work runs as one-shot RunPod jobs: `bitnet/pod_boot.sh` (base64 start command) + `bitnet/jobs/<JOB>.sh`;
  results served read-only on port 8888. Do not use a remote exec server (blocked by policy).
- Tree attention is in the engine too (`TREE_ATTN=`, key file from `bitnet/export_key_trees.py`, needs `-fa on`):
  decode at a 7.5k-token prompt, 4 threads: tau 5 1.17× (+0.5% ppl), tau 4 1.35× (+1.6%); everything on
  (head + sparse MLP + attention) 1.49× at tau 5 (+1.6% ppl) / 1.77× at tau 4 (+2.8%). Notebook Phase 8.
- `TREE_KV8=1` (int8 K/V copy, lossless) lifts everything-on tau 5 to 1.64×. 64 trees/key: fewer reads but no net
  speed (scan/tables double); keep 32. TEAL-style input sparsity fails on BitNet (Phase 9). Notebook Phase 10.
- MLP→tree, last attempt (Phase 11): routing in *neuron space* (clusters by which neurons fire, static exact subsets,
  linear router) is 3× better than input-space trees (single-layer KL 0.012 at 6.6× fewer MLP MACs), and local
  distillation helps modestly, but it is ~10× short of a 30-layer budget.
- Healing (Phase 12): all MLPs replaced by teacher-bootstrapped students + end-to-end KL training, 6M tokens/arm:
  6.8× fewer MLP MACs heals ppl 2506 → 42 (base 12.7), then plateaus (KL 1.37). Neuron-space routing beats a narrow
  MLP 2× at equal compute, but radical MLP cuts need pretraining-scale data. MLP-to-tree is closed at this budget.
- Next: AVX-512 fast-scan and int8 leaves (would make 64 trees pay); per-layer tau; sparse-MLP memory layout.

# Tree-BitNet handoff

**Mission for this session:** run the tree conversion on a real BitNet b1.58 model at scale (`microsoft/bitnet-b1.58-2B-4T`) and measure quality and CPU speed. Everything below was proven only on a tiny 4-layer char-level BitNet transformer (d=128, TinyShakespeare). Treat the tiny results as a working recipe plus a list of traps, not as evidence it scales.

**The idea:** take a normally trained 1.58-bit (ternary) BitNet transformer, keep every BitNet weight frozen, and convert parts of it post hoc so CPU inference runs as decision-tree lookups instead of dense matmuls. Constraint from the owner: *minimal post-training adaptation* ("we're hot-rodding BitNet, not training a frontier model"). Closed-form fitting first; at most a few hundred steps tuning only leaf tables.

Start with `chimera/treeattn/smoke.sh` (about 3 minutes, 2 cores). It builds the C engine and runs every stage end to end.

---

## 1. What exists and what it showed

Two conversions, both in `chimera/treeattn/`.

### A. Tree attention (keys)
Each head's RoPE-rotated keys are encoded by a chain of small decision trees, where each tree fits the residual left by the previous ones (residual boosting). A key becomes S leaf indices. A query builds one table of leaf scores; a key's attention score is the sum of its trees' table entries. The softmax, values, and BitLinear q/k/v/o projections are unchanged.

- Training form (`tree_attn.py`): ordinary attention with `k_hat = sum_s c_s[leaf_s]`. Inference form (`treeattn.c`): byte codes + table lookups. Same function.
- Fast path ("fast-scan"): 16-leaf trees (4 bits), two trees per code byte, `pshufb` scores 32 keys per instruction, 16-bit table entries as two byte planes.
- Fit: no gradients. Per node: top principal direction of the node's data, threshold at the widest gap inside the 45–55% quantile band. Leaves = means. Keys are fitted in the query metric `(k−k̂)ᵀ Σ_q (k−k̂)`.

### B. Tree MLP (model tree)
`leaf = tree(u)`, `out = c[leaf] + Q[leaf] (P[leaf] u)`: a leaf constant plus a rank-r linear map per leaf, all int8. This is the "a ReLU MLP is piecewise linear" view. Fit is closed form (router from regression onto the output's top principal component; leaf maps by ridge + reduced-rank regression).

### Results on the tiny model (base val loss 1.5646; attention removed 3.569; MLPs removed 3.277)

| Conversion | Val loss | Notes |
|---|---|---|
| Keys, oblique 24 trees × 4 bits (12 B/key), no retraining | 1.5795 | int8 tables: 1.5788 |
| Keys, oblique 32 × 4 (16 B/key) | 1.5704 | |
| Keys, axis-aligned 40 × 4 (20 B/key) | 1.5773 | axis trees need ~40 to match 24 oblique |
| Keys + values 16 × 6, leaf-tuned | 1.615 | values are much harder than keys |
| Tree MLP, 1024 leaves, rank 16: closed form / + 300 leaf-tuning steps / int8 | 1.656 / 1.627 / 1.627 | |
| Tree MLP, 256 leaves, rank 16: same | 1.720 / 1.637 / 1.638 | |
| All-tree: keys 24×4 + tree MLP 256 leaves | 1.668 | halves converted separately, then combined |
| All-tree, 1024-leaf MLP | 1.654 | |

Speed (C, 1 thread, per token; `treeattn bench`):

| Model | ctx ≤128 | ≤2048 | ≤8192 |
|---|---|---|---|
| base | 67 µs | 243 | 828 |
| tree MLP + standard attention | 30 | 210 | 771 |
| all-tree | 103 | 184 | 531 |

- Tree MLP: 5.3–7.7× faster per MLP call in the engine here; 12–17× in isolation at d=768 and d=2560 with random weights (speed only, no quality measured at those widths).
- Tree attention scoring alone: 2.4–5.7× at 32-dim heads, **3× at 2k keys and 10–15× at 8k–32k keys at 128-dim heads** (`scanbench 128 32`, random data, vs an AVX2+FMA dot). BitNet 2B4T has 128-dim heads, so that row is the relevant one.
- C matches PyTorch: reference path within float noise; fast path gives the same validation loss within ~0.003.

### What is NOT proven
- Any quality number at real scale. Bits per key and MLP rank/leaf count needed for a 2560-wide model are unknown.
- Tree attention on 128-dim heads of a *trained* model (only random-data speed).
- Anything on subword text, long contexts in-distribution, or downstream tasks.
- The q/k/v/o projections are **not converted at all**. They stay dense ternary (LUT kernel in `chimera/attn/lut.h`).

---

## 2. The recipe (do it in this order)

1. **Base:** load the frozen BitNet model. Keep a frozen copy as the teacher.
2. **Tree attention, alone:** `convert.py`. Layer by layer, fit key trees on calibration activations; each layer is fitted on the outputs of the already-converted attention layers below, with the *original* MLPs in place. Keys only. Int8 the tables (`quant_trees.py`).
3. **Tree MLP, alone:** `convert_mlp.py`. Layer by layer, with the *original* attention in place. Then optionally tune only leaf tables (`--ft_steps 300`, KL to teacher logits, Adam lr 1e-3, cosine). Then int8 (`quant_eval.py`).
4. **Combine:** `plug.py` copies both sets of tree tables into one model. No further tuning (it didn't help).
5. **Export + verify:** `export_tree.py` → `.tre`; `parity_tree.py` (PyTorch vs C; use `EXACT=1` for the float reference path); `evalc.py` (validation loss through C, fast vs reference).

---

## 3. Hard-won rules

**Combining**
- Convert each half **separately against the base model**, then combine. Fitting tree MLPs on top of tree-attention outputs gave 1.79–1.87 instead of 1.67, and leaf tuning could not recover it. Errors of separately converted halves add almost linearly (+0.015 and +0.072 → +0.102).
- Joint leaf tuning after combining: no gain.

**Attention**
- Convert **keys only**. Value trees cost much more quality and compound badly with tree MLPs (1.757 vs 1.654).
- 16-leaf trees with more of them beat 64-leaf trees at equal bits for speed, with equal quality (24×4 ≈ 16×6).
- Bits matter more than cleverness. The query-aware metric helped only slightly because RoPE makes the query covariance near-isotropic.
- Leaf tuning helps at low bit budgets (24–48 bits) and is neutral or slightly negative at 96+ bits with lr 3e-3. Keep the closed-form fit when it is already good.
- An 8-bit shared-step leaf table was too coarse (first trees span a much wider range than later ones). Use 16-bit entries (two `pshufb` planes).
- After fast scoring, the cost is: key encode (~27%), per-query table (~13%), value read (~40%). Values are the wall; nearly every key's value is read because score pruning (`TAU`, default 16 nats) skipped only 3–8% here. Check how peaked the real model's attention is: if it is peaked, lowering `TAU` (and measuring quality) is the cheapest remaining win.
- Fixed cost per token ≈ scoring a few hundred keys. Tree attention only wins past ~1–2k keys at 32-dim heads; crossover should be near 512 keys at 128-dim heads.

**MLP**
- Rank is not the limit; **samples per leaf** are. Rank 64 and 4096 leaves both overfit. Going from 262k to 1M calibration tokens moved the 1024-leaf fit from 1.677 to 1.656. Budget ≥ ~1000 tokens per leaf, ≥ 4·rank minimum.
- Dead ends at equal compute: picking k exact neurons per leaf (activations aren't sparse; a perfect per-token top-64 of 512 only reaches 1.695); a global low-rank linear term; leaf constants only (1.95–2.01, though ~100× cheaper).
- Int8 leaf maps and int8 router cost nothing in quality and matter a lot for speed (memory traffic, not arithmetic, dominates).
- Storage is the price: 1–4 MB per layer here vs 0.03 MB for the ternary MLP.

**Numerical traps (each one cost hours)**
- BitLinear outputs sit on a discrete grid and the same vector reappears with last-bit noise. A median threshold can land on data points or between copies of the same vector, and PyTorch and C then route differently. Fix in place: thresholds at the **widest gap** in a quantile band (`BAND` in `tree_attn.py`).
- Residuals that are pure float noise must be **snapped to exactly zero** (`SNAP = 1e-5·|x|`), identically in fit, PyTorch forward, and C.
- "Degenerate node" detection must use the node's **spread**, not the gap size: with large calibration sets, gaps between neighbors are tiny and a gap test marks every node degenerate (the router silently collapses to one leaf; symptom: identical loss for every depth).
- Chunk calibration forwards (64 sequences at a time) and never materialize per-token gathers of leaf maps over the whole calibration set; that is an instant OOM.
- Small score differences get amplified by BitNet's own int8 activation rounding downstream. Compare *loss over many windows*, not max logit difference.
- Old `.tre` files are invalid after any header change; re-export.

---

## 4. Repo map

```
input.txt                         TinyShakespeare (data.py expects it here)
chimera/model.py                  BitLinear (absmean ternary weights, absmax int8 activations, STE), RMSNorm
chimera/attn/lut.h                exact AVX2 kernels for ternary matvec: T-MAC-style LUT + sign-dot baseline
chimera/treeattn/
  tree_attn.py                    LM (BitNet attention-only or +MLP), Attn, BoostedTrees (oblique or axis=True)
  tree_mlp.py                     TreeMLP, fit_router, reduced-rank regression, convert_mlp()
  train_base.py                   train the tiny base model
  convert.py                      attention conversion grid ("SxD" configs), optional leaf tuning
  convert_mlp.py                  MLP conversion grid ("D:k:r:rg"; use k=0, rg=0), optional leaf tuning
  plug.py                         combine separately converted attention + MLP checkpoints
  convert_both.py                 joint fit (kept as the documented negative result; do not use)
  quant_trees.py / quant_eval.py  int8 the key trees / the MLP leaf tables, with before/after loss
  export_tree.py                  checkpoint -> .tre for the C engine
  treeattn.c                      C engine: gen | bench | dump. EXACT=1 float reference, TAU=<nats>, ATTSTAT=1, MLPTIME=1
  parity_tree.py / evalc.py / fastcheck.py     PyTorch vs C; C loss fast vs reference
  scanbench.c / mlpbench.c / attnbench.c       isolated kernels at arbitrary widths (random data)
  oracle.py / diag.py             the "neurons aren't sparse" ceiling test; per-half ablation
  LAB_NOTEBOOK.md                 chronological results log; early sections are superseded by later ones
  runs/                           tiny base model, one converted attention model, logs, JSON results
history/                          earlier prototypes (FFF "Chimera", routed retention "Pigeon"); context only
```

Checkpoint dict keys: `cfg`, `state`, `chars`, and when converted `S`, `D`, `values`, `axis`, `tm=[D,k,r,rg]`, `tm_q8`. The `.tre` layout is whatever `export_tree.py` writes and `load()` in `treeattn.c` reads; keep the two in lockstep.

Requirements: Python 3, PyTorch, NumPy; gcc; x86-64 with AVX2 + FMA + F16C.

---

## 5. Scaling to BitNet b1.58 2B4T

Config (verified from the model's `config.json` and card): hidden 2560, 30 layers, **20 query heads, 5 key/value heads (grouped-query attention)**, head dim 128, FFN 6912, squared ReLU, SubLN, RoPE (theta 500000), 4096 context, LLaMA-3 tokenizer (vocab 128256), tied embeddings, W1.58A8 with per-token absmax activations. The reference fast engine is bitnet.cpp.

**Read the real modeling code before porting anything.** The tiny `LM` class here is not that model. Check in particular: the exact FFN structure (which projections, where `ffn_sub_norm` sits), where the attention SubLN sits, and the BitLinear quantizer details. Hook the real modules; do not reimplement the model.

### What changes at scale
- **Grouped-query attention.** Keys exist per KV head (5 per layer), each shared by 4 query heads. Fit one tree chain per KV head; build one score table per query head. For the query metric, pool the queries of the 4 heads that share a KV head.
- **Bits per key are unknown at 128 dims.** Sweep roughly 32, 48, 64, 96 trees × 4 bits and plot perplexity against bytes per key. Expect to need more bits than the tiny model did.
- **MLP input is 2560-dim.** The per-leaf closed-form fit as written does ridge regression in input space per leaf; that needs a 2560×2560 solve and thousands of samples per leaf. It will not work as is. First thing to try: project inputs onto a shared PCA subspace (a few hundred dims) and fit per-leaf maps inside it. Hypothesis worth testing early because it also fixes storage: **shared projections with a small per-leaf core**, `out = c[leaf] + Q_g · M[leaf] · (P_g u)` with `M[leaf]` r×r. Storage per leaf drops from 2·r·d to r² + d. Untested; note that a *purely* global low-rank map failed on the tiny model, so the per-leaf core is the part that has to carry it.
- **Storage budget.** As written, 1024 leaves × rank 64 at d=2560 is ~335 MB per layer in int8, ~10 GB over 30 layers. That is not shippable; either the shared-subspace form, fewer leaves, or lower rank has to work.
- **Calibration memory.** Activations are 2560 floats per token. Stream them: fit routers on a subsample, then accumulate per-leaf sufficient statistics in a second pass. Do not hold full activation tensors.
- **Amdahl.** Per layer the dense ternary work is roughly q 6.6M + k 1.6M + v 1.6M + o 6.6M multiply-adds for attention projections, and at least 2 × 6912 × 2560 ≈ 35M for the FFN. Trees remove the FFN and the key scoring, **not the projections**. Even a free MLP caps the per-layer weight work at roughly 3× unless the projections are also addressed.

### Suggested phases and gates
1. **Harness.** Load the model in PyTorch, reproduce a baseline perplexity on held-out text (e.g. WikiText-2 and a slice of something conversational), and add hooks to capture keys, queries, and MLP inputs/outputs per layer. Gate: baseline matches published behavior.
2. **Keys, one layer.** Convert one middle layer's keys; sweep bits. Gate: perplexity change is small and monotone in bits.
3. **Keys, all layers**, sequential calibration. Gate: pick the smallest bit budget within the owner's quality tolerance.
4. **Tree MLP, one layer**, in the shared-subspace form. Measure output relative error and perplexity vs leaves/rank/storage. Gate: one layer costs little; otherwise stop and rethink before doing 30.
5. **Tree MLP, all layers**, then leaf-only tuning (few hundred steps, KL to teacher).
6. **Combine** separately converted halves. Report perplexity, bytes per key, MB per layer.
7. **Engine.** Port the kernels into bitnet.cpp rather than growing `treeattn.c`: fast-scan key scoring (`treeattn.c`: `build_table`, the scan loop, `encode4`/`encode_axis`), the tree-MLP op, and keep their existing ternary kernels for projections. Benchmark tokens/second at several context lengths, multi-threaded, against stock bitnet.cpp.

Report at each gate: perplexity (and KL to the base logits), the storage added, and measured speed, with the base model's numbers beside them. Negative results are useful; say so plainly if a gate fails.

### Open questions, in rough priority
1. Does the tree MLP hold quality at d=2560 at an acceptable storage cost? (Biggest prize, biggest risk.)
2. How many bits per key do 128-dim heads need?
3. Is real-model attention peaked enough that a tighter `TAU` skips most value reads?
4. Can values be compressed without the quality loss seen here (int8 values are an obvious first step; value trees were poor)?
5. Can the q/k/v/o projections be made cheaper, since they become the floor?
