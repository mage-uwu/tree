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
