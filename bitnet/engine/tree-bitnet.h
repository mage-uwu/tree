// Tree-BitNet decode ops for bitnet.cpp (llama.cpp fork): sparse exact MLP and tree output layer.
// Enabled by environment variables (unset = stock behaviour):
//   TREE_MLP_FRAC=0.99   sparse exact MLP: per token keep the fewest neurons whose relu(gate)^2 covers this fraction
//   TREE_MLP_K=1536      ... or a fixed top-k by relu(gate) (FRAC wins if both are set)
//   TREE_MLP_PARTIAL=256:3072  candidates from an exact sum over the 256 largest-|x| input dims, exact gate for the
//                        top 3072 only (measured: no decode speedup over the dense gate, +2.6% ppl; kept for reference)
//   TREE_HEAD=path.bin   tree output layer (file from bitnet/export_vocab_trees.py)
//   TREE_HEAD_N=8192     exact candidates per token
//   TREE_ATTN=path.bin   select+rescore attention (file from bitnet/export_key_trees.py)
//   TREE_TAU=8           keep keys within TAU nats of the best tree score (+ first cell + TREE_RECENT newest)
//   TREE_RECENT=64
//   TREE_ALL=1           also use the ops for multi-token batches (prompt / perplexity); default: decode only
#pragma once
#include "ggml.h"

struct llama_model;

// returns nullptr when disabled; otherwise the MLP output (before the residual add) for input x [n_embd, n_tokens]
ggml_tensor * tree_bitnet_ffn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * gate, ggml_tensor * up,
                              ggml_tensor * down, ggml_tensor * sub_norm, float eps, int il, int n_tokens);

// returns nullptr when disabled; otherwise logits [n_vocab, n_tokens] for normed hidden h [n_embd, n_tokens]
ggml_tensor * tree_bitnet_head(ggml_context * ctx, ggml_tensor * h, ggml_tensor * tok_embd, int n_tokens);

bool tree_bitnet_attn_enabled(int n_tokens);

// encode the ubatch's post-RoPE keys k_cur [hd, n_head_kv, n_tokens] into tree codes at their KV-cache cells
// (k_idxs: int64 cell indices). Returns a dummy tensor the attention op depends on.
ggml_tensor * tree_bitnet_kenc(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int64_t kv_size, int il);

// select+rescore attention for q_cur [hd, n_head, n_tokens]; k/v are the cache views, kq_mask the stock mask.
// deps: tensors that must be computed first (key codes, cache stores). Returns [hd*n_head, n_tokens].
ggml_tensor * tree_bitnet_attn(ggml_context * ctx, ggml_tensor * q_cur, ggml_tensor * k, ggml_tensor * v,
                               ggml_tensor * kq_mask, ggml_tensor ** deps, int n_deps, float kq_scale, int il);
