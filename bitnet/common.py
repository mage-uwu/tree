"""Harness for microsoft/bitnet-b1.58-2B-4T: load the real HF model, data, perplexity + KL eval,
and hooks on the real modules (no reimplementation of the model).

Hooks:
- keys/queries: modeling_bitnet.apply_rotary_pos_emb is wrapped. A forward pre-hook on each
  self_attn records the current layer; the wrapper can capture post-RoPE q/k and replace k with
  a tree approximation (KEYS[layer]).
- MLP: forward hooks on layer.mlp capture (u = MLP input, y = MLP output); MLPS[layer] replaces
  the MLP's output.
STATE.enabled switches every replacement on/off, so the same model is both teacher and student.
"""
import math, os, time, json
import torch
import torch.nn.functional as F
from transformers.models.bitnet import modeling_bitnet as MB

MODEL = os.environ.get("BITNET_MODEL", "microsoft/bitnet-b1.58-2B-4T-bf16")


class _State:
    layer = -1          # layer whose attention is running
    enabled = True      # apply replacements (False = base model)
    capture = None      # fn(layer, q, k) called with post-RoPE q (B,Hq,T,hd), k (B,Hkv,T,hd)
    keys = {}           # layer -> module: k (B,Hkv,T,hd) -> k_hat
    mlps = {}           # layer -> module: u (B,T,d) -> y
    mlp_capture = None  # fn(layer, u, y)
    select = set()      # layers whose key trees only SELECT keys; scores use exact keys (select + rescore)
    khat = {}           # layer -> k_hat stashed for selection
    attn = {}           # layer -> dict(tau_sel=, tau_v=, recent=): custom attention for that layer
    attn_stats = {}     # layer -> [kept, total] over queries in the second half of the window
STATE = _State()

_orig_rope = MB.apply_rotary_pos_emb
def _rope(q, k, cos, sin, unsqueeze_dim=1):
    q, k = _orig_rope(q, k, cos, sin, unsqueeze_dim)
    L = STATE.layer
    if STATE.capture is not None:
        STATE.capture(L, q, k)
    if STATE.enabled and L in STATE.keys:
        if L in STATE.select:
            STATE.khat[L] = STATE.keys[L](k)
        else:
            k = STATE.keys[L](k).to(k.dtype)
    return q, k


from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

def tree_attention(module, q, k, v, mask, dropout=0.0, scaling=None, **kw):
    """sdpa unless STATE.attn has an entry for this layer. Then: exact scores, restricted to keys that are
    (a) within tau_sel of the max *tree* score (if the layer's key trees are in select mode), always including
    the sink (position 0) and the `recent` newest keys, and (b) within tau_v of the max exact score."""
    L = module.layer_idx
    cfg = STATE.attn.get(L) if STATE.enabled else None
    if cfg is None:
        return sdpa_attention_forward(module, q, k, v, mask, dropout=dropout, scaling=scaling, **kw)
    G = module.num_key_value_groups
    kk = MB.repeat_kv(k, G).float(); vv = MB.repeat_kv(v, G).float(); qf = q.float()
    s = qf @ kk.transpose(-1, -2) * scaling
    T, S = s.shape[-2:]
    i = torch.arange(T, device=s.device)[:, None] + (S - T); j = torch.arange(S, device=s.device)[None, :]
    causal = j <= i
    s = s.masked_fill(~causal, float("-inf"))
    keep = causal.expand_as(s)
    kh = STATE.khat.pop(L, None)
    if kh is not None and cfg.get("tau_sel") is not None:
        st = (qf @ MB.repeat_kv(kh.float(), G).transpose(-1, -2) * scaling).masked_fill(~causal, float("-inf"))
        sel = st > st.amax(-1, keepdim=True) - cfg["tau_sel"]
        sel = sel | (j == 0) | ((i - j) < cfg.get("recent", 0))
        keep = keep & sel
    if cfg.get("tau_v") is not None:
        keep = keep & (s > s.amax(-1, keepdim=True) - cfg["tau_v"])
    half = (i[:, 0] >= T // 2) if T > 1 else torch.ones(T, dtype=torch.bool, device=s.device)
    st_ = STATE.attn_stats.setdefault(L, [0.0, 0.0])
    st_[0] += keep[..., half, :].sum().item(); st_[1] += causal.expand_as(s)[..., half, :].sum().item()
    p = torch.softmax(s.masked_fill(~keep, float("-inf")), -1)
    return (p @ vv).transpose(1, 2).contiguous().to(q.dtype), None

AttentionInterface.register("tree", tree_attention)
MB.apply_rotary_pos_emb = _rope


def install_hooks(model):
    for i, layer in enumerate(model.model.layers):
        def pre(mod, args, kwargs, i=i):
            STATE.layer = i
        layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True)

        def mlp_hook(mod, args, out, i=i):
            u = args[0]
            if STATE.mlp_capture is not None:
                STATE.mlp_capture(i, u, out)
            if STATE.enabled and i in STATE.mlps:
                return STATE.mlps[i](u).to(out.dtype)
        layer.mlp.register_forward_hook(mlp_hook)


def load_model(device="cuda", dtype=torch.bfloat16, path=MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    install_hooks(model)
    model.set_attn_implementation("tree")
    return model, tok


# ---------------------------------------------------------------- data
def _parquet(repo, pattern):
    from huggingface_hub import list_repo_files, hf_hub_download
    import pyarrow.parquet as pq
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.startswith(pattern) and f.endswith(".parquet"))
    assert files, (repo, pattern)
    return pq.read_table(hf_hub_download(repo, files[0], repo_type="dataset")).to_pylist()


def wikitext(split):
    rows = _parquet("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-")
    return "".join(r["text"] for r in rows)


def ultrachat(split, tok, n):
    """n conversations rendered with the model's chat template, joined into one stream."""
    rows = _parquet("HuggingFaceH4/ultrachat_200k", f"data/{split}-")[:n]
    return "".join(tok.apply_chat_template(r["messages"], tokenize=False) for r in rows)


def windows(tok, text, T, n=None, bos=True):
    """Non-overlapping T-token windows (each starts with BOS, as the model always sees one)."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    L = T - 1 if bos else T
    w = [ids[i:i + L] for i in range(0, len(ids) - L + 1, L)]
    if n is not None:
        w = w[:n]
    x = torch.tensor(w)
    if bos:
        x = torch.cat([torch.full((len(x), 1), tok.bos_token_id), x], 1)
    return x


# ---------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, X, bs=2, kl=True, device="cuda", chunk=512):
    """Mean NLL (and ppl) of the current model on windows X; if kl, also mean KL(base || current)
    per token, where base = same model with STATE.enabled = False. Loss excludes the BOS position.
    Log-softmax over the 128k vocab is done in position chunks to bound memory."""
    nll = kls = top1 = 0.0; n = 0
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(device)
        if kl:
            STATE.enabled = False
            lb = model(x).logits[:, :-1]
            STATE.enabled = True
        lc = model(x).logits[:, :-1]
        y = x[:, 1:]
        for j in range(0, y.shape[1], chunk):
            lpc = F.log_softmax(lc[:, j:j + chunk].float(), -1)
            nll += -lpc.gather(-1, y[:, j:j + chunk, None]).sum().item()
            if kl:
                lpb = F.log_softmax(lb[:, j:j + chunk].float(), -1)
                kls += (lpb.exp() * (lpb - lpc)).sum().item()
                top1 += (lpb.argmax(-1) == lpc.argmax(-1)).sum().item()
        n += y.numel()
        del lc
        if kl: del lb
    out = {"nll": nll / n, "ppl": math.exp(nll / n), "tokens": n}
    if kl:
        out["kl"] = kls / n; out["top1_agree"] = top1 / n
    return out


def log(path, rec):
    rec = dict(rec, t=time.strftime("%H:%M:%S"))
    print(json.dumps(rec), flush=True)
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
