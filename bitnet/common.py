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
STATE = _State()

_orig_rope = MB.apply_rotary_pos_emb
def _rope(q, k, cos, sin, unsqueeze_dim=1):
    q, k = _orig_rope(q, k, cos, sin, unsqueeze_dim)
    L = STATE.layer
    if STATE.capture is not None:
        STATE.capture(L, q, k)
    if STATE.enabled and L in STATE.keys:
        k = STATE.keys[L](k).to(k.dtype)
    return q, k
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
def evaluate(model, X, bs=2, kl=True, device="cuda"):
    """Mean NLL (and ppl) of the current model on windows X; if kl, also mean KL(base || current)
    per token, where base = same model with STATE.enabled = False. Loss excludes the BOS position."""
    nll = kls = top1 = 0.0; n = 0
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(device)
        if kl:
            STATE.enabled = False
            lb = model(x).logits[:, :-1].float()
            STATE.enabled = True
        lc = model(x).logits[:, :-1].float()
        y = x[:, 1:]
        lpc = F.log_softmax(lc, -1)
        nll += -lpc.gather(-1, y[..., None]).sum().item()
        if kl:
            lpb = F.log_softmax(lb, -1)
            kls += (lpb.exp() * (lpb - lpc)).sum().item()
            top1 += (lb.argmax(-1) == lc.argmax(-1)).sum().item()
            del lb, lpb
        n += y.numel()
        del lc, lpc
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
