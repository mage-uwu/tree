"""Phase 1 (harness + baseline) and phase 2 (key trees on one layer, bit sweep).
Writes JSON lines to --out. Usage: python phase12.py --out runs/phase12.jsonl"""
import argparse, math, time, torch
from common import STATE, load_model, wikitext, ultrachat, windows, evaluate, log
from keytrees import KeyTrees

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase12.jsonl")
ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--configs", default="16,32,48,64,96")          # trees x 4 bits
ap.add_argument("--extra_layers", default="2,28")                 # S=32 at other layers
ap.add_argument("--calib_wiki", type=int, default=96)
ap.add_argument("--calib_chat", type=int, default=32)
ap.add_argument("--eval_wiki", type=int, default=64)              # sweep eval (baseline uses all)
ap.add_argument("--eval_chat", type=int, default=24)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=None)
a = ap.parse_args()
dev = a.device
torch.manual_seed(0)

t0 = time.time()
model, tok = load_model(dev, path=a.model) if a.model else load_model(dev)
cfg = model.config
Hq, Hkv, hd, NL = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.num_hidden_layers
log(a.out, {"phase": 1, "event": "loaded", "s": round(time.time() - t0, 1), "layers": NL, "Hq": Hq, "Hkv": Hkv, "hd": hd})

# ---------------------------------------------------------------- data
wiki_test = windows(tok, wikitext("test"), a.T)
chat_test = windows(tok, ultrachat("test_sft", tok, 400), a.T, 96)
calib = torch.cat([windows(tok, wikitext("train"), a.T, a.calib_wiki),
                   windows(tok, ultrachat("train_sft", tok, 1500), a.T, a.calib_chat)])
log(a.out, {"phase": 1, "event": "data", "wiki_test_windows": len(wiki_test), "chat_test_windows": len(chat_test), "calib_windows": len(calib)})

# ---------------------------------------------------------------- phase 1: baseline
STATE.enabled = False
for name, X in [("wikitext2_test", wiki_test), ("ultrachat_test", chat_test),
                ("wikitext2_test_sweepslice", wiki_test[:a.eval_wiki]), ("ultrachat_test_sweepslice", chat_test[:a.eval_chat])]:
    t = time.time(); r = evaluate(model, X, a.bs, kl=False, device=dev)
    log(a.out, dict(phase=1, event="baseline", data=name, T=a.T, s=round(time.time() - t, 1), **r))

msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to(dev)
with torch.no_grad():
    g = model.generate(ids, max_new_tokens=60, do_sample=False)
log(a.out, {"phase": 1, "event": "generate", "text": tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)})

# ---------------------------------------------------------------- attention peakedness (for TAU)
stats = {}
def peak_capture(L, q, k):
    if L not in (0, 7, 15, 22, NL - 1): return
    G = Hq // Hkv
    kk = k.float().repeat_interleave(G, 1)
    s = (q.float() @ kk.transpose(-1, -2)) / math.sqrt(hd)               # (B,Hq,T,T)
    T = s.shape[-1]
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
    m = s.max(-1, keepdim=True).values
    p = torch.softmax(s, -1)
    pos = torch.arange(T, device=s.device)
    big = pos >= T // 2                                                    # queries with >= T/2 keys
    d = stats.setdefault(L, {})
    for tau in (4, 8, 12, 16):
        keep = (s > m - tau)
        frac = (keep.sum(-1).float() / (pos + 1))[..., big].mean().item()
        mass = (p * keep).sum(-1)[..., big].mean().item()
        d.setdefault(tau, []).append((frac, mass))
    d.setdefault("sink_mass", []).append(p[..., 0][..., big].mean().item())
STATE.capture = peak_capture
with torch.no_grad():
    for i in range(4):
        model(wiki_test[i:i + 1].to(dev))
STATE.capture = None
for L, d in sorted(stats.items()):
    rec = {"phase": 1, "event": "peakedness", "layer": L, "sink_mass": round(sum(d["sink_mass"]) / len(d["sink_mass"]), 4)}
    for tau in (4, 8, 12, 16):
        rec[f"tau{tau}_keys_read"] = round(sum(f for f, _ in d[tau]) / len(d[tau]), 4)
        rec[f"tau{tau}_mass_kept"] = round(sum(m for _, m in d[tau]) / len(d[tau]), 5)
    log(a.out, rec)

# ---------------------------------------------------------------- phase 2: one layer's keys
def capture_layer(L, X):
    K, Sq = [], torch.zeros(Hkv, hd, hd, device=dev, dtype=torch.float64); nq = 0
    def cap(l, q, k):
        nonlocal nq
        if l != L: return
        K.append(k.float().transpose(0, 1).reshape(Hkv, -1, hd))
        qg = q.float().reshape(q.shape[0], Hkv, Hq // Hkv, q.shape[2], hd).transpose(0, 1).reshape(Hkv, -1, hd)
        Sq.add_(torch.einsum("hnd,hne->hde", qg.double(), qg.double())); nq += qg.shape[1]
    STATE.capture = cap
    with torch.no_grad():
        for i in range(0, len(X), a.bs):
            model.model(X[i:i + a.bs].to(dev))   # base model up to layer L is all we need; skip lm_head
    STATE.capture = None
    return torch.cat(K, 1), (Sq / nq).float()

def score_err(kt, L, X):
    """held-out E[(q.(k - k_hat))^2] / E[(q.k)^2] over query heads."""
    num = den = 0.0
    def cap(l, q, k):
        nonlocal num, den
        if l != L: return
        kf = k.float(); kh = kt(kf)
        G = Hq // Hkv
        qf = q.float().reshape(q.shape[0], Hkv, G, q.shape[2], hd)
        num += torch.einsum("bhgtd,bhsd->bhgts", qf, kf - kh).pow(2).sum().item()
        den += torch.einsum("bhgtd,bhsd->bhgts", qf, kf).pow(2).sum().item()
    STATE.capture = cap; en = STATE.enabled; STATE.enabled = False
    with torch.no_grad():
        model.model(X.to(dev))
    STATE.capture = None; STATE.enabled = en
    return num / den

def run(L, S, exact_sink, K, Sq):
    t = time.time()
    kt = KeyTrees(Hkv, S, 4, hd, exact_sink=exact_sink).to(dev)
    kt.fit(K, Sq)
    fit_s = time.time() - t
    STATE.keys = {L: kt}
    rec = {"phase": 2, "layer": L, "S": S, "D": 4, "bytes_per_key_per_kvhead": kt.bytes_per_key(), "exact_sink": exact_sink,
           "fit_s": round(fit_s, 1), "score_rel_err": round(score_err(kt, L, wiki_test[-2:]), 5)}
    for name, X in [("wiki", wiki_test[:a.eval_wiki]), ("chat", chat_test[:a.eval_chat])]:
        r = evaluate(model, X, a.bs, kl=True, device=dev)
        rec.update({f"{name}_ppl": round(r["ppl"], 4), f"{name}_kl": round(r["kl"], 5), f"{name}_top1": round(r["top1_agree"], 4)})
    STATE.keys = {}
    log(a.out, rec)

STATE.keys = {a.layer: lambda k: k.float()}     # hook sanity: identity replacement must give KL exactly 0
r = evaluate(model, wiki_test[:2], a.bs, kl=True, device=dev); STATE.keys = {}
log(a.out, {"phase": 2, "event": "identity_check", "kl": r["kl"], "top1": r["top1_agree"]})
K, Sq = capture_layer(a.layer, calib)
log(a.out, {"phase": 2, "event": "calib", "layer": a.layer, "keys_per_head": K.shape[1]})
for S in map(int, a.configs.split(",")):
    run(a.layer, S, False, K, Sq)
run(a.layer, 32, True, K, Sq)
del K
for L in map(int, filter(None, a.extra_layers.split(","))):
    K, Sq = capture_layer(L, calib)
    run(L, 32, False, K, Sq); run(L, 32, True, K, Sq)
    del K
log(a.out, {"event": "done", "total_s": round(time.time() - t0, 1)})
