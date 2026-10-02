"""Phase 6: combine the three conversions that survived, each alone and all together, on the full WikiText-2 test
set and 64 UltraChat windows:
  attn  select+rescore attention: key trees (S trees x 4 bits per KV head, fitted per layer on base activations)
        choose keys within tau_sel of the max tree score (+ sink + `recent` newest); exact keys score them;
        only their values are read.
  mlp   sparse exact MLP: exact gate, then up/down only for neurons covering `frac` of the token's relu(g)^2
        (or top-k with --mlp_k).
  head  tree output layer: vocab codes (S_v trees), exact logits for the top-N candidates by tree score.
Each piece is converted against the base model (handoff rule), then plugged together."""
import argparse, math, time, torch
import torch.nn.functional as F
from common import STATE, load_model, wikitext, ultrachat, windows, evaluate, log
from keytrees import KeyTrees

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase6.jsonl")
ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--S", type=int, default=32)
ap.add_argument("--tau_sel", type=float, default=8)
ap.add_argument("--recent", type=int, default=64)
ap.add_argument("--mlp_frac", type=float, default=0.99)
ap.add_argument("--mlp_k", type=int, default=0)
ap.add_argument("--Sv", type=int, default=128)
ap.add_argument("--N", type=int, default=8192)
ap.add_argument("--calib_windows", type=int, default=64)
ap.add_argument("--eval_wiki", type=int, default=0)
ap.add_argument("--eval_chat", type=int, default=64)
ap.add_argument("--combos", default="attn,mlp,head,attn+mlp,attn+mlp+head")
ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=None)
a = ap.parse_args()
dev = a.device
t0 = time.time()
model, tok = load_model(dev, path=a.model) if a.model else load_model(dev)
cfg = model.config
Hq, Hkv, NL, d, Fh = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size
hd = d // Hq
wiki_test = windows(tok, wikitext("test"), a.T)
if a.eval_wiki: wiki_test = wiki_test[:a.eval_wiki]
chat_test = windows(tok, ultrachat("test_sft", tok, 400), a.T, a.eval_chat)
calib = torch.cat([windows(tok, wikitext("train"), a.T, a.calib_windows * 3 // 4),
                   windows(tok, ultrachat("train_sft", tok, 1500), a.T, a.calib_windows // 4)])


class Stop(Exception):
    pass


# ---------------------------------------------------------------- attention: key trees per layer (base activations)
def capture(L):
    K, Sq = [], torch.zeros(Hkv, hd, hd, device=dev, dtype=torch.float64); n = [0]
    def cap(l, q, k):
        if l != L: return
        K.append(k.float().transpose(0, 1).reshape(Hkv, -1, hd))
        qg = q.float().reshape(q.shape[0], Hkv, Hq // Hkv, q.shape[2], hd).transpose(0, 1).reshape(Hkv, -1, hd).double()
        Sq.add_(torch.einsum("hnd,hne->hde", qg, qg)); n[0] += qg.shape[1]
        raise Stop
    STATE.capture = cap; STATE.enabled = False
    with torch.no_grad():
        for i in range(0, len(calib), 2):
            try: model.model(calib[i:i + 2].to(dev))
            except Stop: pass
    STATE.capture = None; STATE.enabled = True
    return torch.cat(K, 1), (Sq / n[0]).float()

t = time.time(); KT = {}
for L in range(NL):
    K, Sq = capture(L)
    KT[L] = KeyTrees(Hkv, a.S, 4, hd, exact_sink=True).to(dev); KT[L].fit(K, Sq)
    del K
log(a.out, {"event": "key trees", "S": a.S, "s": round(time.time() - t, 1)})


# ---------------------------------------------------------------- mlp: sparse exact
KSTAT = [0.0, 0]
def sparse_mlp(mlp):
    def f(u):
        sh = u.shape; U = u.reshape(-1, d); out = []
        for i in range(0, len(U), 4096):
            x = U[i:i + 4096]
            g = mlp.gate_proj(x); h = mlp.act_fn(g) * mlp.up_proj(x)
            if a.mlp_k:
                m = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, torch.relu(g).float().topk(a.mlp_k, -1).indices, True)
            else:
                e = torch.relu(g).float().pow(2); v, idx = e.sort(-1, descending=True); c = v.cumsum(-1)
                m = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx, (c - v) < a.mlp_frac * c[:, -1:])
            KSTAT[0] += m.sum().item(); KSTAT[1] += len(x)
            out.append(mlp.down_proj(mlp.ffn_sub_norm(h * m)))
        return torch.cat(out).reshape(sh)
    return f
MLPS = {l: sparse_mlp(model.model.layers[l].mlp) for l in range(NL)}


# ---------------------------------------------------------------- head: tree output layer
E = model.lm_head.weight.float(); V = E.shape[0]
Hc = []
with torch.no_grad():
    STATE.enabled = False
    for i in range(min(32, len(calib))):
        Hc.append(model.model(calib[i:i + 1].to(dev)).last_hidden_state[0].float())
    STATE.enabled = True
Hc = torch.cat(Hc); Sqv = (Hc.T @ Hc / len(Hc))[None]; del Hc
t = time.time()
vt = KeyTrees(1, a.Sv, 4, d).to(dev); vt.fit(E[None], Sqv)
Eh = vt.encode(E[None])[0][0].to(torch.bfloat16); del vt
log(a.out, {"event": "vocab trees", "Sv": a.Sv, "s": round(time.time() - t, 1)})
HEAD = {"on": False}
def head_hook(mod, args, out):
    if not (STATE.enabled and HEAD["on"]):
        return None
    h = args[0]; sh = out.shape; H2 = h.reshape(-1, d); res = []
    for i in range(0, len(H2), 1024):
        hh = H2[i:i + 1024]
        z = out.reshape(-1, V)[i:i + 1024]                      # exact logits (only candidates are kept)
        za = hh.to(torch.bfloat16) @ Eh.T
        top = za.float().topk(a.N, -1).indices
        inc = torch.zeros(len(hh), V, dtype=torch.bool, device=dev).scatter_(1, top, True)
        res.append(torch.where(inc, z, za.to(z.dtype)))
    return torch.cat(res).reshape(sh)
model.lm_head.register_forward_hook(head_hook)


def setup(combo):
    parts = combo.split("+")
    STATE.keys, STATE.select, STATE.attn, STATE.mlps = {}, set(), {}, {}
    HEAD["on"] = "head" in parts
    if "attn" in parts:
        STATE.keys = dict(KT); STATE.select = set(range(NL))
        STATE.attn = {L: {"tau_sel": a.tau_sel, "recent": a.recent} for L in range(NL)}
    if "mlp" in parts:
        STATE.mlps = dict(MLPS)


for combo in a.combos.split(","):
    setup(combo); STATE.attn_stats = {}; KSTAT[0] = KSTAT[1] = 0
    rec = {"phase": 6, "combo": combo, "S": a.S, "tau_sel": a.tau_sel, "mlp": f"k={a.mlp_k}" if a.mlp_k else f"frac={a.mlp_frac}",
           "Sv": a.Sv, "N": a.N}
    for name, X in [("wiki", wiki_test), ("chat", chat_test)]:
        r = evaluate(model, X, a.bs, kl=True, device=dev)
        rec.update({f"{name}_ppl": round(r["ppl"], 4), f"{name}_kl": round(r["kl"], 5), f"{name}_top1": round(r["top1_agree"], 4)})
    if STATE.attn_stats:
        rec["kv_read_frac"] = round(sum(k / n for k, n in STATE.attn_stats.values()) / len(STATE.attn_stats), 4)
    if KSTAT[1]:
        rec["mlp_mean_k"] = round(KSTAT[0] / KSTAT[1], 1)
    log(a.out, rec)
setup("")
log(a.out, {"event": "done", "total_s": round(time.time() - t0, 1)})
