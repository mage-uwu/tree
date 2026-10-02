"""Phase 5: sparse exact MLP on all layers (frozen ternary weights; per token only k of 6912 neurons).
Selectors: gate (exact gate pre-activations, top-k of relu(g)) or lowrank<r> (closed-form rank-r gate predictor in
each layer's input-data metric, fitted against the base model). Optionally combined with select+rescore attention
from saved key trees (phase 6)."""
import argparse, time, torch
from common import STATE, load_model, wikitext, ultrachat, windows, evaluate, log
from transformers.integrations.bitnet import ActQuant

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase5.jsonl")
ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--configs", default="gate:1024,lowrank256:1024,lowrank256:1536")
ap.add_argument("--calib_windows", type=int, default=64)
ap.add_argument("--eval_wiki", type=int, default=0)
ap.add_argument("--eval_chat", type=int, default=64)
ap.add_argument("--keytrees", default="")                         # phase 3 file -> also select+rescore attention
ap.add_argument("--tau_sel", type=float, default=4)
ap.add_argument("--recent", type=int, default=64)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=None)
a = ap.parse_args()
dev = a.device
t0 = time.time()
model, tok = load_model(dev, path=a.model) if a.model else load_model(dev)
d, F, NL = model.config.hidden_size, model.config.intermediate_size, model.config.num_hidden_layers
wiki_test = windows(tok, wikitext("test"), a.T)
if a.eval_wiki: wiki_test = wiki_test[:a.eval_wiki]
chat_test = windows(tok, ultrachat("test_sft", tok, 400), a.T, a.eval_chat)
calib = torch.cat([windows(tok, wikitext("train"), a.T, a.calib_windows * 3 // 4),
                   windows(tok, ultrachat("train_sft", tok, 1500), a.T, a.calib_windows // 4)])

# per-layer input second moments of the int8-quantized MLP input (base model)
C = torch.zeros(NL, d, d, device=dev, dtype=torch.float64); n = [0] * NL
def cap(l, u, y):
    x = ActQuant.apply(u.reshape(-1, d)).double(); C[l] += x.T @ x; n[l] += len(x)
STATE.mlp_capture = cap; STATE.enabled = False
with torch.no_grad():
    for i in range(0, len(calib), a.bs):
        model.model(calib[i:i + a.bs].to(dev))
STATE.mlp_capture = None; STATE.enabled = True
C /= torch.tensor(n, device=dev, dtype=torch.float64)[:, None, None]


def ternary(W):
    s = 1.0 / W.abs().mean().clamp(min=1e-5)
    return (W * s).round().clamp(-1, 1) / s


def lowrank(l, r):
    W = ternary(model.model.layers[l].mlp.gate_proj.weight.float()).double()
    ev, V = torch.linalg.eigh(C[l]); ev = ev.clamp(min=ev.max() * 1e-6)
    Ch, Chi = (V * ev.sqrt()) @ V.T, (V * ev.rsqrt()) @ V.T
    Uw, Sw, Vw = torch.linalg.svd(W @ Ch, full_matrices=False)
    return (Chi @ Vw[:r].T * Sw[:r]).float(), Uw[:, :r].T.float().contiguous()


def topk_mask(score, k):
    return torch.zeros_like(score, dtype=torch.bool).scatter_(-1, score.topk(k, -1).indices, True)


KSTAT = [0.0, 0]


def energy_mask(g, frac):
    """per token: fewest neurons whose relu(g)^2 covers `frac` of the token's total relu(g)^2."""
    e = torch.relu(g).float().pow(2)
    v, idx = e.sort(-1, descending=True)
    c = v.cumsum(-1); keep = (c - v) < frac * c[:, -1:]
    KSTAT[0] += keep.sum().item(); KSTAT[1] += len(g)
    return torch.zeros_like(keep).scatter_(-1, idx, keep)


def sparse_mlp(mlp, k, AB=None, frac=None):
    def f(u):
        sh = u.shape; U = u.reshape(-1, d); out = []
        for i in range(0, len(U), 4096):
            x = U[i:i + 4096]
            g = mlp.gate_proj(x); h = mlp.act_fn(g) * mlp.up_proj(x)
            if frac is not None:
                m = energy_mask(g, frac)
            else:
                score = torch.relu(g).float() if AB is None else torch.relu((ActQuant.apply(x).float() @ AB[0]) @ AB[1])
                m = topk_mask(score, k)
            out.append(mlp.down_proj(mlp.ffn_sub_norm(h * m)))
        return torch.cat(out).reshape(sh)
    return f


if a.keytrees:
    from keytrees import KeyTrees
    sd = torch.load(a.keytrees, map_location=dev)
    Hkv, hd = model.config.num_key_value_heads, d // model.config.num_attention_heads
    for L in range(NL):
        kt = KeyTrees(Hkv, sd["S"], sd["D"], hd, exact_sink=sd["exact_sink"]).to(dev); kt.load_state_dict(sd[L])
        STATE.keys[L] = kt; STATE.select.add(L); STATE.attn[L] = {"tau_sel": a.tau_sel, "recent": a.recent}
    STATE.attn_stats = {}
    rec = {"phase": 6, "config": f"select+rescore S={sd['S']} tau_sel={a.tau_sel} only"}
    for name, X in [("wiki", wiki_test), ("chat", chat_test)]:
        r = evaluate(model, X, a.bs, kl=True, device=dev)
        rec.update({f"{name}_ppl": round(r["ppl"], 4), f"{name}_kl": round(r["kl"], 5), f"{name}_top1": round(r["top1_agree"], 4)})
    rec["read_frac_mean"] = round(sum(k / n for k, n in STATE.attn_stats.values()) / len(STATE.attn_stats), 4)
    log(a.out, rec)

for cfg in a.configs.split(","):
    sel, k = cfg.split(":")
    frac = float(k) if sel == "gatee" else None
    k = int(float(k)) if frac is None else 0
    STATE.mlps = {}
    for l in range(NL):
        mlp = model.model.layers[l].mlp
        AB = lowrank(l, int(sel[len("lowrank"):])) if sel.startswith("lowrank") else None
        STATE.mlps[l] = sparse_mlp(mlp, k, AB, frac)
    r_ = int(sel[len("lowrank"):]) if sel.startswith("lowrank") else 0
    macs = (r_ * (d + F) + 3 * k * d) if r_ else (F * d + 2 * k * d)
    KSTAT[0] = KSTAT[1] = 0
    rec = {"phase": 6 if a.keytrees else 5, "config": cfg, "attn": "select+rescore" if a.keytrees else "exact",
           "mlp_macs": macs, "dense_mlp_macs": 3 * F * d}
    STATE.attn_stats = {}
    for name, X in [("wiki", wiki_test), ("chat", chat_test)]:
        r = evaluate(model, X, a.bs, kl=True, device=dev)
        rec.update({f"{name}_ppl": round(r["ppl"], 4), f"{name}_kl": round(r["kl"], 5), f"{name}_top1": round(r["top1_agree"], 4)})
    if frac is not None:
        rec["mean_k"] = round(KSTAT[0] / max(KSTAT[1], 1), 1); rec["mlp_macs"] = int(F * d + 2 * rec["mean_k"] * d)
    log(a.out, rec)
STATE.mlps = {}
log(a.out, {"event": "done", "total_s": round(time.time() - t0, 1)})
