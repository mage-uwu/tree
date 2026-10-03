"""Phase 14 probe: are attention heads per-token sparse? Skipping a query head saves its q_proj rows and o_proj columns
(13.1M of the 16.4M projection weights per layer are q+o). Oracle: per token keep the m of 20 heads with the largest
output contribution ||W_o[:, h] x_h|| (x = o_proj input, after attn_sub_norm), zero the rest; all 30 layers.
Baseline: a static per-layer head subset (heads ranked by mean contribution on calibration text).
Gives the ceiling for a head selector; CPU-friendly (a few 512-token windows)."""
import argparse, json, torch
from common import STATE, load_model, wikitext, windows, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--ms", default="16,12,10,8")
ap.add_argument("--windows", type=int, default=4)
ap.add_argument("--T", type=int, default=512)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="runs/phase_heads.jsonl")
a = ap.parse_args()
dev = a.device
model, tok = load_model(dev)
cfg = model.config; d, H, NL = cfg.hidden_size, cfg.num_attention_heads, cfg.num_hidden_layers
hd = d // H
X = windows(tok, wikitext("test"), a.T)[:a.windows]
Xc = windows(tok, wikitext("train"), a.T)[:2]

G = []                                                            # per layer: [H, hd, hd] Gram of o_proj column blocks
with torch.no_grad():
    for layer in model.model.layers:
        op = layer.self_attn.o_proj
        Wo = torch.cat([op(torch.eye(d, dtype=torch.bfloat16, device=dev)[i:i + 512]).float() for i in range(0, d, 512)])  # (in, out)
        Wh = Wo.reshape(H, hd, d)
        G.append(Wh @ Wh.transpose(1, 2))


def contrib(L, x):
    xh = x.float().reshape(*x.shape[:-1], H, hd)
    return torch.einsum("...hi,hij,...hj->...h", xh, G[L], xh).clamp_min(0).sqrt()


MODE = {"m": H, "static": None}
mean_c = torch.zeros(NL, H)


def hook(L):
    def pre(mod, args):
        x = args[0]
        c = contrib(L, x)
        if MODE.get("calib"): mean_c[L] += c.reshape(-1, H).mean(0).cpu(); return None
        if not STATE.enabled or MODE["m"] >= H: return None
        if MODE["static"] is not None:
            keep = torch.zeros(H, dtype=torch.bool, device=x.device); keep[MODE["static"][L]] = True
            mask = keep.expand_as(c)
        else:
            mask = torch.zeros_like(c, dtype=torch.bool).scatter_(-1, c.topk(MODE["m"], -1).indices, True)
        xh = x.reshape(*x.shape[:-1], H, hd) * mask[..., None].to(x.dtype)
        return (xh.reshape(x.shape),)
    return pre


for L, layer in enumerate(model.model.layers):
    layer.self_attn.o_proj.register_forward_pre_hook(hook(L))

with torch.no_grad():
    MODE["calib"] = True
    for i in range(len(Xc)): model(Xc[i:i + 1].to(dev))
    MODE["calib"] = False
    # how concentrated is a token's attention output over heads? fraction of contribution energy in the top-m heads
    STATE.enabled = False
    ens = []
    def cap(L):
        def pre(mod, args):
            c = contrib(L, args[0]).reshape(-1, H).pow(2); c = c / c.sum(-1, keepdim=True)
            ens.append(c.sort(-1, descending=True).values.cumsum(-1).mean(0).cpu())
        return pre
    hs = [layer.self_attn.o_proj.register_forward_pre_hook(cap(L)) for L, layer in enumerate(model.model.layers)]
    model(X[:1].to(dev)); [h.remove() for h in hs]
    cum = torch.stack(ens).mean(0)
    rec = {"phase": 14, "event": "head energy concentration", "top_m_energy": {m: round(cum[m - 1].item(), 4) for m in (4, 8, 10, 12, 16)}}
    print(json.dumps(rec), flush=True); open(a.out, "a").write(json.dumps(rec) + "\n")
    for m in [int(s) for s in a.ms.split(",")]:
        for kind in ("oracle", "static"):
            MODE["m"] = m
            MODE["static"] = None if kind == "oracle" else [mean_c[L].topk(m).indices.to(dev) for L in range(NL)]
            r = evaluate(model, X, 1, kl=True, device=dev)
            rec = {"phase": 14, "heads_kept": m, "of": H, "kind": kind, "proj_weights_saved": round((H - m) * 2 * hd * d / (2 * d * d + 2 * (d // 4) * d), 3),
                   "ppl": round(r["ppl"], 3), "kl": round(r["kl"], 4), "top1": round(r["top1_agree"], 4)}
            print(json.dumps(rec), flush=True); open(a.out, "a").write(json.dumps(rec) + "\n")
