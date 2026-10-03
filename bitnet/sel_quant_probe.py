"""How small can the Phase-13 selector be? SVD selectors of one layer's gate at several ranks and weight precisions
(fp32 / int8 / int4 / ternary, per-row absmax or absmean scales), measured on held-out tokens with select-then-rescore
(kc = 2k candidates by the selector, exact gate on them, top-k computed). Bytes = selector storage per layer.
CPU-friendly: a few thousand tokens, one layer."""
import argparse, json, torch
import torch.nn.functional as F
from common import STATE, load_model, wikitext, windows

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--windows", type=int, default=6)
ap.add_argument("--T", type=int, default=512)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
dev = a.device; L = a.layer
model, tok = load_model(dev)
d, Fn = model.config.hidden_size, model.config.intermediate_size
X = windows(tok, wikitext("test"), a.T)[:a.windows]


class Stop(Exception):
    pass


Us = []
def cap(l, u, y):
    if l == L: Us.append(u.reshape(-1, d).float()); raise Stop
STATE.mlp_capture = cap; STATE.enabled = False
with torch.no_grad():
    for i in range(len(X)):
        try: model.model(X[i:i + 1].to(dev))
        except Stop: pass
U = torch.cat(Us)
mlp = model.model.layers[L].mlp
with torch.no_grad():
    probe = lambda lin, n: torch.cat([lin(torch.eye(n, dtype=torch.bfloat16)[i:i + 512]).float() for i in range(0, n, 512)])
    Wg, Wu, Wd = probe(mlp.gate_proj, d), probe(mlp.up_proj, d), probe(mlp.down_proj, Fn)
    snw = mlp.ffn_sub_norm.weight.float()
    h = F.relu(U @ Wg).pow(2) * (U @ Wu)
    hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) * snw
    Y = hn @ Wd; E = (hn * Wd.norm(dim=1)).pow(2); G = U @ Wg
    Uq, Sq, Vh = torch.linalg.svd(Wg, full_matrices=False)


def quant(M, kind):
    """per-row quantisation of M (rows = output units of that factor)."""
    if kind == "fp32": return M
    if kind == "int8": s = M.abs().amax(1, keepdim=True) / 127; return (M / s).round().clamp(-127, 127) * s
    if kind == "int4": s = M.abs().amax(1, keepdim=True) / 7; return (M / s).round().clamp(-7, 7) * s
    if kind == "ternary": s = M.abs().mean(1, keepdim=True); return (M / s).round().clamp(-1, 1) * s
    raise ValueError(kind)


bits = {"fp32": 32, "int8": 8, "int4": 4, "ternary": 1.6}
res = []
with torch.no_grad():
    for k in (1024, 1536):
        top = E.topk(k, -1).indices; m = torch.zeros_like(E).scatter_(-1, top, 1.0)
        hs = h * m; y = (hs * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) * snw) @ Wd
        res.append({"k": k, "selector": "oracle", "rel_err": round(((y - Y).pow(2).sum() / Y.pow(2).sum()).item(), 5)})
        for r in (64, 128, 256):
            A = Uq[:, :r] * Sq[:r].sqrt(); B = Vh[:r].T * Sq[:r].sqrt()          # g ~ (u A) B^T ; A (d,r), B (F,r)
            for kind in ("fp32", "int8", "int4", "ternary"):
                Aq, Bq = quant(A.T, kind).T, quant(B, kind)                      # A per output (rank) row, B per neuron row
                s = (U @ Aq) @ Bq.T
                cand = s.topk(2 * k, -1).indices
                sel = cand.gather(-1, G.gather(-1, cand).topk(k, -1).indices)
                mm = torch.zeros_like(E).scatter_(-1, sel, 1.0)
                hs = h * mm
                c = (hs.pow(2).sum(-1) / h.pow(2).sum(-1)).mean()
                y = (hs * torch.rsqrt(hs.pow(2).sum(-1, keepdim=True) / (Fn * c) + 1e-5) * snw) @ Wd
                rec = ((E * mm).sum(-1) / E.sum(-1)).mean().item()
                nbytes = (d + Fn) * r * bits[kind] / 8
                res.append({"k": k, "r": r, "quant": kind, "selector_MB": round(nbytes / 2**20, 2), "recall": round(rec, 4),
                            "rel_err": round(((y - Y).pow(2).sum() / Y.pow(2).sum()).item(), 5)})
                print(json.dumps(res[-1]), flush=True)
print(json.dumps({"layer": L, "tokens": len(U), "ternary_gate_MB": round(d * Fn / 4 / 2**20, 2)}))
