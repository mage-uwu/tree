"""Phase 4: tree MLP on one layer in shared-subspace form. Sweep subspace type / ranks / leaves.
Reports held-out output relative error, ppl and KL to base with only that layer's MLP replaced, storage and MACs.
Config "P:r_in:r_out:D" with P in {jac, pca}: jac = top eigvecs of E[J^T J] (input directions the MLP is most
sensitive to, J = dy/du); pca = top eigvecs of the input covariance."""
import argparse, time, torch
from common import STATE, load_model, wikitext, ultrachat, windows, evaluate, log
from treemlp import TreeMLP, top_eig

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase4.jsonl")
ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--configs", default="jac:64:64:0,jac:64:0:10,jac:64:64:8,jac:64:64:10,pca:64:64:10,jac:128:128:10,jac:256:64:10,jac:256:256:10")
ap.add_argument("--sub_windows", type=int, default=48)            # shared fit (P, Q, router) subsample
ap.add_argument("--calib_wiki", type=int, default=384)            # per-leaf statistics stream
ap.add_argument("--calib_chat", type=int, default=128)
ap.add_argument("--eval_wiki", type=int, default=32)
ap.add_argument("--eval_chat", type=int, default=16)
ap.add_argument("--quant", type=int, default=1)                   # also eval int8 version of each config
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=None)
a = ap.parse_args()
dev = a.device; L = a.layer
torch.manual_seed(0)
t0 = time.time()
model, tok = load_model(dev, path=a.model) if a.model else load_model(dev)
d = model.config.hidden_size
mlp = model.model.layers[L].mlp

wiki_test = windows(tok, wikitext("test"), a.T)
chat_test = windows(tok, ultrachat("test_sft", tok, 400), a.T, 64)
calib = torch.cat([windows(tok, wikitext("train"), a.T, a.calib_wiki),
                   windows(tok, ultrachat("train_sft", tok, 3000), a.T, a.calib_chat)])
calib = calib[torch.randperm(len(calib), generator=torch.Generator().manual_seed(0))]
log(a.out, {"phase": 4, "event": "data", "layer": L, "calib_windows": len(calib), "calib_tokens": calib.numel()})


class Stop(Exception):
    pass


def stream(X, fn, bs=None):
    """run the base model on windows X up to layer L's MLP; fn(u, y) gets (B*T, d) tensors."""
    def cap(l, u, y):
        if l == L:
            fn(u.reshape(-1, d), y.reshape(-1, d)); raise Stop
    STATE.mlp_capture = cap; en = STATE.enabled; STATE.enabled = False
    with torch.no_grad():
        for i in range(0, len(X), bs or a.bs):
            try:
                model.model(X[i:i + (bs or a.bs)].to(dev))
            except Stop:
                pass
    STATE.mlp_capture = None; STATE.enabled = en


# ---------------------------------------------------------------- subsample + shared bases
Us, Ys = [], []
stream(calib[:a.sub_windows], lambda u, y: (Us.append(u.float()), Ys.append(y.float())))
Us, Ys = torch.cat(Us), torch.cat(Ys)
log(a.out, {"phase": 4, "event": "subsample", "tokens": len(Us), "u_rms": Us.pow(2).mean().sqrt().item(), "y_rms": Ys.pow(2).mean().sqrt().item()})

Uc = Us - Us.mean(0); Yc = Ys - Ys.mean(0)
P_pca, ev_u = top_eig(Uc.T @ Uc / len(Uc), 256)
Q_pca, ev_y = top_eig(Yc.T @ Yc / len(Yc), 256)
Gj = torch.zeros(d, d, device=dev, dtype=torch.float64)
for i in range(0, len(Us), 2048):
    u = Us[i:i + 2048].to(torch.bfloat16).requires_grad_(True)
    with torch.enable_grad():
        y = mlp(u)
        g, = torch.autograd.grad(y, u, torch.randn_like(y))
    Gj += g.double().T @ g.double()
P_jac, ev_j = top_eig(Gj, 256)

# sparsity of the relu^2 intermediate: oracle per-token top-k neurons (how far an exact-sparse MLP could go)
with torch.no_grad():
    ub = Us[:8192].to(torch.bfloat16)
    g = mlp.gate_proj(ub); h = mlp.act_fn(g) * mlp.up_proj(ub); hn = mlp.ffn_sub_norm(h); y = mlp.down_proj(hn).float()
    yd = (y - y.mean(0)).pow(2).sum()
    rec = {"phase": 4, "event": "sparsity", "layer": L, "frac_gate_nonpos": round((g <= 0).float().mean().item(), 4)}
    e = hn.float().pow(2); es = e.sort(-1, descending=True).values.cumsum(-1) / e.sum(-1, keepdim=True)
    for k in [k for k in (128, 512, 1024, 2048) if k <= hn.shape[-1]]:
        top = hn.float().abs().topk(k, -1).indices
        m = torch.zeros_like(hn).scatter_(-1, top, 1)
        rec[f"top{k}_energy"] = round(es[:, k - 1].mean().item(), 4)
        rec[f"top{k}_rel_err"] = round(((mlp.down_proj(hn * m).float() - y).pow(2).sum() / yd).item(), 5)
    log(a.out, rec)
    del g, h, hn, y, e, es
cum = lambda ev, r: (ev[:r].sum() / ev.sum()).item()
log(a.out, {"phase": 4, "event": "spectra",
            **{f"u_var@{r}": round(cum(ev_u, r), 4) for r in (64, 128, 256)},
            **{f"y_var@{r}": round(cum(ev_y, r), 4) for r in (64, 128, 256)},
            **{f"jac@{r}": round(cum(ev_j, r), 4) for r in (64, 128, 256)}})
del Gj, Uc, Yc

# held-out MLP io for relative error
Ut, Yt = [], []
stream(wiki_test[-4:], lambda u, y: (Ut.append(u.float()), Yt.append(y.float())))
Ut, Yt = torch.cat(Ut), torch.cat(Yt)
den = (Yt - Yt.mean(0)).pow(2).sum().item()

def evals(rec):
    for name, X in [("wiki", wiki_test[:a.eval_wiki]), ("chat", chat_test[:a.eval_chat])]:
        r = evaluate(model, X, a.bs, kl=True, device=dev)
        rec.update({f"{name}_ppl": round(r["ppl"], 4), f"{name}_kl": round(r["kl"], 5), f"{name}_top1": round(r["top1_agree"], 4)})
    return rec

# references: base, MLP replaced by its mean output
for name, X in [("wiki", wiki_test[:a.eval_wiki]), ("chat", chat_test[:a.eval_chat])]:
    STATE.enabled = False; r = evaluate(model, X, a.bs, kl=False, device=dev); STATE.enabled = True
    log(a.out, {"phase": 4, "event": "base", "data": name, "ppl": round(r["ppl"], 4)})
ymean = Ys.mean(0)
STATE.mlps = {L: lambda u: ymean.expand(*u.shape[:-1], d)}
log(a.out, evals({"phase": 4, "layer": L, "config": "mean_output (MLP removed)", "rel_err": 1.0}))

for cfgs in a.configs.split(","):
    P, ri, ro, D = cfgs.split(":"); ri, ro, D = int(ri), int(ro), int(D)
    t = time.time()
    tm = TreeMLP(d, ri, ro, D).to(dev)
    Pg = (P_jac if P == "jac" else P_pca)[:ri]
    tm.fit_shared(Us, Ys, Pg, Q_pca[:ro].T.contiguous())
    st = tm.new_stats()
    stream(calib, lambda u, y: tm.accumulate(st, u, y))
    info = tm.solve(st)
    del st
    fit_s = time.time() - t
    for q in ([0, 1] if a.quant else [0]):
        if q: tm.quantize()
        rel = (tm(Ut) - Yt).pow(2).sum().item() / den
        STATE.mlps = {L: tm}
        sb = tm.storage_bytes()
        rec = {"phase": 4, "layer": L, "config": cfgs, "int8": bool(q), "leaves": tm.L, "rel_err": round(rel, 5),
               "MB_per_layer_int8": round((sb["per_leaf"] + sb["shared"]) / 2**20, 2), "macs": tm.macs(),
               "dense_macs": 3 * d * model.config.intermediate_size, "fit_s": round(fit_s, 1), **info}
        log(a.out, evals(rec))
    STATE.mlps = {}
    del tm
    torch.cuda.empty_cache()
log(a.out, {"event": "done", "total_s": round(time.time() - t0, 1)})
