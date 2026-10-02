"""Phase 4b: sparse *exact* MLP. Keep the frozen ternary weights; per token compute only k of the 6912
intermediate neurons (rows of gate/up, columns of down). The question is how to pick the set cheaply.

Selectors (all evaluated as a mask on the intermediate h before ffn_sub_norm, so the norm is computed over the
kept neurons with the full-width denominator, which is what a sparse kernel would do):
  oracle    per-token top-k of |h| (needs the full MLP; upper bound)
  gate      exact gate pre-activations (1/3 of the MLP), top-k of relu(g)
  lowrank   rank-r predictor of the gate, g_hat = (u A) B, closed form: best rank-r approx of W_gate in the
            input-data metric; top-k of g_hat. Cost r*(d + 6912).
  tree      oblique tree on u (D splits, fitted like the key trees on the top PCs of u); per leaf a fixed set =
            the k neurons most often in the token top-k inside that leaf. Cost D*d.
  tree+lowrank  union of a small leaf set and a low-rank top-k.
MACs reported per token: selector + 3*k*d for the exact sparse part (dense MLP = 3*6912*d = 53.1M)."""
import argparse, time, torch
from common import STATE, load_model, wikitext, ultrachat, windows, evaluate, log

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase4b.jsonl")
ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--layers", default="15,2,28")
ap.add_argument("--ks", default="512,1024,1536,2048")
ap.add_argument("--ranks", default="128,256")
ap.add_argument("--depths", default="8,10")
ap.add_argument("--fit_windows", type=int, default=256)           # 512k tokens for leaf neuron statistics
ap.add_argument("--eval_wiki", type=int, default=32)
ap.add_argument("--eval_chat", type=int, default=16)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=None)
a = ap.parse_args()
dev = a.device
t0 = time.time()
model, tok = load_model(dev, path=a.model) if a.model else load_model(dev)
d, F = model.config.hidden_size, model.config.intermediate_size
wiki_test = windows(tok, wikitext("test"), a.T)
chat_test = windows(tok, ultrachat("test_sft", tok, 400), a.T, a.eval_chat)
calib = torch.cat([windows(tok, wikitext("train"), a.T, a.fit_windows * 3 // 4),
                   windows(tok, ultrachat("train_sft", tok, 2000), a.T, a.fit_windows // 4)])
KS = list(map(int, a.ks.split(",")))


class Stop(Exception):
    pass


def stream(L, X, fn):
    def cap(l, u, y):
        if l == L:
            fn(u.reshape(-1, d)); raise Stop
    STATE.mlp_capture = cap; en = STATE.enabled; STATE.enabled = False
    with torch.no_grad():
        for i in range(0, len(X), a.bs):
            try: model.model(X[i:i + a.bs].to(dev))
            except Stop: pass
    STATE.mlp_capture = None; STATE.enabled = en


def parts(mlp, u):
    g = mlp.gate_proj(u); h = mlp.act_fn(g) * mlp.up_proj(u)
    return g, h


def masked_mlp(mlp, select):
    """MLP whose intermediate is masked to the selected neurons. select(u, g, h) -> bool mask (N, F)."""
    def f(u):
        sh = u.shape; U = u.reshape(-1, d); out = []
        for i in range(0, len(U), 4096):
            x = U[i:i + 4096]; g, h = parts(mlp, x)
            m = select(x, g, h)
            out.append(mlp.down_proj(mlp.ffn_sub_norm(h * m)))
        return torch.cat(out).reshape(sh)
    return f


def topk_mask(score, k):
    return torch.zeros_like(score, dtype=torch.bool).scatter_(-1, score.topk(k, -1).indices, True)


def ev(rec, L, fn, Ut, Yt, den):
    with torch.no_grad():
        rec["rel_err"] = round(((fn(Ut) - Yt).float().pow(2).sum() / den).item(), 5)
    STATE.mlps = {L: fn}
    for name, X in [("wiki", wiki_test[:a.eval_wiki]), ("chat", chat_test)]:
        r = evaluate(model, X, a.bs, kl=True, device=dev)
        rec.update({f"{name}_ppl": round(r["ppl"], 4), f"{name}_kl": round(r["kl"], 5)})
    STATE.mlps = {}
    rec["dense_macs"] = 3 * F * d
    log(a.out, rec)


for L in map(int, a.layers.split(",")):
    mlp = model.model.layers[L].mlp
    # held-out io
    Ut = []; stream(L, wiki_test[-4:], lambda u: Ut.append(u)); Ut = torch.cat(Ut)
    with torch.no_grad():
        Yt = torch.cat([mlp(Ut[i:i + 4096]) for i in range(0, len(Ut), 4096)])
    den = (Yt.float() - Yt.float().mean(0)).pow(2).sum()
    # calibration subsample for predictors
    Us = []; stream(L, calib[:48], lambda u: Us.append(u)); Us = torch.cat(Us)

    with torch.no_grad():
        for k in KS:
            ev({"layer": L, "sel": "oracle", "k": k, "macs": 3 * k * d}, L,
               masked_mlp(mlp, lambda x, g, h, k=k: topk_mask(h.abs().float(), k)), Ut, Yt, den)
            ev({"layer": L, "sel": "gate", "k": k, "macs": F * d + 2 * k * d}, L,
               masked_mlp(mlp, lambda x, g, h, k=k: topk_mask(torch.relu(g).float(), k)), Ut, Yt, den)

        # low-rank gate predictor in the input-data metric: W ~ (W C^1/2) svd -> A = C^-1/2 V_r S_r, B = U_r^T
        from transformers.integrations.bitnet import ActQuant
        Xq = ActQuant.apply(Us).float()
        W = mlp.gate_proj.weight.float()
        W = (W * (1.0 / W.abs().mean().clamp(min=1e-5))).round().clamp(-1, 1) / (1.0 / W.abs().mean().clamp(min=1e-5))
        C = Xq.T @ Xq / len(Xq)
        ev_, V = torch.linalg.eigh(C.double()); ev_ = ev_.clamp(min=ev_.max() * 1e-6)
        Ch = (V * ev_.sqrt()) @ V.T; Chi = (V * ev_.rsqrt()) @ V.T
        Uw, Sw, Vw = torch.linalg.svd(W.double() @ Ch, full_matrices=False)       # (F,d)
        for r in map(int, a.ranks.split(",")):
            A = (Chi @ Vw[:r].T * Sw[:r]).float()          # (d, r)
            B = Uw[:, :r].T.float().contiguous()           # (r, F)
            pred = lambda x, A=A, B=B: (ActQuant.apply(x).float() @ A) @ B
            gt = W @ Xq[:8192].T
            corr = torch.corrcoef(torch.stack([pred(Us[:8192]).flatten()[:2_000_000], gt.T.flatten()[:2_000_000]]))[0, 1].item()
            for k in KS:
                ev({"layer": L, "sel": f"lowrank{r}", "k": k, "gate_pred_corr": round(corr, 4), "macs": r * (d + F) + 3 * k * d}, L,
                   masked_mlp(mlp, lambda x, g, h, k=k, pred=pred: topk_mask(torch.relu(pred(x)), k)), Ut, Yt, den)

        # tree leaf sets
        Uc = Us.float() - Us.float().mean(0)
        _, P = torch.linalg.eigh((Uc.T @ Uc).double()); P = P.flip(1)[:, :64].T.float()   # top-64 PCs
        for D in map(int, a.depths.split(",")):
            nL = 2 ** D
            w = torch.zeros(nL - 1, d, device=dev); b = torch.zeros(nL - 1, device=dev)
            # router: at each node split on the node's top input PC (within the top-64 PC space), at the median
            node = torch.zeros(len(Us), dtype=torch.long, device=dev); Z = Uc @ P.T
            for depth in range(D):
                for j in range(2 ** depth - 1, 2 ** (depth + 1) - 1):
                    idx = (node == j).nonzero()[:, 0]
                    if len(idx) < 8:
                        node[idx] = 2 * j + 1; continue
                    Zs = Z[idx] - Z[idx].mean(0)
                    v = torch.linalg.eigh((Zs.T @ Zs).double())[1][:, -1].float()
                    wn = v @ P; p = Us[idx].float() @ wn; t = p.median()
                    w[j], b[j] = wn, t
                    node[idx] = 2 * j + 1 + (p > t).long()
            def _route(x, w, b, D):
                nd = torch.zeros(len(x), dtype=torch.long, device=x.device)
                for _ in range(D):
                    nd = 2 * nd + 1 + ((x.float() * w[nd]).sum(-1) - b[nd] > 0).long()
                return nd - (2 ** D - 1)
            # leaf neuron frequency counts over the calibration stream (top-1024 of |h| per token)
            cnt = torch.zeros(nL, F, device=dev)
            def acc(u):
                for i in range(0, len(u), 4096):
                    x = u[i:i + 4096]; _, h = parts(mlp, x)
                    cnt.index_add_(0, _route(x, w, b, D), topk_mask(h.abs().float(), min(1024, F)).float())
            stream(L, calib, acc)
            leafsets = {k: topk_mask(cnt, k) for k in set(KS) | {k // 2 for k in KS}}
            for k in KS:
                ev({"layer": L, "sel": f"tree_D{D}", "k": k, "macs": D * d + 3 * k * d}, L,
                   masked_mlp(mlp, lambda x, g, h, k=k, ls=leafsets: ls[k][_route(x, w, b, D)]), Ut, Yt, den)
            r = int(a.ranks.split(",")[-1])
            A = (Chi @ Vw[:r].T * Sw[:r]).float(); B = Uw[:, :r].T.float().contiguous()
            for k in KS:
                k2 = k // 2
                sel = lambda x, g, h, k2=k2, A=A, B=B: leafsets[k2][_route(x, w, b, D)] | topk_mask(torch.relu((ActQuant.apply(x).float() @ A) @ B), k2)
                ev({"layer": L, "sel": f"tree_D{D}+lowrank{r}", "k": k, "macs": D * d + r * (d + F) + 3 * k * d}, L,
                   masked_mlp(mlp, sel), Ut, Yt, den)
    del Us, Ut, Yt
    torch.cuda.empty_cache()
log(a.out, {"event": "done", "total_s": round(time.time() - t0, 1)})
