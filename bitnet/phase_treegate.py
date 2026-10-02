"""Tree-routed sparse MLP: select + rescore over neurons.

Picking the active neurons is a maximum-inner-product search: gate_i = w_i . x over the 6912 ternary gate rows.
Encode the gate rows with S boosted 16-leaf trees fitted in the metric of the (int8-quantized) MLP inputs, exactly
like the vocabulary trees. Per token:
  1. tree scores for all 6912 neurons (table build S*16*d MACs + 6912*S/2 byte lookups)
  2. top-C neurons by tree score are candidates; their gate is computed exactly (C*d ternary MACs)
  3. among candidates keep the fewest whose relu(g)^2 covers `frac` of the candidates' total; up/down exact (2k*d)
The dense exact-gate selector (phase 5) costs F*d for step 2 instead; this removes most of it.
Reports per layer and all-layer: rel err, KL, ppl, mean candidates' share of the true relu(g)^2 energy, MACs."""
import argparse, time, torch
from common import STATE, load_model, wikitext, ultrachat, windows, evaluate, log
from keytrees import KeyTrees
from transformers.integrations.bitnet import ActQuant

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase_treegate.jsonl")
ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--layers", default="15,2,28")
ap.add_argument("--trees", default="16,32,64")
ap.add_argument("--cands", default="1536,2048,3072")
ap.add_argument("--frac", type=float, default=0.99)
ap.add_argument("--partial", default="")                          # input dims m for partial-sum scoring, e.g. 256,512
ap.add_argument("--partial_cands", default="2048,3072")
ap.add_argument("--rot", default="")                              # S:C list, trees fitted after a random orthogonal rotation (control)
ap.add_argument("--mix", default="")                              # K:S:C list, K token regimes (spectral/k-means on |x|), one code each
ap.add_argument("--hybrid", default="")                           # m:S:C list, e.g. 128:32:2048,256:32:2048
ap.add_argument("--all_partial", default="")                      # all-layer partial configs m:C
ap.add_argument("--all_hybrid", default="")                       # all-layer hybrid configs m:S:C
ap.add_argument("--all", default="32:2048,32:3072,64:2048")     # all-layer configs S:C ("" to skip)
ap.add_argument("--calib_windows", type=int, default=48)
ap.add_argument("--eval_wiki", type=int, default=32)
ap.add_argument("--eval_chat", type=int, default=16)
ap.add_argument("--all_eval_wiki", type=int, default=0)
ap.add_argument("--all_eval_chat", type=int, default=64)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=None)
a = ap.parse_args()
dev = a.device
t0 = time.time()
model, tok = load_model(dev, path=a.model) if a.model else load_model(dev)
d, F, NL = model.config.hidden_size, model.config.intermediate_size, model.config.num_hidden_layers
wiki_test = windows(tok, wikitext("test"), a.T)
chat_test = windows(tok, ultrachat("test_sft", tok, 400), a.T, max(a.eval_chat, a.all_eval_chat))
calib = torch.cat([windows(tok, wikitext("train"), a.T, a.calib_windows * 3 // 4),
                   windows(tok, ultrachat("train_sft", tok, 1500), a.T, a.calib_windows // 4)])


def ternary(W):
    s = 1.0 / W.abs().mean().clamp(min=1e-5)
    return (W * s).round().clamp(-1, 1) / s


# per-layer second moments of the int8-quantized MLP inputs (one pass over calibration)
C = torch.zeros(NL, d, d, device=dev, dtype=torch.float64); n = [0] * NL
def cap(l, u, y):
    x = ActQuant.apply(u.reshape(-1, d)).double(); C[l] += x.T @ x; n[l] += len(x)
STATE.mlp_capture = cap; STATE.enabled = False
with torch.no_grad():
    for i in range(0, len(calib), a.bs):
        model.model(calib[i:i + a.bs].to(dev))
STATE.mlp_capture = None; STATE.enabled = True
C = (C / torch.tensor(n, device=dev, dtype=torch.float64)[:, None, None]).float()
log(a.out, {"event": "calib", "tokens": n[0], "s": round(time.time() - t0, 1)})

STAT = {"cand_energy": 0.0, "k": 0.0, "n": 0}

CRES = {}
def residual_moments(l, m):
    """E[r r^T] with r = x minus its top-m |x| entries (int8-quantized MLP input of layer l), on 16 calib windows"""
    if (l, m) in CRES: return CRES[(l, m)]
    acc = torch.zeros(d, d, device=dev, dtype=torch.float64); cnt = [0]
    class Stop(Exception): pass
    def cap(ll, u, y):
        if ll != l: return
        x = ActQuant.apply(u.reshape(-1, d)).float()
        J = x.abs().topk(m, -1).indices
        r = x.scatter(-1, J, 0.0).double(); acc.add_(r.T @ r); cnt[0] += len(r)
        raise Stop
    STATE.mlp_capture = cap; STATE.enabled = False
    with torch.no_grad():
        for i in range(0, 16, 2):
            try: model.model(calib[i:i + 2].to(dev))
            except Stop: pass
    STATE.mlp_capture = None; STATE.enabled = True
    CRES[(l, m)] = (acc / cnt[0]).float()
    return CRES[(l, m)]


def fit_res_trees(l, S, m):
    W = ternary(model.model.layers[l].mlp.gate_proj.weight.float())
    kt = KeyTrees(1, S, 4, d).to(dev); kt.fit(W[None], residual_moments(l, m)[None])
    return kt.encode(W[None])[0][0], W


def fit_trees(l, S):
    W = ternary(model.model.layers[l].mlp.gate_proj.weight.float())
    kt = KeyTrees(1, S, 4, d).to(dev); kt.fit(W[None], C[l][None])
    return kt.encode(W[None])[0][0], W                           # W_hat (F,d), W


def capture_inputs(l, nwin=24):
    xs = []
    class Stop(Exception): pass
    def cap(ll, u, y):
        if ll != l: return
        xs.append(ActQuant.apply(u.reshape(-1, d)).float()); raise Stop
    STATE.mlp_capture = cap; STATE.enabled = False
    with torch.no_grad():
        for i in range(0, nwin, 2):
            try: model.model(calib[i:i + 2].to(dev))
            except Stop: pass
    STATE.mlp_capture = None; STATE.enabled = True
    return torch.cat(xs)


def regime_features(x):
    f = x.abs(); return f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def fit_mixture(l, K, S):
    """spectral-style regimes: k-means on the normalized |x| profile (which dims are outliers); one tree code per
    regime fitted in that regime's input metric. Router = nearest regime centroid."""
    X = capture_inputs(l); Fe = regime_features(X)
    g = torch.Generator(device=dev).manual_seed(0)
    mu = Fe[torch.randperm(len(Fe), generator=g, device=dev)[:K]].clone()
    for _ in range(25):
        a_ = (Fe @ mu.T).argmax(1)
        for k in range(K):
            sel = a_ == k
            if sel.any(): mu[k] = Fe[sel].mean(0); mu[k] /= mu[k].norm().clamp(min=1e-12)
    a_ = (Fe @ mu.T).argmax(1)
    W = ternary(model.model.layers[l].mlp.gate_proj.weight.float())
    codes, sizes = [], []
    for k in range(K):
        Xk = X[a_ == k]; sizes.append(len(Xk))
        Sk = (Xk.T @ Xk / max(len(Xk), 1)) if len(Xk) > 10 else C[l]
        kt = KeyTrees(1, S, 4, d).to(dev); kt.fit(W[None], Sk[None])
        codes.append(kt.encode(W[None])[0][0])
    return mu, torch.stack(codes), sizes


def tree_mlp(mlp, What, Cn, frac, mdim=0, Wres=None, mix=None):
    def f(u):
        sh = u.shape; U = u.reshape(-1, d); out = []
        for i in range(0, len(U), 4096):
            x = U[i:i + 4096]
            g = mlp.gate_proj(x); h = mlp.act_fn(g) * mlp.up_proj(x)
            if mdim:                                               # exact ternary sum over the m largest-|x| input dims
                xq = ActQuant.apply(x).float()
                J = xq.abs().topk(mdim, -1).indices
                xs = torch.zeros_like(xq).scatter_(-1, J, xq.gather(-1, J))
                gh = xs @ What.T
                if Wres is not None: gh = gh + (xq - xs) @ Wres.T             # tree estimate of the remainder
                cand = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, gh.topk(Cn, -1).indices, True)
            elif mix is not None:
                mu, codes = mix
                xq = ActQuant.apply(x).float(); k_ = (regime_features(xq) @ mu.T).argmax(1)
                gh = torch.einsum("nd,nfd->nf", xq, codes[k_]) if len(xq) <= 256 else torch.cat(
                    [torch.einsum("nd,nfd->nf", xq[j:j + 256], codes[k_[j:j + 256]]) for j in range(0, len(xq), 256)])
                cand = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, gh.topk(Cn, -1).indices, True)
            elif What is None:
                cand = torch.ones_like(g, dtype=torch.bool)
            else:
                gh = ActQuant.apply(x).float() @ What.T
                cand = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, gh.topk(Cn, -1).indices, True)
            e_all = torch.relu(g).float().pow(2)
            e = e_all * cand
            v, idx = e.sort(-1, descending=True); c = v.cumsum(-1)
            m = torch.zeros_like(cand).scatter_(-1, idx, ((c - v) < frac * c[:, -1:]) & (v > 0))
            STAT["cand_energy"] += (e.sum(-1) / e_all.sum(-1).clamp(min=1e-20)).sum().item()
            STAT["k"] += m.sum().item(); STAT["n"] += len(x)
            out.append(mlp.down_proj(mlp.ffn_sub_norm(h * m)))
        return torch.cat(out).reshape(sh)
    return f


def ev(rec, wiki, chat):
    STAT.update(cand_energy=0.0, k=0.0, n=0)
    for name, X in [("wiki", wiki), ("chat", chat)]:
        r = evaluate(model, X, a.bs, kl=True, device=dev)
        rec.update({f"{name}_ppl": round(r["ppl"], 4), f"{name}_kl": round(r["kl"], 5), f"{name}_top1": round(r["top1_agree"], 4)})
    if STAT["n"]:
        rec["cand_energy_share"] = round(STAT["cand_energy"] / STAT["n"], 5); rec["mean_k"] = round(STAT["k"] / STAT["n"], 1)
        S, Cn, m = rec.get("S", 0) or 0, rec.get("C", F), rec.get("m", 0)
        rec["mlp_macs"] = int(S * 16 * d + m * F + (Cn if (S or m) else F) * d + 2 * rec["mean_k"] * d)
        rec["lookup_bytes"] = F * S // 2
    rec["dense_mlp_macs"] = 3 * F * d
    log(a.out, rec)


W8, C8 = wiki_test[:a.eval_wiki], chat_test[:a.eval_chat]
for L in map(int, filter(None, a.layers.split(","))):
    mlp = model.model.layers[L].mlp
    STATE.mlps = {L: tree_mlp(mlp, None, F, a.frac)}
    ev({"layer": L, "sel": f"exact gate, energy {a.frac}"}, W8, C8)
    for m in map(int, filter(None, a.partial.split(","))):
        W = ternary(mlp.gate_proj.weight.float())
        for Cn in map(int, a.partial_cands.split(",")):
            STATE.mlps = {L: tree_mlp(mlp, W, Cn, a.frac, m)}
            ev({"layer": L, "sel": "partial-input exact sum", "m": m, "C": Cn}, W8, C8)
    for cfg in filter(None, a.rot.split(",")):
        S, Cn = map(int, cfg.split(":"))
        Q = torch.linalg.qr(torch.randn(d, d, device=dev, generator=torch.Generator(device=dev).manual_seed(1)))[0]
        W = ternary(mlp.gate_proj.weight.float())
        kt = KeyTrees(1, S, 4, d).to(dev); kt.fit((W @ Q)[None], (Q.T @ C[L] @ Q)[None])
        What = kt.encode((W @ Q)[None])[0][0] @ Q.T                     # back to the original basis
        STATE.mlps = {L: tree_mlp(mlp, What, Cn, a.frac)}
        ev({"layer": L, "sel": "trees after random rotation", "S": S, "C": Cn}, W8, C8)
    for cfg in filter(None, a.mix.split(",")):
        K, S, Cn = map(int, cfg.split(":"))
        mu, codes, sizes = fit_mixture(L, K, S)
        STATE.mlps = {L: tree_mlp(mlp, None, Cn, a.frac, mix=(mu, codes))}
        ev({"layer": L, "sel": "regime mixture of tree codes", "K": K, "S": S, "C": Cn, "regime_sizes": sizes}, W8, C8)
        del codes
    for cfg in filter(None, a.hybrid.split(",")):
        m, S, Cn = map(int, cfg.split(":"))
        Wres, W = fit_res_trees(L, S, m)
        STATE.mlps = {L: tree_mlp(mlp, W, Cn, a.frac, m, Wres)}
        ev({"layer": L, "sel": "hybrid partial+tree", "m": m, "S": S, "C": Cn}, W8, C8)
    for S in map(int, filter(None, a.trees.split(","))):
        t = time.time(); What, W = fit_trees(L, S); fs = time.time() - t
        # score quality on held-out-ish inputs: correlation of tree gate scores with exact
        xs = ActQuant.apply(torch.randn(4096, d, device=dev) @ torch.linalg.cholesky(C[L].double() + 1e-6 * torch.eye(d, device=dev, dtype=torch.float64)).float().T)
        g, gh = xs @ W.T, xs @ What.T
        corr = torch.corrcoef(torch.stack([g.flatten()[:4_000_000], gh.flatten()[:4_000_000]]))[0, 1].item()
        for Cn in map(int, a.cands.split(",")):
            STATE.mlps = {L: tree_mlp(mlp, What, Cn, a.frac)}
            ev({"layer": L, "sel": "tree select+rescore", "S": S, "C": Cn, "fit_s": round(fs, 1), "gate_corr": round(corr, 4),
                "degenerate_nodes": None}, W8, C8)
    STATE.mlps = {}

for cfg in filter(None, a.all.split(",")):
    S, Cn = map(int, cfg.split(":"))
    t = time.time(); mlps = {}
    for l in range(NL):
        What, _ = fit_trees(l, S)
        mlps[l] = tree_mlp(model.model.layers[l].mlp, What, Cn, a.frac)
    STATE.mlps = mlps
    Wa = wiki_test[:a.all_eval_wiki] if a.all_eval_wiki else wiki_test
    ev({"layer": "all", "sel": "tree select+rescore", "S": S, "C": Cn, "fit_s": round(time.time() - t, 1)}, Wa, chat_test[:a.all_eval_chat])
    STATE.mlps = {}
Wa = wiki_test[:a.all_eval_wiki] if a.all_eval_wiki else wiki_test
for cfg in filter(None, a.all_partial.split(",")):
    m, Cn = map(int, cfg.split(":"))
    STATE.mlps = {l: tree_mlp(model.model.layers[l].mlp, ternary(model.model.layers[l].mlp.gate_proj.weight.float()), Cn, a.frac, m) for l in range(NL)}
    ev({"layer": "all", "sel": "partial-input exact sum", "m": m, "C": Cn}, Wa, chat_test[:a.all_eval_chat])
    STATE.mlps = {}
for cfg in filter(None, a.all_hybrid.split(",")):
    m, S, Cn = map(int, cfg.split(":"))
    mlps = {}
    for l in range(NL):
        Wres, W = fit_res_trees(l, S, m)
        mlps[l] = tree_mlp(model.model.layers[l].mlp, W, Cn, a.frac, m, Wres)
    STATE.mlps = mlps
    ev({"layer": "all", "sel": "hybrid partial+tree", "m": m, "S": S, "C": Cn}, Wa, chat_test[:a.all_eval_chat])
    STATE.mlps = {}
log(a.out, {"event": "done", "total_s": round(time.time() - t0, 1)})
