"""Phase 11: the MLP in neuron space. Cluster tokens by WHICH NEURONS carry their output (not by where the input sits),
give each cluster a static set of m exact neurons, and route to a cluster from the input with a cheap classifier, so
the full 6912-wide gate is never computed. Then distill the routed leaves (init = exact teacher rows) against the
teacher's outputs. One layer at a time; reports held-out output relative error, single-layer KL, MACs and storage.

Stages
  0  sanity: float student with all neurons == teacher MLP (the BitLinear int8 activation rounding is the only gap)
  1  references: oracle per-token top-m neurons; MLP removed
  2  probe: k-NN regression of y in input space and in gate space (how locally constant is y? bounds any tree on u)
  3  neuron-space clusters (spherical k-means on per-token neuron contribution profiles), static top-m per cluster:
     ceiling (cluster known) and routed (linear router from u; also top-2 cluster union)
  4  distillation of routed leaves (all clusters in parallel), vs a single narrow MLP of the same per-token MACs
"""
import argparse, math, time, torch
import torch.nn.functional as F
from common import STATE, load_model, wikitext, ultrachat, windows, evaluate, log

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase_geo.jsonl")
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--calib_wiki", type=int, default=160)
ap.add_argument("--calib_chat", type=int, default=40)
ap.add_argument("--clu_tokens", type=int, default=131072)      # tokens used for k-means in neuron space
ap.add_argument("--Ks", default="1,16,64,256")
ap.add_argument("--ms", default="512,1024,2048")
ap.add_argument("--kl_cfgs", default="16:1024,64:1024,64:512,256:1024")   # K:m evaluated by single-layer KL
ap.add_argument("--distill", default="1:1024,16:1024,64:1024,64:512")    # K:m distilled
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--eval_wiki", type=int, default=16)
ap.add_argument("--eval_chat", type=int, default=8)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--device", default="cuda")
ap.add_argument("--nref", type=int, default=262144)
a = ap.parse_args()
dev = a.device; L = a.layer
torch.manual_seed(0)
t0 = time.time()
R = lambda x, n=5: round(float(x), n)

model, tok = load_model(dev)
d = model.config.hidden_size
mlp = model.model.layers[L].mlp
Fn = mlp.gate_proj.out_features if hasattr(mlp.gate_proj, "out_features") else model.config.intermediate_size
eps = getattr(mlp.ffn_sub_norm, "variance_epsilon", getattr(mlp.ffn_sub_norm, "eps", 1e-5))

wiki_test = windows(tok, wikitext("test"), a.T)
chat_test = windows(tok, ultrachat("test_sft", tok, 400), a.T, 64)
calib = torch.cat([windows(tok, wikitext("train"), a.T, a.calib_wiki),
                   windows(tok, ultrachat("train_sft", tok, 3000), a.T, a.calib_chat)])
calib = calib[torch.randperm(len(calib), generator=torch.Generator().manual_seed(0))]


class Stop(Exception):
    pass


def capture(X, bs=2):
    """MLP inputs u (bf16) and teacher outputs y (bf16) of layer L on windows X (base model)."""
    Us, Ys = [], []
    def cap(l, u, y):
        if l == L:
            Us.append(u.reshape(-1, d).to(torch.bfloat16)); Ys.append(y.reshape(-1, d).to(torch.bfloat16)); raise Stop
    STATE.mlp_capture = cap; STATE.enabled = False
    with torch.no_grad():
        for i in range(0, len(X), bs):
            try:
                model.model(X[i:i + bs].to(dev))
            except Stop:
                pass
    STATE.mlp_capture = None
    return torch.cat(Us), torch.cat(Ys)


U, Y = capture(calib)
Ut, Yt = capture(torch.cat([wiki_test[-6:], chat_test[-2:]]))
N, Nt = len(U), len(Ut)
den_t = (Yt.float() - Yt.float().mean(0)).pow(2).sum().item()
rel = lambda yh, yt=None: ((yh.float() - (Yt if yt is None else yt).float()).pow(2).sum().item() / den_t)
log(a.out, {"phase": 11, "event": "data", "layer": L, "calib_tokens": N, "heldout_tokens": Nt, "F": Fn})

# ---------------------------------------------------------------- effective float weights (identity-probe)
with torch.no_grad():
    def probe(lin, n):
        out = []
        for i in range(0, n, 1024):
            e = torch.eye(n, device=dev, dtype=torch.bfloat16)[i:i + 1024]
            out.append(lin(e).float())
        return torch.cat(out)                                   # (n_in, n_out) = W^T
    Wg, Wu, Wd = probe(mlp.gate_proj, d), probe(mlp.up_proj, d), probe(mlp.down_proj, Fn)   # (d,F) (d,F) (F,d)
    snw = mlp.ffn_sub_norm.weight.float()
    dnorm = Wd.norm(dim=1)                                      # (F,) output norm of each neuron


def mlp_part(u, gi, ui, di, sw, cov=1.0, chunk=8192):
    """float gated relu^2 MLP over a neuron subset: gi/ui (d,m), di (m,d), sw (m,). The sub-norm RMS is estimated
    from the subset as sqrt(sum_S h^2 / (F * cov)), cov = the subset's mean share of the energy (calibration)."""
    out = []
    for i in range(0, len(u), chunk):
        x = u[i:i + chunk].float()
        h = F.relu(x @ gi).pow(2) * (x @ ui)
        hn = h * torch.rsqrt(h.pow(2).sum(-1, keepdim=True) / (Fn * cov) + eps) * sw
        out.append(hn @ di)
    return torch.cat(out)


def contrib(u, chunk=8192):
    """per-token |hn_i| (fp16); the contribution of neuron i to the output is |hn_i| * |down_i|."""
    out = []
    for i in range(0, len(u), chunk):
        x = u[i:i + chunk].float()
        h = F.relu(x @ Wg).pow(2) * (x @ Wu)
        hn = h * torch.rsqrt(h.pow(2).sum(-1, keepdim=True) / Fn + eps) * snw
        out.append(hn.abs().half())
    return torch.cat(out)


with torch.no_grad():
    # stage 0: float student with every neuron
    y_full = mlp_part(Ut, Wg, Wu, Wd, snw)
    log(a.out, {"phase": 11, "stage": 0, "config": "float student, all neurons", "rel_err": R(rel(y_full))})

    # stage 1: references
    At = contrib(Ut)
    for m in [int(x) for x in a.ms.split(",")]:
        top = (At.float() * dnorm).topk(m, -1).indices
        yh = []
        for i in range(0, Nt, 4096):
            x = Ut[i:i + 4096].float(); idx = top[i:i + 4096]
            h = F.relu(x @ Wg).pow(2) * (x @ Wu)
            mask = torch.zeros_like(h).scatter_(-1, idx, 1.0)
            hn = h * torch.rsqrt(h.pow(2).sum(-1, keepdim=True) / Fn + eps) * snw * mask      # true sub-norm
            yh.append(hn @ Wd)
        log(a.out, {"phase": 11, "stage": 1, "config": f"oracle per-token top-{m}", "rel_err": R(rel(torch.cat(yh)))})
    log(a.out, {"phase": 11, "stage": 1, "config": "mean output (MLP removed)", "rel_err": R(rel(Y.float().mean(0).expand(Nt, d)))})

    # stage 2: k-NN probe (is y locally constant in some geometry?)
    nref, nq = min(N, a.nref), min(Nt, 4096)
    def knn(Qf, Rf, ks=(1, 16, 64)):
        Qn, Rn = F.normalize(Qf.float(), dim=-1).half(), F.normalize(Rf.float(), dim=-1).half()
        res = {}
        idx_all = []
        for i in range(0, len(Qn), 512):
            s = (Qn[i:i + 512].float() @ Rn.float().T) if dev == "cpu" else Qn[i:i + 512] @ Rn.T
            idx_all.append(s.topk(max(ks), -1).indices)
        idx = torch.cat(idx_all)
        for k in ks:
            yh = Y[:nref][idx[:, :k]].float().mean(1)
            res[f"k{k}"] = R(((yh - Yt[:nq].float()).pow(2).sum() / (Yt[:nq].float() - Yt.float().mean(0)).pow(2).sum()).item())
        return res
    log(a.out, {"phase": 11, "stage": 2, "geometry": "input u (cosine)", **knn(Ut[:nq], U[:nref])})
    G_ref = torch.cat([(U[i:min(i + 8192, nref)].float() @ Wg).half() for i in range(0, nref, 8192)])
    log(a.out, {"phase": 11, "stage": 2, "geometry": "gate pre-activations g = W_gate u (cosine)", **knn(Ut[:nq].float() @ Wg, G_ref)})
    del G_ref
    A_ref = contrib(U[:nref])
    log(a.out, {"phase": 11, "stage": 2, "geometry": "neuron contribution profile (cosine; needs the full MLP)", **knn(At[:nq].float() * dnorm, A_ref.float() * dnorm)})
    del A_ref

    # stage 3: neuron-space clusters
    nc = min(N, a.clu_tokens)
    A = contrib(U[:nc])
    An = torch.cat([F.normalize(A[i:i + 16384].float() * dnorm, dim=-1).half() for i in range(0, nc, 16384)])
    Atn = F.normalize(At.float() * dnorm, dim=-1).half()


def kmeans(X, K, iters=25):
    g = torch.Generator(device=dev).manual_seed(0)
    C = X[torch.randperm(len(X), generator=g, device=dev)[:K]].float()
    for _ in range(iters):
        asg = torch.cat([(X[i:i + 16384] @ C.half().T).argmax(-1) for i in range(0, len(X), 16384)])
        Cn = torch.zeros_like(C).index_add_(0, asg, X.float())
        cnt = torch.bincount(asg, minlength=K)
        dead = cnt == 0
        if dead.any():
            Cn[dead] = X[torch.randint(len(X), (int(dead.sum()),), device=dev, generator=g)].float()
        C = F.normalize(Cn, dim=-1)
    asg = torch.cat([(X[i:i + 16384] @ C.half().T).argmax(-1) for i in range(0, len(X), 16384)])
    return C, asg


def subsets(asg, K, m):
    """per cluster: top-m neurons by summed contribution energy, and their mean share of the sub-norm energy."""
    E = torch.zeros(K, Fn, device=dev)
    for i in range(0, nc, 16384):
        E.index_add_(0, asg[i:i + 16384], (A[i:i + 16384].float() * dnorm).pow(2))
    S = E.topk(m, -1).indices                                   # (K, m)
    cov = torch.zeros(K, device=dev); cnt = torch.zeros(K, device=dev)
    for i in range(0, nc, 16384):
        e = A[i:i + 16384].float().pow(2); k = asg[i:i + 16384]
        share = e.gather(-1, S[k]).sum(-1) / e.sum(-1).clamp_min(1e-12)
        cov.index_add_(0, k, share); cnt.index_add_(0, k, torch.ones_like(share))
    return S, (cov / cnt.clamp_min(1)).clamp(0.05, 1.0)


def routed_eval(u, asg_u, SC):
    """each token uses its cluster's static neuron subset (exact teacher rows)."""
    S, cov = SC
    yh = torch.empty(len(u), d, device=dev)
    for k in asg_u.unique().tolist():
        sel = (asg_u == k).nonzero().squeeze(1)
        idx = S[k]
        yh[sel] = mlp_part(u[sel], Wg[:, idx], Wu[:, idx], Wd[idx], snw[idx], cov[k].item())
    return yh


def union_eval(u, top2, SC):
    """top-2 clusters: union of their subsets (coverage: the larger of the two)."""
    S, cov = SC
    yh = torch.empty(len(u), d, device=dev)
    keys = top2[:, 0] * 100000 + top2[:, 1]
    for kk in keys.unique().tolist():
        sel = (keys == kk).nonzero().squeeze(1)
        k1, k2 = kk // 100000, kk % 100000
        idx = torch.unique(torch.cat([S[k1], S[k2]]))
        yh[sel] = mlp_part(u[sel], Wg[:, idx], Wu[:, idx], Wd[idx], snw[idx], max(cov[k1].item(), cov[k2].item()))
    return yh


def train_router(K, asg, steps=400):
    """linear classifier u -> cluster (K x d MACs per token)."""
    Wr = torch.zeros(d, K, device=dev, requires_grad=True); br = torch.zeros(K, device=dev, requires_grad=True)
    opt = torch.optim.Adam([Wr, br], lr=3e-3)
    X = U[:nc]; mu, sd = X.float().mean(0), X.float().std(0) + 1e-6
    for s in range(steps):
        i = torch.randint(nc, (16384,), device=dev)
        loss = F.cross_entropy(((X[i].float() - mu) / sd) @ Wr + br, asg[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda u: ((u.float() - mu) / sd) @ Wr.detach() + br.detach()


Ks = [int(x) for x in a.Ks.split(",")]; ms = [int(x) for x in a.ms.split(",")]
dense_macs = 3 * Fn * d
CL = {}
for K in Ks:
    with torch.no_grad():
        C, asg = kmeans(An, K) if K > 1 else (None, torch.zeros(nc, dtype=torch.long, device=dev))
        asg_or = (Atn @ C.half().T).argmax(-1) if K > 1 else torch.zeros(Nt, dtype=torch.long, device=dev)
    router = train_router(K, asg) if K > 1 else (lambda u: torch.zeros(len(u), 1, device=dev))
    with torch.no_grad():
        logits = torch.cat([router(Ut[i:i + 8192]) for i in range(0, Nt, 8192)])
        asg_r = logits.argmax(-1); top2 = logits.topk(min(2, K), -1).indices
        acc = (asg_r == asg_or).float().mean().item()
        for m in ms:
            SC = subsets(asg, K, m)
            rec = {"phase": 11, "stage": 3, "K": K, "m": m, "router_acc": R(acc, 4),
                   "macs": K * d + 3 * m * d, "mlp_speedup_macs": R(dense_macs / (K * d + 3 * m * d), 2),
                   "mean_coverage": R(SC[1].mean().item(), 4),
                   "rel_err_ceiling": R(rel(routed_eval(Ut, asg_or, SC))),
                   "rel_err_routed": R(rel(routed_eval(Ut, asg_r, SC)))}
            if K > 1:
                rec["rel_err_routed_top2"] = R(rel(union_eval(Ut, top2, SC)))
            log(a.out, rec)
            CL[(K, m)] = (router, SC)

# ---------------------------------------------------------------- single-layer KL
X_eval = [("wiki", wiki_test[:a.eval_wiki]), ("chat", chat_test[:a.eval_chat])]


def kl_rec(rec):
    for name, X in X_eval:
        r = evaluate(model, X, a.bs, kl=True, device=dev)
        rec.update({f"{name}_ppl": R(r["ppl"], 4), f"{name}_kl": R(r["kl"]), f"{name}_top1": R(r["top1_agree"], 4)})
    return rec


class Routed(torch.nn.Module):
    def __init__(self, router, leaves):
        super().__init__(); self.router, self.leaves = router, leaves     # leaves: list of (gi, ui, di, sw)
    def forward(self, u):
        sh = u.shape; x = u.reshape(-1, d)
        k = self.router(x).argmax(-1)
        y = torch.empty(len(x), d, device=dev)
        for kk in k.unique().tolist():
            sel = (k == kk).nonzero().squeeze(1)
            y[sel] = mlp_part(x[sel], *self.leaves[kk])
        return y.reshape(*sh[:-1], d).to(u.dtype)


def exact_leaves(SC):
    S, cov = SC
    return [(Wg[:, S[k]], Wu[:, S[k]], Wd[S[k]], snw[S[k]], cov[k].item()) for k in range(len(S))]


STATE.mlps = {L: lambda u: Y.float().mean(0).to(u.dtype).expand(*u.shape[:-1], d)}
log(a.out, kl_rec({"phase": 11, "stage": "kl", "config": "MLP removed (mean output)"}))
for cfg in filter(None, a.kl_cfgs.split(",")):
    K, m = map(int, cfg.split(":"))
    if (K, m) not in CL: continue
    router, SC = CL[(K, m)]
    STATE.mlps = {L: Routed(router, exact_leaves(SC))}
    log(a.out, kl_rec({"phase": 11, "stage": "kl", "config": f"routed exact subsets K={K} m={m}"}))

# ---------------------------------------------------------------- stage 4: distillation of the routed leaves
for cfg in filter(None, a.distill.split(",")):
    K, m = map(int, cfg.split(":"))
    if (K, m) not in CL: continue
    router, (S, cov) = CL[(K, m)]
    t1 = time.time()
    P = [torch.nn.Parameter(t.clone()) for t in (Wg[:, S].permute(1, 0, 2).contiguous(), Wu[:, S].permute(1, 0, 2).contiguous(),
                                                 Wd[S].contiguous(), snw[S].contiguous())]   # (K,d,m) (K,d,m) (K,m,d) (K,m)
    opt = torch.optim.Adam(P, lr=a.lr)
    with torch.no_grad():
        asg_all = torch.cat([router(U[i:i + 8192]).argmax(-1) for i in range(0, N, 8192)])
    ysq = Y.float().pow(2).sum(-1).mean()
    B = 16384
    for s in range(a.steps):
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * s / a.steps)); [g.update(lr=lr) for g in opt.param_groups]
        i = torch.randint(N, (B,), device=dev); k = asg_all[i]
        loss = 0.0
        for kk in k.unique().tolist():
            sel = i[k == kk]
            x = U[sel].float()
            h = F.relu(x @ P[0][kk]).pow(2) * (x @ P[1][kk])
            hn = h * torch.rsqrt(h.pow(2).sum(-1, keepdim=True) / (Fn * cov[kk]) + eps) * P[3][kk]
            loss = loss + (hn @ P[2][kk] - Y[sel].float()).pow(2).sum()
        loss = loss / (B * ysq)
        opt.zero_grad(); loss.backward(); opt.step()
    leaves = [(P[0][k].detach(), P[1][k].detach(), P[2][k].detach(), P[3][k].detach(), cov[k].item()) for k in range(K)]
    with torch.no_grad():
        logits = torch.cat([router(Ut[i:i + 8192]) for i in range(0, Nt, 8192)]).argmax(-1)
        yh = torch.empty(Nt, d, device=dev)
        for kk in logits.unique().tolist():
            sel = (logits == kk).nonzero().squeeze(1)
            yh[sel] = mlp_part(Ut[sel], *leaves[kk])
    STATE.mlps = {L: Routed(router, leaves)}
    log(a.out, kl_rec({"phase": 11, "stage": 4, "config": f"distilled K={K} m={m}", "steps": a.steps,
                       "rel_err": R(rel(yh)), "macs": K * d + 3 * m * d,
                       "MB_int8_if_leaves_duplicated": R(K * 3 * m * d / 2 ** 20, 1), "train_s": R(time.time() - t1, 1)}))
STATE.mlps = {}
log(a.out, {"event": "done", "total_s": R(time.time() - t0, 1)})
