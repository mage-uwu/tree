"""Phase 15 step 1: FFF-style tree router over one layer's own exact BitNet neurons (CPU, single layer).
Neurons are grouped into L leaves of B neurons by recursive balanced bisection of their co-activation profiles (which
tokens they carry energy on); a soft binary tree (one hyperplane per internal node, leaf log-prob = sum of log-sigmoids
along the path) is trained on the teacher's leaf energy shares (dense labels, listwise CE). A token computes the top-m
leaves' neurons exactly (k = m*B), or with rescore takes 2k candidate neurons and keeps the top k by exact gate.
Compared at equal k with: the Phase-13 low-rank selector (SVD init, trained the same way, per-neuron), a flat leaf router
(one linear score per leaf, no tree), and the leaf oracle (true best leaves; the cost of block granularity alone).
Metric: MLP output relative error on held-out tokens (coverage-corrected sub-norm, as the engine runs it) + energy recall."""
import argparse, json, math, time, torch
import torch.nn.functional as F
from common import STATE, load_model, wikitext, windows

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--train_windows", type=int, default=64)
ap.add_argument("--test_windows", type=int, default=8)
ap.add_argument("--T", type=int, default=512)
ap.add_argument("--depths", default="6,8,10")                   # 64 / 256 / 1024 leaves (~108 / 27 / 6.75 neurons)
ap.add_argument("--ks", default="1024,1536")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--bs", type=int, default=1024)
ap.add_argument("--r", type=int, default=256)
ap.add_argument("--out", default="runs/phase_ftree.jsonl")
a = ap.parse_args()
torch.manual_seed(0)
t0 = time.time()
L_ = a.layer
model, tok = load_model("cpu")
d, Fn = model.config.hidden_size, model.config.intermediate_size


def log(rec):
    rec = dict(rec, s=round(time.time() - t0)); print(json.dumps(rec), flush=True)
    open(a.out, "a").write(json.dumps(rec) + "\n")


class Stop(Exception):
    pass


def capture(X):
    Us = []
    def cap(l, u, y):
        if l == L_: Us.append(u.reshape(-1, d).float()); raise Stop
    STATE.mlp_capture = cap; STATE.enabled = False
    with torch.no_grad():
        for i in range(len(X)):
            try: model.model(X[i:i + 1])
            except Stop: pass
    STATE.mlp_capture = None
    return torch.cat(Us)


Utr = capture(windows(tok, wikitext("train"), a.T)[:a.train_windows])
Ute = capture(windows(tok, wikitext("test"), a.T)[:a.test_windows])
log({"phase": 15, "event": "captured", "train_tokens": len(Utr), "test_tokens": len(Ute)})

mlp = model.model.layers[L_].mlp
with torch.no_grad():
    probe = lambda lin, n: torch.cat([lin(torch.eye(n, dtype=torch.bfloat16)[i:i + 512]).float() for i in range(0, n, 512)])
    Wg, Wu, Wd = probe(mlp.gate_proj, d), probe(mlp.up_proj, d), probe(mlp.down_proj, Fn)
    snw = mlp.ffn_sub_norm.weight.float(); dn = Wd.norm(dim=1)
del model


def teacher(U):
    G = U @ Wg; h = F.relu(G).pow(2) * (U @ Wu)
    hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) * snw
    return G, h, hn @ Wd, (hn * dn).pow(2)


with torch.no_grad():
    Etr = torch.cat([teacher(Utr[i:i + 4096])[3] for i in range(0, len(Utr), 4096)])
    Gte, Hte, Yte, Ete = teacher(Ute)


def assess(mask):
    """rel err of the MLP output computed on the masked neurons (coverage-corrected RMS), and energy recall."""
    hs = Hte * mask
    c = (hs.pow(2).sum(-1) / Hte.pow(2).sum(-1).clamp_min(1e-12)).mean()
    y = (hs * torch.rsqrt(hs.pow(2).sum(-1, keepdim=True) / (Fn * c) + 1e-5) * snw) @ Wd
    rec = ((Ete * mask).sum(-1) / Ete.sum(-1).clamp_min(1e-12)).mean().item()
    return round(((y - Yte).pow(2).sum() / Yte.pow(2).sum()).item(), 5), round(rec, 4)


def mask_from(idx):
    """idx may contain -1 padding (ragged leaves)."""
    return torch.zeros(len(Ute), Fn + 1).scatter_(-1, torch.where(idx < 0, Fn, idx), 1.0)[:, :Fn]


Gpad = None
def rescore(cand, k):
    global Gpad
    if Gpad is None: Gpad = torch.cat([Gte, torch.full((len(Gte), 1), -float("inf"))], 1)
    return cand.gather(-1, Gpad.gather(-1, torch.where(cand < 0, Fn, cand)).topk(k, -1).indices)


def train(params, loss_fn):
    opt = torch.optim.Adam(params, lr=1e-3)
    for step in range(a.steps):
        for g in opt.param_groups: g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        i = torch.randint(0, len(Utr), (a.bs,))
        loss = loss_fn(Utr[i], Etr[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


ks = [int(x) for x in a.ks.split(",")]
# ---- reference: low-rank per-neuron selector (Phase 13 "lr")
Us_, S_, Vh_ = torch.linalg.svd(Wg, full_matrices=False)
A = (Us_[:, :a.r] * S_[:a.r].sqrt()).clone().requires_grad_(True)
Bm = (S_[:a.r, None].sqrt() * Vh_[:a.r]).clone().requires_grad_(True)
ce = lambda logits, E: -(E / E.sum(-1, keepdim=True).clamp_min(1e-12) * F.log_softmax(logits, -1)).sum(-1).mean()
fl = train([A, Bm], lambda U, E: ce((U @ A) @ Bm, E))
with torch.no_grad():
    S = (Ute @ A) @ Bm
    for k in ks:
        for rs in (0, 1):
            idx = rescore(S.topk(2 * k, -1).indices, k) if rs else S.topk(k, -1).indices
            err, rec = assess(mask_from(idx))
            log({"phase": 15, "router": "lowrank", "r": a.r, "k": k, "rescore": rs, "rel_err": err, "recall": rec,
                 "router_macs": (d + Fn) * a.r, "final_loss": round(fl, 4)})
    for k in ks:
        err, rec = assess(mask_from(Ete.topk(k, -1).indices))
        log({"phase": 15, "router": "neuron_oracle", "k": k, "rel_err": err, "recall": rec})


def bisect(P, idx, depth):
    """balanced recursive bisection of neurons idx by the top principal direction of their profiles P[idx]."""
    if depth == 0: return [idx]
    X = P[idx] - P[idx].mean(0)
    v = torch.linalg.svd(X[:, :], full_matrices=False)[2][0] if len(idx) <= 4096 else torch.pca_lowrank(X, q=1, center=False)[2][:, 0]
    o = (X @ v).argsort()
    h = len(idx) // 2
    return bisect(P, idx[o[:h]], depth - 1) + bisect(P, idx[o[h:]], depth - 1)


# co-activation profile of each neuron: sqrt of its energy share per token (a random 8k-token subsample)
sub = torch.randperm(len(Utr))[:8192]
P = (Etr[sub] / Etr[sub].sum(-1, keepdim=True)).sqrt().T.contiguous()        # (F, tokens)
P = P / P.norm(dim=1, keepdim=True).clamp_min(1e-12)

for D in [int(x) for x in a.depths.split(",")]:
    Lv = 2 ** D; B = Fn / Lv
    leaves = bisect(P, torch.arange(Fn), D)                                   # Lv index tensors of floor/ceil(F/Lv)
    leaf_of = torch.empty(Fn, dtype=torch.long)
    for j, lf in enumerate(leaves): leaf_of[lf] = j
    mB = max(len(lf) for lf in leaves); members = torch.full((Lv, mB), -1, dtype=torch.long)
    for j, lf in enumerate(leaves): members[j, :len(lf)] = lf
    blk = lambda E: torch.zeros(len(E), Lv).index_add_(1, leaf_of, E)         # leaf energy (sum over its neurons)
    # path matrices: leaf j at depth D descends node n (heap order) left/right
    Nn = Lv - 1; Pos = torch.zeros(Lv, Nn); Neg = torch.zeros(Lv, Nn)
    for j in range(Lv):
        n = 0
        for lvl in range(D):
            bit = (j >> (D - 1 - lvl)) & 1
            (Pos if bit else Neg)[j, n] = 1.0; n = 2 * n + 1 + bit
    Wt = torch.zeros(d, Nn, requires_grad=True); bt = torch.zeros(Nn, requires_grad=True)
    tree_logp = lambda U: F.logsigmoid(U @ Wt + bt) @ Pos.T + F.logsigmoid(-(U @ Wt + bt)) @ Neg.T
    blk_ce = lambda logp, E: -(blk(E) / E.sum(-1, keepdim=True).clamp_min(1e-12) * logp).sum(-1).mean()
    fl_t = train([Wt, bt], lambda U, E: blk_ce(tree_logp(U), E))
    Wf = torch.zeros(d, Lv, requires_grad=True); bf = torch.zeros(Lv, requires_grad=True)
    fl_f = train([Wf, bf], lambda U, E: blk_ce(F.log_softmax(U @ Wf + bf, -1), E))
    with torch.no_grad():
        scores = {"tree": tree_logp(Ute), "flat": Ute @ Wf + bf, "leaf_oracle": blk(Ete)}
        for k in ks:
            for name, sc in scores.items():
                for rs in ((0, 1) if name != "leaf_oracle" else (0,)):
                    m = int(round((2 * k if rs else k) / B))
                    cand = members[sc.topk(m, -1).indices].reshape(len(Ute), -1)
                    idx = rescore(cand, min(k, cand.shape[1])) if rs else cand
                    err, rec = assess(mask_from(idx))
                    log({"phase": 15, "router": name, "B": round(B, 2), "leaves": Lv, "depth": D, "k": k, "neurons_used": round((idx >= 0).sum(-1).float().mean().item()),
                         "rescore": rs, "rel_err": err, "recall": rec,
                         "router_macs": {"tree": Nn * d, "flat": Lv * d}.get(name, 0),
                         "router_macs_beam": D * d if name == "tree" else None,
                         "final_loss": round({"tree": fl_t, "flat": fl_f}.get(name, 0.0), 4)})
log({"phase": 15, "event": "done"})
