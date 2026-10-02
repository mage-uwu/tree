"""Phase 12: healing. Replace ALL 30 MLPs with cheap students bootstrapped from the teacher's own neurons, then train
only the students end to end so the full model matches the original's next-token distribution (KL to teacher logits).
Attention, embeddings and norms stay frozen (HF BitLinear uses straight-through estimators, so gradients flow).

Student per layer (bootstrapped as in phase 11): K clusters of tokens by which neurons carry their output, a linear
router u -> cluster, and per cluster a gated relu^2 MLP of m neurons initialised with the exact teacher rows of that
cluster's top-m neurons (sub-norm RMS corrected by the cluster's calibration coverage). K=1 is a narrow MLP holding
the globally most important m neurons. Per-token MLP MACs: K*d + 3*m*d (dense: 3*6912*d).
--ternary trains the leaves with BitNet absmean ternary weights (straight-through), so they stay CPU-ternary.
"""
import argparse, math, time, torch
import torch.nn as nn
import torch.nn.functional as F
from common import STATE, load_model, wikitext, ultrachat, windows, evaluate, log

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase_heal.jsonl")
ap.add_argument("--K", type=int, default=1)
ap.add_argument("--m", type=int, default=1024)
ap.add_argument("--ternary", type=int, default=0)
ap.add_argument("--calib_windows", type=int, default=32)       # 2048-token windows for clustering/subsets (all layers)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--T", type=int, default=1024)
ap.add_argument("--B", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--eval_wiki", type=int, default=8)
ap.add_argument("--eval_chat", type=int, default=4)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
dev = a.device
torch.manual_seed(0)
t0 = time.time()
R = lambda x, n=5: round(float(x), n)
tag = f"K={a.K} m={a.m}" + (" ternary" if a.ternary else "")


def retry(fn, n=5):
    for i in range(n):
        try:
            return fn()
        except Exception as e:                                   # flaky pod network to the HF hub
            if i == n - 1: raise
            print("retry", i, repr(e)[:200], flush=True); time.sleep(10 * (i + 1))


model, tok = retry(lambda: load_model(dev))
cfg = model.config; d, Fn, NL = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
wiki_tr = retry(lambda: wikitext("train")); wiki_te = retry(lambda: wikitext("test"))
chat_tr = retry(lambda: ultrachat("train_sft", tok, 12000)); chat_te = retry(lambda: ultrachat("test_sft", tok, 400))
calib = windows(tok, wiki_tr, 2048, a.calib_windows)
train = torch.cat([windows(tok, wiki_tr, a.T), windows(tok, chat_tr, a.T)])
train = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(1))]
X_eval = [("wiki", windows(tok, wiki_te, 2048)[:a.eval_wiki]), ("chat", windows(tok, chat_te, 2048, 64)[:a.eval_chat])]
log(a.out, {"phase": 12, "event": "data", "arm": tag, "train_windows": len(train), "train_tokens": train.numel()})

# ---------------------------------------------------------------- capture MLP inputs of every layer (base model)
Ucap = [[] for _ in range(NL)]
STATE.mlp_capture = lambda l, u, y: Ucap[l].append(u.reshape(-1, d).to(torch.bfloat16).cpu())
STATE.enabled = False
with torch.no_grad():
    for i in range(0, len(calib), 2):
        model.model(calib[i:i + 2].to(dev))
STATE.mlp_capture = None


def ternarize(w, step):
    """ternary weights {-step, 0, +step} with a straight-through gradient. step is fixed at the teacher's own scale,
    so the bootstrapped leaves start exactly at the teacher's ternary rows; sw (learnable) absorbs rescaling."""
    return w + ((w / step).round().clamp(-1, 1) * step - w).detach()


class Student(nn.Module):
    def __init__(self, K, m, G, Uw, D, sw, cov, Wr, br, mu, sd, ternary):
        super().__init__()
        self.K, self.m, self.ternary = K, m, ternary
        self.G, self.U, self.D, self.sw = (nn.Parameter(t.contiguous()) for t in (G, Uw, D, sw))   # (K,d,m)(K,d,m)(K,m,d)(K,m)
        self.register_buffer("cov", cov); self.register_buffer("Wr", Wr); self.register_buffer("br", br)
        self.register_buffer("mu", mu); self.register_buffer("sd", sd)
        self.register_buffer("steps", torch.stack([G.abs().amax((1, 2)), Uw.abs().amax((1, 2)), D.abs().amax((1, 2))], 1))

    def forward(self, u):
        sh = u.shape; x = u.reshape(-1, sh[-1]).float()
        k = (((x - self.mu) / self.sd) @ self.Wr + self.br).argmax(-1) if self.K > 1 else None
        y = torch.zeros_like(x)
        for kk in range(self.K):
            sel = (k == kk).nonzero().squeeze(1) if self.K > 1 else None
            xs = x if sel is None else x[sel]
            if len(xs) == 0: continue
            G, U, D = self.G[kk], self.U[kk], self.D[kk]
            if self.ternary:
                G, U, D = ternarize(G, self.steps[kk, 0]), ternarize(U, self.steps[kk, 1]), ternarize(D, self.steps[kk, 2])
            h = F.relu(xs @ G).pow(2) * (xs @ U)
            hn = h * torch.rsqrt(h.pow(2).sum(-1, keepdim=True) / (Fn * self.cov[kk]) + 1e-5) * self.sw[kk]
            out = hn @ D
            y = out if sel is None else y.index_copy(0, sel, out)
        return y.reshape(sh).to(u.dtype)


def kmeans(X, K, iters=25):
    g = torch.Generator(device=dev).manual_seed(0)
    C = X[torch.randperm(len(X), generator=g, device=dev)[:K]].float()
    for _ in range(iters):
        asg = torch.cat([(X[i:i + 16384].float() @ C.T).argmax(-1) for i in range(0, len(X), 16384)])
        Cn = torch.zeros_like(C).index_add_(0, asg, X.float())
        dead = torch.bincount(asg, minlength=K) == 0
        if dead.any(): Cn[dead] = X[torch.randint(len(X), (int(dead.sum()),), device=dev, generator=g)].float()
        C = F.normalize(Cn, dim=-1)
    return torch.cat([(X[i:i + 16384].float() @ C.T).argmax(-1) for i in range(0, len(X), 16384)])


def build(L):
    mlp = model.model.layers[L].mlp
    with torch.no_grad():
        def probe(lin, n):
            return torch.cat([lin(torch.eye(n, device=dev, dtype=torch.bfloat16)[i:i + 1024]).float() for i in range(0, n, 1024)])
        Wg, Wu, Wd = probe(mlp.gate_proj, d), probe(mlp.up_proj, d), probe(mlp.down_proj, Fn)
        snw = mlp.ffn_sub_norm.weight.float(); dnorm = Wd.norm(dim=1)
        U = torch.cat(Ucap[L]).to(dev); N = len(U)
        A = []                                                   # |hn| per token (fp16)
        for i in range(0, N, 8192):
            x = U[i:i + 8192].float(); h = F.relu(x @ Wg).pow(2) * (x @ Wu)
            A.append((h * torch.rsqrt(h.pow(2).sum(-1, keepdim=True) / Fn + 1e-5) * snw).abs().half())
        A = torch.cat(A)
        if a.K > 1:
            An = torch.cat([F.normalize(A[i:i + 16384].float() * dnorm, dim=-1).half() for i in range(0, N, 16384)])
            asg = kmeans(An, a.K); del An
        else:
            asg = torch.zeros(N, dtype=torch.long, device=dev)
        E = torch.zeros(a.K, Fn, device=dev)
        for i in range(0, N, 16384):
            E.index_add_(0, asg[i:i + 16384], (A[i:i + 16384].float() * dnorm).pow(2))
        S = E.topk(a.m, -1).indices                              # (K, m)
        cov = torch.zeros(a.K, device=dev); cnt = torch.zeros(a.K, device=dev)
        for i in range(0, N, 16384):
            e = A[i:i + 16384].float().pow(2); k = asg[i:i + 16384]
            cov.index_add_(0, k, e.gather(-1, S[k]).sum(-1) / e.sum(-1).clamp_min(1e-12)); cnt.index_add_(0, k, torch.ones(len(k), device=dev))
        cov = (cov / cnt.clamp_min(1)).clamp(0.05, 1.0)
        mu, sd = U.float().mean(0), U.float().std(0) + 1e-6
    Wr = torch.zeros(d, a.K, device=dev); br = torch.zeros(a.K, device=dev)
    if a.K > 1:                                                  # linear router u -> cluster
        Wr.requires_grad_(True); br.requires_grad_(True)
        opt = torch.optim.Adam([Wr, br], lr=3e-3)
        for _ in range(300):
            i = torch.randint(N, (16384,), device=dev)
            loss = F.cross_entropy(((U[i].float() - mu) / sd) @ Wr + br, asg[i])
            opt.zero_grad(); loss.backward(); opt.step()
        Wr, br = Wr.detach(), br.detach()
        with torch.no_grad():
            acc = ((((U.float() - mu) / sd) @ Wr + br).argmax(-1) == asg).float().mean().item()
    else:
        acc = 1.0
    st = Student(a.K, a.m, Wg[:, S].permute(1, 0, 2), Wu[:, S].permute(1, 0, 2), Wd[S], snw[S], cov, Wr, br, mu, sd, a.ternary).to(dev)
    del U, A, Wg, Wu, Wd
    return st, acc, cov.mean().item()


students = {}
for L in range(NL):
    st, acc, cv = build(L)
    students[L] = st; Ucap[L] = None
    log(a.out, {"phase": 12, "event": "built", "arm": tag, "layer": L, "router_acc": R(acc, 4), "coverage": R(cv, 4)})
torch.cuda.empty_cache() if dev == "cuda" else None
for L in range(NL):                                              # student replaces mlp.forward (no wasted teacher MLP)
    mlp = model.model.layers[L].mlp
    mlp.forward = (lambda orig, st: (lambda u: st(u) if STATE.enabled else orig(u)))(mlp.forward, students[L])
params = [p for st in students.values() for p in st.parameters()]
n_params = sum(p.numel() for p in params)
macs = a.K * d + 3 * a.m * d


def evals(step, extra=None):
    rec = {"phase": 12, "arm": tag, "step": step, "student_params_M": R(n_params / 1e6, 1), "mlp_macs": macs,
           "mlp_fewer_macs": R(3 * Fn * d / macs, 2), **(extra or {})}
    with torch.no_grad():
        for name, X in X_eval:
            r = evaluate(model, X, 1, kl=True, device=dev)
            rec.update({f"{name}_ppl": R(r["ppl"], 4), f"{name}_kl": R(r["kl"]), f"{name}_top1": R(r["top1_agree"], 4)})
    log(a.out, rec)


STATE.enabled = False
with torch.no_grad():
    for name, X in X_eval:
        r = evaluate(model, X, 1, kl=False, device=dev)
        log(a.out, {"phase": 12, "event": "base", "data": name, "ppl": R(r["ppl"], 4)})
STATE.enabled = True
evals(0)

opt = torch.optim.Adam(params, lr=a.lr)
run = 0.0; nb = 0
for step in range(1, a.steps + 1):
    lr = a.lr * min(1.0, step / a.warmup) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
    for g in opt.param_groups: g["lr"] = lr
    x = train[((step - 1) * a.B) % len(train):][:a.B].to(dev)
    with torch.no_grad():
        STATE.enabled = False
        lt = model(x).logits[:, :-1]
        STATE.enabled = True
    ls = model(x).logits[:, :-1]
    loss = 0.0
    for j in range(0, ls.shape[1], 256):
        lpt = F.log_softmax(lt[:, j:j + 256].float(), -1)
        lps = F.log_softmax(ls[:, j:j + 256].float(), -1)
        loss = loss + (lpt.exp() * (lpt - lps)).sum()
    loss = loss / (ls.shape[0] * ls.shape[1])
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    run += loss.item(); nb += 1
    del lt, ls, loss
    if step % a.eval_every == 0 or step == a.steps:
        evals(step, {"train_kl": R(run / nb), "tokens_seen": step * a.B * a.T, "elapsed_s": R(time.time() - t0, 1)})
        run = 0.0; nb = 0
log(a.out, {"event": "done", "arm": tag, "total_s": R(time.time() - t0, 1)})
