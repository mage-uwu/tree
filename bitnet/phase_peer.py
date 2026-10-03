"""Phase 13: PEER-style neuron selection initialised from BitNet's own gate, trained against the teacher's exact
per-neuron energies, then healed end to end.

Trick: keep WHAT is computed exact (the real BitNet neurons), learn only WHICH neurons to compute, and supervise
that choice neuron by neuron: for every token the teacher's exact gate gives the energy of all 6912 neurons
(e_i = (|hn_i| * |down_i|)^2), i.e. thousands of labels per token, so modest data suffices.

Selectors (all score every neuron, the top-k are computed exactly):
  svd   untrained: s = (u A) B, A B = rank-r SVD of the gate matrix (scores ~ gate pre-activations)
  lr    same form, trained (listwise KL to the energy distribution)                 cost r*d + r*F
  peer  product keys: q = u A (r dims), split in halves; neuron i lives in grid cell (a_i, b_i) found by product
        quantisation of its SVD gate embedding; s_i = q1.K1[a_i] + q2.K2[b_i]; A, K1, K2 trained   cost r*d + ~F adds
The sub-norm RMS of the selected neurons is corrected by the layer's mean energy coverage c (calibration).

Stage A: chosen layers alone (recall, output rel err, single-layer KL).  Stage B: all 30 layers, whole-model KL,
then healing: the selected neurons' weights (init = teacher) are trained end to end on KL to the teacher logits.
"""
import argparse, math, time, torch
import torch.nn.functional as F
from common import STATE, load_model, wikitext, ultrachat, windows, evaluate, log

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase_peer.jsonl")
ap.add_argument("--stageA_layers", default="15,2,25")
ap.add_argument("--selectors", default="svd,lr,peer")
ap.add_argument("--r_lr", type=int, default=256)
ap.add_argument("--r_peer", type=int, default=512)
ap.add_argument("--ks", default="512,1024,2048")
ap.add_argument("--calib_windows", type=int, default=96)        # 2048-token windows (~196k tokens)
ap.add_argument("--sel_steps", type=int, default=1500)
ap.add_argument("--sel_bs", type=int, default=2048)
ap.add_argument("--stageB", type=int, default=1)
ap.add_argument("--stageB_selector", default="peer")
ap.add_argument("--stageB_ks", default="1024,1536")
ap.add_argument("--heal_k", type=int, default=1024)
ap.add_argument("--kc_mult", type=int, default=2)                 # select+rescore: kc = kc_mult*k candidates, exact gate on them
ap.add_argument("--stageB_rescore", type=int, default=1)
ap.add_argument("--stageB_layers", type=int, default=0)          # 0 = all layers (smoke tests: first N only)
ap.add_argument("--heal_steps", type=int, default=1500)
ap.add_argument("--heal_lr", type=float, default=1e-4)
ap.add_argument("--heal_ternary", type=int, default=0)            # keep healed neuron weights ternary (STE, fixed step)
ap.add_argument("--T", type=int, default=1024)
ap.add_argument("--B", type=int, default=4)
ap.add_argument("--eval_wiki", type=int, default=8)
ap.add_argument("--eval_chat", type=int, default=4)
ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
dev = a.device
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
t0 = time.time()
R = lambda x, n=5: round(float(x), n)


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
calib = torch.cat([windows(tok, wiki_tr, 2048, a.calib_windows * 3 // 4), windows(tok, chat_tr, 2048, a.calib_windows // 4)])
calib = calib[torch.randperm(len(calib), generator=torch.Generator().manual_seed(0))]
held = torch.cat([windows(tok, wiki_te, 2048)[-6:], windows(tok, chat_te, 2048, 64)[-2:]])
X_eval = [("wiki", windows(tok, wiki_te, 2048)[:a.eval_wiki]), ("chat", windows(tok, chat_te, 2048, 64)[:a.eval_chat])]
log(a.out, {"phase": 13, "event": "data", "calib_tokens": calib.numel(), "held_tokens": held.numel()})


class Stop(Exception):
    pass


def capture(L, X, bs=2):
    Us = []
    def cap(l, u, y):
        if l == L:
            Us.append(u.reshape(-1, d).to(torch.bfloat16)); raise Stop
    STATE.mlp_capture = cap; en = STATE.enabled; STATE.enabled = False
    with torch.no_grad():
        for i in range(0, len(X), bs):
            try: model.model(X[i:i + bs].to(dev))
            except Stop: pass
    STATE.mlp_capture = None; STATE.enabled = en
    return torch.cat(Us)


def teacher_weights(L):
    mlp = model.model.layers[L].mlp
    with torch.no_grad():
        def probe(lin, n):
            return torch.cat([lin(torch.eye(n, device=dev, dtype=torch.bfloat16)[i:i + 1024]).float() for i in range(0, n, 1024)])
        Wg, Wu, Wd = probe(mlp.gate_proj, d), probe(mlp.up_proj, d), probe(mlp.down_proj, Fn)
    return Wg, Wu, Wd, mlp.ffn_sub_norm.weight.float().detach().clone()


def energies(x, W):
    """teacher per-neuron energy e (B,F), h (B,F) and exact output y (B,d) for MLP inputs x."""
    Wg, Wu, Wd, snw = W
    x = x.float(); h = F.relu(x @ Wg).pow(2) * (x @ Wu)
    hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) * snw
    return (hn * Wd.norm(dim=1)).pow(2), h, hn @ Wd


def sel_output(x, mask, W, c):
    """output with only the masked neurons computed; sub-norm RMS from the subset / coverage c."""
    Wg, Wu, Wd, snw = W
    x = x.float(); h = F.relu(x @ Wg).pow(2) * (x @ Wu) * mask
    hn = h * torch.rsqrt(h.pow(2).sum(-1, keepdim=True) / (Fn * c) + 1e-5) * snw
    return hn @ Wd


class Selector(torch.nn.Module):
    def __init__(self, kind, Wg, r):
        super().__init__()
        self.kind = kind
        U_, S_, Vh = torch.linalg.svd(Wg, full_matrices=False)        # Wg (d,F) = U S Vh
        A = U_[:, :r] * S_[:r].sqrt(); Z = (Vh[:r].T * S_[:r].sqrt())   # g ~ u A Z^T ; Z (F, r) neuron embeddings
        self.A = torch.nn.Parameter(A.contiguous())
        if kind in ("svd", "lr"):
            self.B = torch.nn.Parameter(Z.T.contiguous())
        else:                                                     # product keys over a ~sqrt(F) x sqrt(F) grid
            n1 = int(math.ceil(math.sqrt(Fn))); n2 = int(math.ceil(Fn / n1)); h_ = r // 2
            def km(X, K, it=30):
                C = X[torch.randperm(len(X), device=X.device)[:K]].clone()
                for _ in range(it):
                    asg = torch.cdist(X, C).argmin(-1)
                    Cn = torch.zeros_like(C).index_add_(0, asg, X); cnt = torch.bincount(asg, minlength=K)
                    C = torch.where(cnt[:, None] > 0, Cn / cnt.clamp_min(1)[:, None], C)
                return C, torch.cdist(X, C).argmin(-1)
            K1, a1 = km(Z[:, :h_], n1); K2, a2 = km(Z[:, h_:], n2)
            self.K1, self.K2 = torch.nn.Parameter(K1), torch.nn.Parameter(K2)
            self.register_buffer("a1", a1); self.register_buffer("a2", a2)
            self.cells = n1 * n2

    def forward(self, x):
        q = x.float() @ self.A
        if self.kind in ("svd", "lr"):
            return q @ self.B
        h_ = q.shape[-1] // 2
        s1 = q[:, :h_] @ self.K1.T; s2 = q[:, h_:] @ self.K2.T          # (B,n1) (B,n2): cost ~ sqrt(F) * r
        return s1[:, self.a1] + s2[:, self.a2]                            # per-neuron score: F gathers + adds

    def macs(self):
        r = self.A.shape[1]
        return r * d + (r * Fn if self.kind in ("svd", "lr") else (self.K1.shape[0] + self.K2.shape[0]) * r // 2)


def train_selector(sel, U, W, steps, bs):
    opt = torch.optim.Adam(sel.parameters(), lr=1e-3)
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * s / steps))
        i = torch.randint(len(U), (bs,), device=dev)
        with torch.no_grad():
            e = energies(U[i], W)[0]; p = e / e.sum(-1, keepdim=True).clamp_min(1e-12)
        loss = -(p * F.log_softmax(sel(U[i]), -1)).sum(-1).mean()       # listwise: CE(p, softmax(s))
        opt.zero_grad(); loss.backward(); opt.step()
    return sel


@torch.no_grad()
def choose(sel, x, W, k, kc=0):
    """top-k neurons by the selector, or (kc > 0) top-kc candidates by the selector rescored by their exact gate
    pre-activation (exact gate computed only for the kc candidates: kc*d MACs), keeping the top-k."""
    s = sel(x)
    if not kc:
        return s.topk(k, -1).indices
    cand = s.topk(kc, -1).indices
    gc = (x.float() @ W[0]).gather(-1, cand)                 # experiment: full gate then gather == gate rows of cand
    return cand.gather(-1, gc.topk(k, -1).indices)


@torch.no_grad()
def coverage(sel, U, W, k, n=16384, kc=0):
    x = U[:n]; e, h, _ = energies(x, W)
    m = torch.zeros_like(e).scatter_(-1, choose(sel, x, W, k, kc), 1.0)
    return ((h.pow(2) * m).sum(-1) / h.pow(2).sum(-1).clamp_min(1e-12)).mean().item()


@torch.no_grad()
def assess(sel, Ut, W, k, c, kc=0):
    """energy recall of the selected top-k and output rel err (exact neurons, coverage-corrected sub-norm)."""
    rec, num, den = 0.0, 0.0, 0.0
    for i in range(0, len(Ut), 4096):
        x = Ut[i:i + 4096]; e, h, y = energies(x, W)
        idx = choose(sel, x, W, k, kc) if sel is not None else e.topk(k, -1).indices
        m = torch.zeros_like(e).scatter_(-1, idx, 1.0)
        rec += ((e * m).sum(-1) / e.sum(-1).clamp_min(1e-12)).sum().item()
        num += (sel_output(x, m, W, c) - y).pow(2).sum().item(); den += y.pow(2).sum().item()
    return rec / len(Ut), num / den


def ternarize(w, step):
    """ternary {-step,0,+step} with a straight-through gradient; step fixed at the teacher's scale, so step 0 is exact."""
    return w + ((w / step).round().clamp(-1, 1) * step - w).detach()


class SelMLP(torch.nn.Module):
    """exact BitNet neurons, top-k chosen by the selector; optionally trainable neuron weights (healing)."""
    def __init__(self, sel, W, k, c, train_neurons=False, kc=0, ternary=False):
        super().__init__()
        self.sel, self.k, self.c, self.kc, self.ternary = sel, k, c, kc, ternary
        self.steps = [W[0].abs().max().item(), W[1].abs().max().item(), W[2].abs().max().item()]
        Wg, Wu, Wd, snw = W
        mk = torch.nn.Parameter if train_neurons else (lambda t: t)
        self.Wg, self.Wu, self.Wd, self.snw = mk(Wg.clone()), mk(Wu.clone()), mk(Wd.clone()), mk(snw.clone())
        if train_neurons:
            for p in self.sel.parameters(): p.requires_grad_(False)

    def forward(self, u):
        sh = u.shape; x = u.reshape(-1, d)
        Wg, Wu, Wd = self.Wg, self.Wu, self.Wd
        if self.ternary:
            Wg, Wu, Wd = ternarize(Wg, self.steps[0]), ternarize(Wu, self.steps[1]), ternarize(Wd, self.steps[2])
        with torch.no_grad():
            idx = choose(self.sel, x, (Wg.detach(), None, None, None), self.k, self.kc)
        m = torch.zeros(len(x), Fn, device=x.device).scatter_(-1, idx, 1.0)
        return sel_output(x, m, (Wg, Wu, Wd, self.snw), self.c).reshape(sh).to(u.dtype)


def kl_rec(rec):
    for name, X in X_eval:
        r = evaluate(model, X, 1, kl=True, device=dev)
        rec.update({f"{name}_ppl": R(r["ppl"], 4), f"{name}_kl": R(r["kl"]), f"{name}_top1": R(r["top1_agree"], 4)})
    return rec


def install(mods):
    """replace mlp.forward (no wasted teacher MLP when enabled)."""
    for L, mod in mods.items():
        mlp = model.model.layers[L].mlp
        if not hasattr(mlp, "_orig_forward"): mlp._orig_forward = mlp.forward
        mlp.forward = (lambda orig, st: (lambda u: st(u) if STATE.enabled else orig(u)))(mlp._orig_forward, mod)


dense_macs = 3 * Fn * d
ks = [int(x) for x in a.ks.split(",")]

# ---------------------------------------------------------------- stage A: single layers
for L in [int(x) for x in a.stageA_layers.split(",") if x]:
    U = capture(L, calib); Ut = capture(L, held); W = teacher_weights(L)
    for k in ks:
        rec, rel = assess(None, Ut, W, k, 1.0)
        log(a.out, {"phase": 13, "stage": "A", "layer": L, "selector": "oracle (needs full gate)", "k": k, "recall": R(rec, 4), "rel_err": R(rel)})
    best = None
    for kind in a.selectors.split(","):
        t1 = time.time()
        sel = Selector(kind, W[0], a.r_peer if kind == "peer" else a.r_lr).to(dev)
        if kind != "svd": train_selector(sel, U, W, a.sel_steps, a.sel_bs)
        for k in ks:
            c = coverage(sel, U, W, k)
            rec, rel = assess(sel, Ut, W, k, c)
            macs = sel.macs() + 3 * k * d
            log(a.out, {"phase": 13, "stage": "A", "layer": L, "selector": kind, "k": k, "recall": R(rec, 4), "rel_err": R(rel),
                        "coverage": R(c, 4), "selector_macs": sel.macs(), "mlp_fewer_macs": R(dense_macs / macs, 2), "train_s": R(time.time() - t1, 1)})
            kc = a.kc_mult * k
            c2 = coverage(sel, U, W, k, kc=kc)
            rec2, rel2 = assess(sel, Ut, W, k, c2, kc)
            macs2 = sel.macs() + kc * d + 2 * k * d
            log(a.out, {"phase": 13, "stage": "A", "layer": L, "selector": kind + "+rescore", "k": k, "kc": kc, "recall": R(rec2, 4),
                        "rel_err": R(rel2), "coverage": R(c2, 4), "mlp_fewer_macs": R(dense_macs / macs2, 2)})
            if k == 1024 and kind == a.stageB_selector: best = (sel, c2 if a.stageB_rescore else c)
    if best is not None and L == int(a.stageA_layers.split(",")[0]):
        install({L: SelMLP(best[0], W, 1024, best[1], kc=a.kc_mult * 1024 if a.stageB_rescore else 0)}); STATE.enabled = True
        log(a.out, kl_rec({"phase": 13, "stage": "A", "layer": L, "event": "single-layer KL", "selector": a.stageB_selector, "k": 1024, "rescore": a.stageB_rescore}))
        STATE.enabled = False; model.model.layers[L].mlp.forward = model.model.layers[L].mlp._orig_forward
    del U, Ut

if not a.stageB:
    log(a.out, {"event": "done", "total_s": R(time.time() - t0, 1)}); raise SystemExit

# ---------------------------------------------------------------- stage B: all layers, then healing
sels = {}
LB = range(a.stageB_layers or NL)
for L in LB:
    t1 = time.time()
    U = capture(L, calib); W = teacher_weights(L)
    sel = train_selector(Selector(a.stageB_selector, W[0], a.r_peer if a.stageB_selector == "peer" else a.r_lr).to(dev), U, W, a.sel_steps, a.sel_bs)
    for p in sel.parameters(): p.requires_grad_(False)
    KC = lambda k: a.kc_mult * k if a.stageB_rescore else 0
    cs = {k: coverage(sel, U, W, k, kc=KC(k)) for k in [int(x) for x in a.stageB_ks.split(",")] + [a.heal_k]}
    sels[L] = (sel, cs)
    log(a.out, {"phase": 13, "stage": "B", "event": "selector", "layer": L, "coverage": {str(k): R(v, 4) for k, v in cs.items()}, "s": R(time.time() - t1, 1)})
    del U
STATE.enabled = False
with torch.no_grad():
    for name, X in X_eval:
        r = evaluate(model, X, 1, kl=False, device=dev)
        log(a.out, {"phase": 13, "event": "base", "data": name, "ppl": R(r["ppl"], 4)})
for k in [int(x) for x in a.stageB_ks.split(",")]:
    install({L: SelMLP(sels[L][0], teacher_weights(L), k, sels[L][1][k], kc=KC(k)) for L in LB}); STATE.enabled = True
    m = sels[0][0].macs() + (KC(k) * d + 2 * k * d if KC(k) else 3 * k * d)
    log(a.out, kl_rec({"phase": 13, "stage": "B", "event": "all layers, no healing", "k": k, "mlp_fewer_macs": R(dense_macs / m, 2)}))
    STATE.enabled = False

# healing: selected neurons' weights trained end to end (selectors frozen)
k = a.heal_k
mods = {L: SelMLP(sels[L][0], teacher_weights(L), k, sels[L][1][k], train_neurons=True, kc=KC(k), ternary=bool(a.heal_ternary)) for L in LB}
install(mods); STATE.enabled = True
params = [p for mmod in mods.values() for p in (mmod.Wg, mmod.Wu, mmod.Wd, mmod.snw)]
train = torch.cat([windows(tok, wiki_tr, a.T), windows(tok, chat_tr, a.T)])
train = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(1))]
opt = torch.optim.Adam(params, lr=a.heal_lr)
mm = sels[0][0].macs() + (KC(k) * d + 2 * k * d if KC(k) else 3 * k * d)
log(a.out, kl_rec({"phase": 13, "stage": "heal", "k": k, "ternary": a.heal_ternary, "step": 0, "mlp_fewer_macs": R(dense_macs / mm, 2)}))
run = 0.0; nb = 0
for step in range(1, a.heal_steps + 1):
    for g in opt.param_groups: g["lr"] = a.heal_lr * min(1.0, step / 50) * 0.5 * (1 + math.cos(math.pi * step / a.heal_steps))
    x = train[((step - 1) * a.B) % len(train):][:a.B].to(dev)
    with torch.no_grad():
        STATE.enabled = False; lt = model(x).logits[:, :-1]; STATE.enabled = True
    ls = model(x).logits[:, :-1]
    loss = 0.0
    for j in range(0, ls.shape[1], 256):
        lpt = F.log_softmax(lt[:, j:j + 256].float(), -1); lps = F.log_softmax(ls[:, j:j + 256].float(), -1)
        loss = loss + (lpt.exp() * (lpt - lps)).sum()
    loss = loss / (ls.shape[0] * ls.shape[1])
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    run += loss.item(); nb += 1; del lt, ls, loss
    if step % a.eval_every == 0 or step == a.heal_steps:
        log(a.out, kl_rec({"phase": 13, "stage": "heal", "k": k, "ternary": a.heal_ternary, "step": step, "train_kl": R(run / nb), "tokens_seen": step * a.B * a.T, "elapsed_s": R(time.time() - t0, 1)}))
        run = 0.0; nb = 0
log(a.out, {"event": "done", "total_s": R(time.time() - t0, 1)})
