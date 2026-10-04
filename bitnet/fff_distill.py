"""Phase 17: FFF distillation, fast path. Every MLP of BitNet 2B4T becomes an FFF layer over its own exact neurons:
neurons are grouped into contiguous leaves of B_leaf (co-activation bisection, then permuted so leaf l = neurons
[l*B, (l+1)*B)); a router (low-rank flat, or soft binary tree) picks the top m leaves per token (k = m*B neurons); the
selected neurons run exactly. Trained end to end by KL to the frozen teacher (the same model with its original MLPs),
neuron weights kept ternary (STE, fixed step = teacher scale), router trained *jointly* on the student's own exact leaf
energies (dense labels, listwise CE), k annealed from k_start to k.

Fast path: ternary weights quantized once (HF re-quantizes the bf16 masters on every forward); dense masked MLP (fast
matmuls, no gathers) in training; long packed sequences streamed from FineWeb-Edu (+ UltraChat); periodic checkpoints in
the patch_gguf_mlp.py format (+ routers, permutations) and jsonl logs with tokens/s.

  python3 fff_distill.py --k 1024 --tokens 100e6 --save /root/out/fff        (GPU)
  python3 fff_distill.py --device cpu --layers 0,1 --data wiki --T 64 --mb 1 --steps 3 --calib_tokens 2048 ...  (smoke)
"""
import argparse, json, math, os, queue, random, threading, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.integrations.bitnet import ActQuant
from common import load_model, wikitext, windows

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=1024)                   # target neurons per token
ap.add_argument("--k_start", type=int, default=1536)             # annealed from here to --k
ap.add_argument("--anneal_frac", type=float, default=0.3)
ap.add_argument("--B_leaf", type=int, default=8)                 # neurons per leaf (divides 6912)
ap.add_argument("--router", default="flat", choices=["flat", "tree"])
ap.add_argument("--router_rank", type=int, default=256)
ap.add_argument("--router_init", default="svd", choices=["svd", "zero"])  # flat router: SVD of the gate, summed per leaf
ap.add_argument("--layers", default="")                          # "" = all
ap.add_argument("--tokens", type=float, default=100e6)
ap.add_argument("--steps", type=int, default=0)                  # overrides --tokens
ap.add_argument("--T", type=int, default=1024)
ap.add_argument("--mb", type=int, default=8)                     # sequences per micro-batch
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--router_lr", type=float, default=1e-3)
ap.add_argument("--aux", type=float, default=0.05)               # weight of router CE (per layer, summed)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--data", default="fineweb", choices=["fineweb", "wiki"])
ap.add_argument("--chat_frac", type=float, default=0.1)
ap.add_argument("--calib_tokens", type=int, default=65536)
ap.add_argument("--router_warm_steps", type=int, default=1500)
ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--eval_windows", type=int, default=16)
ap.add_argument("--save_every", type=int, default=500)
ap.add_argument("--save", default="")
ap.add_argument("--out", default="runs/fff_distill.jsonl")
ap.add_argument("--ckpt", type=int, default=0)                   # activation checkpointing per FFF layer
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
dev = a.device
t0 = time.time()


def rss_gb():
    try: return round(int([l for l in open("/proc/self/status") if l.startswith("VmRSS")][0].split()[1]) / 2**20, 2)
    except Exception: return None


def log(rec):
    rec = dict(rec, t=time.strftime("%H:%M:%S"), s=round(time.time() - t0, 1), rss_gb=rss_gb(),
               gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None)
    print(json.dumps(rec), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "a").write(json.dumps(rec) + "\n")


model, tok = load_model(dev)
cfg = model.config; d, Fn, NL, eps = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers, cfg.rms_norm_eps
LAYERS = [int(x) for x in a.layers.split(",")] if a.layers else list(range(NL))
assert Fn % a.B_leaf == 0
NLEAF = Fn // a.B_leaf


# ------------------------------------------------------------------ fast path 1: quantize every BitLinear once
@torch.no_grad()
def quantize_once(model):
    n = 0
    for m in model.modules():
        if getattr(m, "online_quant", False):
            w = m.weight.data.float(); s = 1.0 / w.abs().mean().clamp(min=1e-5)
            m.weight.data = ((w * s).round().clamp(-1, 1) / s).to(m.weight.dtype)
            m.online_quant = False
            m.weight_scale = torch.ones(1, dtype=m.weight.dtype, device=m.weight.device)
            n += 1
    return n


log({"phase": 17, "event": "quantized BitLinears once", "n": quantize_once(model), "layers": len(LAYERS), "leaves": NLEAF})


# ------------------------------------------------------------------ data
def text_stream():
    """endless documents: FineWeb-Edu sample-10BT (+ UltraChat with prob chat_frac), or wikitext train (smoke)."""
    if a.data == "wiki":
        txt = wikitext("train"); paras = [txt[i:i + 4000] for i in range(0, len(txt), 4000)]   # ~1k-token chunks
        while True:
            random.shuffle(paras)
            yield from paras
    from huggingface_hub import list_repo_files, hf_hub_download
    import pyarrow.parquet as pq
    fw = sorted(f for f in list_repo_files("HuggingFaceFW/fineweb-edu", repo_type="dataset") if f.startswith("sample/10BT/") and f.endswith(".parquet"))
    uc = sorted(f for f in list_repo_files("HuggingFaceH4/ultrachat_200k", repo_type="dataset") if f.startswith("data/train_sft") and f.endswith(".parquet"))
    chat = pq.read_table(hf_hub_download("HuggingFaceH4/ultrachat_200k", uc[0], repo_type="dataset")).to_pylist()
    chat = [tok.apply_chat_template(r["messages"], tokenize=False) for r in chat]
    for f in fw:
        pf = pq.ParquetFile(hf_hub_download("HuggingFaceFW/fineweb-edu", f, repo_type="dataset"))
        for batch in pf.iter_batches(batch_size=1024, columns=["text"]):
            for t in batch.column(0).to_pylist():
                if random.random() < a.chat_frac: yield random.choice(chat)
                yield t


def batches():
    """packed (mb, T) token batches, each sequence starts with BOS; documents separated by EOS. Background thread."""
    q = queue.Queue(maxsize=8)
    def work():
        buf = []; L = a.T - 1
        docs = text_stream(); pend = []
        while True:
            pend.append(next(docs))
            if len(pend) < 64: continue
            for ids in tok(pend, add_special_tokens=False)["input_ids"]:
                buf.extend(ids + [tok.eos_token_id])
            pend = []
            while len(buf) >= L * a.mb:
                x = torch.tensor([buf[i * L:(i + 1) * L] for i in range(a.mb)])
                buf = buf[L * a.mb:]
                q.put(torch.cat([torch.full((a.mb, 1), tok.bos_token_id), x], 1))
    threading.Thread(target=work, daemon=True).start()
    while True:
        yield q.get()


# ------------------------------------------------------------------ eval data
wiki_te = windows(tok, wikitext("test"), a.T)[:a.eval_windows]
X_eval = [("wiki", wiki_te)]
if a.data == "fineweb":
    from common import ultrachat
    X_eval.append(("chat", windows(tok, ultrachat("test_sft", tok, 400), a.T)[:max(4, a.eval_windows // 2)]))


# ------------------------------------------------------------------ the FFF layer
def ternarize(w, step):
    """ternary {-step,0,+step} with straight-through gradient; fixed step = teacher scale (step 0 is exact)."""
    return w + ((w / step).round().clamp(-1, 1) * step - w).detach()


class FlatRouter(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d, r) / math.sqrt(d)); self.B = nn.Parameter(torch.zeros(r, NLEAF))
        self.b = nn.Parameter(torch.zeros(NLEAF))
    def forward(self, x):                                            # log-probs over leaves
        return F.log_softmax((x @ self.A) @ self.B + self.b, -1)


class TreeRouter(nn.Module):
    def __init__(self):
        super().__init__()
        D = int(round(math.log2(NLEAF))); assert 2 ** D == NLEAF, "tree router needs 2^D leaves (e.g. --B_leaf 27)"
        self.W = nn.Parameter(torch.zeros(d, NLEAF - 1)); self.b = nn.Parameter(torch.zeros(NLEAF - 1))
        Pos = torch.zeros(NLEAF, NLEAF - 1); Neg = torch.zeros(NLEAF, NLEAF - 1)
        for j in range(NLEAF):
            n = 0
            for lvl in range(D):
                bit = (j >> (D - 1 - lvl)) & 1
                (Pos if bit else Neg)[j, n] = 1.0; n = 2 * n + 1 + bit
        self.register_buffer("Pos", Pos.T.contiguous()); self.register_buffer("Neg", Neg.T.contiguous())
    def forward(self, x):
        z = x @ self.W + self.b
        return F.logsigmoid(z) @ self.Pos + F.logsigmoid(-z) @ self.Neg


class FFFMLP(nn.Module):
    """exact (permuted) BitNet neurons, top-m leaves per token by the router; dense masked compute for training."""
    def __init__(self, Wg, Wu, Wd, snw, perm, router):
        super().__init__()
        self.register_buffer("perm", perm)
        self.steps = [Wg.abs().max().item(), Wu.abs().max().item(), Wd.abs().max().item()]
        self.wg = nn.Parameter(Wg[perm].float().clone()); self.wu = nn.Parameter(Wu[perm].float().clone())
        self.wd = nn.Parameter(Wd[:, perm].float().clone())         # (d, F)
        self.snw = nn.Parameter(snw[perm].float().clone())
        self.router = router
        self.m = NLEAF; self.aux = None
        self.register_buffer("cov", torch.ones(()))                  # running coverage of the selected neurons
    training_cov = True
    def weights(self, dt):
        return (ternarize(self.wg, self.steps[0]).to(dt), ternarize(self.wu, self.steps[1]).to(dt), ternarize(self.wd, self.steps[2]).to(dt))
    def forward(self, x):
        wg, wu, wd = self.weights(x.dtype)
        xq = ActQuant.apply(x)
        g = F.linear(xq, wg); h = F.relu(g).pow(2) * F.linear(xq, wu)                    # (..., F), all neurons
        lp = self.router(x.float())                                                      # (..., NLEAF) log-probs
        with torch.no_grad():                                                            # student's own exact leaf energies
            hf = h.float(); hn = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps) * self.snw
            e = (hn * wd.float().norm(dim=0)).pow(2).reshape(*h.shape[:-1], NLEAF, a.B_leaf).sum(-1)
            p = e / e.sum(-1, keepdim=True).clamp_min(1e-12)
            top = lp.topk(self.m, -1).indices
            mask = torch.zeros_like(lp, dtype=h.dtype).scatter_(-1, top, 1.0).repeat_interleave(a.B_leaf, -1)
        self.aux = -(p * lp).sum(-1).mean()
        hs = (h * mask).float()
        if self.training_cov:                                    # running coverage c (Phase 13: RMS = sum/(F c)); k anneals
            with torch.no_grad():
                c = (hs.pow(2).sum(-1) / hf.pow(2).sum(-1).clamp_min(1e-12)).mean()
                self.cov.mul_(0.95).add_(0.05 * c)
        # RMS over all F (engine-identical) with the coverage folded in; at export snw * sqrt(cov) -> GGUF ffn_sub_norm
        hn = (hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) / self.cov + eps) * self.snw).to(x.dtype)
        return F.linear(ActQuant.apply(hn), wd)


class Switch(nn.Module):
    """teacher = original MLP, student = FFF; STUDENT flag selects."""
    STUDENT = False
    def __init__(self, orig, fff):
        super().__init__(); self.orig = orig; self.fff = fff
    def forward(self, x):
        if not Switch.STUDENT: return self.orig(x)
        if a.ckpt and self.training_ckpt: return torch.utils.checkpoint.checkpoint(self.fff, x, use_reentrant=False)
        return self.fff(x)
    training_ckpt = True


# ------------------------------------------------------------------ init: calibration capture, leaves, router warm-up
def split_leaves(P, idx, n):
    """balanced split of neurons idx into n leaves of equal size by recursive top-PC bisection (odd n: PC chunking)."""
    if n == 1: return [idx]
    X = P[idx] - P[idx].mean(0)
    v = torch.linalg.svd(X, full_matrices=False)[2][0] if X.shape[0] <= 4096 else torch.pca_lowrank(X, q=2, center=False)[2][:, 0]
    o = idx[(X @ v).argsort()]
    if n % 2 == 0:
        h = len(idx) // 2
        return split_leaves(P, o[:h], n // 2) + split_leaves(P, o[h:], n // 2)
    return list(o.reshape(n, -1))


@torch.no_grad()
def capture_inputs(ntok):
    Us = {L: [] for L in LAYERS}; hooks = []
    for L in LAYERS:
        hooks.append(model.model.layers[L].mlp.register_forward_pre_hook(lambda m, args, L=L: Us[L].append(args[0].reshape(-1, d))))
    X = windows(tok, wikitext("train"), min(a.T, 1024))
    for i in range(0, max(1, ntok // X.shape[1])):
        model.model(X[i % len(X)][None].to(dev))
    [h.remove() for h in hooks]
    return {L: torch.cat(v) for L, v in Us.items()}


Ucal = capture_inputs(a.calib_tokens)
mods = {}
for L in LAYERS:
    mlp = model.model.layers[L].mlp
    Wg, Wu, Wd = mlp.gate_proj.weight.data, mlp.up_proj.weight.data, mlp.down_proj.weight.data   # (F,d), (F,d), (d,F)
    snw = mlp.ffn_sub_norm.weight.data
    U = Ucal[L]
    with torch.no_grad():
        Uq = ActQuant.apply(U)
        h = (F.relu(F.linear(Uq, Wg)).pow(2) * F.linear(Uq, Wu)).float()
        hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * snw.float()
        E = (hn * Wd.float().norm(dim=0)).pow(2)                                          # teacher neuron energies
        sub = torch.randperm(len(E), device=E.device)[:8192]
        P = (E[sub] / E[sub].sum(-1, keepdim=True).clamp_min(1e-12)).sqrt().T.contiguous()
        P = P / P.norm(dim=1, keepdim=True).clamp_min(1e-12)
        leaves = split_leaves(P, torch.arange(Fn, device=P.device), NLEAF)
        perm = torch.cat(leaves)
    router = (FlatRouter(a.router_rank) if a.router == "flat" else TreeRouter()).to(dev)
    if a.router == "flat" and a.router_init == "svd":                                  # g = x Wg^T ~ (x A) B, summed per leaf
        with torch.no_grad():
            Us, Ss, Vh = torch.linalg.svd(Wg.float().T, full_matrices=False)               # (d,F) = U S Vh
            r = a.router_rank
            router.A.copy_(Us[:, :r] * Ss[:r].sqrt())
            router.B.copy_((Ss[:r, None].sqrt() * Vh[:r])[:, perm].reshape(r, NLEAF, a.B_leaf).sum(-1))
            z = (U.float()[:4096] @ router.A) @ router.B                                # temperature: logit std ~2
            router.B.mul_(2.0 / z.std().clamp_min(1e-6))
    fff = FFFMLP(Wg, Wu, Wd, snw, perm, router).to(dev)
    # router warm-up on teacher leaf energies (dense labels), then fold coverage of the initial k into the sub-norm
    Ep = E[:, perm].reshape(len(E), NLEAF, a.B_leaf).sum(-1); Pl = Ep / Ep.sum(-1, keepdim=True).clamp_min(1e-12)
    opt = torch.optim.Adam(router.parameters(), lr=1e-3)
    Uf = U.float()
    for s in range(a.router_warm_steps):
        for gp in opt.param_groups: gp["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * s / a.router_warm_steps))
        i = torch.randint(0, len(Uf), (min(1024, len(Uf)),), device=Uf.device)
        loss = -(Pl[i] * router(Uf[i])).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        m0 = max(1, round(a.k_start / a.B_leaf))
        top = router(Uf).topk(m0, -1).indices
        msk = torch.zeros(len(Uf), NLEAF, device=Uf.device).scatter_(-1, top, 1.0).repeat_interleave(a.B_leaf, -1)
        hp = h[:, perm]
        c = ((hp * msk).pow(2).sum(-1) / hp.pow(2).sum(-1).clamp_min(1e-12)).mean().item()
        rec = ((E[:, perm] * msk).sum(-1) / E.sum(-1).clamp_min(1e-12)).mean().item()
        fff.cov.fill_(c)                                                                  # engine: folded into ffn_sub_norm at export
    fff.m = m0
    model.model.layers[L].mlp = Switch(mlp, fff)
    mods[L] = fff
    log({"phase": 17, "event": "layer init", "layer": L, "router_ce": round(loss.item(), 4), "coverage_c": round(c, 4), "recall_k_start": round(rec, 4)})
del Ucal

params_n = [p for f in mods.values() for p in (f.wg, f.wu, f.wd, f.snw)]
params_r = [p for f in mods.values() for p in f.router.parameters()]
for p in params_n + params_r: p.requires_grad_(True)
opt = torch.optim.AdamW([{"params": params_n, "lr": a.lr}, {"params": params_r, "lr": a.router_lr}], betas=(0.9, 0.95), weight_decay=0.0)
tok_per_step = a.mb * a.T * a.accum
steps = a.steps or max(1, int(a.tokens // tok_per_step))


def set_k(step):
    frac = min(1.0, step / max(1, a.anneal_frac * steps))
    k = a.k_start + (a.k - a.k_start) * frac
    for f in mods.values(): f.m = max(1, round(k / a.B_leaf))
    return round(k)


def kl_loss(lt, ls):
    tot = 0.0
    for j in range(0, ls.shape[1], 256):
        lpt = F.log_softmax(lt[:, j:j + 256].float(), -1); lps = F.log_softmax(ls[:, j:j + 256].float(), -1)
        tot = tot + (lpt.exp() * (lpt - lps)).sum()
    return tot / (ls.shape[0] * ls.shape[1])


@torch.no_grad()
def evaluate(step, k):
    FFFMLP.training_cov = False
    rec = {"phase": 17, "event": "eval", "step": step, "k": k, "tokens_seen": step * tok_per_step}
    for name, X in X_eval:
        nll = kls = top1 = n = 0.0
        for i in range(len(X)):
            x = X[i:i + 1].to(dev)
            Switch.STUDENT = False; lt = model(x).logits[:, :-1].float()
            Switch.STUDENT = True; ls = model(x).logits[:, :-1].float()
            y = x[:, 1:]; lpt = F.log_softmax(lt, -1); lps = F.log_softmax(ls, -1)
            nll += -lps.gather(-1, y[..., None]).sum().item(); kls += (lpt.exp() * (lpt - lps)).sum().item()
            top1 += (lt.argmax(-1) == ls.argmax(-1)).sum().item(); n += y.numel()
        rec.update({f"{name}_ppl": round(math.exp(nll / n), 4), f"{name}_kl": round(kls / n, 5), f"{name}_top1": round(top1 / n, 4)})
    Switch.STUDENT = False; FFFMLP.training_cov = True
    log(rec)


@torch.no_grad()
def save(step, k):
    if not a.save: return
    ck = {"k": k, "B_leaf": a.B_leaf, "router": a.router, "ternary": 1, "step": step, "layers": {}}
    for L, f in mods.items():
        ent = {"snw": f.snw.detach().float().cpu(), "c": float(f.cov), "perm": f.perm.cpu(),
               "router_state": {n: t.detach().float().cpu() for n, t in f.router.state_dict().items()}}
        # patch_gguf_mlp.py format: gate/up as (d,F) "in x out", down as (F,d); 2-bit codes (value+1), 4/byte, last dim
        for name, w, st in (("gate", f.wg.T, f.steps[0]), ("up", f.wu.T, f.steps[1]), ("down", f.wd.T, f.steps[2])):
            q = ((w / st).round().clamp(-1, 1) + 1).to(torch.uint8)
            ent[name] = (q[..., 0::4] | (q[..., 1::4] << 2) | (q[..., 2::4] << 4) | (q[..., 3::4] << 6)).cpu()
            ent[name + "_scale"] = st; ent[name + "_shape"] = tuple(w.shape)
        ck["layers"][L] = ent
    tmp = a.save + ".tmp"; torch.save(ck, tmp); os.replace(tmp, a.save)
    log({"phase": 17, "event": "saved", "path": a.save, "step": step})


# ------------------------------------------------------------------ train
k = set_k(0)
if os.environ.get("FFF_MEMDEBUG"):
    for i, lay in enumerate(model.model.layers):
        lay.register_forward_pre_hook(lambda m, args, i=i: print("layer", i, "rss", rss_gb(), "grad", torch.is_grad_enabled(), flush=True) if Switch.STUDENT and torch.is_grad_enabled() else None)
evaluate(0, k)
data = batches()
log({"phase": 17, "event": "start", "steps": steps, "tokens_per_step": tok_per_step, "k": a.k, "k_start": a.k_start,
     "B_leaf": a.B_leaf, "router": a.router})
run_kl = run_aux = 0.0; nb = 0; tw = time.time(); tok_w = 0
for step in range(1, steps + 1):
    k = set_k(step)
    sched = min(1.0, step / a.warmup) * 0.5 * (1 + math.cos(math.pi * step / steps))
    opt.param_groups[0]["lr"] = a.lr * sched; opt.param_groups[1]["lr"] = a.router_lr * sched
    for _ in range(a.accum):
        x = next(data).to(dev)
        with torch.no_grad():
            Switch.STUDENT = False; lt = model(x).logits[:, :-1]
        Switch.STUDENT = True
        ls = model(x).logits[:, :-1]
        kl = kl_loss(lt, ls); aux = sum(f.aux for f in mods.values())
        if step == 1: log({"phase": 17, "event": "debug fwd done"})
        ((kl + a.aux * aux) / a.accum).backward()
        if step == 1: log({"phase": 17, "event": "debug bwd done"})
        run_kl += kl.item() / a.accum; run_aux += aux.item() / a.accum / len(mods)
        del lt, ls, kl, aux
    Switch.STUDENT = False
    torch.nn.utils.clip_grad_norm_(params_n + params_r, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    nb += 1; tok_w += tok_per_step
    if step % 10 == 0 or step == steps:
        el = time.time() - tw
        log({"phase": 17, "event": "train", "step": step, "k": k, "train_kl": round(run_kl / nb, 5), "router_ce": round(run_aux / nb, 4),
             "tok_per_s": round(tok_w / el, 1), "tokens_seen": step * tok_per_step})
        run_kl = run_aux = 0.0; nb = 0; tw = time.time(); tok_w = 0
    if step % a.eval_every == 0 or step == steps: evaluate(step, k)
    if step % a.save_every == 0 or step == steps: save(step, k)
log({"phase": 17, "event": "done"})
