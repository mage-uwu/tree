"""Convert a trained BitNet attention LM to boosted-tree attention; zero-shot + leaf-only distillation."""
import argparse, copy, json, math, time
import torch, torch.nn.functional as F
from tree_attn import LM
import data

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--configs", default="1x4,2x4,4x4,8x4,4x6")   # S x D
p.add_argument("--values", type=int, default=1)
p.add_argument("--qaware", type=int, default=1)
p.add_argument("--axis", type=int, default=0)
p.add_argument("--ft_steps", type=int, default=300)
p.add_argument("--ft_lr", type=float, default=3e-3)
p.add_argument("--calib_seqs", type=int, default=64)
p.add_argument("--save", default=None, help="save the converted model for config S x D given here")
p.add_argument("--out", required=True)
a = p.parse_args()
torch.manual_seed(0); torch.set_num_threads(1)

ck = torch.load(a.base); cfg = ck["cfg"]
base = LM(cfg); base.load_state_dict(ck["state"]); base.eval()
for q in base.parameters(): q.requires_grad_(False)
task = cfg["task"]
get, _, _ = data.shakespeare(128) if task == "shakespeare" else data.mqar(32)


@torch.no_grad()
def evaluate(m, ablate=False):
    m.eval(); g = torch.Generator().manual_seed(123)
    if task == "shakespeare":
        tot = 0
        for _ in range(40):
            x, y = get("val", 32, g); tot += F.cross_entropy(m(x, ablate).flatten(0, 1), y.flatten()).item()
        return {"val_loss": round(tot / 40, 4)}
    out = {}
    for P in (32, 48, 64):
        acc = 0
        for _ in range(8):
            x, y = get("val", 32, g, P); lg = m(x, ablate); mk = y != -100
            acc += (lg.argmax(-1)[mk] == y[mk]).float().mean().item()
        out[f"acc@{P}"] = round(acc / 8, 4)
    return out


@torch.no_grad()
def score_fidelity(m):
    """relative error of q.k_hat vs q.k (layer 0, val batch)."""
    x, _ = get("val", 16, torch.Generator().manual_seed(5))
    at = m.attn[0]; q, k, _ = at.qkv(m.norms[0](m.emb(x)))
    kh = at.kq(k)[0]
    s, sh = q @ k.transpose(-1, -2), q @ kh.transpose(-1, -2)
    return round(((s - sh).norm() / s.norm()).item(), 4)


g = torch.Generator().manual_seed(999)
calib = torch.cat([get("train", 1, g)[0] for _ in range(a.calib_seqs)])
res = {"cfg": cfg, "base": evaluate(base), "attention_ablated": evaluate(base, ablate=True), "runs": []}
print(json.dumps(res), flush=True)

for c in a.configs.split(","):
    S, D = map(int, c.split("x"))
    m = copy.deepcopy(base); m.convert(S, D, bool(a.values), bool(a.qaware), bool(a.axis)); m.calibrate(calib)
    r = {"S": S, "D": D, "qaware": bool(a.qaware), "bits_per_key": S * D, "leaves": 2 ** D, "values": bool(a.values),
         "cache_bytes_per_token_per_head": S * (2 if a.values else 1) + (0 if a.values else 2 * m.attn[0].hd),
         "score_rel_err": score_fidelity(m), "zero_shot": evaluate(m)}
    t0 = time.time()
    if a.ft_steps:
        params = [t for at in m.attn for t in ([at.kq.c] + ([at.vq.c] if at.vq is not None else []))]
        for t in params: t.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=a.ft_lr)
        m.train()
        for step in range(a.ft_steps):
            for gp in opt.param_groups: gp["lr"] = a.ft_lr * 0.5 * (1 + math.cos(math.pi * step / a.ft_steps))
            x, _ = get("train", 32)
            with torch.no_grad(): tl = base(x)
            sl = m(x)
            loss = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction="batchmean") / x.shape[1]
            loss.backward(); opt.step(); opt.zero_grad()
        r["distilled"] = evaluate(m); r["score_rel_err_after"] = score_fidelity(m)
        r["trainable_params"] = sum(t.numel() for t in params); r["ft_seconds"] = round(time.time() - t0)
    print(json.dumps(r), flush=True); res["runs"].append(r)
    if a.save and c == a.save.split(":")[0]:
        torch.save({"cfg": cfg, "chars": ck["chars"], "state": m.state_dict(), "S": S, "D": D, "values": bool(a.values), "axis": bool(a.axis)},
                   a.save.split(":")[1])
json.dump(res, open(a.out, "w"), indent=1)
