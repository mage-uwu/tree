"""Grid: tree-MLP conversion of a trained BitNet transformer (frozen weights), closed-form fit, no training."""
import argparse, copy, json, time
import torch, torch.nn.functional as F
from tree_attn import LM
from tree_mlp import convert_mlp
import data

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--configs", default="8:0:0:0")            # D:k:r:rg
p.add_argument("--calib_seqs", type=int, default=512)
p.add_argument("--savedir", default=None)
p.add_argument("--ft_steps", type=int, default=0)          # leaf-only distillation (c, P, Q); BitNet weights frozen
p.add_argument("--ft_lr", type=float, default=1e-3)
p.add_argument("--out", required=True)
a = p.parse_args()
torch.manual_seed(0); torch.set_num_threads(2)
ck = torch.load(a.base); cfg = ck["cfg"]
base = LM(cfg); base.load_state_dict(ck["state"]); base.eval()
get, _, _ = data.shakespeare(128)

@torch.no_grad()
def evaluate(m, **kw):
    m.eval(); g = torch.Generator().manual_seed(123); tot = 0
    for _ in range(40):
        x, y = get("val", 32, g); tot += F.cross_entropy(m(x, **kw).flatten(0, 1), y.flatten()).item()
    return round(tot / 40, 4)

g = torch.Generator().manual_seed(999)
calib = torch.cat([get("train", 1, g)[0] for _ in range(a.calib_seqs)])
H = cfg["mlp"] * cfg["d"]
res = {"base": evaluate(base), "mlp_ablated": evaluate(base, ablate_mlp=True), "runs": []}
print(json.dumps(res), flush=True)
for c in a.configs.split(","):
    c = c.strip()
    D, k, r, rg = map(int, c.split(":"))
    t0 = time.time()
    m = convert_mlp(copy.deepcopy(base), calib, D, k, r, rg)
    u = m.tm[0].row_units()
    rr = {"D": D, "leaves": 2 ** D, "k": k, "r": r, "rg": rg, "row_units": u, "predicted_speedup": round(2 * H / u, 1),
          "fit_s": round(time.time() - t0), "val_loss": evaluate(m)}
    if a.ft_steps:
        import math
        for q in m.parameters(): q.requires_grad_(False)
        params = [q for t in m.tm for n_, q in t.named_parameters()]
        for q in params: q.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=a.ft_lr); m.train(); t1 = time.time()
        for step in range(a.ft_steps):
            for gp in opt.param_groups: gp["lr"] = a.ft_lr * 0.5 * (1 + math.cos(math.pi * step / a.ft_steps))
            x, _ = get("train", 32)
            with torch.no_grad(): tl = base(x)
            loss = F.kl_div(F.log_softmax(m(x), -1), F.log_softmax(tl, -1), log_target=True, reduction="batchmean") / x.shape[1]
            loss.backward(); opt.step(); opt.zero_grad()
        rr["distilled"] = evaluate(m); rr["ft_s"] = round(time.time() - t1)
        rr["leaf_params"] = sum(q.numel() for q in params)
    print(json.dumps(rr), flush=True); res["runs"].append(rr)
    if a.savedir:
        torch.save({"cfg": cfg, "chars": ck["chars"], "state": m.state_dict(), "tm": [D, k, r, rg]}, f"{a.savedir}/full_mlptree_{D}_{k}_{r}_{rg}.pt")
json.dump(res, open(a.out, "w"), indent=1)
