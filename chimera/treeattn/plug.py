"""Combine a separately converted tree-attention model and tree-MLP model (same frozen BitNet base) into one
all-tree model; optionally tune only the leaf tables jointly against the frozen base; int8 the MLP leaves."""
import argparse, json, math, time
import torch, torch.nn as nn, torch.nn.functional as F
from tree_attn import LM
from tree_mlp import TreeMLP
import data
p = argparse.ArgumentParser()
p.add_argument("--attn", required=True); p.add_argument("--mlp", required=True)
p.add_argument("--base", default="runs/full_shakespeare.pt")
p.add_argument("--ft_steps", type=int, default=0); p.add_argument("--save", default=None)
a = p.parse_args()
torch.manual_seed(0); torch.set_num_threads(2)
A = torch.load(a.attn); M = torch.load(a.mlp); B = torch.load(a.base); cfg = A["cfg"]
base = LM(cfg); base.load_state_dict(B["state"]); base.eval()
m = LM(cfg); m.convert(A["S"], A["D"], A["values"], axis=bool(A.get("axis")))
D, k, r, rg = M["tm"]
m.tm = nn.ModuleList(TreeMLP(cfg["d"], cfg["mlp"] * cfg["d"], D, k, r, rg) for _ in range(cfg["layers"]))
sd = dict(A["state"]); sd.update({k_: v for k_, v in M["state"].items() if k_.startswith("tm.")})
m.load_state_dict(sd); m.eval()
get, _, _ = data.shakespeare(128)
@torch.no_grad()
def ev(mm):
    mm.eval(); g = torch.Generator().manual_seed(123); tot = 0
    for _ in range(40):
        x, y = get("val", 32, g); tot += F.cross_entropy(mm(x).flatten(0, 1), y.flatten()).item()
    return round(tot / 40, 4)
res = {"attn": a.attn, "mlp": a.mlp, "base": ev(base), "plugged": ev(m)}
print(json.dumps(res), flush=True)
if a.ft_steps:
    for q in list(m.parameters()) + list(base.parameters()): q.requires_grad_(False)
    pa = [t for at in m.attn for t in ([at.kq.c] + ([at.vq.c] if at.vq is not None else []))]
    pm = [q for t in m.tm for _, q in t.named_parameters()]
    for q in pa + pm: q.requires_grad_(True)
    lrs = [1e-3, 5e-4]; opt = torch.optim.Adam([{"params": pa, "lr": lrs[0]}, {"params": pm, "lr": lrs[1]}])
    m.train(); t1 = time.time()
    for step in range(a.ft_steps):
        for gp, bl in zip(opt.param_groups, lrs): gp["lr"] = bl * 0.5 * (1 + math.cos(math.pi * step / a.ft_steps))
        x, _ = get("train", 32)
        with torch.no_grad(): tl = base(x)
        loss = F.kl_div(F.log_softmax(m(x), -1), F.log_softmax(tl, -1), log_target=True, reduction="batchmean") / x.shape[1]
        loss.backward(); opt.step(); opt.zero_grad()
        if step % 50 == 0: print(f"step {step} kl {loss.item():.4f} {time.time()-t1:.0f}s", flush=True)
    res["joint_leaf_tuned"] = ev(m); res["ft_s"] = round(time.time() - t1)
    print(json.dumps(res), flush=True)
for t in m.tm: t.quantize()
res["int8_mlp"] = ev(m)
print(json.dumps(res), flush=True)
if a.save:
    torch.save({"cfg": cfg, "chars": B["chars"], "state": m.state_dict(), "S": A["S"], "D": A["D"], "values": A["values"], "axis": bool(A.get("axis")),
                "tm": M["tm"], "tm_q8": True, "res": res}, a.save)
    json.dump(res, open(a.save.replace(".pt", ".json"), "w"), indent=1)
