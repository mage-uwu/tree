"""All-tree conversion of a trained BitNet transformer: tree attention (boosted trees on K, optionally V)
+ tree-MLP (model tree: leaf constant + rank-r map). BitNet weights frozen throughout.

  1. closed-form fit, layer by layer, each block calibrated on the outputs of the converted blocks below it
  2. optional minimal adaptation: distil ONLY the leaf tables (attention leaf values, MLP c/P/Q) to the frozen base
  3. int8 MLP leaf maps + router
"""
import argparse, copy, json, math, time
import torch, torch.nn as nn, torch.nn.functional as F
from tree_attn import LM, BoostedTrees
from tree_mlp import TreeMLP
import data

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--attn", default="16x6")          # S x D
p.add_argument("--values", type=int, default=0)
p.add_argument("--mlp", default="8:16")           # D:r
p.add_argument("--calib_seqs", type=int, default=6000)
p.add_argument("--attn_calib", type=int, default=384)
p.add_argument("--ft_steps", type=int, default=300)
p.add_argument("--teacher_target", type=int, default=1)   # fit each tree-MLP to the base model's stream (absorbs drift)
p.add_argument("--save", required=True)
a = p.parse_args()
torch.manual_seed(0); torch.set_num_threads(2)
ck = torch.load(a.base); cfg = ck["cfg"]
base = LM(cfg); base.load_state_dict(ck["state"]); base.eval()
for q in base.parameters(): q.requires_grad_(False)
get, _, _ = data.shakespeare(128)


@torch.no_grad()
def evaluate(m):
    m.eval(); g = torch.Generator().manual_seed(123); tot = 0
    for _ in range(40):
        x, y = get("val", 32, g); tot += F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item()
    return round(tot / 40, 4)


S, D = map(int, a.attn.split("x")); mD, mr = map(int, a.mlp.split(":"))
d, H = cfg["d"], cfg["heads"]
m = copy.deepcopy(base)
m.convert(S, D, bool(a.values))
m.tm = nn.ModuleList(TreeMLP(d, cfg["mlp"] * d, mD, 0, mr, 0) for _ in range(cfg["layers"])); m.tm_ready = 0
g = torch.Generator().manual_seed(999)
calib = torch.cat([get("train", 1, g)[0] for _ in range(a.calib_seqs)])
res = {"base": evaluate(base), "attn": a.attn, "values": bool(a.values), "mlp": a.mlp}
print(json.dumps(res), flush=True)

chunk = lambda f, t: torch.cat([f(t[i:i + 64]) for i in range(0, len(t), 64)])
t0 = time.time()
with torch.no_grad():
    x = m.emb(calib); xb = x.clone()                              # converted stream, base (teacher) stream
    for l, (n, at) in enumerate(zip(m.norms, m.attn)):
        q, k, v = at.qkv(n(x[:a.attn_calib]))
        at.kq.fit(k.transpose(0, 1).flatten(1, 2), q.transpose(0, 1).flatten(1, 2))
        if at.vq is not None: at.vq.fit(v.transpose(0, 1).flatten(1, 2))
        del q, k, v
        x = chunk(lambda t: t + at(n(t)), x)                      # through the converted attention
        if a.teacher_target:
            bl = base.attn[l]
            xb = chunk(lambda t: base._mlp(l, t + bl(base.norms[l](t))), xb)   # teacher stream after block l
            m.tm[l].fit(m.n2[l](x), m.up[l], m.down[l], target=xb - x)
        else:
            m.tm[l].fit(m.n2[l](x), m.up[l], m.down[l])
        m.tm_ready = l + 1
        x = chunk(lambda t: m._mlp(l, t), x)                      # through the converted MLP
        print(f"layer {l} fitted {time.time()-t0:.0f}s", flush=True)
    del x
res["fit_s"] = round(time.time() - t0); res["closed_form"] = evaluate(m)
print(json.dumps(res), flush=True)

if a.ft_steps:
    for q in m.parameters(): q.requires_grad_(False)
    pa = [t for at in m.attn for t in ([at.kq.c] + ([at.vq.c] if at.vq is not None else []))]
    pm = [q for t in m.tm for _, q in t.named_parameters()]
    for q in pa + pm: q.requires_grad_(True)
    opt = torch.optim.Adam([{"params": pa, "lr": 3e-3}, {"params": pm, "lr": 1e-3}])
    base_lr = [3e-3, 1e-3]; m.train(); t1 = time.time()
    for step in range(a.ft_steps):
        for gp, bl in zip(opt.param_groups, base_lr): gp["lr"] = bl * 0.5 * (1 + math.cos(math.pi * step / a.ft_steps))
        x, _ = get("train", 32)
        with torch.no_grad(): tl = base(x)
        loss = F.kl_div(F.log_softmax(m(x), -1), F.log_softmax(tl, -1), log_target=True, reduction="batchmean") / x.shape[1]
        loss.backward(); opt.step(); opt.zero_grad()
        if step % 100 == 0: print(f"distil step {step} kl {loss.item():.4f} {time.time()-t1:.0f}s", flush=True)
    res["leaf_tuned"] = evaluate(m); res["ft_s"] = round(time.time() - t1)
    res["leaf_params"] = sum(q.numel() for q in pa + pm)
    print(json.dumps(res), flush=True)

for t in m.tm: t.quantize()
res["int8_mlp"] = evaluate(m)
print(json.dumps(res), flush=True)
torch.save({"cfg": cfg, "chars": ck["chars"], "state": m.state_dict(), "S": S, "D": D, "values": bool(a.values),
            "tm": [mD, 0, mr, 0], "tm_q8": True, "res": res}, a.save)
json.dump(res, open(a.save.replace(".pt", ".json"), "w"), indent=1)
