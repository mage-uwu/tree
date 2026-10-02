import sys, torch, torch.nn as nn, torch.nn.functional as F
from tree_attn import LM
from tree_mlp import TreeMLP
import data
torch.set_num_threads(1)
ck = torch.load(sys.argv[1]); cfg = ck["cfg"]
m = LM(cfg); m.convert(ck["S"], ck["D"], ck["values"])
D, k, r, rg = ck["tm"]
m.tm = nn.ModuleList(TreeMLP(cfg["d"], cfg["mlp"] * cfg["d"], D, k, r, rg) for _ in range(cfg["layers"]))
m.load_state_dict(ck["state"]); m.eval()
for t in m.tm: t.q8 = True
get, _, _ = data.shakespeare(128)
@torch.no_grad()
def ev():
    g = torch.Generator().manual_seed(123); tot = 0
    for _ in range(20):
        x, y = get("val", 32, g); tot += F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item()
    return tot / 20
print("both trees        ", round(ev(), 4))
kq = [a.kq for a in m.attn]; vq = [a.vq for a in m.attn]
for a in m.attn: a.kq = None; a.vq = None
print("tree MLP only     ", round(ev(), 4))
for a, k_, v_ in zip(m.attn, kq, vq): a.kq, a.vq = k_, v_
m.tm_ready = 0
print("tree attention only", round(ev(), 4))
for a in m.attn: a.kq = None; a.vq = None
print("neither (base)    ", round(ev(), 4))
# per-layer: tree MLP in only layer l (base attention)
for l in range(cfg["layers"]):
    tm_all = m.tm
    class One(nn.Module): pass
    m.tm_ready = 4
    saved = m._mlp
    def mlp_one(i, x, ablate=False, l=l, saved=saved):
        if i == l: return saved(i, x, ablate)
        return x + m.down[i](F.gelu(m.up[i](m.n2[i](x))))
    m._mlp = mlp_one
    print(f"tree MLP in layer {l} only", round(ev(), 4))
    m._mlp = saved
