"""load a tree-MLP checkpoint, quantize leaf maps to int8, evaluate, save *_q8.pt"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
from tree_attn import LM
from tree_mlp import TreeMLP
import data
torch.set_num_threads(1)
ck = torch.load(sys.argv[1]); cfg = ck["cfg"]; D, k, r, rg = ck["tm"]
m = LM(cfg); m.tm = nn.ModuleList(TreeMLP(cfg["d"], cfg["mlp"] * cfg["d"], D, k, r, rg) for _ in range(cfg["layers"]))
m.load_state_dict(ck["state"]); m.eval()
get, _, _ = data.shakespeare(128)
@torch.no_grad()
def ev():
    g = torch.Generator().manual_seed(123); tot = 0
    for _ in range(40):
        x, y = get("val", 32, g); tot += F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item()
    return tot / 40
a = ev()
for t in m.tm: t.quantize()
b = ev()
print(f"{sys.argv[1]}: fp32 leaf maps {a:.4f} -> int8 leaf maps {b:.4f}")
ck["state"] = m.state_dict(); ck["tm_q8"] = True
torch.save(ck, sys.argv[1].replace(".pt", "_q8.pt"))
