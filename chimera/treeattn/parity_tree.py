"""torch (training form, parallel attention with tree-quantized K/V) vs C (byte-code cache, lookup scoring)."""
import subprocess, sys
import numpy as np
import torch, torch.nn.functional as F
from tree_attn import LM
import data
torch.set_num_threads(1)

ck = torch.load(sys.argv[1]); tre = sys.argv[2]; T = int(sys.argv[3]) if len(sys.argv) > 3 else 256
m = LM(ck["cfg"])
if ck.get("S"): m.convert(ck["S"], ck["D"], ck["values"], axis=bool(ck.get("axis")))
if ck.get("tm"):
    import torch.nn as nn
    from tree_mlp import TreeMLP
    D_, k_, r_, rg_ = ck["tm"]; d_ = ck["cfg"]["d"]
    m.tm = nn.ModuleList(TreeMLP(d_, ck["cfg"]["mlp"] * d_, D_, k_, r_, rg_) for _ in range(ck["cfg"]["layers"]))
m.load_state_dict(ck["state"]); m.eval()
if ck.get("tm_q8"):
    for t_ in m.tm: t_.q8 = True
if ck["cfg"]["task"] == "shakespeare":
    get, _, _ = data.shakespeare(T)
else:
    get, _, _ = data.mqar(32)
x, y = get("val", 1, torch.Generator().manual_seed(3)); x, y = x[0], y[0]; T = len(x)
with torch.no_grad():
    lt = m(x[None])[0]
x.numpy().astype("<i4").tofile("/tmp/tt.bin")
subprocess.run(["./treeattn", tre, "dump", "/tmp/tt.bin", "/tmp/tl.bin"], check=True)
lc = torch.from_numpy(np.fromfile("/tmp/tl.bin", "<f4").reshape(T, -1))
ce = lambda l: F.cross_entropy(l, y, ignore_index=-100).item()
dd = (lt - lc).abs()
print(f"S={ck.get('S', 0)} D={ck.get('D', 0)} values={ck.get('values', False)} T={T}: loss torch {ce(lt):.4f} C {ce(lc):.4f} | "
      f"max|dlogit| {dd.max():.2e} median {dd.median():.1e} | argmax agree {(lt.argmax(-1) == lc.argmax(-1)).float().mean()*100:.1f}%")
