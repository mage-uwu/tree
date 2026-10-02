"""Compare C engine logits to the PyTorch parallel (training) form and recurrent form."""
import subprocess, sys
import numpy as np
import torch, torch.nn.functional as F
torch.set_num_threads(1)
from model import Chimera

ck = torch.load(sys.argv[1]); chim = sys.argv[2]
md = int(sys.argv[3]) if len(sys.argv) > 3 else None
T = 512
m = Chimera(ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
stoi = {c: i for i, c in enumerate(ck["chars"])}
text = open("../input.txt").read()
toks = torch.tensor([stoi[c] for c in text[-20000:-20000 + T + 1]])
x, y = toks[:-1], toks[1:]

with torch.no_grad():
    par = m(x[None], md)[0]
    S = m.init_state(1); rec = []
    for t in range(T):
        l, S = m.step(x[t:t + 1], S, md); rec.append(l[0])
    rec = torch.stack(rec)

x.numpy().astype("<i4").tofile("/tmp/toks.bin")
args = ["./chimera", chim, "dump", "/tmp/toks.bin", "/tmp/logits.bin"] + ([str(md)] if md else [])
subprocess.run(args, check=True)
c = torch.from_numpy(np.fromfile("/tmp/logits.bin", "<f4").reshape(T, -1))

ce = lambda l: F.cross_entropy(l, y).item()
print(f"depth={md or ck['cfg']['depth']}  loss: parallel {ce(par):.4f} | torch-recurrent {ce(rec):.4f} | C {ce(c):.4f}")
for name, a, b in [("parallel vs recurrent", par, rec), ("recurrent vs C", rec, c), ("parallel vs C", par, c)]:
    d = (a - b).abs()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"  {name:22s} max|dlogit| {d.max():.4f}  median {d.median():.2e}  argmax agree {agree*100:.1f}%")
