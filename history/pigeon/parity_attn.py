"""torch parallel / chunkwise / recurrent  vs  C engine (3 projection backends)."""
import subprocess, sys
import numpy as np
import torch, torch.nn.functional as F
from attn_model import AttnLM
torch.set_num_threads(1)

ck = torch.load(sys.argv[1]); cha = sys.argv[2]; T = int(sys.argv[3]) if len(sys.argv) > 3 else 384
m = AttnLM(ck["cfg"]); m.load_state_dict(ck["state"], strict=False); m.eval()
if ck["cfg"]["task"] == "shakespeare":
    stoi = {c: i for i, c in enumerate(ck["chars"])}
    text = open("../../input.txt").read()
    toks = torch.tensor([stoi[c] for c in text[-30000:-30000 + T + 1]])
    x, y = toks[:-1], toks[1:]
else:
    import random; g = torch.Generator().manual_seed(7)
    P = 32; keys = torch.randperm(64, generator=g)[:P]; vals = torch.randint(64, (P,), generator=g) + 64
    qi = torch.randint(P, (P,), generator=g)
    seq = torch.cat([torch.stack([keys, vals], -1).flatten(), torch.stack([keys[qi], vals[qi]], -1).flatten()])
    x, y = seq[:-1], seq[1:].clone(); y[: 2 * P] = -100; y[2 * P + 1::2] = -100; T = len(x)

with torch.no_grad():
    par = m(x[None])[0]; chk = m(x[None], "chunk", 32)[0]
    st = m.init_state(1); rec = []
    for t in range(T):
        l, st = m.step(x[t:t + 1], st); rec.append(l[0])
    rec = torch.stack(rec)
x.numpy().astype("<i4").tofile("/tmp/toks.bin")
C = []
for be in (0, 1, 2):
    subprocess.run(["./attn", cha, "dump", "/tmp/toks.bin", f"/tmp/lg{be}.bin", str(be)], check=True)
    C.append(torch.from_numpy(np.fromfile(f"/tmp/lg{be}.bin", "<f4").reshape(T, -1)))

ce = lambda l: F.cross_entropy(l, y, ignore_index=-100).item()
print(f"{ck['cfg']['kinds']} depth={ck['cfg']['rdepth']} loss: parallel {ce(par):.4f} chunk {ce(chk):.4f} recurrent {ce(rec):.4f} C {ce(C[2]):.4f}")
print(f"  C backends bit-identical: naive==sign {torch.equal(C[0], C[1])}, naive==lut {torch.equal(C[0], C[2])}")
for nm, a, b in [("parallel~chunk", par, chk), ("parallel~recurrent", par, rec), ("recurrent~C", rec, C[2])]:
    dd = (a - b).abs()
    print(f"  {nm:20s} max {dd.max():.2e} median {dd.median():.1e} argmax agree {(a.argmax(-1) == b.argmax(-1)).float().mean()*100:.1f}%")
