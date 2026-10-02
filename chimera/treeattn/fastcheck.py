"""fast-scan path vs the float reference path of the same C engine (same key codes): isolates the scan's own error."""
import subprocess, sys, os, numpy as np, torch, torch.nn.functional as F
import data
tre = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
get, _, _ = data.shakespeare(T); x, y = get("val", 1, torch.Generator().manual_seed(5)); x, y = x[0], y[0]
x.numpy().astype("<i4").tofile("/tmp/fc.bin")
out = []
for env in ({}, {"EXACT": "1"}):
    subprocess.run(["./treeattn", tre, "dump", "/tmp/fc.bin", "/tmp/fc_out.bin"], check=True, env={**os.environ, **env})
    out.append(torch.from_numpy(np.fromfile("/tmp/fc_out.bin", "<f4").reshape(T, -1).copy()))
f, e = out; d = (f - e).abs()
ce = lambda l: F.cross_entropy(l, y).item()
print(f"{tre} T={T}: loss fast {ce(f):.4f} reference {ce(e):.4f} | max|dlogit| {d.max():.2e} median {d.median():.1e} | argmax agree {(f.argmax(-1)==e.argmax(-1)).float().mean()*100:.2f}%")
