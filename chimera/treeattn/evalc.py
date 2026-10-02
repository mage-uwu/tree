"""validation loss through the C engine (fast path and float reference path) over many 128-token windows."""
import subprocess, sys, os, numpy as np, torch, torch.nn.functional as F
import data
tre = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 64
get, _, _ = data.shakespeare(128); g = torch.Generator().manual_seed(123)
x, y = get("val", W, g)
res = {}
for name, env in (("fast", {}), ("reference", {"EXACT": "1"})):
    tot = 0
    for i in range(W):
        x[i].numpy().astype("<i4").tofile("/tmp/ev.bin")
        subprocess.run(["./treeattn", tre, "dump", "/tmp/ev.bin", "/tmp/ev_out.bin"], check=True, env={**os.environ, **env})
        lg = torch.from_numpy(np.fromfile("/tmp/ev_out.bin", "<f4").reshape(128, -1).copy())
        tot += F.cross_entropy(lg, y[i]).item()
    res[name] = tot / W
print(f"{tre}: C val loss over {W} windows: fast {res['fast']:.4f} | reference {res['reference']:.4f}")
