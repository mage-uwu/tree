"""int8 the key trees: each split vector and each leaf value -> int8 with its own absmax scale (stored dequantised).
Evaluates val loss before/after (torch) and saves *_q8.pt."""
import sys, torch, torch.nn.functional as F
from tree_attn import LM
import data
torch.set_num_threads(2)
ck = torch.load(sys.argv[1]); cfg = ck["cfg"]
m = LM(cfg); m.convert(ck["S"], ck["D"], ck["values"], axis=bool(ck.get("axis"))); m.load_state_dict(ck["state"]); m.eval()
get, _, _ = data.shakespeare(128)
@torch.no_grad()
def ev():
    g = torch.Generator().manual_seed(123); tot = 0
    for _ in range(40):
        x, y = get("val", 32, g); tot += F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item()
    return tot / 40
a = ev()
with torch.no_grad():
    for at in m.attn:
        for t in [at.kq] + ([at.vq] if at.vq is not None else []):
            for p in (t.w, t.c):
                s = p.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127
                p.copy_((p / s).round().clamp(-127, 127) * s)
b = ev()
print(f"{sys.argv[1]}: fp32 trees {a:.4f} -> int8 trees {b:.4f}")
ck["state"] = m.state_dict(); ck["trees_q8"] = True
torch.save(ck, sys.argv[1].replace(".pt", "_q8.pt"))
