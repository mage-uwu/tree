"""checkpoint -> .cha2 for attn.c. Ternary tensors: fp32 absmean scale + 2-bit codes (q+1), 4/byte, row-major [out,in]."""
import struct, sys
import numpy as np
import torch
sys.path.insert(0, "..")
from model import weight_quant

ck = torch.load(sys.argv[1]); out = sys.argv[2]
cfg, sd = ck["cfg"], ck["state"]
H = len(cfg["kinds"])


def f32(fh, t): fh.write(t.detach().float().contiguous().numpy().astype("<f4").tobytes())


def tern(fh, w):
    _, q, s = weight_quant(w.float())
    c = (q.flatten().to(torch.int64) + 1).numpy().astype(np.uint8)
    c = np.concatenate([c, np.ones((-len(c)) % 4, np.uint8)]).reshape(-1, 4)
    fh.write(struct.pack("<f", s.item()))
    fh.write((c[:, 0] | c[:, 1] << 2 | c[:, 2] << 4 | c[:, 3] << 6).astype(np.uint8).tobytes())


mask = sum(1 << h for h, k in enumerate(cfg["kinds"]) if k == "S")
with open(out, "wb") as fh:
    fh.write(b"CHA2")
    local = int(bool(cfg.get("local")) and cfg["rdepth"] > 0)
    fh.write(struct.pack("<8i", 2, cfg["vocab"], cfg["d"], H, cfg["layers"], cfg["rdepth"], mask, local))
    f32(fh, sd["mixers.0.gamma"])
    f32(fh, sd["mixers.0.gamma_leaf"] if "mixers.0.gamma_leaf" in sd else sd["mixers.0.gamma"])
    f32(fh, sd["emb.weight"])
    for l in range(cfg["layers"]):
        p = f"mixers.{l}."
        f32(fh, sd[f"norms.{l}.weight"]); f32(fh, sd[p + "mu_k"]); f32(fh, sd[p + "mu_v"])
        for nm in "qkvgo":
            tern(fh, sd[p + f"{nm}.weight"])
        f32(fh, sd[p + "head_norm"])
        if cfg["rdepth"] > 0:
            f32(fh, sd[p + "router.w"])
    f32(fh, sd["norm.weight"]); f32(fh, sd["head.weight"])
    chars = ck.get("chars") or [chr(0)] * cfg["vocab"]
    fh.write("".join(chars).encode("latin-1"))
print("wrote", out)
