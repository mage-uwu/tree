"""
Export a checkpoint to the .chim binary format read by chimera.c.

Ternary tensors use a BitNet b1.58-compatible-ish layout:
  float32 scale (absmean, dequant multiplier) | packed 2-bit codes
  code = q + 1  (0 -> -1, 1 -> 0, 2 -> +1), 4 codes per byte,
  element i stored in bits [2*(i%4), 2*(i%4)+1] of byte i//4, row-major [rows, cols].
This matches the {0,1,2} ternary encoding used by bitnet.cpp's I2_S-style kernels
in spirit; re-packing to a specific upstream kernel layout is a permutation away.
"""
import struct, sys
import numpy as np
import torch
from model import weight_quant

ck = torch.load(sys.argv[1])
out = sys.argv[2]
cfg, sd, chars = ck["cfg"], ck["state"], ck["chars"]


def f32(fh, t):
    fh.write(t.detach().float().contiguous().numpy().astype("<f4").tobytes())


def tern(fh, w):
    _, q, s = weight_quant(w.float())
    codes = (q.flatten().to(torch.int64) + 1).numpy().astype(np.uint8)
    pad = (-len(codes)) % 4
    codes = np.concatenate([codes, np.ones(pad, np.uint8)]).reshape(-1, 4)
    packed = codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
    fh.write(struct.pack("<f", s.item()))
    fh.write(packed.astype(np.uint8).tobytes())


with open(out, "wb") as fh:
    fh.write(b"CHIM")
    fh.write(struct.pack("<8i", 1, cfg["vocab"], cfg["d"], cfg["layers"], cfg["heads"],
                         cfg["depth"], cfg["trees"], 0 if cfg["ffn"] == "fff" else 1))
    f32(fh, sd["blocks.0.ret.gamma"])
    f32(fh, sd["emb.weight"])
    for l in range(cfg["layers"]):
        p = f"blocks.{l}."
        f32(fh, sd[p + "n1.weight"])
        for nm in "qkvgo":
            tern(fh, sd[p + f"ret.{nm}.weight"])
        f32(fh, sd[p + "ret.head_norm"])
        f32(fh, sd[p + "n2.weight"])
        if cfg["ffn"] == "fff":
            tern(fh, sd[p + "ffn.w_in"]); tern(fh, sd[p + "ffn.w_out"])
        else:
            tern(fh, sd[p + "ffn.up.weight"]); tern(fh, sd[p + "ffn.down.weight"])
    f32(fh, sd["norm.weight"])
    f32(fh, sd["head.weight"])
    fh.write("".join(chars).encode("latin-1"))
print("wrote", out)
