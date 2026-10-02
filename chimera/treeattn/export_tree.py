"""converted (or base) checkpoint -> .tre for treeattn.c"""
import struct, sys
import numpy as np
import torch
sys.path.insert(0, "..")
from model import weight_quant

ck = torch.load(sys.argv[1]); out = sys.argv[2]
cfg, sd = ck["cfg"], ck["state"]
S, D, values = ck.get("S", 0), ck.get("D", 0), int(ck.get("values", False))


def f32(fh, t): fh.write(t.detach().float().contiguous().numpy().astype("<f4").tobytes())


def tern(fh, w):
    _, q, s = weight_quant(w.float())
    c = (q.flatten().to(torch.int64) + 1).numpy().astype(np.uint8)
    c = np.concatenate([c, np.ones((-len(c)) % 4, np.uint8)]).reshape(-1, 4)
    fh.write(struct.pack("<f", s.item()))
    fh.write((c[:, 0] | c[:, 1] << 2 | c[:, 2] << 4 | c[:, 3] << 6).astype(np.uint8).tobytes())


with open(out, "wb") as fh:
    fh.write(b"TRE1")
    mlp = cfg.get("mlp", 0)
    tm = ck.get("tm")                                  # [D, k, r, rg] of a tree-MLP, if converted
    assert tm is None or (tm[1] == 0 and tm[3] == 0), "C engine supports leaf constant + leaf low-rank only"
    q8 = int(bool(ck.get("tm_q8")))
    axis = int(bool(ck.get("axis")))
    fh.write(struct.pack("<13i", 5, cfg["vocab"], cfg["d"], cfg["heads"], cfg["layers"], S, D, values, mlp,
                         tm[0] if tm else -1, tm[2] if tm else 0, q8, axis))
    f32(fh, sd["emb.weight"])
    for l in range(cfg["layers"]):
        p = f"attn.{l}."
        f32(fh, sd[f"norms.{l}.weight"])
        for nm in "qkvo":
            tern(fh, sd[p + f"{nm}.weight"])
        f32(fh, sd[p + "sub.weight"])
        if S:
            for t in ["kq"] + (["vq"] if values else []):
                f32(fh, sd[p + t + ".w"]); f32(fh, sd[p + t + ".b"]); f32(fh, sd[p + t + ".c"])
                if axis and t == "kq": f32(fh, sd[p + "kq.G"]); f32(fh, sd[p + "kq.Gi"])
        if mlp:
            f32(fh, sd[f"n2.{l}.weight"]); tern(fh, sd[f"up.{l}.weight"]); tern(fh, sd[f"down.{l}.weight"])
            if tm:
                if q8:
                    W = sd[f"tm.{l}.w"]; sw = W.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127
                    fh.write((W / sw).round().clamp(-127, 127).numpy().astype(np.int8).tobytes()); f32(fh, sw.flatten())
                else:
                    f32(fh, sd[f"tm.{l}.w"])
                f32(fh, sd[f"tm.{l}.b"]); f32(fh, sd[f"tm.{l}.c"])
                if tm[2] and q8:
                    P, Q = sd[f"tm.{l}.P"], sd[f"tm.{l}.Q"]
                    sP = P.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127
                    sQ = Q.abs().amax(-2, keepdim=True).clamp(min=1e-12) / 127
                    fh.write((P / sP).round().clamp(-127, 127).numpy().astype(np.int8).tobytes()); f32(fh, sP.flatten())
                    fh.write((Q / sQ).round().clamp(-127, 127).transpose(1, 2).contiguous().numpy().astype(np.int8).tobytes()); f32(fh, sQ.flatten())
                elif tm[2]: f32(fh, sd[f"tm.{l}.P"]); f32(fh, sd[f"tm.{l}.Q"])
    f32(fh, sd["norm.weight"]); f32(fh, sd["head.weight"])
    fh.write("".join(ck.get("chars") or [chr(0)] * cfg["vocab"]).encode("latin-1"))
print("wrote", out, f"S={S} D={D} values={values}")
