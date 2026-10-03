"""Write Phase-13 healed MLP weights into a copy of the bitnet.cpp I2_S GGUF, in place (sizes don't change).
Per layer: ffn_gate / ffn_up / ffn_down ternary codes + per-tensor scale (from a phase_peer.py --save ckpt with
--heal_ternary 1), and ffn_sub_norm = snw * sqrt(c): the engine normalises the k selected neurons over all F (zeros
elsewhere) while training used the coverage-corrected RMS sum/(F c), so sqrt(c) folds the difference into the weight.
--sel <.sel.pt> --k K: only the sub-norm fold (trained selectors on the original weights).
--hf_codes 1: write the HF model's own MLP codes (the GGUF's differ: ~2% of HF nonzeros are 0 there), the
control for healed weights, which start from the HF codes.
--verify L: decode the GGUF's layer-L gate/up/down and compare to the HF model's (checks the packing; no writing).
I2_S: per 128 weights 32 bytes, byte j holds weights j, 32+j, 64+j, 96+j at bits 6,4,2,0; code = value+1; the tensor's
codes are followed by its float scale (tensor bytes = n/4 + 32). gate/up rows are the F neurons (d long), down rows d."""
import argparse, shutil, numpy as np, torch
from gguf import GGUFReader

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--out", default="")
ap.add_argument("--ckpt", default="")
ap.add_argument("--sel", default="")
ap.add_argument("--k", type=int, default=1536)
ap.add_argument("--verify", type=int, default=-1)
ap.add_argument("--hf_codes", type=int, default=0)                 # control: the HF model's own (unhealed) MLP codes
a = ap.parse_args()


def offsets(path):
    r = GGUFReader(path)
    return {t.name: (int(t.data_offset), [int(x) for x in t.shape]) for t in r.tensors}


def unpack_i2s(buf, rows, n):
    """bytes -> int8 values (rows, n) in {-1,0,1}."""
    b = np.frombuffer(buf, np.uint8, rows * n // 4).reshape(rows, n // 128, 32)
    v = np.stack([(b >> 6) & 3, (b >> 4) & 3, (b >> 2) & 3, b & 3], 2)          # (rows, groups, 4, 32): weight 32*p + j
    return v.reshape(rows, n).astype(np.int8) - 1


def pack_i2s(vals):
    """int (rows, n) in {-1,0,1} -> I2_S bytes."""
    rows, n = vals.shape
    c = (vals + 1).astype(np.uint8).reshape(rows, n // 128, 4, 32)
    return ((c[:, :, 0] << 6) | (c[:, :, 1] << 4) | (c[:, :, 2] << 2) | c[:, :, 3]).tobytes()


def unpack_ckpt(packed, shape):
    """phase_peer 2-bit codes (value+1), 4 per byte along the last dim -> int8 values of `shape`."""
    p = packed.numpy()
    q = np.stack([p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], -1).reshape(*shape[:-1], shape[-1])
    return q.astype(np.int8) - 1


off = offsets(a.gguf)
if a.verify >= 0:
    from common import load_model
    L = a.verify
    model, _ = load_model("cpu"); mlp = model.model.layers[L].mlp
    d, Fn = model.config.hidden_size, model.config.intermediate_size
    probe = lambda lin, n: torch.cat([lin(torch.eye(n, dtype=torch.bfloat16)[i:i + 512]).float() for i in range(0, n, 512)])
    with open(a.gguf, "rb") as f:
        for name, lin, n_in, rows in (("gate", mlp.gate_proj, d, Fn), ("up", mlp.up_proj, d, Fn), ("down", mlp.down_proj, Fn, d)):
            o, _ = off[f"blk.{L}.ffn_{name}.weight"]
            f.seek(o); buf = f.read(rows * n_in // 4 + 4)
            v = unpack_i2s(buf, rows, n_in); s = np.frombuffer(buf[-4:], np.float32)[0]
            W = probe(lin, n_in).T.numpy()                                        # (rows, n_in): out x in
            st = np.abs(W).max()
            print(name, "scale gguf", s, "hf", st, "code agreement", (np.round(W / st).astype(np.int8) == v).mean())
    raise SystemExit

assert a.out, "--out required"
shutil.copyfile(a.gguf, a.out)
with open(a.out, "r+b") as f:
    if a.ckpt:
        ck = torch.load(a.ckpt)
        for L, ent in ck["layers"].items():
            for name in ("gate", "up", "down"):
                v = unpack_ckpt(ent[name], ent[name + "_shape"])                 # gate/up (d,F), down (F,d): in x out
                o, _ = off[f"blk.{L}.ffn_{name}.weight"]
                f.seek(o); f.write(pack_i2s(np.ascontiguousarray(v.T))); f.write(np.float32(ent[name + "_scale"]).tobytes())
            o, _ = off[f"blk.{L}.ffn_sub_norm.weight"]
            f.seek(o); f.write((ent["snw"].float() * float(ent["c"]) ** 0.5).numpy().astype("<f4").tobytes())
            print("layer", L, "c", round(float(ent["c"]), 4), flush=True)
    elif a.hf_codes:
        from common import load_model
        model, _ = load_model("cpu"); d, Fn = model.config.hidden_size, model.config.intermediate_size
        probe = lambda lin, n: torch.cat([lin(torch.eye(n, dtype=torch.bfloat16)[i:i + 512]).float() for i in range(0, n, 512)])
        with torch.no_grad():
            for L, layer in enumerate(model.model.layers):
                sites = [("ffn_gate", layer.mlp.gate_proj, d), ("ffn_up", layer.mlp.up_proj, d), ("ffn_down", layer.mlp.down_proj, Fn)]
                if a.hf_codes == 2:                                    # 2: attention projections too
                    at = layer.self_attn
                    sites += [("attn_q", at.q_proj, d), ("attn_k", at.k_proj, d), ("attn_v", at.v_proj, d), ("attn_output", at.o_proj, d)]
                for name, lin, n_in in sites:
                    W = probe(lin, n_in).T.numpy(); st = np.abs(W).max()
                    o, _ = off[f"blk.{L}.{name}.weight"]
                    f.seek(o); f.write(pack_i2s(np.round(W / st).astype(np.int8))); f.write(np.float32(st).tobytes())
                print("layer", L, flush=True)
    else:
        sel = torch.load(a.sel)
        for L, ent in sel.items():
            o, shp = off[f"blk.{L}.ffn_sub_norm.weight"]
            f.seek(o); w = np.frombuffer(f.read(shp[0] * 4), "<f4").copy()
            f.seek(o); f.write((w * float(ent["cov"][a.k]) ** 0.5).astype("<f4").tobytes())
            print("layer", L, "c", round(float(ent["cov"][a.k]), 4), flush=True)
print("wrote", a.out)
