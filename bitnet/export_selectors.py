"""Export Phase-13 neuron selectors for the engine: per layer, the rank-r SVD of the MLP gate (g ~ (u A) B^T, A = U_r S^1/2,
B = V_r S^1/2), quantised per row: A^T int8 [r][d] + f32 scale[r], B int4 [F][r/2] (two per byte, value+8, low nibble =
even index) + f32 scale[F]. Optional --ckpt: selectors from a phase_peer.py --save checkpoint instead of plain SVD.
File: "TSEL", int32 version=1, L, d, F, r, then per layer: A8, sA, B4, sB."""
import argparse, struct, time, torch
from common import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--r", type=int, default=256)
ap.add_argument("--out", default="sel_r256.bin")
ap.add_argument("--ckpt", default="")
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
t0 = time.time()
model, tok = load_model(a.device)
cfg = model.config; d, Fn, NL = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
ck = torch.load(a.ckpt) if a.ckpt else None
r = a.r if ck is None else ck["layers"][0]["A"].shape[1]
with open(a.out, "wb") as f:
    f.write(b"TSEL"); f.write(struct.pack("<5i", 1, NL, d, Fn, r))
    for L in range(NL):
        with torch.no_grad():
            if ck is None:
                gp = model.model.layers[L].mlp.gate_proj
                Wg = torch.cat([gp(torch.eye(d, dtype=torch.bfloat16, device=a.device)[i:i + 512]).float() for i in range(0, d, 512)])
                U, S, Vh = torch.linalg.svd(Wg, full_matrices=False)
                A = U[:, :r] * S[:r].sqrt(); B = Vh[:r].T * S[:r].sqrt()          # (d,r), (F,r)
            else:
                A = ck["layers"][L]["A"].float(); B = ck["layers"][L]["B"].float().T
            At = A.T.contiguous()
            sA = At.abs().amax(1).clamp_min(1e-12) / 127; A8 = (At / sA[:, None]).round().clamp(-127, 127).to(torch.int8)
            sB = B.abs().amax(1).clamp_min(1e-12) / 7; B4 = (B / sB[:, None]).round().clamp(-7, 7).to(torch.int16) + 8
            packed = (B4[:, 0::2] | (B4[:, 1::2] << 4)).to(torch.uint8)
        f.write(A8.cpu().numpy().tobytes()); f.write(sA.float().cpu().numpy().astype("<f4").tobytes())
        f.write(packed.cpu().numpy().tobytes()); f.write(sB.float().cpu().numpy().astype("<f4").tobytes())
        print("layer", L, round(time.time() - t0, 1), "s", flush=True)
print("wrote", a.out)
