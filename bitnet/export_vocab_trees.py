"""Fit vocabulary trees on the real model and export them for the engine.
File layout (little endian):
  magic "TVOC", int32 version=1, V, d, S
  float32 leaves[S][16][d]          leaf vectors (residual chain, folded to the original space)
  uint8   codes[NB][S/2][32]        NB = ceil(V/32) blocks of 32 tokens; byte p of token j in block b holds
                                    tree 2p in the low nibble and tree 2p+1 in the high nibble (pad tokens: 0)
Approximate logit of token v = sum_s h . leaves[s][code_s(v)]."""
import argparse, struct, time, torch
import numpy as np
from common import STATE, load_model, wikitext, windows
from keytrees import KeyTrees

ap = argparse.ArgumentParser()
ap.add_argument("--S", type=int, default=128)
ap.add_argument("--calib_windows", type=int, default=32)
ap.add_argument("--out", default="/root/out/vocab_trees_S128.bin")
a = ap.parse_args()
model, tok = load_model("cuda")
E = model.lm_head.weight.float(); V, d = E.shape
calib = windows(tok, wikitext("train"), 2048, a.calib_windows)
Hc = []
with torch.no_grad():
    STATE.enabled = False
    for i in range(len(calib)):
        Hc.append(model.model(calib[i:i + 1].cuda()).last_hidden_state[0].float())
Hc = torch.cat(Hc); Sq = (Hc.T @ Hc / len(Hc))[None]
t = time.time()
kt = KeyTrees(1, a.S, 4, d).cuda(); kt.fit(E[None], Sq)
Eh, codes = kt.encode(E[None]); Eh, codes = Eh[0], codes[0]                 # (V,d), (V,S)
print("fit", round(time.time() - t, 1), "s; rel err", ((Eh - E) @ Sq[0] * (Eh - E)).sum().item() / ((E @ Sq[0]) * E).sum().item())
# sanity: leaves reproduce Eh
leaves = kt.c[0]                                                               # (S,16,d)
chk = leaves[torch.arange(a.S, device="cuda")[None, :], codes[:4]].sum(1)
print("leaf-sum check max abs diff", (chk - Eh[:4]).abs().max().item())
NB = (V + 31) // 32
cpad = torch.zeros(NB * 32, a.S, dtype=torch.uint8, device="cuda"); cpad[:V] = codes.to(torch.uint8)
packed = (cpad[:, 0::2] | (cpad[:, 1::2] << 4))                                # (NB*32, S/2)
packed = packed.reshape(NB, 32, a.S // 2).transpose(1, 2).contiguous()        # (NB, S/2, 32)
with open(a.out, "wb") as f:
    f.write(b"TVOC"); f.write(struct.pack("<4i", 1, V, d, a.S))
    f.write(leaves.float().cpu().numpy().astype("<f4").tobytes())
    f.write(packed.cpu().numpy().tobytes())
print("wrote", a.out)
