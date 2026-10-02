"""Fit per-layer key trees (select+rescore attention) on base activations and export them for the engine.
File (little endian): magic "TKEY", int32 version=1, L, H (kv heads), S, D, hd; then for each layer, each kv head:
  float32 w[S][2^D-1][hd], float32 b[S][2^D-1], float32 c[S][2^D][hd]
Encoding a key k: r = k; for each tree s: walk D splits (go right if w.r - b > 0), leaf l; code_s = l; r -= c[s][l];
snap r to 0 if |r| < 1e-5 |k|.  Approximate score of query q: sum_s q . c[s][code_s]."""
import argparse, struct, time, torch
from common import STATE, load_model, wikitext, ultrachat, windows
from keytrees import KeyTrees

ap = argparse.ArgumentParser()
ap.add_argument("--S", type=int, default=32)
ap.add_argument("--calib_windows", type=int, default=64)
ap.add_argument("--out", default="/root/out/key_trees_S32.bin")
a = ap.parse_args()
model, tok = load_model("cuda")
cfg = model.config
Hq, Hkv, NL = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_hidden_layers
hd = cfg.hidden_size // Hq
calib = torch.cat([windows(tok, wikitext("train"), 2048, a.calib_windows * 3 // 4),
                   windows(tok, ultrachat("train_sft", tok, 1500), 2048, a.calib_windows // 4)])


class Stop(Exception):
    pass


def capture(L):
    K, Sq = [], torch.zeros(Hkv, hd, hd, device="cuda", dtype=torch.float64); n = [0]
    def cap(l, q, k):
        if l != L: return
        K.append(k.float().transpose(0, 1).reshape(Hkv, -1, hd))
        qg = q.float().reshape(q.shape[0], Hkv, Hq // Hkv, q.shape[2], hd).transpose(0, 1).reshape(Hkv, -1, hd).double()
        Sq.add_(torch.einsum("hnd,hne->hde", qg, qg)); n[0] += qg.shape[1]
        raise Stop
    STATE.capture = cap; STATE.enabled = False
    with torch.no_grad():
        for i in range(0, len(calib), 2):
            try: model.model(calib[i:i + 2].cuda())
            except Stop: pass
    STATE.capture = None
    return torch.cat(K, 1), (Sq / n[0]).float()


t = time.time()
with open(a.out, "wb") as f:
    f.write(b"TKEY"); f.write(struct.pack("<6i", 1, NL, Hkv, a.S, 4, hd))
    for L in range(NL):
        K, Sq = capture(L)
        kt = KeyTrees(Hkv, a.S, 4, hd).cuda(); kt.fit(K, Sq)
        # sanity: leaf sums reproduce the encoder's reconstruction
        xh, codes = kt.encode(K[:, :1000])
        rec = kt.c[torch.arange(Hkv, device="cuda")[:, None, None], torch.arange(a.S, device="cuda")[None, None, :], codes].sum(2)
        assert (rec - xh).abs().max() < 1e-3, "leaf-sum check failed"
        for h in range(Hkv):
            f.write(kt.w[h].float().cpu().numpy().astype("<f4").tobytes())
            f.write(kt.b[h].float().cpu().numpy().astype("<f4").tobytes())
            f.write(kt.c[h].float().cpu().numpy().astype("<f4").tobytes())
        print("layer", L, "degenerate", round(kt.degenerate_frac(), 4), round(time.time() - t, 1), "s", flush=True)
        del K
print("wrote", a.out)
