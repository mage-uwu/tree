"""Tree output layer (select + rescore over the vocabulary). The tied output layer (128256 x 2560, f16 in bitnet.cpp) is ~40% of CPU decode
time. Cluster the vocabulary embeddings (k-means, C clusters); per token score the final hidden state against the
C centroids, take the top-m clusters as candidates and compute *exact* logits only for their tokens. Tokens in the
other clusters get the approximate logit h.mu_c + log-normal correction (so the distribution is defined over the
whole vocabulary and ppl / KL are measurable).

Reports: top-1 agreement with the exact head, exact probability mass covered by the candidates, KL(exact || tree),
ppl. Per-token cost: table build S*16*d MACs + V*S/2 byte lookups + N*d MACs (vs V*d)."""
import argparse, math, time, torch
import torch.nn.functional as F
from common import STATE, load_model, wikitext, ultrachat, windows, log
from keytrees import KeyTrees

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase_vocab.jsonl")
ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--trees", default="16,32,64")
ap.add_argument("--cands", default="64,256,1024")
ap.add_argument("--calib_windows", type=int, default=32)
ap.add_argument("--eval_wiki", type=int, default=16)
ap.add_argument("--eval_chat", type=int, default=16)
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=None)
a = ap.parse_args()
dev = a.device
t0 = time.time()
model, tok = load_model(dev, path=a.model) if a.model else load_model(dev)
E = model.lm_head.weight.float()                      # (V, d) tied embeddings
V, d = E.shape
wiki = windows(tok, wikitext("test"), a.T)[:a.eval_wiki]
chat = windows(tok, ultrachat("test_sft", tok, 400), a.T, a.eval_chat)

def hidden(X):
    hs = []
    with torch.no_grad():
        STATE.enabled = False
        for i in range(len(X)):
            hs.append(model.model(X[i:i + 1].to(dev)).last_hidden_state[0, :-1].float())
    return torch.cat(hs), X[:, 1:].reshape(-1).to(dev)


H = {name: hidden(X) for name, X in [("wiki", wiki), ("chat", chat)]}
log(a.out, {"event": "hidden", "tokens": {k: len(v[0]) for k, v in H.items()}, "V": V, "d": d})

calib = windows(tok, wikitext("train"), a.T, a.calib_windows)
Hc, _ = hidden(calib)
Sq = (Hc.T @ Hc / len(Hc))[None]                      # hidden-state second moments (query metric)
del Hc
for S in map(int, a.trees.split(",")):
    t = time.time()
    kt = KeyTrees(1, S, 4, d).to(dev); kt.fit(E[None], Sq)
    Eh = kt.encode(E[None])[0][0]                     # tree reconstruction of every embedding (V, d)
    log(a.out, {"event": "fit", "S": S, "bytes_per_token": S // 2, "s": round(time.time() - t, 1),
                "degenerate_nodes": round(kt.degenerate_frac(), 4),
                "rel_err": round(((Eh - E) @ Sq[0] * (Eh - E)).sum().item() / ((E @ Sq[0]) * E).sum().item(), 5)})
    for N in map(int, a.cands.split(",")):
        rec = {"S": S, "bytes_per_token": S // 2, "cands": N}
        for name, (Hs, y) in H.items():
            agree = mass = kl = nll = 0.0; n = 0
            for i in range(0, len(Hs), 1024):
                h = Hs[i:i + 1024]; yy = y[i:i + 1024]
                z = h @ E.T; lp = F.log_softmax(z, -1)
                za = h @ Eh.T                                                   # approximate logits (tree scores)
                top = za.topk(N, -1).indices
                incand = torch.zeros_like(z, dtype=torch.bool).scatter_(1, top, True)
                zt = torch.where(incand, z, za)
                lq = F.log_softmax(zt, -1)
                agree += (zt.argmax(-1) == z.argmax(-1)).sum().item()
                mass += (lp.exp() * incand).sum().item()
                kl += (lp.exp() * (lp - lq)).sum().item()
                nll += -lq.gather(1, yy[:, None]).sum().item(); n += len(h)
            rec.update({f"{name}_top1": round(agree / n, 5), f"{name}_mass": round(mass / n, 5), f"{name}_kl": round(kl / n, 5),
                        f"{name}_ppl": round(math.exp(nll / n), 4)})
        rec["macs"] = S * 16 * d + N * d; rec["lookup_bytes"] = V * S // 2; rec["dense_macs"] = V * d
        log(a.out, rec)
    del Eh, kt
# reference: exact head ppl on the same tokens
for name, (Hs, y) in H.items():
    nll = sum(-F.log_softmax(Hs[i:i + 1024] @ E.T, -1).gather(1, y[i:i + 1024, None]).sum().item() for i in range(0, len(Hs), 1024))
    log(a.out, {"event": "exact", "data": name, "ppl": round(math.exp(nll / len(Hs)), 4)})
log(a.out, {"event": "done", "total_s": round(time.time() - t0, 1)})
