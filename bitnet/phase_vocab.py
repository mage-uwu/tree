"""Tree (two-level) output layer. The tied output layer (128256 x 2560, f16 in bitnet.cpp) is ~40% of CPU decode
time. Cluster the vocabulary embeddings (k-means, C clusters); per token score the final hidden state against the
C centroids, take the top-m clusters as candidates and compute *exact* logits only for their tokens. Tokens in the
other clusters get the approximate logit h.mu_c + log-normal correction (so the distribution is defined over the
whole vocabulary and ppl / KL are measurable).

Reports: top-1 agreement with the exact head, exact probability mass covered by the candidates, KL(exact || tree),
ppl, and MACs per token (C*d + candidates*d vs V*d)."""
import argparse, math, time, torch
import torch.nn.functional as F
from common import STATE, load_model, wikitext, ultrachat, windows, log

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase_vocab.jsonl")
ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--clusters", default="512,1024,2048")
ap.add_argument("--tops", default="4,8,16,32")
ap.add_argument("--iters", type=int, default=20)
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

@torch.no_grad()
def kmeans(X, C, iters, w=None):
    g = torch.Generator(device=X.device).manual_seed(0)
    mu = X[torch.randperm(len(X), generator=g, device=X.device)[:C]].clone()
    for _ in range(iters):
        assign = torch.cat([(X[i:i + 16384] @ mu.T - 0.5 * (mu * mu).sum(1)).argmax(1) for i in range(0, len(X), 16384)])
        ww = torch.ones(len(X), device=X.device) if w is None else w
        cnt = torch.zeros(C, device=X.device).index_add_(0, assign, ww)
        s = torch.zeros(C, X.shape[1], device=X.device).index_add_(0, assign, X * ww[:, None])
        empty = cnt == 0
        mu = torch.where(empty[:, None], X[torch.randint(len(X), (C,), generator=g, device=X.device)], s / cnt.clamp(min=1e-9)[:, None])
    assign = torch.cat([(X[i:i + 16384] @ mu.T - 0.5 * (mu * mu).sum(1)).argmax(1) for i in range(0, len(X), 16384)])
    return mu, assign


def hidden(X):
    hs = []
    with torch.no_grad():
        STATE.enabled = False
        for i in range(len(X)):
            hs.append(model.model(X[i:i + 1].to(dev)).last_hidden_state[0, :-1].float())
    return torch.cat(hs), X[:, 1:].reshape(-1).to(dev)


H = {name: hidden(X) for name, X in [("wiki", wiki), ("chat", chat)]}
log(a.out, {"event": "hidden", "tokens": {k: len(v[0]) for k, v in H.items()}, "V": V, "d": d})

for C in map(int, a.clusters.split(",")):
    t = time.time()
    mu, assign = kmeans(E, C, a.iters)
    size = torch.bincount(assign, minlength=C).float()
    # per-cluster spread along the hidden direction is unknown; use isotropic variance for the log-normal term
    var = torch.zeros(C, device=dev).index_add_(0, assign, (E - mu[assign]).pow(2).sum(1)) / size.clamp(min=1) / d
    log(a.out, {"event": "kmeans", "C": C, "s": round(time.time() - t, 1), "max_size": int(size.max()), "median_size": int(size.median())})
    order = assign.argsort(); starts = torch.cumsum(size.long(), 0) - size.long()
    for m in map(int, a.tops.split(",")):
        rec = {"C": C, "top_clusters": m}
        for name, (Hs, y) in H.items():
            agree = mass = kl = nll = cand = 0.0; n = 0
            for i in range(0, len(Hs), 1024):
                h = Hs[i:i + 1024]; yy = y[i:i + 1024]
                z = h @ E.T                                                     # exact logits (B, V)
                lp = F.log_softmax(z, -1)
                sc = h @ mu.T                                                   # (B, C)
                top = sc.topk(m, -1).indices
                incand = torch.zeros(len(h), C, dtype=torch.bool, device=dev).scatter_(1, top, True)[:, assign]   # (B, V)
                approx = (sc + 0.5 * var[None] * h.pow(2).sum(1, keepdim=True))[:, assign]
                zt = torch.where(incand, z, approx)
                lq = F.log_softmax(zt, -1)
                agree += (zt.argmax(-1) == z.argmax(-1)).sum().item()
                mass += (lp.exp() * incand).sum().item()
                kl += (lp.exp() * (lp - lq)).sum().item()
                nll += -lq.gather(1, yy[:, None]).sum().item()
                cand += incand.sum().item(); n += len(h)
            rec.update({f"{name}_top1": round(agree / n, 4), f"{name}_mass": round(mass / n, 5), f"{name}_kl": round(kl / n, 5),
                        f"{name}_ppl": round(math.exp(nll / n), 4), f"{name}_cand_tokens": int(cand / n)})
        rec["macs"] = C * d + rec["wiki_cand_tokens"] * d
        rec["dense_macs"] = V * d
        log(a.out, rec)
# reference: exact head ppl on the same tokens
for name, (Hs, y) in H.items():
    nll = sum(-F.log_softmax(Hs[i:i + 1024] @ E.T, -1).gather(1, y[i:i + 1024, None]).sum().item() for i in range(0, len(Hs), 1024))
    log(a.out, {"event": "exact", "data": name, "ppl": round(math.exp(nll / len(Hs)), 4)})
log(a.out, {"event": "done", "total_s": round(time.time() - t0, 1)})
