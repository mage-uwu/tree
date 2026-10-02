"""Tree-routed sparse MLP: select + rescore over neurons.

Picking the active neurons is a maximum-inner-product search: gate_i = w_i . x over the 6912 ternary gate rows.
Encode the gate rows with S boosted 16-leaf trees fitted in the metric of the (int8-quantized) MLP inputs, exactly
like the vocabulary trees. Per token:
  1. tree scores for all 6912 neurons (table build S*16*d MACs + 6912*S/2 byte lookups)
  2. top-C neurons by tree score are candidates; their gate is computed exactly (C*d ternary MACs)
  3. among candidates keep the fewest whose relu(g)^2 covers `frac` of the candidates' total; up/down exact (2k*d)
The dense exact-gate selector (phase 5) costs F*d for step 2 instead; this removes most of it.
Reports per layer and all-layer: rel err, KL, ppl, mean candidates' share of the true relu(g)^2 energy, MACs."""
import argparse, time, torch
from common import STATE, load_model, wikitext, ultrachat, windows, evaluate, log
from keytrees import KeyTrees
from transformers.integrations.bitnet import ActQuant

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase_treegate.jsonl")
ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--layers", default="15,2,28")
ap.add_argument("--trees", default="16,32,64")
ap.add_argument("--cands", default="1536,2048,3072")
ap.add_argument("--frac", type=float, default=0.99)
ap.add_argument("--all", default="32:2048,32:3072,64:2048")     # all-layer configs S:C ("" to skip)
ap.add_argument("--calib_windows", type=int, default=48)
ap.add_argument("--eval_wiki", type=int, default=32)
ap.add_argument("--eval_chat", type=int, default=16)
ap.add_argument("--all_eval_wiki", type=int, default=0)
ap.add_argument("--all_eval_chat", type=int, default=64)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=None)
a = ap.parse_args()
dev = a.device
t0 = time.time()
model, tok = load_model(dev, path=a.model) if a.model else load_model(dev)
d, F, NL = model.config.hidden_size, model.config.intermediate_size, model.config.num_hidden_layers
wiki_test = windows(tok, wikitext("test"), a.T)
chat_test = windows(tok, ultrachat("test_sft", tok, 400), a.T, max(a.eval_chat, a.all_eval_chat))
calib = torch.cat([windows(tok, wikitext("train"), a.T, a.calib_windows * 3 // 4),
                   windows(tok, ultrachat("train_sft", tok, 1500), a.T, a.calib_windows // 4)])


def ternary(W):
    s = 1.0 / W.abs().mean().clamp(min=1e-5)
    return (W * s).round().clamp(-1, 1) / s


# per-layer second moments of the int8-quantized MLP inputs (one pass over calibration)
C = torch.zeros(NL, d, d, device=dev, dtype=torch.float64); n = [0] * NL
def cap(l, u, y):
    x = ActQuant.apply(u.reshape(-1, d)).double(); C[l] += x.T @ x; n[l] += len(x)
STATE.mlp_capture = cap; STATE.enabled = False
with torch.no_grad():
    for i in range(0, len(calib), a.bs):
        model.model(calib[i:i + a.bs].to(dev))
STATE.mlp_capture = None; STATE.enabled = True
C = (C / torch.tensor(n, device=dev, dtype=torch.float64)[:, None, None]).float()
log(a.out, {"event": "calib", "tokens": n[0], "s": round(time.time() - t0, 1)})

STAT = {"cand_energy": 0.0, "k": 0.0, "n": 0}


def fit_trees(l, S):
    W = ternary(model.model.layers[l].mlp.gate_proj.weight.float())
    kt = KeyTrees(1, S, 4, d).to(dev); kt.fit(W[None], C[l][None])
    return kt.encode(W[None])[0][0], W                           # W_hat (F,d), W


def tree_mlp(mlp, What, Cn, frac):
    def f(u):
        sh = u.shape; U = u.reshape(-1, d); out = []
        for i in range(0, len(U), 4096):
            x = U[i:i + 4096]
            g = mlp.gate_proj(x); h = mlp.act_fn(g) * mlp.up_proj(x)
            if What is None:
                cand = torch.ones_like(g, dtype=torch.bool)
            else:
                gh = ActQuant.apply(x).float() @ What.T
                cand = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, gh.topk(Cn, -1).indices, True)
            e_all = torch.relu(g).float().pow(2)
            e = e_all * cand
            v, idx = e.sort(-1, descending=True); c = v.cumsum(-1)
            m = torch.zeros_like(cand).scatter_(-1, idx, ((c - v) < frac * c[:, -1:]) & (v > 0))
            STAT["cand_energy"] += (e.sum(-1) / e_all.sum(-1).clamp(min=1e-20)).sum().item()
            STAT["k"] += m.sum().item(); STAT["n"] += len(x)
            out.append(mlp.down_proj(mlp.ffn_sub_norm(h * m)))
        return torch.cat(out).reshape(sh)
    return f


def ev(rec, wiki, chat):
    STAT.update(cand_energy=0.0, k=0.0, n=0)
    for name, X in [("wiki", wiki), ("chat", chat)]:
        r = evaluate(model, X, a.bs, kl=True, device=dev)
        rec.update({f"{name}_ppl": round(r["ppl"], 4), f"{name}_kl": round(r["kl"], 5), f"{name}_top1": round(r["top1_agree"], 4)})
    if STAT["n"]:
        rec["cand_energy_share"] = round(STAT["cand_energy"] / STAT["n"], 5); rec["mean_k"] = round(STAT["k"] / STAT["n"], 1)
        S, Cn = rec.get("S", 0), rec.get("C", F)
        rec["mlp_macs"] = int(S * 16 * d + (Cn if S else F) * d + 2 * rec["mean_k"] * d)
        rec["lookup_bytes"] = F * S // 2
    rec["dense_mlp_macs"] = 3 * F * d
    log(a.out, rec)


W8, C8 = wiki_test[:a.eval_wiki], chat_test[:a.eval_chat]
for L in map(int, filter(None, a.layers.split(","))):
    mlp = model.model.layers[L].mlp
    STATE.mlps = {L: tree_mlp(mlp, None, F, a.frac)}
    ev({"layer": L, "sel": f"exact gate, energy {a.frac}"}, W8, C8)
    for S in map(int, a.trees.split(",")):
        t = time.time(); What, W = fit_trees(L, S); fs = time.time() - t
        # score quality on held-out-ish inputs: correlation of tree gate scores with exact
        xs = ActQuant.apply(torch.randn(4096, d, device=dev) @ torch.linalg.cholesky(C[L].double() + 1e-6 * torch.eye(d, device=dev, dtype=torch.float64)).float().T)
        g, gh = xs @ W.T, xs @ What.T
        corr = torch.corrcoef(torch.stack([g.flatten()[:4_000_000], gh.flatten()[:4_000_000]]))[0, 1].item()
        for Cn in map(int, a.cands.split(",")):
            STATE.mlps = {L: tree_mlp(mlp, What, Cn, a.frac)}
            ev({"layer": L, "sel": "tree select+rescore", "S": S, "C": Cn, "fit_s": round(fs, 1), "gate_corr": round(corr, 4),
                "degenerate_nodes": None}, W8, C8)
    STATE.mlps = {}

for cfg in filter(None, a.all.split(",")):
    S, Cn = map(int, cfg.split(":"))
    t = time.time(); mlps = {}
    for l in range(NL):
        What, _ = fit_trees(l, S)
        mlps[l] = tree_mlp(model.model.layers[l].mlp, What, Cn, a.frac)
    STATE.mlps = mlps
    Wa = wiki_test[:a.all_eval_wiki] if a.all_eval_wiki else wiki_test
    ev({"layer": "all", "sel": "tree select+rescore", "S": S, "C": Cn, "fit_s": round(time.time() - t, 1)}, Wa, chat_test[:a.all_eval_chat])
    STATE.mlps = {}
log(a.out, {"event": "done", "total_s": round(time.time() - t0, 1)})
