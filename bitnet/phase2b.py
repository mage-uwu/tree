"""Phase 2b: attention-side experiments with the custom `tree` attention.
1. floor: custom exact attention (fp32, no pruning) on all layers vs the stock sdpa path.
2. value pruning by exact score (TAU) on all layers: quality vs fraction of keys/values read.
3. select + rescore on one layer: key trees choose keys (within tau_sel of the max tree score, plus the
   sink and the `recent` newest keys), exact keys score them, only their values are read."""
import argparse, time, torch
from common import STATE, load_model, wikitext, ultrachat, windows, evaluate, log
from keytrees import KeyTrees

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase2b.jsonl")
ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--taus", default="4,6,8,10,12")
ap.add_argument("--layers", default="15,2")
ap.add_argument("--S", default="16,32")
ap.add_argument("--tau_sel", default="4,8,12")
ap.add_argument("--recent", type=int, default=64)
ap.add_argument("--calib_wiki", type=int, default=96)
ap.add_argument("--calib_chat", type=int, default=32)
ap.add_argument("--eval_wiki", type=int, default=32)
ap.add_argument("--eval_chat", type=int, default=16)
ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=None)
a = ap.parse_args()
dev = a.device
t0 = time.time()
model, tok = load_model(dev, path=a.model) if a.model else load_model(dev)
cfg = model.config
Hq, Hkv, NL = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_hidden_layers
hd = cfg.hidden_size // Hq
wiki_test = windows(tok, wikitext("test"), a.T)[:a.eval_wiki]
chat_test = windows(tok, ultrachat("test_sft", tok, 400), a.T, a.eval_chat)
calib = torch.cat([windows(tok, wikitext("train"), a.T, a.calib_wiki),
                   windows(tok, ultrachat("train_sft", tok, 1500), a.T, a.calib_chat)])


def ev(rec):
    STATE.attn_stats = {}
    for name, X in [("wiki", wiki_test), ("chat", chat_test)]:
        r = evaluate(model, X, a.bs, kl=True, device=dev)
        rec.update({f"{name}_ppl": round(r["ppl"], 4), f"{name}_kl": round(r["kl"], 5), f"{name}_top1": round(r["top1_agree"], 4)})
    if STATE.attn_stats:
        fr = {L: k / n for L, (k, n) in STATE.attn_stats.items()}
        rec["read_frac_mean"] = round(sum(fr.values()) / len(fr), 4)
        rec["read_frac_by_layer"] = {L: round(f, 3) for L, f in sorted(fr.items()) if L % 5 == 0 or L == NL - 1}
    log(a.out, rec)


STATE.attn = {L: {} for L in range(NL)}
ev({"exp": "floor: custom exact attention, all layers"})
for tau in map(float, a.taus.split(",")):
    STATE.attn = {L: {"tau_v": tau} for L in range(NL)}
    ev({"exp": "value pruning, all layers", "tau_v": tau})
STATE.attn = {}


def capture_layer(L, X):
    K, Sq = [], torch.zeros(Hkv, hd, hd, device=dev, dtype=torch.float64); n = [0]
    def cap(l, q, k):
        if l != L: return
        K.append(k.float().transpose(0, 1).reshape(Hkv, -1, hd))
        qg = q.float().reshape(q.shape[0], Hkv, Hq // Hkv, q.shape[2], hd).transpose(0, 1).reshape(Hkv, -1, hd).double()
        Sq.add_(torch.einsum("hnd,hne->hde", qg, qg)); n[0] += qg.shape[1]
    STATE.capture = cap; STATE.enabled = False
    with torch.no_grad():
        for i in range(0, len(X), 2):
            model.model(X[i:i + 2].to(dev))
    STATE.capture = None; STATE.enabled = True
    return torch.cat(K, 1), (Sq / n[0]).float()


for L in map(int, a.layers.split(",")):
    K, Sq = capture_layer(L, calib)
    STATE.attn = {L: {}}
    ev({"exp": "floor one layer", "layer": L})
    for S in map(int, a.S.split(",")):
        kt = KeyTrees(Hkv, S, 4, hd, exact_sink=True).to(dev); kt.fit(K, Sq)
        STATE.keys = {L: kt}; STATE.select = set(); STATE.attn = {L: {}}
        ev({"exp": "replace (k_hat scores)", "layer": L, "S": S})
        STATE.select = {L}
        for ts in map(float, a.tau_sel.split(",")):
            STATE.attn = {L: {"tau_sel": ts, "recent": a.recent}}
            ev({"exp": "select+rescore", "layer": L, "S": S, "tau_sel": ts, "recent": a.recent})
        STATE.keys = {}; STATE.select = set(); STATE.attn = {}
    del K
log(a.out, {"event": "done", "total_s": round(time.time() - t0, 1)})
