"""Phase 3: key trees on all layers, sequential calibration (layer l is fitted on keys produced with the
already-converted key trees of layers < l in place; MLPs are the original ones). One full conversion per S."""
import argparse, time, torch
from common import STATE, load_model, wikitext, ultrachat, windows, evaluate, log
from keytrees import KeyTrees

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase3.jsonl")
ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--configs", default="32,48,64")
ap.add_argument("--exact_sink", type=int, default=0)
ap.add_argument("--calib_wiki", type=int, default=96)
ap.add_argument("--calib_chat", type=int, default=32)
ap.add_argument("--eval_wiki", type=int, default=0)               # 0 = all
ap.add_argument("--eval_chat", type=int, default=64)
ap.add_argument("--save", default="")                             # dir to save fitted trees
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=None)
a = ap.parse_args()
dev = a.device
t0 = time.time()
model, tok = load_model(dev, path=a.model) if a.model else load_model(dev)
cfg = model.config
Hq, Hkv, NL = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_hidden_layers
hd = cfg.hidden_size // Hq

wiki_test = windows(tok, wikitext("test"), a.T)
if a.eval_wiki: wiki_test = wiki_test[:a.eval_wiki]
chat_test = windows(tok, ultrachat("test_sft", tok, 400), a.T, a.eval_chat)
calib = torch.cat([windows(tok, wikitext("train"), a.T, a.calib_wiki),
                   windows(tok, ultrachat("train_sft", tok, 1500), a.T, a.calib_chat)])


class Stop(Exception):
    pass


def capture(L, X):
    """keys and pooled query second moments of layer L, with the current STATE.keys (layers < L) active."""
    K, Sq = [], torch.zeros(Hkv, hd, hd, device=dev, dtype=torch.float64); n = [0]
    def cap(l, q, k):
        if l != L: return
        K.append(k.float().transpose(0, 1).reshape(Hkv, -1, hd))
        qg = q.float().reshape(q.shape[0], Hkv, Hq // Hkv, q.shape[2], hd).transpose(0, 1).reshape(Hkv, -1, hd).double()
        Sq.add_(torch.einsum("hnd,hne->hde", qg, qg)); n[0] += qg.shape[1]
        raise Stop
    STATE.capture = cap; STATE.enabled = True
    with torch.no_grad():
        for i in range(0, len(X), a.bs):
            try:
                model.model(X[i:i + a.bs].to(dev))
            except Stop:
                pass
    STATE.capture = None
    return torch.cat(K, 1), (Sq / n[0]).float()


for S in map(int, a.configs.split(",")):
    t = time.time(); STATE.keys = {}
    for L in range(NL):
        K, Sq = capture(L, calib)
        kt = KeyTrees(Hkv, S, 4, hd, exact_sink=bool(a.exact_sink)).to(dev)
        kt.fit(K, Sq)
        STATE.keys[L] = kt
        del K
    fit_s = time.time() - t
    rec = {"phase": 3, "S": S, "D": 4, "exact_sink": bool(a.exact_sink), "bytes_per_key_per_kvhead": S / 2,
           "kv_cache_key_bytes_per_token": NL * Hkv * S // 2, "fp16_key_bytes_per_token": NL * Hkv * hd * 2, "fit_s": round(fit_s, 1)}
    for name, X in [("wiki", wiki_test), ("chat", chat_test)]:
        r = evaluate(model, X, a.bs, kl=True, device=dev)
        rec.update({f"{name}_ppl": round(r["ppl"], 4), f"{name}_kl": round(r["kl"], 5), f"{name}_top1": round(r["top1_agree"], 4)})
    log(a.out, rec)
    if a.save:
        torch.save({L: kt.state_dict() for L, kt in STATE.keys.items()} | {"S": S, "D": 4, "exact_sink": bool(a.exact_sink)},
                   f"{a.save}/keytrees_S{S}.pt")
    STATE.keys = {}
log(a.out, {"event": "done", "total_s": round(time.time() - t0, 1)})
