import argparse, json, math, random, time
import torch
import torch.nn.functional as F
from attn_model import AttnLM

p = argparse.ArgumentParser()
p.add_argument("--task", default="shakespeare", choices=["shakespeare", "mqar"])
p.add_argument("--kinds", default="RRRR")
p.add_argument("--rdepth", type=int, default=4)
p.add_argument("--layers", type=int, default=1)
p.add_argument("--local", type=int, default=0)
p.add_argument("--d", type=int, default=128)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--ctx", type=int, default=128)
p.add_argument("--pairs", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--aux", type=float, default=0.05)
p.add_argument("--threads", type=int, default=1)
p.add_argument("--out", required=True)
args = p.parse_args()
torch.manual_seed(0); random.seed(0); torch.set_num_threads(args.threads)

# ------------------------------------------------------------------ data
if args.task == "shakespeare":
    text = open("../../input.txt").read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text]); n = int(0.9 * len(data))
    tr, va = data[:n], data[n:]
    vocab = len(chars)

    def get(split, bs, g=None):
        src = tr if split == "train" else va
        ix = torch.randint(len(src) - args.ctx - 1, (bs,), generator=g)
        x = torch.stack([src[i:i + args.ctx] for i in ix]); y = torch.stack([src[i + 1:i + args.ctx + 1] for i in ix])
        return x, y
else:
    # MQAR: k1 v1 ... kN vN | kq vq kq vq ...; loss only on predicting vq after kq
    NK, NV = 64, 64
    vocab, chars = NK + NV, None

    def mqar(bs, pairs, queries, g=None):
        keys = torch.argsort(torch.rand(bs, NK, generator=g), -1)[:, :pairs]
        vals = torch.randint(NV, (bs, pairs), generator=g) + NK
        qi = torch.randint(pairs, (bs, queries), generator=g)
        qk, qv = keys.gather(1, qi), vals.gather(1, qi)
        seq = torch.cat([torch.stack([keys, vals], -1).flatten(1), torch.stack([qk, qv], -1).flatten(1)], 1)
        x, y = seq[:, :-1], seq[:, 1:].clone()
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[:, 2 * pairs::2] = True                # positions whose target is a query value
        y[~mask] = -100
        return x, y

    def get(split, bs, g=None, pairs=None):
        p_ = pairs or args.pairs
        return mqar(bs, p_, p_, g)

cfg = dict(vocab=vocab, d=args.d, layers=args.layers, kinds=args.kinds, rdepth=args.rdepth, task=args.task, local=bool(args.local))
model = AttnLM(cfg)
print(cfg, sum(p.numel() for p in model.parameters()), "params", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)


@torch.no_grad()
def evaluate(iters=20, pairs=None):
    model.eval(); g = torch.Generator().manual_seed(123)
    loss = acc = 0
    for _ in range(iters):
        x, y = get("val", 32, g, pairs) if args.task == "mqar" else get("val", 32, g)
        lg = model(x)
        loss += F.cross_entropy(lg.flatten(0, 1), y.flatten(), ignore_index=-100).item()
        m = y != -100
        acc += (lg.argmax(-1)[m] == y[m]).float().mean().item()
    model.train()
    return loss / iters, acc / iters


t0 = time.time()
for step in range(args.steps + 1):
    lr = args.lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
    for gp in opt.param_groups:
        gp["lr"] = lr
    x, y = get("train", args.bs)
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
    aux = model.aux_loss()
    (loss + args.aux * aux).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if step % 250 == 0:
        print(f"step {step} loss {loss.item():.3f} aux {float(aux):.4f} {time.time()-t0:.0f}s", flush=True)

res = {"cfg": cfg, "train_time_s": time.time() - t0}
if args.task == "shakespeare":
    res["val_loss"], _ = evaluate(50)
else:
    res["acc_by_pairs"] = {pp: evaluate(10, pp)[1] for pp in [8, 16, 32, 48, 64]}
# leaf usage (keys) on a val batch
with torch.no_grad():
    model.eval()
    x, _ = get("val", 32, torch.Generator().manual_seed(1))
    m0 = model.mixers[0]; u = model.norms[0](model.emb(x))
    k = m0._heads(m0.k(u * m0.mu_k + F.pad(u, (0, 0, 1, 0))[:, :-1] * (1 - m0.mu_k)))
    P, _ = m0.router(k)
    res["leaves_used_per_head"] = [(P[:, h].sum((0, 1)) > 0).sum().item() for h in range(P.shape[1])]
print(json.dumps(res), flush=True)
torch.save({"cfg": cfg, "chars": chars, "state": model.state_dict(), "res": res}, args.out)
