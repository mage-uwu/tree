import argparse, json, math, random, time
import torch
import torch.nn.functional as F
from model import Chimera

p = argparse.ArgumentParser()
p.add_argument("--ffn", default="fff", choices=["fff", "dense"])
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--d", type=int, default=128)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--depth", type=int, default=8)
p.add_argument("--trees", type=int, default=2)
p.add_argument("--ctx", type=int, default=128)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--aux", type=float, default=0.05)
p.add_argument("--matryoshka", type=float, default=0.5, help="prob. of random depth truncation per step")
p.add_argument("--out", default=None)
args = p.parse_args()
torch.manual_seed(0); random.seed(0)

text = open("../input.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
n = int(0.9 * len(data))
tr, va = data[:n], data[n:]

cfg = dict(vocab=len(chars), d=args.d, layers=args.layers, heads=args.heads,
           depth=args.depth, trees=args.trees, ffn=args.ffn, ctx=args.ctx)
model = Chimera(cfg)
print(cfg, sum(p.numel() for p in model.parameters()), "params")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)


def batch(src, bs):
    ix = torch.randint(len(src) - args.ctx - 1, (bs,))
    x = torch.stack([src[i:i + args.ctx] for i in ix])
    y = torch.stack([src[i + 1:i + args.ctx + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(max_depth=None, iters=20):
    model.eval()
    g = torch.Generator().manual_seed(123)
    tot = 0
    for _ in range(iters):
        ix = torch.randint(len(va) - args.ctx - 1, (32,), generator=g)
        x = torch.stack([va[i:i + args.ctx] for i in ix])
        y = torch.stack([va[i + 1:i + args.ctx + 1] for i in ix])
        tot += F.cross_entropy(model(x, max_depth).flatten(0, 1), y.flatten()).item()
    model.train()
    return tot / iters


t0 = time.time()
warm = 100
for step in range(args.steps + 1):
    lr = args.lr * min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
    for g in opt.param_groups:
        g["lr"] = lr
    md = None
    if args.ffn == "fff" and random.random() < args.matryoshka:
        md = random.randint(2, args.depth)
    x, y = batch(tr, args.bs)
    loss = F.cross_entropy(model(x, md).flatten(0, 1), y.flatten())
    aux = model.aux_loss()
    (loss + args.aux * aux).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.3f} aux {aux.item():.4f} lr {lr:.2e} {time.time()-t0:.0f}s", flush=True)
    if step % 500 == 0 and step:
        print(f"  val {evaluate():.3f}", flush=True)

res = {"cfg": cfg, "val_full": evaluate(iters=50), "train_time_s": time.time() - t0}
if args.ffn == "fff":
    res["val_by_depth"] = {k: evaluate(k, iters=50) for k in range(2, args.depth + 1)}
print(json.dumps(res, indent=1))
out = args.out or f"ckpt_{args.ffn}.pt"
torch.save({"cfg": cfg, "chars": chars, "state": model.state_dict(), "res": res}, out)
