"""Train a standard BitNet b1.58 attention-only LM (the thing we will convert)."""
import argparse, json, math, time
import torch, torch.nn.functional as F
from tree_attn import LM
import data

p = argparse.ArgumentParser()
p.add_argument("--task", default="shakespeare")
p.add_argument("--layers", type=int, default=1)
p.add_argument("--d", type=int, default=128)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--mlp", type=int, default=0)
p.add_argument("--threads", type=int, default=1)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--out", required=True)
a = p.parse_args()
torch.manual_seed(0); torch.set_num_threads(a.threads)
get, vocab, chars = data.shakespeare(128) if a.task == "shakespeare" else data.mqar(32)
cfg = dict(vocab=vocab, d=a.d, heads=a.heads, layers=a.layers, task=a.task, mlp=a.mlp)
m = LM(cfg); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
t0 = time.time()
for step in range(a.steps + 1):
    lr = a.lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
    for g in opt.param_groups: g["lr"] = lr
    x, y = get("train", 32)
    loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten(), ignore_index=-100)
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
    if step % 250 == 0: print(f"step {step} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
torch.save({"cfg": cfg, "chars": chars, "state": m.state_dict()}, a.out)
print("saved", a.out)
