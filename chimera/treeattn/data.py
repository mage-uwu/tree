import torch

_TEXT = None


def shakespeare(ctx):
    global _TEXT
    text = open(__file__.rsplit("/", 2)[0] + "/../input.txt").read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text]); n = int(0.9 * len(data))
    tr, va = data[:n], data[n:]

    def get(split, bs, g=None):
        src = tr if split == "train" else va
        ix = torch.randint(len(src) - ctx - 1, (bs,), generator=g)
        return (torch.stack([src[i:i + ctx] for i in ix]), torch.stack([src[i + 1:i + ctx + 1] for i in ix]))
    return get, len(chars), chars


def mqar(pairs_default):
    """k1 v1 .. kN vN | kq vq ..., loss only on query values. vocab 128 (64 keys, 64 values)."""
    NK, NV = 64, 64

    def get(split, bs, g=None, pairs=None):
        P = pairs or pairs_default
        keys = torch.argsort(torch.rand(bs, NK, generator=g), -1)[:, :P]
        vals = torch.randint(NV, (bs, P), generator=g) + NK
        qi = torch.randint(P, (bs, P), generator=g)
        seq = torch.cat([torch.stack([keys, vals], -1).flatten(1),
                         torch.stack([keys.gather(1, qi), vals.gather(1, qi)], -1).flatten(1)], 1)
        x, y = seq[:, :-1], seq[:, 1:].clone()
        m = torch.zeros_like(y, dtype=torch.bool); m[:, 2 * P::2] = True; y[~m] = -100
        return x, y
    return get, NK + NV, None
