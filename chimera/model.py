"""
Chimera: train as a dense-ish GPU-friendly net, infer as a CPU tree + RNN.

  - BitLinear: BitNet b1.58 conventions (absmean ternary weights per tensor,
    absmax int8 activations per token, STE for both). Layout [out, in].
  - Retention (RetNet-style, simplified): parallel form for training,
    recurrent form for inference. Per-head RMSNorm makes them exactly equal
    without RetNet's extra normalisations.
  - FFF: Fast Feedforward tree (UltraFastBERT-style) with ternary node
    weights and a hard-route STE: forward takes one path, backward uses the
    sigmoid gradient. Training computes all nodes densely (tensor-core food),
    inference walks one root->leaf path per tree.
  - Matryoshka depth: random depth truncation during training so inference
    can exit early (anytime / min-path knob).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- quant (b1.58)
def weight_quant(w):
    """absmean ternary. returns (dequantised-with-STE, codes, scale)."""
    s = w.abs().mean().clamp(min=1e-5)
    q = (w / s).round().clamp(-1, 1)
    return w + (q * s - w).detach(), q, s


def act_quant(x):
    """absmax int8 per token (last dim). returns STE-dequantised x."""
    s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    q = (x * s).round().clamp(-128, 127)
    return x + (q / s - x).detach()


class BitLinear(nn.Linear):
    """BitNet b1.58 linear (no bias). Input should already be RMSNorm'd."""
    def __init__(self, i, o):
        super().__init__(i, o, bias=False)

    def forward(self, x):
        w, _, _ = weight_quant(self.weight)
        return F.linear(act_quant(x), w)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def hard_route(logit):
    """forward: 1 if logit>0 else 0. backward: d sigmoid."""
    soft = torch.sigmoid(logit)
    return (logit > 0).to(logit.dtype) + (soft - soft.detach())


# ---------------------------------------------------------------- retention
class Retention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.hd = n_heads, d // n_heads
        self.q, self.k, self.v, self.g, self.o = (BitLinear(d, d) for _ in range(5))
        self.head_norm = nn.Parameter(torch.ones(n_heads, self.hd))
        gam = 1 - 2.0 ** (-5 - torch.arange(n_heads, dtype=torch.float32))
        self.register_buffer("gamma", gam)

    def _norm_gate_out(self, o, g):           # o: (..., H, hd)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * self.head_norm
        o = o.flatten(-2) * F.silu(g)
        return self.o(o)

    def forward(self, x):                      # parallel form
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.hd).transpose(1, 2) * self.hd ** -0.5
        v = self.v(x).view(B, T, self.h, self.hd).transpose(1, 2)
        n = torch.arange(T, device=x.device)
        diff = (n[:, None] - n[None, :]).float()
        D = torch.where(diff >= 0, self.gamma[:, None, None] ** diff.clamp(min=0),
                        torch.zeros((), device=x.device))
        o = ((q @ k.transpose(-1, -2)) * D) @ v           # (B,H,T,hd)
        return self._norm_gate_out(o.transpose(1, 2), self.g(x))

    def step(self, x, S):                      # recurrent form, x: (B,d), S: (B,H,hd,hd)
        B = x.shape[0]
        q = self.q(x).view(B, self.h, self.hd)
        k = self.k(x).view(B, self.h, self.hd) * self.hd ** -0.5
        v = self.v(x).view(B, self.h, self.hd)
        S = self.gamma[None, :, None, None] * S + k[..., :, None] * v[..., None, :]
        o = (q[..., None, :] @ S).squeeze(-2)
        return self._norm_gate_out(o, self.g(x)), S


# ---------------------------------------------------------------- FFF tree
class FFF(nn.Module):
    def __init__(self, d, depth, n_trees):
        super().__init__()
        self.depth, self.nt = depth, n_trees
        self.nn = 2 ** depth - 1
        N = n_trees * self.nn
        self.w_in = nn.Parameter(torch.randn(N, d) / math.sqrt(d))    # [node, d]  (BitLinear layout)
        self.w_out = nn.Parameter(torch.randn(N, d) / math.sqrt(depth))  # [node, d] row = output vector
        lvl = torch.cat([torch.full((2 ** l,), l) for l in range(depth)])
        self.register_buffer("level", lvl)
        self.aux = {}

    def forward(self, x, max_depth=None):
        lead = x.shape[:-1]
        wi, _, _ = weight_quant(self.w_in)
        wo, _, _ = weight_quant(self.w_out)
        a = F.linear(act_quant(x), wi).view(*lead, self.nt, self.nn)
        r = hard_route(a)
        pps = [torch.ones_like(a[..., :1])]
        for l in range(self.depth - 1):
            s, e = 2 ** l - 1, 2 ** (l + 1) - 1
            p, rl = pps[-1], r[..., s:e]
            pps.append(torch.stack([p * (1 - rl), p * rl], -1).flatten(-2))
        pp = torch.cat(pps, -1)                                   # path prob, hard in fwd
        h = pp * F.gelu(a)
        if max_depth is not None and max_depth < self.depth:
            h = h * (self.level < max_depth).to(h.dtype)
        if self.training:
            vis = pp.detach()
            self.aux["margin"] = (vis * F.relu(0.25 - a.abs())).sum() / vis.sum()
            soft = torch.sigmoid(a)
            frac = (vis * soft).flatten(0, -3).sum(0) / vis.flatten(0, -3).sum(0).clamp(min=1)
            used = vis.flatten(0, -3).sum(0) > 0
            self.aux["balance"] = ((frac - 0.5) ** 2)[used].mean()
        return h.flatten(-2) @ wo


class DenseFFN(nn.Module):
    """ternary baseline with the same parameter count as the tree."""
    def __init__(self, d, hidden):
        super().__init__()
        self.up, self.down = BitLinear(d, hidden), BitLinear(hidden, d)
        self.aux = {}

    def forward(self, x, max_depth=None):
        return self.down(F.gelu(self.up(x)))


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d"]
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.ret = Retention(d, cfg["heads"])
        if cfg["ffn"] == "fff":
            self.ffn = FFF(d, cfg["depth"], cfg["trees"])
        else:
            self.ffn = DenseFFN(d, cfg["trees"] * (2 ** cfg["depth"] - 1))

    def forward(self, x, max_depth=None):
        x = x + self.ret(self.n1(x))
        return x + self.ffn(self.n2(x), max_depth)

    def step(self, x, S, max_depth=None):
        y, S = self.ret.step(self.n1(x), S)
        x = x + y
        return x + self.ffn(self.n2(x), max_depth), S


class Chimera(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg["vocab"], cfg["d"])          # fp, as in BitNet
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg["layers"]))
        self.norm = RMSNorm(cfg["d"])
        self.head = nn.Linear(cfg["d"], cfg["vocab"], bias=False)  # fp, as in BitNet

    def forward(self, idx, max_depth=None):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, max_depth)
        return self.head(self.norm(x))

    def init_state(self, B):
        c = self.cfg
        hd = c["d"] // c["heads"]
        return [torch.zeros(B, c["heads"], hd, hd) for _ in range(c["layers"])]

    @torch.no_grad()
    def step(self, tok, states, max_depth=None):
        x = self.emb(tok)
        new = []
        for b, S in zip(self.blocks, states):
            x, S = b.step(x, S, max_depth)
            new.append(S)
        return self.head(self.norm(x)), new

    def aux_loss(self):
        m = [b.ffn.aux for b in self.blocks if b.ffn.aux]
        if not m:
            return torch.zeros(())
        return sum(a["margin"] + a["balance"] for a in m) / len(m)
