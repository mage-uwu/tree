"""
Routed Retention: a RetNet-style mixer whose recurrent state is split across the
leaves of a learned hyperplane tree, composable head-by-head with standard
BitNet-style softmax attention.

Per head h (retention-family), with leaf maps b(.) from a shared tree on q/k:

  parallel   O = (Q~ K~^T * D * M) V,   D[n,m] = gamma^(n-m) [n>=m],
                                         M[n,m] = [b(q_n) == b(k_m)]  (= Pq Pk^T)
  chunkwise  inner-chunk parallel + cross-chunk via per-leaf states
  recurrent  S_b <- gamma^dt S_b + k~^T v   (only the written leaf, lazy decay)
             o   = gamma^dt' q~ S_{b(q)}    (only the read leaf)

Q~, K~ are RoPE-rotated (absolute positions), so all three forms are the same
function. Routing uses the un-rotated q/k (content, not position).
With tree depth 0 (one leaf) this is exactly RetNet retention (+RoPE).

Head kinds: 'S' = causal softmax attention (BitNet b1.58 BitLinear projections,
RoPE, KV cache at inference), 'R' = routed retention. All heads share the same
BitLinear q/k/v/g/o weights and per-head subLN, so a layer can mix them freely
and a BitNet attention checkpoint's projections drop straight in.

Forward of the router is hard (one leaf); backward is the soft leaf
distribution (product of sigmoids on the path) via STE.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import BitLinear, RMSNorm  # b1.58 BitLinear, shared with the FFF prototype


def rope(x, pos):
    """x (..., T, hd), pos (T,) absolute positions. rotate-half convention."""
    half = x.shape[-1] // 2
    inv = 10000.0 ** (-torch.arange(half, dtype=x.dtype, device=x.device) / half)
    ang = pos.to(x.dtype)[:, None] * inv[None, :]
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)


class Router(nn.Module):
    """per-head bias-free hyperplane tree over head_dim vectors."""
    def __init__(self, H, depth, hd):
        super().__init__()
        self.depth = depth
        self.w = nn.Parameter(torch.randn(H, max(2 ** depth - 1, 1), hd) / math.sqrt(hd))

    def forward(self, x):                      # x (B,H,T,hd) -> P (B,H,T,L), logits
        if self.depth == 0:
            return torch.ones_like(x[..., :1]), None
        a = torch.einsum("bhtd,hnd->bhtn", x, self.w)
        rs, rh = torch.sigmoid(a), (a > 0).to(x.dtype)
        soft = hard = torch.ones_like(x[..., :1])
        for l in range(self.depth):
            s, e = 2 ** l - 1, 2 ** (l + 1) - 1
            soft = torch.stack([soft * (1 - rs[..., s:e]), soft * rs[..., s:e]], -1).flatten(-2)
            hard = torch.stack([hard * (1 - rh[..., s:e]), hard * rh[..., s:e]], -1).flatten(-2)
        return hard + (soft - soft.detach()), a


class Mixer(nn.Module):
    def __init__(self, d, kinds, rdepth, local=False):
        super().__init__()
        self.local = local and rdepth > 0
        self.H = len(kinds); self.hd = d // self.H; self.kinds = kinds
        self.q, self.k, self.v, self.g, self.o = (BitLinear(d, d) for _ in range(5))
        self.mu_k = nn.Parameter(torch.full((d,), 0.5))   # token shift (RWKV-style) for k, v
        self.mu_v = nn.Parameter(torch.full((d,), 0.5))
        self.head_norm = nn.Parameter(torch.ones(self.H, self.hd))
        self.router = Router(self.H, rdepth, self.hd)
        self.register_buffer("gamma", 1 - 2.0 ** (-5 - torch.arange(self.H, dtype=torch.float32)))
        # slow decay for routed leaf memories when a fast local state is also present
        self.register_buffer("gamma_leaf", 1 - 2.0 ** (-8 - torch.arange(self.H, dtype=torch.float32)))
        self.register_buffer("is_soft", torch.tensor([c == "S" for c in kinds]))
        self.aux = {}

    # ---- shared projections
    def _proj(self, u, u_prev):
        q, g = self.q(u), self.g(u)
        k = self.k(u * self.mu_k + u_prev * (1 - self.mu_k))
        v = self.v(u * self.mu_v + u_prev * (1 - self.mu_v))
        return q, k, v, g

    def _heads(self, t):                      # (B,T,d) -> (B,H,T,hd)
        B, T, _ = t.shape
        return t.view(B, T, self.H, self.hd).transpose(1, 2)

    def _out(self, o, g):                     # o (B,H,T,hd)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * self.head_norm[:, None, :]
        o = o.transpose(1, 2).flatten(-2) * F.silu(g)
        return self.o(o)

    def _route(self, q, k):
        Pq, aq = self.router(q)
        Pk, ak = self.router(k)
        if self.training and aq is not None:
            R = ~self.is_soft
            use = 0.5 * (Pq + Pk).mean((0, 2))[R]                  # (Hr, L) soft usage
            L = use.shape[-1]
            self.aux["balance"] = (use * (use * L + 1e-9).log()).sum(-1).mean()
            cq = aq / (q.norm(dim=-1, keepdim=True) * self.router.w.norm(dim=-1)[None, :, None] + 1e-6)
            ck = ak / (k.norm(dim=-1, keepdim=True) * self.router.w.norm(dim=-1)[None, :, None] + 1e-6)
            self.aux["margin"] = (F.relu(0.05 - cq.abs()).mean() + F.relu(0.05 - ck.abs()).mean())
        return Pq, Pk

    def forward(self, u, mode="parallel", chunk=32):
        B, T, d = u.shape
        u_prev = F.pad(u, (0, 0, 1, 0))[:, :-1]
        q, k, v, g = (self._heads(t) if i < 3 else t for i, t in enumerate(self._proj(u, u_prev)))
        Pq, Pk = self._route(q, k)
        pos = torch.arange(T, device=u.device)
        qr, kr = rope(q, pos), rope(k, pos) * self.hd ** -0.5
        # softmax heads (standard causal attention)
        o_s = F.scaled_dot_product_attention(qr, kr * self.hd ** 0.5, v, is_causal=True)
        # retention-family heads
        f = self._ret_parallel if mode == "parallel" else (lambda *a: self._ret_chunkwise(*a, C=chunk))
        if self.local:   # fast global state + slow routed leaf states: QK^T * (D_loc + D_leaf * M)
            one = torch.ones_like(Pq[..., :1])
            o_r = f(qr, kr, v, one, one, self.gamma) + f(qr, kr, v, Pq, Pk, self.gamma_leaf)
        else:
            o_r = f(qr, kr, v, Pq, Pk, self.gamma)
        o = torch.where(self.is_soft[None, :, None, None], o_s, o_r)
        return self._out(o, g)

    def _decay(self, n, dev, dt, gamma):
        i = torch.arange(n, device=dev)
        diff = (i[:, None] - i[None, :]).to(dt)
        return torch.where(diff >= 0, gamma.to(dt)[:, None, None] ** diff.clamp(min=0),
                           torch.zeros((), dtype=dt, device=dev))

    def _ret_parallel(self, qr, kr, v, Pq, Pk, gamma):
        D = self._decay(qr.shape[2], qr.device, qr.dtype, gamma)
        M = Pq @ Pk.transpose(-1, -2)
        return ((qr @ kr.transpose(-1, -2)) * D * M) @ v

    def _ret_chunkwise(self, qr, kr, v, Pq, Pk, gamma, C=32):
        B, H, T, hd = qr.shape
        L = Pq.shape[-1]
        gam = gamma.to(qr.dtype)
        S = qr.new_zeros(B, H, L, hd, hd)
        outs = []
        for s in range(0, T, C):
            e = min(s + C, T); n = e - s
            qc, kc, vc, pq, pk = qr[:, :, s:e], kr[:, :, s:e], v[:, :, s:e], Pq[:, :, s:e], Pk[:, :, s:e]
            idx = torch.arange(n, device=qr.device).to(qr.dtype)
            inner = ((qc @ kc.transpose(-1, -2)) * self._decay(n, qr.device, qr.dtype, gamma)
                     * (pq @ pk.transpose(-1, -2))) @ vc
            cross = torch.einsum("bhnl,bhnd,bhlde->bhne", pq, qc, S) * (gam[:, None] ** (idx + 1))[None, :, :, None]
            outs.append(inner + cross)
            wk = pk * (gam[:, None] ** (n - 1 - idx))[None, :, :, None]
            S = S * (gam ** n)[None, :, None, None, None] + torch.einsum("bhml,bhmd,bhme->bhlde", wk, kc, vc)
        return torch.cat(outs, 2)

    # ---- recurrent (one token), state = dict
    def init_state(self, B, dev=None, dt=torch.float32):
        L = max(2 ** self.router.depth, 1)
        return dict(t=0, u_prev=torch.zeros(B, self.H * self.hd, device=dev, dtype=dt),
                    S=torch.zeros(B, self.H, L, self.hd, self.hd, device=dev, dtype=dt),
                    Sl=torch.zeros(B, self.H, self.hd, self.hd, device=dev, dtype=dt),
                    K=[], V=[])

    def step(self, u, st):                    # u (B,d)
        B = u.shape[0]
        q, k, v, g = self._proj(u, st["u_prev"])
        sh = lambda t: t.view(B, self.H, 1, self.hd)
        q, k, v = sh(q), sh(k), sh(v)
        Pq, Pk = self.router(q)[0], self.router(k)[0]            # (B,H,1,L)
        pos = torch.tensor([st["t"]], device=u.device)
        qr, kr = rope(q, pos), rope(k, pos) * self.hd ** -0.5
        # softmax heads with KV cache
        st["K"].append(kr * self.hd ** 0.5); st["V"].append(v)
        K, V = torch.cat(st["K"], 2), torch.cat(st["V"], 2)
        att = torch.softmax((qr @ K.transpose(-1, -2)) / math.sqrt(self.hd), -1)
        o_s = att @ V
        # routed retention: per-leaf states
        kv = kr[:, :, 0, :, None] * v[:, :, 0, None, :]           # (B,H,hd,hd)
        gl = self.gamma_leaf if self.local else self.gamma
        S = gl.to(u.dtype)[None, :, None, None, None] * st["S"] + Pk[:, :, 0, :, None, None] * kv[:, :, None]
        o_r = torch.einsum("bhl,bhd,bhlde->bhe", Pq[:, :, 0], qr[:, :, 0], S)[:, :, None]
        if self.local:
            st["Sl"] = self.gamma.to(u.dtype)[None, :, None, None] * st["Sl"] + kv
            o_r = o_r + (qr[:, :, 0, None, :] @ st["Sl"])
        o = torch.where(self.is_soft[None, :, None, None], o_s, o_r)
        st.update(t=st["t"] + 1, u_prev=u, S=S)
        return self._out(o, g[:, None])[:, 0], st


class AttnLM(nn.Module):
    """embedding -> [x + Mixer(norm(x))] * layers -> norm -> head. No MLP."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg["vocab"], cfg["d"])
        self.norms = nn.ModuleList(RMSNorm(cfg["d"]) for _ in range(cfg["layers"]))
        self.mixers = nn.ModuleList(Mixer(cfg["d"], cfg["kinds"], cfg["rdepth"], cfg.get("local", False))
                                    for _ in range(cfg["layers"]))
        self.norm = RMSNorm(cfg["d"])
        self.head = nn.Linear(cfg["d"], cfg["vocab"], bias=False)

    def forward(self, idx, mode="parallel", chunk=32):
        x = self.emb(idx)
        for n, m in zip(self.norms, self.mixers):
            x = x + m(n(x), mode, chunk)
        return self.head(self.norm(x))

    def init_state(self, B, dt=torch.float32):
        return [m.init_state(B, dt=dt) for m in self.mixers]

    @torch.no_grad()
    def step(self, tok, states):
        x = self.emb(tok)
        for n, m, st in zip(self.norms, self.mixers, states):
            y, _ = m.step(n(x), st)
            x = x + y
        return self.head(self.norm(x)), states

    def aux_loss(self):
        a = [m.aux for m in self.mixers if m.aux]
        return sum(x["balance"] + x["margin"] for x in a) / len(a) if a else torch.zeros(())
