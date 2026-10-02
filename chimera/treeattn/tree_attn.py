"""
Tree attention: take a normally trained BitNet b1.58 softmax attention layer and
convert it to a form that inferences as a boosted tree ensemble.

Base layer (standard, unchanged):   q,k,v = BitLinear(norm(x));  RoPE(q,k);
                                    o = softmax(q k^T / sqrt(hd) + causal) v;  o_proj(subLN(o))

Conversion: each head's rotated keys (and optionally values) are encoded by a
*boosted* sequence of S oblique decision trees (residual boosting):

    r_0 = k;   leaf_s = tree_s(r_{s-1});   r_s = r_{s-1} - c_s[leaf_s]       (each tree fits the residual)
    k_hat = sum_s c_s[leaf_s]

Score of query n against key m:
    q.k_hat = sum_s  q.c_s[leaf_s(k_m)]  =  sum_s  Tq_s[leaf_s(k_m)]

i.e. every key is scored by a boosted tree ensemble whose leaf values Tq_s[.] are set by the
query (built once per query: S*L small dots). Per key at inference: S byte lookups + adds,
no dot products, and the KV cache holds S bytes per key (and S bytes per value).
Values: o = sum_s sum_l (sum_{m: leaf_s(v_m)=l} p_m) c^v_s[l]  -> accumulate probability per leaf.

Training form == inference form exactly (it is attention with k_hat, v_hat). Trees are fit
training-free from calibration activations (PCA split + median threshold per node, i.e. a
balanced hierarchical 2-means), then optionally only the leaf values are distilled for a few
hundred steps. All BitNet weights stay frozen.
"""
import math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import BitLinear, RMSNorm


def rope(x, pos):
    half = x.shape[-1] // 2
    inv = 10000.0 ** (-torch.arange(half, dtype=x.dtype, device=x.device) / half)
    ang = pos.to(x.dtype)[:, None] * inv[None, :]
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)


# ---------------------------------------------------------------- boosted tree quantizer
SNAP = 1e-5   # residuals smaller than SNAP*|x| are float noise: snap to exactly 0 (same rule in C)
BAND = (0.45, 0.55)   # split threshold = widest gap inside this quantile band (max-margin, near-balanced)
class BoostedTrees(nn.Module):
    """per head: S oblique trees of depth D, applied to successive residuals."""
    def __init__(self, H, S, D, hd, axis=False):
        super().__init__()
        self.H, self.S, self.D, self.L = H, S, D, 2 ** D
        self.axis = axis          # axis-aligned splits (one coordinate vs threshold) in a fixed per-head rotated basis
        if axis:
            self.register_buffer("G", torch.eye(hd).repeat(H, 1, 1))     # key basis:   k' = G k
            self.register_buffer("Gi", torch.eye(hd).repeat(H, 1, 1))    # inverse:     k  = Gi k'   (query side uses Gi^T q)
        self.register_buffer("w", torch.zeros(H, S, self.L - 1, hd))   # split directions
        self.register_buffer("b", torch.zeros(H, S, self.L - 1))       # thresholds
        self.c = nn.Parameter(torch.zeros(H, S, self.L, hd))           # leaf values

    def _route(self, r, s):                                            # r (B,H,T,hd) -> leaf (B,H,T)
        B, H, T, hd = r.shape
        hi = torch.arange(H, device=r.device)[None, :, None].expand(B, H, T)
        node = torch.zeros(B, H, T, dtype=torch.long, device=r.device)
        for _ in range(self.D):
            a = (r * self.w[hi, s, node]).sum(-1) - self.b[hi, s, node]
            node = 2 * node + 1 + (a > 0).long()
        return node - (self.L - 1)

    def forward(self, x):                                              # -> x_hat, codes (B,H,T,S)
        B, H, T, hd = x.shape
        hi = torch.arange(H, device=x.device)[None, :, None].expand(B, H, T)
        if self.axis: x = torch.einsum("bhtd,hed->bhte", x, self.G)
        r, xh, codes = x, torch.zeros_like(x), []
        tiny = SNAP * x.detach().norm(dim=-1, keepdim=True)
        for s in range(self.S):
            leaf = self._route(r.detach(), s)
            cs = self.c[hi, s, leaf]
            xh = xh + cs; r = r - cs; codes.append(leaf)
            r = torch.where(r.detach().norm(dim=-1, keepdim=True) < tiny, torch.zeros_like(r), r)   # snap float noise
        if self.axis: xh = torch.einsum("bhte,hde->bhtd", xh, self.Gi)
        return xh, torch.stack(codes, -1)

    @torch.no_grad()
    def fit(self, X, Q=None):
        """X (H,N,hd) calibration keys/values. If Q (H,M,hd) queries are given, fit in the
        query metric: minimise E[(q.(k - k_hat))^2] = (k-k_hat)^T Sigma_q (k-k_hat).
        Trees are fit on k' = A k (A = Sigma_q^1/2); the transform is folded back into the
        split directions (w = A w') and leaf values (c = A^-1 c'), so inference is unchanged."""
        if self.axis: return self._fit_axis(X, Q)
        for h in range(self.H):
            if Q is not None:
                Sq = Q[h].T @ Q[h] / len(Q[h])
                ev, U = torch.linalg.eigh(Sq)
                ev = ev.clamp(min=ev.max() * 1e-4)
                A, Ai = U @ torch.diag(ev.sqrt()) @ U.T, U @ torch.diag(ev.rsqrt()) @ U.T
            else:
                A = Ai = torch.eye(X.shape[-1])
            R = X[h] @ A.T
            tiny = SNAP * X[h].norm(dim=-1, keepdim=True)
            for s in range(self.S):
                self._fit_tree(R, h, s)
                leaf = self._route(R[None, None].expand(1, self.H, -1, -1).contiguous(), s)[0, h]
                R = R - self.c.data[h, s, leaf]
                R = torch.where((R @ Ai.T).norm(dim=-1, keepdim=True) < tiny, torch.zeros_like(R), R)
            # fold transform: w.(A k) = (A^T w).k ;  c = A^-1 c'
            self.w[h] = self.w[h] @ A
            self.c.data[h] = self.c.data[h] @ Ai.T

    def _fit_axis(self, X, Q):
        """Basis per head: G = U^T A, with A = Sigma_q^1/2 (query metric) and U the PCA basis of A k.
        In that basis every node splits ONE coordinate (the one with most variance in the node) at a
        max-margin threshold near its median. Split vectors are one-hot, so routing is compare-only."""
        hd = X.shape[-1]
        for h in range(self.H):
            if Q is not None:
                Sq = Q[h].T @ Q[h] / len(Q[h]); ev, U = torch.linalg.eigh(Sq); ev = ev.clamp(min=ev.max() * 1e-4)
                A, Ai = U @ torch.diag(ev.sqrt()) @ U.T, U @ torch.diag(ev.rsqrt()) @ U.T
            else:
                A = Ai = torch.eye(hd)
            Ka = X[h] @ A.T
            _, P = torch.linalg.eigh(Ka.T @ Ka / len(Ka)); P = P.flip(1)        # columns = principal directions, descending
            self.G[h] = P.T @ A; self.Gi[h] = Ai @ P
            R = X[h] @ self.G[h].T
            tiny = SNAP * R.norm(dim=-1, keepdim=True)
            for s in range(self.S):
                tol = 1e-4 * R.norm(dim=1).mean().clamp(min=1e-12)
                stack = [(0, torch.arange(len(R)))]; leaf = torch.zeros(len(R), dtype=torch.long)
                while stack:
                    node, idx = stack.pop(); Xs = R[idx]
                    if node >= self.L - 1:
                        self.c.data[h, s, node - (self.L - 1)] = Xs.mean(0) if len(idx) else 0
                        leaf[idx] = node - (self.L - 1); continue
                    w = torch.zeros(hd); t = 0.0
                    if len(idx) >= 2:
                        f = int(Xs.var(0).argmax()); p = Xs[:, f].sort().values; n = len(p)
                        if p[-1] - p[0] >= tol:
                            i0 = int(BAND[0] * (n - 1)); i1 = max(int(BAND[1] * (n - 1)), i0 + 1)
                            j = i0 + int((p[i0 + 1:i1 + 1] - p[i0:i1]).argmax())
                            w[f] = 1.0; t = ((p[j] + p[j + 1]) / 2).item()
                    self.w[h, s, node] = w; self.b[h, s, node] = t
                    right = (Xs @ w - t) > 0
                    stack += [(2 * node + 1, idx[~right]), (2 * node + 2, idx[right])]
                R = R - self.c.data[h, s, leaf]
                R = torch.where(R.norm(dim=-1, keepdim=True) < tiny, torch.zeros_like(R), R)

    def _fit_tree(self, R, h, s):
        tol = 1e-4 * R.norm(dim=1).mean().clamp(min=1e-12)       # below this spread a split is float noise
        stack = [(0, torch.arange(len(R)))]
        while stack:
            node, idx = stack.pop()
            Xs = R[idx]
            if node >= self.L - 1:                                     # leaf
                self.c.data[h, s, node - (self.L - 1)] = Xs.mean(0) if len(idx) else 0
                continue
            spread = (Xs - Xs.mean(0)).norm(dim=1).max() if len(idx) else torch.tensor(0.)
            if len(idx) < 2 or spread < tol:
                w = torch.zeros(R.shape[1]); t = 0.0                    # degenerate: 0 > 0 is false -> always left
            else:
                Xc = Xs - Xs.mean(0)
                _, _, V = torch.linalg.svd(Xc[:4096], full_matrices=False)
                w = V[0]; p = (Xs @ w).sort().values; n = len(p)
                # max-margin balanced split: threshold in the widest gap within the 30-70% quantile band
                i0, i1 = int(BAND[0] * (n - 1)), max(int(BAND[1] * (n - 1)), int(BAND[0] * (n - 1)) + 1)
                gaps = p[i0 + 1:i1 + 1] - p[i0:i1]
                j = i0 + int(gaps.argmax())
                if gaps.max() < tol:
                    w = torch.zeros(R.shape[1]); t = 0.0
                else:
                    t = ((p[j] + p[j + 1]) / 2).item()
            self.w[h, s, node] = w; self.b[h, s, node] = t
            right = (Xs @ w - t) > 0
            stack += [(2 * node + 1, idx[~right]), (2 * node + 2, idx[right])]


# ---------------------------------------------------------------- standard BitNet attention
class Attn(nn.Module):
    def __init__(self, d, H):
        super().__init__()
        self.H, self.hd = H, d // H
        self.q, self.k, self.v, self.o = (BitLinear(d, d) for _ in range(4))
        self.sub = RMSNorm(d)                     # BitNet b1.58 attn_sub_norm before o_proj
        self.kq = None; self.vq = None            # BoostedTrees once converted

    def qkv(self, u):
        B, T, _ = u.shape
        sh = lambda t: t.view(B, T, self.H, self.hd).transpose(1, 2)
        pos = torch.arange(T, device=u.device)
        return rope(sh(self.q(u)), pos), rope(sh(self.k(u)), pos), sh(self.v(u))

    def forward(self, u):
        q, k, v = self.qkv(u)
        if self.kq is not None: k = self.kq(k)[0]
        if self.vq is not None: v = self.vq(v)[0]
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(self.sub(o.transpose(1, 2).flatten(-2)))


class LM(nn.Module):
    """embedding -> [x + Attn(norm(x))] * layers -> norm -> head   (pure attention, no MLP)"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg["vocab"], cfg["d"])
        self.norms = nn.ModuleList(RMSNorm(cfg["d"]) for _ in range(cfg["layers"]))
        self.attn = nn.ModuleList(Attn(cfg["d"], cfg["heads"]) for _ in range(cfg["layers"]))
        self.norm = RMSNorm(cfg["d"])
        self.head = nn.Linear(cfg["d"], cfg["vocab"], bias=False)
        self.mlp = cfg.get("mlp", 0)                       # 0 = pure attention; else hidden = mlp*d (ternary MLP)
        if self.mlp:
            d = cfg["d"]
            self.n2 = nn.ModuleList(RMSNorm(d) for _ in range(cfg["layers"]))
            self.up = nn.ModuleList(BitLinear(d, self.mlp * d) for _ in range(cfg["layers"]))
            self.down = nn.ModuleList(BitLinear(self.mlp * d, d) for _ in range(cfg["layers"]))

    def _mlp(self, l, x, ablate=False):
        if not self.mlp or ablate: return x
        tm = getattr(self, "tm", None)
        if tm is not None and l < getattr(self, "tm_ready", len(tm)):      # converted tree-MLP
            return x + tm[l](self.n2[l](x), self.up[l], self.down[l])
        return x + self.down[l](F.gelu(self.up[l](self.n2[l](x))))

    def forward(self, idx, ablate=False, ablate_mlp=False):
        x = self.emb(idx)
        for l, (n, a) in enumerate(zip(self.norms, self.attn)):
            if not ablate: x = x + a(n(x))
            x = self._mlp(l, x, ablate_mlp)
        return self.head(self.norm(x))

    def convert(self, S, D, values=True, qaware=True, axis=False):
        c = self.cfg; hd = c["d"] // c["heads"]; self.qaware = qaware
        for a in self.attn:
            a.kq = BoostedTrees(c["heads"], S, D, hd, axis)
            a.vq = BoostedTrees(c["heads"], S, D, hd) if values else None

    @torch.no_grad()
    def calibrate(self, idx):
        """fit trees layer by layer on activations produced by the already-converted earlier layers."""
        x = self.emb(idx)
        for l, (n, a) in enumerate(zip(self.norms, self.attn)):
            q, k, v = a.qkv(n(x))
            a.kq.fit(k.transpose(0, 1).flatten(1, 2), q.transpose(0, 1).flatten(1, 2) if self.qaware else None)
            if a.vq is not None: a.vq.fit(v.transpose(0, 1).flatten(1, 2))
            x = self._mlp(l, x + a(n(x)))
