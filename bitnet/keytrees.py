"""Boosted oblique key trees for grouped-query attention (one tree chain per KV head).
Same algorithm as chimera/treeattn/tree_attn.py BoostedTrees (top-PC splits, threshold at the widest
gap inside the 45-55% quantile band, leaf = mean, residual boosting, SNAP), but device-agnostic and
fitted in the query metric pooled over the query heads that share each KV head."""
import torch
import torch.nn as nn

SNAP = 1e-5
BAND = (0.45, 0.55)


class KeyTrees(nn.Module):
    def __init__(self, H, S, D, hd, exact_sink=False):
        super().__init__()
        self.H, self.S, self.D, self.L = H, S, D, 2 ** D
        self.exact_sink = exact_sink      # keep position 0 (BOS, the attention sink) exact
        self.register_buffer("w", torch.zeros(H, S, self.L - 1, hd))
        self.register_buffer("b", torch.zeros(H, S, self.L - 1))
        self.register_buffer("c", torch.zeros(H, S, self.L, hd))

    def degenerate_frac(self):
        return (self.w.abs().sum(-1) == 0).float().mean().item()

    def bytes_per_key(self):
        return self.S * self.D / 8

    def _route(self, r, s):
        """r (H,N,hd) residuals -> leaf (H,N) of tree s."""
        a = torch.einsum("hnd,hjd->hnj", r, self.w[:, s]) - self.b[:, s, None, :]   # all node scores
        node = torch.zeros(r.shape[:2], dtype=torch.long, device=r.device)
        for _ in range(self.D):
            node = 2 * node + 1 + (a.gather(-1, node[..., None])[..., 0] > 0).long()
        return node - (self.L - 1)

    def encode(self, x):
        """x (H,N,hd) float32 -> k_hat, codes (H,N,S)."""
        r, xh, codes = x, torch.zeros_like(x), []
        tiny = SNAP * x.norm(dim=-1, keepdim=True)
        hi = torch.arange(self.H, device=x.device)[:, None]
        for s in range(self.S):
            leaf = self._route(r, s)
            cs = self.c[hi, s, leaf]
            xh = xh + cs; r = r - cs; codes.append(leaf)
            r = torch.where(r.norm(dim=-1, keepdim=True) < tiny, torch.zeros_like(r), r)
        return xh, torch.stack(codes, -1)

    def forward(self, k):
        """k (B,H,T,hd) post-RoPE keys -> k_hat, same shape."""
        B, H, T, hd = k.shape
        x = k.float().transpose(0, 1).reshape(H, B * T, hd)
        xh = self.encode(x)[0].reshape(H, B, T, hd).transpose(0, 1)
        if self.exact_sink:
            xh[:, :, 0] = k[:, :, 0].float()
        return xh

    @torch.no_grad()
    def fit(self, X, Sq=None):
        """X (H,N,hd) float32 calibration keys; Sq (H,hd,hd) query second moments (pooled per KV head)
        -> fit in the metric (k-k_hat)^T Sq (k-k_hat); the transform is folded back afterwards."""
        H, N, hd = X.shape
        for h in range(H):
            if Sq is not None:
                ev, U = torch.linalg.eigh(Sq[h].double())
                ev = ev.clamp(min=ev.max() * 1e-4)
                A = (U @ torch.diag(ev.sqrt()) @ U.T).float(); Ai = (U @ torch.diag(ev.rsqrt()) @ U.T).float()
            else:
                A = Ai = torch.eye(hd, device=X.device)
            R = X[h] @ A.T
            tiny = SNAP * X[h].norm(dim=-1, keepdim=True)
            for s in range(self.S):
                leaf = self._fit_tree(R, h, s)
                R = R - self.c[h, s, leaf]
                R = torch.where((R @ Ai.T).norm(dim=-1, keepdim=True) < tiny, torch.zeros_like(R), R)
            self.w[h] = self.w[h] @ A
            self.c[h] = self.c[h] @ Ai.T

    def _fit_tree(self, R, h, s):
        N, hd = R.shape
        tol = 1e-4 * R.norm(dim=1).mean().clamp(min=1e-12)
        node = torch.zeros(N, dtype=torch.long, device=R.device)
        for depth in range(self.D):
            for j in range(2 ** depth - 1, 2 ** (depth + 1) - 1):
                idx = (node == j).nonzero()[:, 0]
                w = torch.zeros(hd, device=R.device); t = 0.0
                if len(idx) >= 2:
                    Xs = R[idx]; Xc = Xs - Xs.mean(0)
                    if Xc.norm(dim=1).max() >= tol:
                        w = torch.linalg.eigh((Xc.T @ Xc).double())[1][:, -1].float()
                        p = (Xs @ w).sort().values; n = len(p)
                        i0 = int(BAND[0] * (n - 1)); i1 = max(int(BAND[1] * (n - 1)), i0 + 1)
                        gaps = p[i0 + 1:i1 + 1] - p[i0:i1]
                        g = int(gaps.argmax())
                        # degenerate only if the node has no spread (checked above); a tiny widest gap is
                        # normal with many samples. Require the gap to exceed float resolution so the
                        # threshold never sits on (copies of) a data point.
                        if gaps[g] <= 1e-6 * p.abs().max().clamp(min=1e-12):
                            w = torch.zeros(hd, device=R.device)
                        else:
                            t = ((p[i0 + g] + p[i0 + g + 1]) / 2).item()
                    self.w[h, s, j] = w; self.b[h, s, j] = t
                    node[idx] = 2 * j + 1 + ((Xs @ w - t) > 0).long()
                else:
                    self.w[h, s, j] = w; self.b[h, s, j] = t
                    node[idx] = 2 * j + 1
        leaf = node - (self.L - 1)
        cnt = torch.bincount(leaf, minlength=self.L).clamp(min=1).float()
        c = torch.zeros(self.L, hd, device=R.device).index_add_(0, leaf, R)
        self.c[h, s] = c / cnt[:, None]
        return leaf
