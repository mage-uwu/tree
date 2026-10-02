"""
Tree-MLP: convert a trained BitNet MLP  f(u) = down(gelu(up(u)))  (frozen ternary weights) into a model tree:

    leaf = tree(u)                                   D oblique splits on the normed input
    out  = c[leaf]                                   leaf constant
         + Q[leaf] (P[leaf] u)                       optional rank-r linear map per leaf   (2r row units)
         + down[:, set[leaf]] gelu(up[set[leaf]] u)  optional k exact neurons per leaf     (2k row units)
         + Qg (Pg u)                                 optional global rank-rg linear map    (2rg row units)

A GELU/ReLU MLP is (nearly) piecewise linear, so "tree + small linear map per leaf" is its natural tree form.
Everything is fit in closed form from calibration activations (ridge + reduced-rank regression, means,
top-k statistics): no gradient training, BitNet weights untouched.

Cost per token in row units (one length-d dot): dense MLP = 2*hidden;  tree = D + 2r + 2k + 2rg.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tree_attn import BAND
from model import act_quant


def _top_pc(Yc, iters=8):
    v = torch.randn(Yc.shape[1], generator=torch.Generator().manual_seed(0))
    for _ in range(iters):
        v = Yc.T @ (Yc @ v); v = v / v.norm().clamp(min=1e-12)
    return v


def _rrr(Xc, Yc, r, lam=1e-2):
    """reduced-rank ridge regression Yc ~ Xc P^T Q^T. returns P (r,d_in), Q (d_out,r)."""
    d = Xc.shape[1]
    G = Xc.T @ Xc
    W = torch.linalg.solve(G + lam * G.diagonal().mean() * torch.eye(d), Xc.T @ Yc)      # (d_in, d_out)
    Fit = Xc @ W
    _, _, Vh = torch.linalg.svd(Fit[:8192], full_matrices=False)
    V = Vh[:r].T                                                                          # (d_out, r)
    return (W @ V).T.contiguous(), V.contiguous()


@torch.no_grad()
def fit_router(X, Y, D):
    """split direction = direction of X that best predicts the top principal component of Y (what leaves
    should be homogeneous in); threshold = widest gap inside the BAND quantiles (max-margin, near-balanced)."""
    N, d = X.shape; L = 2 ** D
    w, b, leaf = torch.zeros(max(L - 1, 1), d), torch.zeros(max(L - 1, 1)), torch.zeros(N, dtype=torch.long)
    tol = 1e-4 * X.norm(dim=1).mean()
    stack = [(0, torch.arange(N))]
    while stack:
        node, idx = stack.pop()
        if node >= L - 1:
            leaf[idx] = node - (L - 1); continue
        Xs = X[idx]; wn, t = torch.zeros(d), 0.0
        if len(idx) >= 8:
            Xc = Xs - Xs.mean(0); Yc = Y[idx] - Y[idx].mean(0)
            s = Yc @ _top_pc(Yc)
            G = Xc.T @ Xc
            wn = torch.linalg.solve(G + 1e-3 * G.diagonal().mean() * torch.eye(d), Xc.T @ s)
            wn = wn / wn.norm().clamp(min=1e-12)
            p = (Xs @ wn).sort().values; n = len(p)
            i0 = int(BAND[0] * (n - 1)); i1 = max(int(BAND[1] * (n - 1)), i0 + 1)
            gaps = p[i0 + 1:i1 + 1] - p[i0:i1]; j = i0 + int(gaps.argmax())
            if p[-1] - p[0] < tol: wn = torch.zeros(d)          # no real spread: pass-through node
            else: t = ((p[j] + p[j + 1]) / 2).item()
        w[node], b[node] = wn, t
        right = (Xs @ wn - t) > 0
        stack += [(2 * node + 1, idx[~right]), (2 * node + 2, idx[right])]
    return w, b, leaf


class TreeMLP(nn.Module):
    def __init__(self, d, hidden, D, k=0, r=0, rg=0):
        super().__init__()
        self.D, self.L, self.k, self.r, self.rg = D, 2 ** D, k, r, rg
        self.register_buffer("w", torch.zeros(max(self.L - 1, 1), d))
        self.register_buffer("b", torch.zeros(max(self.L - 1, 1)))
        self.c = nn.Parameter(torch.zeros(self.L, d))
        if k:
            self.register_buffer("idx", torch.zeros(self.L, k, dtype=torch.long))
            self.register_buffer("mask", torch.zeros(self.L, hidden))
        if r:
            self.P = nn.Parameter(torch.zeros(self.L, r, d)); self.Q = nn.Parameter(torch.zeros(self.L, d, r))
        if rg:
            self.Pg = nn.Parameter(torch.zeros(rg, d)); self.Qg = nn.Parameter(torch.zeros(d, rg))

    def route(self, u):
        if getattr(self, "q8", False): u = act_quant(u)                  # int8 router reads int8 activations
        node = torch.zeros(u.shape[:-1], dtype=torch.long, device=u.device)
        for _ in range(self.D):
            node = 2 * node + 1 + ((u * self.w[node]).sum(-1) - self.b[node] > 0).long()
        return node - (self.L - 1)

    def parts(self, u, up, down, leaf):
        out = 0
        if self.rg: out = out + (u @ self.Pg.T) @ self.Qg.T
        if self.k: out = out + down(F.gelu(up(u)) * self.mask[leaf])     # torch computes all rows; C only the k listed
        if self.r:
            ui = act_quant(u) if getattr(self, "q8", False) else u       # int8 leaf maps read int8 activations
            out = out + (self.Q[leaf] @ (self.P[leaf] @ ui[..., None]))[..., 0]
        return out

    @torch.no_grad()
    def quantize(self):
        """int8 leaf maps: each row of P[leaf] and each column of Q[leaf] -> int8 with its own absmax scale.
        4x less memory to move per token; combined with BitNet's int8 activations the dot is integer."""
        sP = self.P.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127
        self.P.data = (self.P / sP).round().clamp(-127, 127) * sP
        sQ = self.Q.abs().amax(-2, keepdim=True).clamp(min=1e-12) / 127
        self.Q.data = (self.Q / sQ).round().clamp(-127, 127) * sQ
        sw = self.w.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127   # int8 split directions too
        self.w.copy_((self.w / sw).round().clamp(-127, 127) * sw)
        self.q8 = True

    def forward(self, u, up, down):
        leaf = self.route(u)
        return self.parts(u, up, down, leaf) + self.c[leaf]

    @torch.no_grad()
    def fit(self, u, up, down, target=None):
        """target (same shape as u): what this block should output. Default = the MLP's own output on u.
        Passing (teacher stream after this MLP) - (converted stream before it) makes the tree also absorb
        the drift accumulated by the converted blocks below it."""
        U = u.flatten(0, -2)
        if self.k:
            g = F.gelu(up(U)); Y = down(g)
        elif target is not None:
            Y = target.flatten(0, -2)
        else:                                                             # chunked: never hold the hidden layer
            Y = torch.cat([down(F.gelu(up(U[i:i + 65536]))) for i in range(0, len(U), 65536)])
        res = Y.clone()
        if self.rg:
            P, Q = _rrr(U - U.mean(0), Y - Y.mean(0), self.rg)
            self.Pg.data, self.Qg.data = P, Q
            res = res - (U @ P.T) @ Q.T
        if self.D: self.w[:], self.b[:], leaf = fit_router(U, res, self.D)
        else: leaf = torch.zeros(len(U), dtype=torch.long)
        cn = down.weight.abs().sum(0)
        for l in range(self.L):
            sel = leaf == l
            if not sel.any(): continue
            Ul, rl = U[sel], res[sel]
            if self.k:                                   # neurons whose contribution VARIES most inside the leaf
                gl = g[sel]
                score = gl.std(0) * cn if sel.sum() > 1 else gl.abs().mean(0) * cn
                top = score.topk(self.k).indices.sort().values
                self.idx[l] = top; self.mask[l] = 0; self.mask[l, top] = 1
                rl = rl - down(gl * self.mask[l])
            if self.r and sel.sum() >= 4 * self.r:
                mu = Ul.mean(0); P, Q = _rrr(Ul - mu, rl - rl.mean(0), self.r, lam=1e-1)
                self.P.data[l], self.Q.data[l] = P, Q
                rl = rl - (Ul @ P.T) @ Q.T
            self.c.data[l] = rl.mean(0)

    def row_units(self):
        return self.D + 2 * self.k + 2 * self.r + 2 * self.rg


def convert_mlp(model, calib, D, k=0, r=0, rg=0):
    """layer by layer; each layer is calibrated on the outputs of the already-converted layers below."""
    cfg = model.cfg; d = cfg["d"]
    model.tm = nn.ModuleList(TreeMLP(d, cfg["mlp"] * d, D, k, r, rg) for _ in range(cfg["layers"]))
    model.tm_ready = 0
    chunk = lambda f, t: torch.cat([f(t[i:i + 64]) for i in range(0, len(t), 64)])      # keep memory flat
    with torch.no_grad():
        x = model.emb(calib)
        for l, (n, a) in enumerate(zip(model.norms, model.attn)):
            x = chunk(lambda t: t + a(n(t)), x)
            model.tm[l].fit(model.n2[l](x), model.up[l], model.down[l])
            model.tm_ready = l + 1
            x = chunk(lambda t: model._mlp(l, t), x)
    return model
