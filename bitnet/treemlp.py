"""Tree MLP in shared-subspace form for d=2560:

    z    = P_g (u - mu)                    shared input projection  (r_in x d)
    leaf = tree(z)                         D oblique splits in z-space (router reuses z: D dots of length r_in)
    out  = c[leaf] + Q_g M[leaf] z         per-leaf core M (r_out x r_in), shared output basis Q_g (d x r_out)

Storage per leaf: r_out*r_in + d (int8), shared: (r_in + r_out) * d.
Fit is closed form and streaming: P_g/Q_g/router from a token subsample, then per-leaf sufficient statistics
(count, sum z, sum z z^T, sum y_out z^T, sum y) accumulated over the full calibration stream, then per-leaf
ridge regression shrunk toward the global map (leaves with few samples fall back to it)."""
import torch
import torch.nn as nn

BAND = (0.45, 0.55)


def top_eig(C, r):
    ev, U = torch.linalg.eigh(C.double())
    return U[:, -r:].flip(1).T.float().contiguous(), ev.flip(0).float()


class TreeMLP(nn.Module):
    def __init__(self, d, r_in, r_out, D):
        super().__init__()
        self.d, self.r_in, self.r_out, self.D, self.L = d, r_in, r_out, D, 2 ** D
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("Pg", torch.zeros(r_in, d))
        self.register_buffer("Qg", torch.zeros(d, r_out))
        self.register_buffer("w", torch.zeros(max(self.L - 1, 1), r_in))
        self.register_buffer("b", torch.zeros(max(self.L - 1, 1)))
        self.M = nn.Parameter(torch.zeros(self.L, r_out, r_in))
        self.c = nn.Parameter(torch.zeros(self.L, d))

    def storage_bytes(self):
        """int8 storage: per-leaf cores + constants, shared projections, router."""
        return dict(per_leaf=self.L * (self.r_out * self.r_in + self.d),
                    shared=(self.r_in + self.r_out) * self.d + (self.L - 1) * (self.r_in + 4))

    def macs(self):
        return self.r_in * self.d + self.D * self.r_in + self.r_out * self.r_in + self.d * self.r_out

    def project(self, u):
        return (u.float() - self.mu) @ self.Pg.T

    def route(self, z):
        node = torch.zeros(z.shape[:-1], dtype=torch.long, device=z.device)
        for _ in range(self.D):
            node = 2 * node + 1 + ((z * self.w[node]).sum(-1) - self.b[node] > 0).long()
        return node - (self.L - 1)

    def forward(self, u, chunk=2048):
        sh = u.shape; U = u.reshape(-1, sh[-1]); out = []
        for i in range(0, len(U), chunk):
            z = self.project(U[i:i + chunk]); leaf = self.route(z)
            yo = torch.bmm(self.M[leaf], z[:, :, None])[:, :, 0]
            out.append(self.c[leaf] + yo @ self.Qg.T)
        return torch.cat(out).reshape(sh)

    # ------------------------------------------------------------ fitting
    @torch.no_grad()
    def fit_shared(self, U, Y, Pg, Qg):
        """U, Y (N,d) float32 token subsample. Pg (r_in,d), Qg (d,r_out) chosen by the caller."""
        self.mu.copy_(U.mean(0)); self.Pg.copy_(Pg); self.Qg.copy_(Qg)
        Z = self.project(U)
        # global map (in z space) on the subsample: residual target for the router
        Zc = Z - Z.mean(0); Yc = Y - Y.mean(0)
        G = Zc.T @ Zc
        Wg = torch.linalg.solve(G + 1e-3 * G.diagonal().mean() * torch.eye(self.r_in, device=G.device), Zc.T @ Yc)
        Rres = Yc - Zc @ Wg
        self._fit_router(Z, Rres)

    def _fit_router(self, Z, R):
        """split direction = ridge regression (in z space) of the node's residual top PC; threshold = widest
        gap inside the BAND quantiles. R (N,d) residual outputs."""
        N, r = Z.shape
        node = torch.zeros(N, dtype=torch.long, device=Z.device)
        I = torch.eye(r, device=Z.device)
        for depth in range(self.D):
            for j in range(2 ** depth - 1, 2 ** (depth + 1) - 1):
                idx = (node == j).nonzero()[:, 0]
                wn = torch.zeros(r, device=Z.device); t = 0.0
                if len(idx) >= 16:
                    Zs = Z[idx]; Zc = Zs - Zs.mean(0)
                    Rs = R[idx]; Rc = Rs - Rs.mean(0)
                    v = torch.randn(Rc.shape[1], device=Z.device, generator=torch.Generator(Z.device).manual_seed(j))
                    for _ in range(8):
                        v = Rc.T @ (Rc @ v); v = v / v.norm().clamp(min=1e-12)
                    s = Rc @ v
                    G = Zc.T @ Zc
                    wn = torch.linalg.solve(G + 1e-3 * G.diagonal().mean() * I, Zc.T @ s)
                    wn = wn / wn.norm().clamp(min=1e-12)
                    p = (Zs @ wn).sort().values; n = len(p)
                    i0 = int(BAND[0] * (n - 1)); i1 = max(int(BAND[1] * (n - 1)), i0 + 1)
                    gaps = p[i0 + 1:i1 + 1] - p[i0:i1]; g = int(gaps.argmax())
                    if p[-1] - p[0] < 1e-6 * p.abs().max().clamp(min=1e-12):
                        wn = torch.zeros(r, device=Z.device)
                    else:
                        t = ((p[i0 + g] + p[i0 + g + 1]) / 2).item()
                self.w[j] = wn; self.b[j] = t
                if len(idx):
                    node[idx] = 2 * j + 1 + ((Z[idx] @ wn - t) > 0).long()

    def new_stats(self):
        dev = self.Pg.device; L, ri, ro = self.L, self.r_in, self.r_out
        f64 = dict(device=dev, dtype=torch.float64)
        return dict(n=torch.zeros(L, **f64), sz=torch.zeros(L, ri, **f64), zz=torch.zeros(L, ri, ri, **f64),
                    yz=torch.zeros(L, ro, ri, **f64), sy=torch.zeros(L, self.d, **f64), syo=torch.zeros(L, ro, **f64))

    @torch.no_grad()
    def accumulate(self, st, U, Y, chunk=8192):
        for i in range(0, len(U), chunk):
            u, y = U[i:i + chunk].float(), Y[i:i + chunk].float()
            z = self.project(u); leaf = self.route(z); yo = y @ self.Qg
            zd, yod = z.double(), yo.double()
            st["n"].index_add_(0, leaf, torch.ones_like(leaf, dtype=torch.float64))
            st["sz"].index_add_(0, leaf, zd)
            st["sy"].index_add_(0, leaf, y.double())
            st["syo"].index_add_(0, leaf, yod)
            # per-leaf outer products: sort by leaf and use segment sums via bmm over one-hot would be huge;
            # loop over the leaves present in this chunk instead
            order = leaf.argsort(); ls = leaf[order]; cnt = torch.bincount(ls, minlength=self.L)
            zs, ys = zd[order], yod[order]; start = 0
            for l in cnt.nonzero()[:, 0].tolist():
                e = start + int(cnt[l]); a, b = zs[start:e], ys[start:e]
                st["zz"][l] += a.T @ a; st["yz"][l] += b.T @ a; start = e

    @torch.no_grad()
    def solve(self, st, lam=1e-2, shrink=1.0):
        """per-leaf centered ridge regression of y_out on z, shrunk toward the pooled (global) map."""
        n = st["n"].clamp(min=1)[:, None]
        mz, myo, my = st["sz"] / n, st["syo"] / n, st["sy"] / n
        Czz = st["zz"] / n[..., None] - mz[:, :, None] * mz[:, None, :]          # (L,ri,ri)
        Cyz = st["yz"] / n[..., None] - myo[:, :, None] * mz[:, None, :]         # (L,ro,ri)
        N = st["n"].sum()
        gz = st["sz"].sum(0) / N; gyo = st["syo"].sum(0) / N
        Gzz = st["zz"].sum(0) / N - torch.outer(gz, gz); Gyz = st["yz"].sum(0) / N - torch.outer(gyo, gz)
        I = torch.eye(self.r_in, device=Czz.device, dtype=Czz.dtype)
        sg = Gzz.diagonal().mean()
        Mg = torch.linalg.solve(Gzz + lam * sg * I, Gyz.T).T                       # (ro, ri)
        # shrinkage: weight of the prior = shrink * r_in samples
        k = shrink * self.r_in / st["n"].clamp(min=1)                             # (L,)
        A = Czz + (lam * sg + k[:, None, None] * sg) * I
        Bm = Cyz + (k[:, None, None] * sg) * Mg
        M = torch.linalg.solve(A, Bm.transpose(1, 2)).transpose(1, 2)            # (L,ro,ri)
        empty = st["n"] == 0
        M[empty] = Mg
        self.M.data.copy_(M.float())
        c = my - (M @ mz[..., None])[..., 0] @ self.Qg.double().T                 # c = mean y - Q M mean z
        c[empty] = (st["sy"].sum(0) / N - (Mg @ gz) @ self.Qg.double().T)
        self.c.data.copy_(c.float())
        return dict(empty_leaves=int(empty.sum()), min_count=int(st["n"].min()), median_count=int(st["n"].median()))

    @torch.no_grad()
    def quantize(self):
        """int8: per-row absmax for Pg, w, M rows; per-column for Qg; per-leaf for c."""
        def q(t, dim):
            s = t.abs().amax(dim, keepdim=True).clamp(min=1e-12) / 127
            return (t / s).round().clamp(-127, 127) * s
        self.Pg.copy_(q(self.Pg, -1)); self.Qg.copy_(q(self.Qg, 0)); self.w.copy_(q(self.w, -1))
        self.M.data.copy_(q(self.M.data, -1)); self.c.data.copy_(q(self.c.data, -1))
