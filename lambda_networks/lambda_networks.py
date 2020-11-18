import torch
from torch import nn, einsum
from einops import rearrange

# helpers functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]                  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1                                       # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos

# lambda layer

class LambdaLayer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_k,
        n = None,
        r = None,
        heads = 4,
        dim_out = None,
        dim_u = 1):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.u = dim_u # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        dim_v = dim_out // heads

        self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_k * dim_u, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_v * dim_u, 1, bias = False)

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

        self.local_contexts = exists(r)
        if exists(r):
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding = (0, r // 2, r // 2))
        else:
            assert exists(n), 'You must specify the window size (n=h=w)'
            rel_lengths = 2 * n - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_u))
            self.rel_pos = calc_rel_pos(n)

    def forward(self, x):
        b, c, hh, ww, u, h = *x.shape, self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
        k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = u)
        v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u = u)

        k = k.softmax(dim=-1)

        λc = einsum('b u k m, b u v m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b h v n', q, λc)

        if self.local_contexts:
            v = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
            λp = self.pos_conv(v)
            Yp = einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))
        else:
            n, m = self.rel_pos.unbind(dim = -1)
            rel_pos_emb = self.rel_pos_emb[n, m]
            λp = einsum('n m k u, b u v m -> b n k v', rel_pos_emb, v)
            Yp = einsum('b h k n, b n k v -> b h v n', q, λp)

        Y = Yc + Yp
        out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        return out
