import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

# helpers functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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
            self.padding = r // 2
            self.R = nn.Parameter(torch.randn(dim_k, dim_u, 1, r, r))
        else:
            assert exists(n), 'You must specify the total sequence length (h x w)'
            self.pos_emb = nn.Parameter(torch.randn(n, n, dim_k, dim_u))


    def forward(self, x):
        b, c, hh, ww, u, h = *x.shape, self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
        k = rearrange(k, 'b (k u) hh ww -> b k u (hh ww)', u = u)
        v = rearrange(v, 'b (v u) hh ww -> b v u (hh ww)', u = u)

        k = k.softmax(dim=-1)

        λc = einsum('b k u m, b v u m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b n h v', q, λc)

        if self.local_contexts:
            v = rearrange(v, 'b v u (hh ww) -> b u v hh ww', hh = hh, ww = ww)
            λp = F.conv3d(v, self.R, padding = (0, self.padding, self.padding))
            Yp = einsum('b h k n, b k v n -> b n h v', q, λp.flatten(3))
        else:
            λp = einsum('n m k u, b v u m -> b n k v', self.pos_emb, v)
            Yp = einsum('b h k n, b n k v -> b n h v', q, λp)

        Y = Yc + Yp
        out = rearrange(Y, 'b (hh ww) h v -> b (h v) hh ww', hh = hh, ww = ww)
        return out
