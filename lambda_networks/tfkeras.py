from einops.layers.tensorflow import Rearrange
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv3D, ZeroPadding3D, Softmax, Lambda, Add, Layer
from tensorflow.keras import initializers
from tensorflow import einsum, nn

# helpers functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# lambda layer

class LambdaLayer(Layer):
    def __init__(
        self,
        *,
        dim_k,
        n = None,
        r = None,
        heads = 4,
        dim_out = None,
        dim_u = 1):
        super(LambdaLayer, self).__init__()

        self.out_dim = dim_out
        self.u = dim_u  # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        self.dim_v = dim_out // heads
        self.dim_k = dim_k
        self.heads = heads

        self.to_q = Conv2D(self.dim_k * heads, 1, use_bias=False)
        self.to_k = Conv2D(self.dim_k * dim_u, 1, use_bias=False)
        self.to_v = Conv2D(self.dim_v * dim_u, 1, use_bias=False)

        self.norm_q = BatchNormalization()
        self.norm_v = BatchNormalization()

        self.local_contexts = exists(r)
        if exists(r):
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = Conv3D(dim_k, (1, r, r), padding='same')
        else:
            assert exists(n), 'You must specify the total sequence length (h x w)'
            self.pos_emb = self.add_weight(name='pos_emb',
                                           shape=(n, n, dim_k, dim_u),
                                           initializer=initializers.random_normal,
                                           trainable=True)

    def call(self, x, **kwargs):
        b, hh, ww, c, u, h = *x.get_shape().as_list(), self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = Rearrange('b hh ww (h k) -> b h k (hh ww)', h=h)(q)
        k = Rearrange('b hh ww (u k) -> b u k (hh ww)', u=u)(k)
        v = Rearrange('b hh ww (u v) -> b u v (hh ww)', u=u)(v)

        k = nn.softmax(k)

        Lc = einsum('b u k m, b u v m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b n h v', q, Lc)

        if self.local_contexts:
            v = Rearrange('b u v (hh ww) -> b v hh ww u', hh=hh, ww=ww)(v)
            Lp = self.pos_conv(v)
            Lp = Rearrange('b v h w k -> b v k (h w)')(Lp)
            Yp = einsum('b h k n, b v k n -> b n h v', q, Lp)
        else:
            Lp = einsum('n m k u, b u v m -> b n k v', self.pos_emb, v)
            Yp = einsum('b h k n, b n k v -> b n h v', q, Lp)

        Y = Yc + Yp
        out = Rearrange('b (hh ww) h v -> b hh ww (h v)', hh = hh, ww = ww)(Y)
        return out

    def compute_output_shape(self, input_shape):
        return (*input_shape[:2], self.out_dim)

    def get_config(self):
        config = {'output_dim': (*self.input_shape[:2], self.out_dim)}
        base_config = super(LambdaLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
