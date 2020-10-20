from einops.layers.keras import Rearrange
from keras.layers import Conv2D, BatchNormalization, Conv3D, ZeroPadding3D, Softmax, Lambda, Add, Layer
from keras import initializers
from tensorflow import einsum

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
            self.pos_padding = ZeroPadding3D(padding=(0, r//2, r//2))
            self.pos_conv = Conv3D(dim_k, (1, r, r), padding='valid')
        else:
            assert exists(n), 'You must specify the total sequence length (h x w)'
            self.pos_emb = self.add_weight(name='pos_emb',
                                           shape=(n, n, dim_k, dim_u),
                                           initializer=initializers.random_normal,
                                           trainable=True)

    def call(self, inputs, **kwargs):
        b, c, hh, ww = inputs.get_shape().as_list()
        u, h = self.u, self.heads
        x = inputs

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = Rearrange('b (h k) hh ww -> b h k (hh ww)', h=h)(q)
        k = Rearrange('b (u k) hh ww -> b u k (hh ww)', u=u)(k)
        v = Rearrange('b (u v) hh ww -> b u v (hh ww)', u=u)(v)

        k = Softmax()(k)

        Lc = Lambda(lambda x: einsum('b u k m, b u v m -> b k v', x[0], x[1]))([k, v])
        Yc = Lambda(lambda x: einsum('b h k n, b k v -> b n h v', x[0], x[1]))([q, Lc])

        if self.local_contexts:
            v = Rearrange('b u v (hh ww) -> b u v hh ww', hh=hh, ww=ww)(v)
            Lp = self.pos_padding(v)
            Lp = self.pos_conv(Lp)
            Lp = Rearrange('b c k h w -> b c k (h w)')(Lp)
            Yp = Lambda(lambda x: einsum('b h k n, b k v n -> b n h v', x[0], x[1]))([q, Lp])
        else:
            Lp = Lambda(lambda x: einsum('n m k u, b u v m -> b n k v', x[0], x[1]))([self.pos_emb, v])
            Yp = Lambda(lambda x: einsum('b h k n, b n k v -> b n h v', x[0], x[1]))([q, Lp])

        Y = Add()([Yc, Yp])
        out = Rearrange('b (hh ww) h v -> b (h v) hh ww', hh = hh, ww = ww)(Y)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_dim, input_shape[2], input_shape[3])

    def get_config(self):
        config = {'output_dim': (self.input_shape[0], self.out_dim, self.input_shape[2], self.input_shape[3])}
        base_config = super(LambdaLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
