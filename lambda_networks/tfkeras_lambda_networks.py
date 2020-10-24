from einops.layers.tensorflow import Rearrange
from tensorflow.keras import initializers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv3D, ZeroPadding3D, Softmax, Lambda, Add, Layer
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
            n=None,
            r=None,
            heads=4,
            dim_out=None,
            dim_u=1,
            data_format='channels_last'):
        super(LambdaLayer, self).__init__()
        self.last_channel = data_format == 'channels_last'
        self.out_dim = dim_out
        self.u = dim_u  # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        self.dim_v = dim_out // heads
        self.dim_k = dim_k
        self.heads = heads
        self.dim_u = dim_u
        self.r = r
        self.n = n

        self.local_contexts = exists(self.r)
        if self.local_contexts:
            assert (self.r % 2) == 1, 'Receptive kernel size should be odd'
            padding = (self.r // 2, self.r // 2, 0) if self.last_channel else (0, self.r // 2, self.r // 2)
            kernel_size = (self.r, self.r, 1) if self.last_channel else (1, self.r, self.r)
            self.pos_padding = ZeroPadding3D(padding=padding, data_format=data_format)
            self.pos_conv = Conv3D(dim_k, kernel_size, padding='valid', data_format=data_format)
        else:
            assert exists(n), 'You must specify the total sequence length (h x w)'
            self.pos_emb = self.add_weight(name='pos_emb',
                                           shape=(n, n, dim_k, dim_u),
                                           initializer=initializers.random_normal,
                                           trainable=True)

        self.to_q = Conv2D(self.dim_k * self.heads, 1, use_bias=False, data_format=data_format)
        self.to_k = Conv2D(self.dim_k * self.dim_u, 1, use_bias=False, data_format=data_format)
        self.to_v = Conv2D(self.dim_v * self.dim_u, 1, use_bias=False, data_format=data_format)

        axis = -1 if self.last_channel else 1
        # should use the same parameters as in Pytorch (and in the paper):
        # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        self.norm_q = BatchNormalization(axis=axis, epsilon=1e-5, momentum=0.1)
        self.norm_v = BatchNormalization(axis=axis, epsilon=1e-5, momentum=0.1)

        if self.local_contexts:
            assert (self.r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_padding = ZeroPadding3D(padding=padding, data_format=data_format)
            self.pos_conv = Conv3D(self.dim_k, kernel_size, padding='valid', data_format=data_format)
        else:
            assert exists(self.n), 'You must specify the total sequence length (h x w)'
            self.pos_emb = self.add_weight(name='pos_emb',
                                           shape=(self.n, self.n, self.dim_k, self.dim_u),
                                           initializer=initializers.random_normal,
                                           trainable=True)

    def call(self, inputs, **kwargs):
        b, c, ww, hh = inputs.get_shape().as_list()
        if self.last_channel:
            hh, c = c, hh 
        u, h = self.u, self.heads
        x = inputs
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        pattern = 'b hh ww (h k) -> b h k (hh ww)' if self.last_channel else 'b (h k) hh ww -> b h k (hh ww)'
        q = Rearrange(pattern, h=h)(q)

        pattern = 'b hh ww (u k) -> b k u (hh ww)' if self.last_channel else 'b (u k) hh ww -> b u k (hh ww)'
        k = Rearrange(pattern, u=u)(k)

        pattern = 'b hh ww (u v) -> b v u (hh ww)' if self.last_channel else 'b (u v) hh ww -> b u v (hh ww)'
        v = Rearrange(pattern, u=u)(v)

        k = Softmax()(k)

        pattern = 'b k m u, b v m u -> b k v' if self.last_channel else 'b u k m, b u v m -> b k v'
        Lc = Lambda(lambda x: einsum(pattern, x[0], x[1]))([k, v])

        pattern = 'b h k n, b k v -> b h v n' if self.last_channel else 'b h k n, b k v -> b n h v'
        Yc = Lambda(lambda x: einsum(pattern, x[0], x[1]))([q, Lc])

        if self.local_contexts:
            pattern = 'b v u (hh ww) -> b v hh ww u' if self.last_channel else 'b u v (hh ww) -> b u v hh ww'
            v = Rearrange(pattern, hh=hh, ww=ww)(v)
            Lp = self.pos_padding(v)
            Lp = self.pos_conv(Lp)

            pattern = 'b k h w c -> b k c (h w)' if self.last_channel else 'b c k h w -> b c k (h w)'
            Lp = Rearrange(pattern)(Lp)

            pattern = 'b h k n, b v k n -> b h v n' if self.last_channel else 'b h k n, b k v n -> b n h v'
            Yp = Lambda(lambda x: einsum(pattern, x[0], x[1]))([q, Lp])
        else:
            pattern = 'n m k u, b v m u -> b v k n' if self.last_channel else 'n m k u, b u v m -> b n k v'
            Lp = Lambda(lambda x: einsum(pattern, x[0], x[1]))([self.pos_emb, v])

            pattern = 'b h k n, b v k n -> b h v n' if self.last_channel else 'b h k n, b n k v -> b n h v'
            Yp = Lambda(lambda x: einsum(pattern, x[0], x[1]))([q, Lp])

        Y = Add()([Yc, Yp])

        pattern = 'b h v (hh ww) -> b hh ww (h v)' if self.last_channel else 'b (hh ww) h v -> b (h v) hh ww'
        out = Rearrange(pattern, hh = hh, ww = ww)(Y)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_dim, input_shape[2], input_shape[3])

    def get_config(self):
        config = {'output_dim': (self.input_shape[0], self.out_dim, self.input_shape[2], self.input_shape[3])}
        base_config = super(LambdaLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
