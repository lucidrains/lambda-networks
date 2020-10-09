## LambdaNetworks - Pytorch

Implementation of LambdaNetworks, an attention-based solution for image recognition that reaches SOTA. The title of the paper claims it is free of attention when in fact it converged on the linear-attention solution other groups have been working on, with the (key x values) rebranded as λ.

## Install

```bash
$ pip install lambda-networks
```

## Usage

```python
import torch
from lambda_networks import LambdaLayer

layer = LambdaLayer(
    dim = 32,       # channels going in
    dim_out = 32,   # channels out
    n = 64 * 64,    # number of input pixels (64 x 64 image)
    m = 64 * 64,    # number of context (64 x 64 global)
    dim_k = 16,     # key dimension
    dim_v = 64,     # value dimension
    heads = 4,      # number of heads, for multi-query
    dim_u = 1       # 'intra-depth' dimension
)

x = torch.randn(1, 32, 64, 64)
layer(x) # (1, 32, 64, 64)
```

## Todo

[] Lambda layers with structured context
[] Document hyperparameters and put some sensible defaults
[] Test it out

## Citations

```bibtex
@inproceedings{
    anonymous2021lambdanetworks,
    title={LambdaNetworks: Modeling long-range Interactions without Attention},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=xTJEN-ggl1b},
    note={under review}
}
```
