from functools import partial
from torch.nn.functional import relu, leaky_relu
from torch import sigmoid, tanh

wiki = {'relu': relu, 'sigmoid': sigmoid, 'tanh': tanh,
         'lrelu': partial(leaky_relu, negative_slope=0.2), None: lambda x: x}
