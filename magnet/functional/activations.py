from functools import partial
from torch.nn.functional import relu, sigmoid, tanh, leaky_relu

wiki = {'relu': relu, 'sigmoid': sigmoid, 'tanh': tanh,
         'lrelu': partial(leaky_relu, leak=0.2), None: lambda x: x}
