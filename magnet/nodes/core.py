# coding=utf-8
from torch import nn

from .nodes import Node

class Lambda(Node):
    r"""Wraps a Node around any function.

        Args:
            fn (callable): The function which gets called in the forward pass

        Examples::
            >>> import magnet.nodes as mn
            >>>
            >>> import torch
            >>>
            >>> model = mn.Lambda(lambda x: x.mean())
            >>>
            >>> model(torch.arange(5, dtype=torch.float)).item()
            >>> # Output: 2.0
            >>>
            >>> def subtract(x, y):
            >>>     return x - y
            >>>
            >>> model = mn.Lambda(subtract)
            >>>
            >>> model(2 * torch.ones(1), torch.ones(1)).item()
            >>> # Output: 1.0
        """

    def __init__(self, fn, **kwargs):
        from magnet.utils.misc import get_function_name

        super().__init__(fn, **kwargs)

        if self.name == self.__class__.__name__:
            self.name = get_function_name(self._args['fn'])
        if self.name is None:
            self.name = self.__class__.__name__

    def forward(self, *args, **kwargs):
        return self._args['fn'](*args, **kwargs)

class Conv(Node):
    def __init__(self, c=None, k=3, p='half', s=1, d=1, g=1, b=True, ic=None, act='relu', bn=False, **kwargs):
        super().__init__(c, k, p, s, d, g, b, ic, act, bn, **kwargs)

    def build(self, x):
        from magnet.nodes.functional import wiki

        self._set_padding(x)
        if self._args['ic'] is None: self._args['ic'] = x.shape[1]

        self._activation = wiki['activations'][self._args['act']]
        layer_class = self._find_layer(x)
        self._layer = layer_class(kernel_size=self._args['k'], out_channels=self._args['c'],
                                stride=self._args['s'], padding=self._args['p'], dilation=self._args['d'],
                                groups=self._args['g'], bias=self._args['b'], in_channels=self._args['ic'])

        if self._args['bn']: self._batch_norm = BatchNorm()
        super().build(x)

    def forward(self, x):
        if hasattr(self, '_upsample'): x = F.upsample(x, scale_factor=self._upsample)
        x = self._activation(self._layer(x))
        if self._args['bn']: x = self._batch_norm(x)

        return x

    def _find_layer(self, x):
        shape_dict = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        ndim = len(x.shape) - 2
        return shape_dict[ndim - 1]

    def _set_padding(self, x):
        in_shape = x.shape

        p = self._args['p']

        if p == 'half': f = 0.5
        elif p == 'same': f = 1
        elif p == 'double':
            self._upsample = 2
            if self._args['c'] is None:
                self._args['c'] = in_shape[1] // 2
            f = 1
        else: return

        s = 1 / f
        if not s.is_integer():
            raise RuntimeError("Padding value won't hold for all vector sizes")

        self._args['d'] = 1
        self._args['s'] = int(s)
        self._args['p'] = int(self._args['k'] // 2)
        if self._args['c'] is None:
            self._args['c'] = self._args['s'] * in_shape[1]

    def _mul_list(self, n):
        convs = [self]
        self._args['c'] = n[0]
        kwargs = self._args.copy()
        for c in n[1:]:
            kwargs['c'] = c
            convs.append(self.__class__(**kwargs))

        return convs

class Linear(Node):
    def __init__(self, o=None, b=True, flat=True, i=None, act='relu', bn=False, **kwargs):
        super().__init__(o, b, flat, i, act, bn, **kwargs)

    def build(self, x):
        from numpy import prod
        from magnet.nodes.functional import wiki

        if self._args['i'] is None: self._args['i'] = prod(x.shape[1:]) if self._args['flat'] else x.shape[-1]


        self._activation = wiki['activations'][self._args['act']]

        self._layer = nn.Linear(*[self._args[k] for k in ('i', 'o', 'b')])

        if self._args['bn']: self._batch_norm = BatchNorm()
        super().build(x)

    def forward(self, x):
        if self._args['flat']: x = x.view(x.size(0), -1)
        x = self._activation(self._layer(x))
        if self._args['bn']: x = self._batch_norm(x)

        return x

    def _mul_list(self, n):
        lins = [self]
        self._args['o'] = n[0]
        kwargs = self._args.copy()
        for o in n[1:]:
            kwargs['o'] = o
            lins.append(self.__class__(**kwargs))

        return lins

class _RNNBase(Node):
    def __init__(self, mode, n=1, h=None, b=False, bi=False, d=0, batch_first=False, i=None, act='tanh', **kwargs):
        self._layer = mode
        super().__init__(n, h, b, bi, d, batch_first, act, **kwargs)

    def build(self, x, h=None):
        if self._args['i'] is None: self._args['i'] =  x.shape[-1]

        self._layer = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self._layer.lower()]

        kwargs = {'nonlinearity': self._args['act'], 'bias': self._args['b'],
                'batch_first': self._args['batch_first'],
                'dropout': self._args['d'], 'bidirectional': self._args['bi']}
        if not isinstance(self._layer, nn.RNN): kwargs.pop('nonlinearity')
        self._layer = self._layer(*[self._args[k] for k in ('i', 'h', 'n')], **kwargs)

        super().build(x, h)

    def forward(self, x, h=None):
        return self._layer(x, h)

    def _mul_list(self, n):
        rnns = [self]
        self._args['h'] = n[0]
        kwargs = self._args.copy()
        for h in n[1:]:
            kwargs['h'] = h
            rnns.append(self.__class__(**kwargs))

        return rnns

class RNN(_RNNBase):
    def __init__(self, n=1, h=None, b=False, bi=False, act='tanh', d=0, batch_first=False, i=None, **kwargs):
        super().__init__('rnn', n, h, b, bi, act, d, batch_first, **kwargs)

class LSTM(_RNNBase):
    def __init__(self, n=1, h=None, b=False, bi=False, act='tanh', d=0, batch_first=False, i=None, **kwargs):
        super().__init__('lstm', n, h, b, bi, act, d, batch_first, **kwargs)

class GRU(_RNNBase):
    def __init__(self, n=1, h=None, b=False, bi=False, act='tanh', d=0, batch_first=False, i=None, **kwargs):
        super().__init__('gru', n, h, b, bi, act, d, batch_first, **kwargs)

class BatchNorm(Node):
    def __init__(self, e=1e-05, m=0.1, a=True, track=True, i=None, **kwargs):
        super().__init__(e, m, a, track, i, **kwargs)

    def build(self, x):
        self._args['i'] = x.shape[1]
        layer_class = self._find_layer(x)
        self._layer = layer_class(*[self._args[k] for k in ('i', 'e', 'm', 'a', 'track')])
        super().build(x)

    def forward(self, x):
        return self._layer(x)

    def _find_layer(self, x):
        shape_dict = [nn.BatchNorm1d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        ndim = len(x.shape) - 1
        return shape_dict[ndim - 1]