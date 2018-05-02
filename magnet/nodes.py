import torch

from torch import nn

from ._utils import caller_locals

class Node(nn.Module):
    def __init__(self, *args, **kwargs):
        self._parse_params()
        super().__init__()

    def build(self, in_shape):
        pass

    @property
    def _default_params(self):
        return {}

    def _parse_params(self):
        args = caller_locals(ancestor=True)
        if 'args' not in args.keys(): args['args'] = []
        if 'kwargs' not in args.keys(): args['kwargs'] = {}
        
        default_param_list = list(self._default_params.items())

        for i, arg_val in enumerate(args['args']):
            param_name = default_param_list[i][0]
            args[param_name] = arg_val
        args.pop('args')
        
        for param_name, default in default_param_list:
            if param_name in args['kwargs'].keys():
                args[param_name] = args['kwargs'][param_name]
            elif param_name not in args.keys():
                args[param_name] = default
        args.pop('kwargs')

        self._args = args

    def get_args(self):
        return ', '.join(str(k) + '=' + str(v) for k, v in self._args.items())

    def get_output_shape(self, in_shape):
        with torch.no_grad(): return tuple(self(torch.randn(in_shape)).size())

    def  _mul_int(self, n):
        return [self] + [self.__class__(**self._args) for _ in range(n - 1)]

    def __mul__(self, n):
        if type(n) is int or (type(n) is float and n.is_integer()):
            return self._mul_int(n)

class MonoNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._set_activation()

    def build(self, in_shape):
        self._layer_class = self._find_layer(in_shape)
        kwargs = self._get_kwargs(in_shape)
        self._layer = self._layer_class(**kwargs)

    def forward(self, x):
        return self._activation(self._layer(x))

    def _find_layer(self, in_shape):
        pass

    def _get_kwargs(self, in_shape):
        return {k: self._args[v] for k, v in self._kwargs_dict.items()}

    @property
    def _kwargs_dict(self):
        pass

    def _set_activation(self):
        from torch.nn import functional as F
        from functools import partial

        activation_dict = {'relu': F.relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh,
                            'lrelu': partial(F.leaky_relu, leak=0.2), None: lambda x: x}
        self._activation = activation_dict[self._args['act']]

    @property
    def _default_params(self):
        p = {'act': 'relu'}
        p.update(super()._default_params)
        return p

class Conv(MonoNode):
    def build(self, in_shape):
        self._set_padding(in_shape)

        super().build(in_shape)

    def _find_layer(self, in_shape):
        shape_dict = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        ndim = len(in_shape) - 2
        return shape_dict[ndim - 1]

    def _get_kwargs(self, in_shape):
        kwargs = super()._get_kwargs(in_shape)
        kwargs['in_channels'] = in_shape[1]
        return kwargs

    @property
    def _kwargs_dict(self):
        return {'kernel_size': 'k', 'out_channels': 'c','stride': 's',
                'padding': 'p', 'dilation': 'd', 'groups': 'g', 'bias': 'b'}
    
    def _set_padding(self, in_shape):
        p = self._args['p']
        if p == 'half': f = 0.5
        elif p == 'same': f = 1
        else: return
        
        s = 1 / f
        if not s.is_integer(): 
            raise RuntimeError("Padding value won't hold for all vector sizes")
            
        self._args['d'] = 1
        self._args['s'] = int(s)
        self._args['p'] = int(self._args['k'] // 2)
        if self._args['c'] is None: 
            self._args['c'] = self._args['s'] * in_shape[1]
        
    @property
    def _default_params(self):
        p = {'c': None, 'k': 3, 'p': 'half', 's': 1, 'd': 1, 'g': 1, 'b': True}
        p.update(super()._default_params)
        return p

class Linear(MonoNode):
    def __init__(self, o, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self._args['flat']: x = x.view(x.size(0), -1)

        return super().forward(x)

    def _find_layer(self, in_shape):
        return nn.Linear

    def _get_kwargs(self, in_shape):
        kwargs = super()._get_kwargs(in_shape)

        from numpy import prod
        kwargs['in_features'] = prod(in_shape[1:]) if self._args['flat'] else in_shape[-1]
        return kwargs

    @property
    def _kwargs_dict(self):
        return {'out_features': 'o', 'bias': 'b'}

    @property
    def _default_params(self):
        p = {'b': True, 'act': 'relu', 'flat': True}
        p.update(super()._default_params)
        return p

class Lambda(Node):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)