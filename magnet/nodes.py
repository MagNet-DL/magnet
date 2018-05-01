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
        for k in ('args', 'kwargs'):
            if k not in args.keys(): args[k] = []
        
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


    def get_output_shape(self, in_shape):
        with torch.no_grad(): return tuple(self(torch.randn(in_shape)).size())

class Conv(Node):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._set_activation()

    def build(self, in_shape):
        shape_dict = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        ndim = len(in_shape) - 2
        module = shape_dict[ndim - 1]
        
        self._set_padding(in_shape)
        
        kw_map = {'kernel_size': 'k', 'out_channels': 'c','stride': 's',
                 'padding': 'p', 'dilation': 'd', 'groups': 'g', 'bias': 'b'}
        kwargs = {k: self._args[v] for k, v in kw_map.items()}
        kwargs['in_channels'] = in_shape[1]
        self.conv = module(**kwargs)
        
    def forward(self, x):
        return self._activation(self.conv(x))
    
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
        return {'c': None, 'k': 3, 'p': 'half', 's': 1, 'd': 1, 'g': 1, 'b': True, 'act': 'relu'}

    def _set_activation(self):
        from torch.nn import functional as F
        from functools import partial

        activation_dict = {'relu': F.relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh,
                            'lrelu': partial(F.leaky_relu, leak=0.2), None: lambda x: x}
        self._activation = activation_dict[self._args['act']]

class Linear(Node):
    def __init__(self, o, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_activation()

    def build(self, in_shape):
        kw_map = {'out_features': 'o', 'bias': 'b'}
        kwargs = {k: self._args[v] for k, v in kw_map.items()}
        kwargs['in_features'] = in_shape[-1]
        self.fc = nn.Linear(**kwargs)

    def forward(self, x):
        return self._activation(self.fc(x))

    @property
    def _default_params(self):
        return {'b': True, 'act': 'relu'}

    def _set_activation(self):
        from torch.nn import functional as F
        from functools import partial

        activation_dict = {'relu': F.relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh,
                            'lrelu': partial(F.leaky_relu, leak=0.2), None: lambda x: x}
        self._activation = activation_dict[self._args['act']]

class Lambda(Node):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)