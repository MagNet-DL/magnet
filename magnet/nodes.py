import torch
import magnet as mag

from torch import nn
from torch.nn import functional as F

from magnet._utils import caller_locals, get_function_name

class Node(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._parse_args()
        self._built = False

    def build(self, *args, **kwargs):
        if self._built and mag.build_lock: return

        [c.build(*args, **kwargs) for c in self.modules() if isinstance(c, Node) and c != self]

        self._built = True
        self.to(mag.device)

    def __call__(self, *args, **kwargs):
        self.build(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def _check_parameters(self, return_val):
        import warnings
        if not self._built: raise RuntimeError(f'Node {self.name} not built yet')
        if not mag.build_lock: warnings.warn('Build-lock disabled. The node may be re-built', RuntimeWarning)

        return return_val

    def parameters(self):
        return self._check_parameters(super().parameters())

    def named_parameters(self, memo=None, prefix=''):
        return self._check_parameters(super().named_parameters())

    def _parse_args(self):
        args = caller_locals(ancestor=True)
        args.update(args.pop('kwargs', {}))

        self.name = args.pop('name', self.__class__.__name__)

        self._args = args

    def get_args(self):
        return ', '.join(str(k) + '=' + str(v) for k, v in self._args.items())

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        try: self.device = next(self.parameters())[0].device
        except StopIteration: pass
        return self

    def load_state_dict(self, f):
        from pathlib import Path
        if isinstance(f, (str, Path)):
            device = self.device.type
            if device == 'cuda': device = 'cuda:0'

            return super().load_state_dict(torch.load(f, map_location=device))
        else:
            return super().load_state_dict(f)

    def  _mul_int(self, n):
        return [self] + [self.__class__(**self._args) for _ in range(n - 1)]

    def  _mul_list(self, n):
        pass

    def __mul__(self, n):
        if type(n) is int or (type(n) is float and n.is_integer()):
            return self._mul_int(n)

        if type(n) is tuple or type(n) is list:
            return self._mul_list(n)

    def _get_name(self):
        return self.name

class Lambda(Node):
    def __init__(self, fn, **kwargs):
        super().__init__(fn, **kwargs)

        if self.name == self.__class__.__name__:
            self.name = get_function_name(self._args['fn'])
        if self.name is None:
            self.name = self.__class__.__name__

    def forward(self, x):
        return self._args['fn'](x)

class Conv(Node):
    def __init__(self, c=None, k=3, p='half', s=1, d=1, g=1, b=True, ic=None, act='relu', **kwargs):
        super().__init__(c, k, p, s, d, g, b, ic, act, **kwargs)

    def build(self, x):
        from magnet.functional import activation_wiki
        self._set_padding(x)
        self._args['ic'] = x.shape[1]

        layer_class = self._find_layer(x)
        layers = [layer_class(kernel_size=self._args['k'], out_channels=self._args['c'],
                                stride=self._args['s'], padding=self._args['p'], dilation=self._args['d'],
                                groups=self._args['g'], bias=self._args['b'], in_channels=self._args['ic']),
                    Lambda(activation_wiki[self._args['act']], name='Activation')]
        if hasattr(self, '_upsample'): layers.insert(0, Lambda(lambda x: F.upsample(x, scale_factor=self._upsample), name='Upsample'))
        self._layer = nn.Sequential(*layers)
        super().build(x)

    def forward(self, x):
        return self._layer(x)

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

    def  _mul_list(self, n):
        convs = [self]
        self._args['c'] = n[0]
        kwargs = self._args.copy()
        for c in n[1:]:
            kwargs['c'] = c
            convs.append(self.__class__(**kwargs))

        return convs

class Linear(Node):
    def __init__(self, o=None, b=True, flat=True, i=None, act='relu', **kwargs):
        super().__init__(o, b, flat, i, act, **kwargs)

    def build(self, x):
        from numpy import prod
        from magnet.functional import activation_wiki

        self._args['i'] = prod(x.shape[1:]) if self._args['flat'] else x.shape[-1]

        layers = [nn.Linear(*[self._args[k] for k in ('i', 'o', 'b')]),
                    Lambda(activation_wiki[self._args['act']], name='Activation')]
        if self._args['flat']: layers.insert(0, Lambda(lambda x: x.view(x.size(0), -1), name='Flatten'))

        self._layer = nn.Sequential(*layers)
        super().build(x)

    def forward(self, x):
        return self._layer(x)

    def  _mul_list(self, n):
        lins = [self]
        self._args['o'] = n[0]
        kwargs = self._args.copy()
        for o in n[1:]:
            kwargs['o'] = o
            lins.append(self.__class__(**kwargs))

        return lins