import torch
import magnet as mag

from torch import nn
from torch.nn import functional as F

from ._utils import caller_locals, get_function_name

class Module(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		try: self.device = next(self.parameters())[0].device
		except StopIteration: pass
		return self

	def load_state_dict(self, f):
		device = self.device.type
		if device == 'cuda': device = 'cuda:0'

		super().load_state_dict(torch.load(f, map_location=device))

class Node(Module):
	configuration = {}

	def __init__(self, *args, **kwargs):
		self._parse_args()
		self._init_config()
		super().__init__()

		self._built = False

	def build(self, *args, **kwargs):
		self._built = True
		self.to(mag.device)

	def forward(self, *args, **kwargs):
		if not (self._built and mag.build_lock): self.build(*args, **kwargs)

	def _check_parameters(self, return_val):
		import warnings
		if not self._built: raise RuntimeError(f'Node {self.name} not built yet')
		if not mag.build_lock: warnings.warn('Build-lock disabled. The node may be re-built', RuntimeWarning)

		return return_val

	def parameters(self):
		return self._check_parameters(super().parameters())

	def named_parameters(self, memo=None, prefix=''):
		return self._check_parameters(super().named_parameters())

	@property
	def _args_order(self):
		return []

	def _parse_args(self):
		args = caller_locals(ancestor=True)
		if 'args' not in args.keys(): args['args'] = ()
		args['args'] = list(args['args'])
		if 'kwargs' not in args.keys(): args['kwargs'] = {}

		if len(args['args']) > 0 and type(args['args'][0]) is str:
			self.name = args['args'].pop(0)
		elif 'name' in args['kwargs'].keys():
			self.name = args['kwargs'].pop('name')
		else:
			self.name = self.__class__.__name__

		for name, val in zip(self._args_order, args['args']): args[name] = val
		args.pop('args')

		args.update(args['kwargs'])
		args.pop('kwargs')

		self._args = args

	def _init_config(self):
		for k in self.configuration.keys():
			if k in self._args.keys(): self.configuration[k] = self._args[k]

	def get_args(self):
		return ', '.join(str(k) + '=' + str(v) for k, v in self._args.items())

	def  _mul_int(self, n):
		return [self] + [self.__class__(**self._args) for _ in range(n - 1)]

	def  _mul_list(self, n):
		pass

	def __mul__(self, n):
		if type(n) is int or (type(n) is float and n.is_integer()):
			return self._mul_int(n)

		if type(n) is tuple or type(n) is list:
			return self._mul_list(n)

class MonoNode(Node):
	configuration = {'act': 'relu'}

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._set_activation()

    def build(self, x):
        layer_class = self._find_layer(x)
        kwargs = self._get_kwargs()
        self._layer = layer_class(**kwargs)

        super().build(x)

	def forward(self, x):
		super().forward(x)
		return self._activation(self._layer(x))

	def _find_layer(self, x):
		pass

	def _get_kwargs(self):
		return {k: self.configuration[v] for k, v in self._kwargs_dict.items()}

	@property
	def _kwargs_dict(self):
		pass

	def _set_activation(self):
		from functools import partial

		activation_dict = {'relu': F.relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh,
							'lrelu': partial(F.leaky_relu, leak=0.2), None: lambda x: x}
		self._activation = activation_dict[self.configuration['act']]

class Conv(MonoNode):
    configuration = {'c': None, 'k': 3, 'p': 'half', 's': 1, 'd': 1, 'g': 1, 'b': True, 'ic': None, 'dims': None}
    configuration.update(MonoNode.configuration)

    def build(self, x):
        self._set_padding(x)
        self.configuration['ic'] = x.shape[1]
        super().build(x)

    def forward(self, x):
        if hasattr(self, '_upsample'): x = F.upsample(x, scale_factor=self._upsample)
        return super().forward(x)

    def _find_layer(self, x):
        shape_dict = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        ndim = len(x.shape) - 2
        self.configuration['dims'] = ndim
        return shape_dict[ndim - 1]

    @property
    def _kwargs_dict(self):
        return {'kernel_size': 'k', 'out_channels': 'c', 'stride': 's',
                'padding': 'p', 'dilation': 'd', 'groups': 'g', 'bias': 'b', 'in_channels': 'ic'}

    def _set_padding(self, x):
        in_shape = x.shape

        p = self.configuration['p']
        if p == 'half': f = 0.5
        elif p == 'same': f = 1
        elif p == 'double':
            self._upsample = 2
            self.configuration['c'] = in_shape[1] // 2
            f = 1
        else: return

        s = 1 / f
        if not s.is_integer():
            raise RuntimeError("Padding value won't hold for all vector sizes")

        self.configuration['d'] = 1
        self.configuration['s'] = int(s)
        self.configuration['p'] = int(self.configuration['k'] // 2)
        if self.configuration['c'] is None:
            self.configuration['c'] = self.configuration['s'] * in_shape[1]

    @property
    def _args_order(self):
        return ['c', 'k', 'p', 's', 'd', 'g', 'b', 'ic', 'dims']

    def  _mul_list(self, n):
        convs = [self]
        self.configuration['c'] = n[0]
        kwargs = self._args.copy()
        for c in n[1:]:
            kwargs['c'] = c
            convs.append(self.__class__(**kwargs))

        return convs

class Linear(MonoNode):
	configuration = {'o': None, 'b': True, 'act': 'relu', 'flat': True, 'i': None}
	configuration.update(MonoNode.configuration)

	def build(self, x):
		from numpy import prod
		self.configuration['i'] = prod(x.shape[1:]) if self.configuration['flat'] else x.shape[-1]
		super().build(x)

	def forward(self, x):
		if self.configuration['flat']: x = x.view(x.size(0), -1)

		return super().forward(x)

	def _find_layer(self, x):
		return nn.Linear

	@property
	def _kwargs_dict(self):
		return {'in_features': 'i', 'out_features': 'o', 'bias': 'b'}

	@property
	def _args_order(self):
		return ['o', 'b', 'act', 'flat', 'i']

	def  _mul_list(self, n):
		lins = [self]
		self.configuration['o'] = n[0]
		kwargs = self._args.copy()
		for o in n[1:]:
			kwargs['o'] = o
			lins.append(self.__class__(**kwargs))

		return lins

class Lambda(Node):
	configuration = {'fn': None}

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		if self.name == self.__class__.__name__:
			self.name = get_function_name(self.configuration['fn'])
			if self.name is None: self.name = 'Lambda'

	def forward(self, x):
		super().forward(x)
		return self.configuration['fn'](x)

	@property
	def _args_order(self):
		return ['fn']