import torch
import magnet as mag

from torch import nn

from .nodes import Node, Module

class MonoSequential(nn.Sequential):
	def __init__(self, node, *args, **kwargs):
		self._node = node

		sizes, args = self._find_sizes(*args)

		layers = self._make_layers(*sizes, **kwargs)

		args = layers + args

		super().__init__(*args, **kwargs)

	def _find_sizes(self, *args):
		i = [i for i, s in enumerate(args) if type(s) is int][-1] + 1
		return args[:i], list(args[i:])

	def _make_layers(self, *sizes, **kwargs):
		node = self._node

		layers = node(**kwargs) * sizes[:-1]
		kwargs.pop('act', None)
		layers.append(node(sizes[-1], act=None, **kwargs))

		return layers

class DNN(MonoSequential):
	def __init__(self, *sizes, **kwargs):
		from .nodes import Linear

		super().__init__(Linear, *sizes, **kwargs)

class CNN(MonoSequential):
	def __init__(self, *sizes, **kwargs):
		from .nodes import Conv

		super().__init__(Conv, *sizes, **kwargs)

	def _make_layers(self, *sizes, **kwargs):
		from .functional import global_avg_pool
		return super()._make_layers(*sizes, **kwargs) + [global_avg_pool]
