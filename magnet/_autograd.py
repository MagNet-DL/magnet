import torch

from contextlib import contextmanager

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

build_lock = True

def eval(*modules):
	"""A Context Manager that makes it easy to run statements in eval mode.
	It sets ```modules``` in eval() mode and ensures that gradients are not computed"""

	from inspect import isfunction
	if not isfunction(modules[0]) or len(modules) > 1:
		return _eval_context_manager(*modules)

	from functools import wraps

	fn = modules[0]
	@wraps(fn)
	def new_fn(*args, **kwargs):
		from torch.nn import Module

		arg_list = list(args) + list(kwargs.values())
		modules = [a for a in arg_list if isinstance(a, Module)]

		with _eval_context_manager(*modules):
			return fn(*args, **kwargs)

	return new_fn

@contextmanager
def _eval_context_manager(*modules):
	states = []
	modules = [module for module in modules if module.training]
	for module in modules:
		states.append(module.training)
		module.eval()

	with torch.no_grad(): 
		try:
			yield
		finally:
			for module, state in zip(modules, states): module.train(state)