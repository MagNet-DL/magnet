import torch

from contextlib import contextmanager

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

build_lock = True

def eval(*modules):
    r"""A Context Manger that makes it easy to run
    computations in ``eval`` mode.

    It sets modules in their ```eval`` mode and ensures
    that gradients are not computed.

    This is a more wholesome option than torch.no_grad() since many Modules
    (BatchNorm, Dropout etc.) behave differently while training and testing.

    Examples::

    >>> import magnet as mag

    >>> import magnet.nodes as mn

    >>> import torch

    >>> model = mn.Linear(10)

    >>> x = torch.randn(4, 3)

    >>> # Using eval() as context manager
    >>> with mag.eval(model):
    >>>     model(x)

    >>> # Use as decorator
    >>> @mag.eval(model)
    >>> def foo():
    >>>     return model(x)
    >>> foo()

    >>> # The modules can also be given at runtime by specifying no arguments
    >>> @mag.eval
    >>> def foo(model):
    >>>     return model(x)
    >>> foo()
    >>> # The method then takes modules from the arguments
    >>> # to the decorated function.
    """

    from inspect import isfunction

    # Check if called as decorator
    if not isfunction(modules[0]) or len(modules) > 1:
        return _eval_context_manager(*modules)

    from functools import wraps

    fn = modules[0] # The decorated function
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