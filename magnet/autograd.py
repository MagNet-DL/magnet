import torch

from contextlib import contextmanager

device = 'cuda' if torch.cuda.is_available() else 'cpu'

lock_build = True

@contextmanager
def eval(*modules):
    """A Context Manager that makes it easy to run statements in eval mode.
    It sets ```modules``` in eval() mode and ensures that gradients are not computed"""

    states = []
    for module in modules:
        states.append(module.training)
        module.eval()
        
    with torch.no_grad(): yield

    for module, state in zip(modules, states): module.training = state