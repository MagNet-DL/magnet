import torch

from contextlib import contextmanager

def accelerate(yes=True):
    """Accelerates computations using a CUDA compatible GPU if available"""
    default_type = torch.zeros(1).type()
    if not yes:
        message = 'Using CPU as requested'
        device = 'cpu'
    else:
        if not torch.cuda.is_available():
            message = 'A GPU could not be found!\nUsing the slow, boring CPU.'
            device = 'cpu'
        else:
            message = 'Accelerating your code using a GPU.'
            device = 'cuda'

        print(message)
        torch.set_default_tensor_type(torch.zeros(1).to(device).type())

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