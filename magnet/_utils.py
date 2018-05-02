def caller_locals(ancestor=False):
    """Print the local variables in the caller's frame."""
    import inspect
    frame = inspect.currentframe().f_back.f_back
    
    try:
        l = frame.f_locals
        
        if ancestor:
            f_class = l.pop('__class__', None)
            caller = l.pop('self')
            while f_class is not None and isinstance(caller, f_class):
                l = frame.f_locals
                l.pop('self', None)
                frame = frame.f_back
                f_class = frame.f_locals.pop('__class__', None)

        l.pop('self', None)
        l.pop('__class__', None)
        return l
    finally: del frame

def num_params(module):
    from numpy import prod

    trainable, non_trainable = 0, 0
    for p in module.parameters():
        n = prod(p.size())
        if p.requires_grad:
            trainable += n
        else:
            non_trainable += n

    return trainable, non_trainable

def get_output_shape(module, input_shape):
    from torch import no_grad, randn
    
    with no_grad(): return tuple(module(randn(input_shape)).size())