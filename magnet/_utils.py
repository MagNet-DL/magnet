def caller_locals(ancestor=False):
    """Print the local variables in the caller's frame."""
    import inspect
    frame = inspect.currentframe().f_back
    
    try:
        l = frame.f_back.f_locals
        if not ancestor:
            true_caller = l.pop('self', None)
            true_class = l.pop('__class__', None)
        else:
            f_class = None
            caller = l.pop('self')
            while f_class != caller.__class__:
                frame = frame.f_back
                l = frame.f_locals
                f_class = l.pop('__class__', None)
                l.pop('self', None)
        return l
    finally: del frame