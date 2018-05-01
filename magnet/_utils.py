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