def caller_locals():
    """Print the local variables in the caller's frame."""
    import inspect
    frame = inspect.currentframe()
    
    try:
        l = frame.f_back.f_back.f_locals
        l.pop('self', None); l.pop('__class__', None)
        return l
    finally: del frame