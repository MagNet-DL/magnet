def checkif(arg, types, exclude=False, name=None):
    if name is None: name = 'One of the arguments'

    def error_message():
        negation = 'NOT ' if exclude else ''

        got_type = type(arg).__name__

        if len(types) > 1:
            arg_string = 'one of (' + ', '.join(str(t.__name__) for t in types) + ')'
        else:
            arg_string = str(types[0].__name__)

        return f'{name} should ' + negation + f'be ' + arg_string + f'. Got {got_type}'

    if not isinstance(types, (tuple, list)): types = (types, )

    if None in types:
        if arg is None: return
        types = tuple(typ for typ in types if typ is not None)

    if exclude:
        if isinstance(arg, types): raise TypeError(error_message())
    else:
        if not isinstance(arg, types): raise TypeError(error_message())

def typecheck(include=None, exclude=None, **kwargs):
    if len(kwargs) > 1:
        for k, v in kwargs.items(): typecheck(**{k: v}, include=include, exclude=exclude)
        return

    name, val = tuple(kwargs.items())[0]

    if include is not None: checkif(val, include, name=name)
    if exclude is not None: checkif(val, exclude, exclude=True, name=name)