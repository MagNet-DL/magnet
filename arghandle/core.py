import inspect, importlib.util

from pathlib import Path
from functools import wraps

def arghandle(fn):
    handler_fn = __get_handler(fn)

    @wraps(fn)
    def new_fn(*args, **kwargs):
        args, kwargs = handler_fn(*args, **kwargs)

        return fn(*args, **kwargs)

    return new_fn

def args():
    frame = inspect.currentframe().f_back

    fn = frame.f_globals[frame.f_code.co_name]

    arg_specs = inspect.getfullargspec(fn)
    local = frame.f_locals

    varargs = local[arg_specs.varargs] if arg_specs.varargs is not None else []
    kwargs = local[arg_specs.varkw] if arg_specs.varkw is not None else {}
    if arg_specs.args is not None: kwargs.update({arg: local[arg] for arg in arg_specs.args})

    return varargs, kwargs

def __get_handler(fn):
    filepath = Path(inspect.getsourcefile(fn))
    new_filepath = filepath.parent / '__arghandle__' / filepath.name

    spec = importlib.util.spec_from_file_location(filepath.name, new_filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    handler_fn = getattr(module, fn.__name__)

    return handler_fn