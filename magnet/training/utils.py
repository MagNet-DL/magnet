import torch, pickle

import magnet as mag

def load_state(module, path, alternative_name=None):
    r"""Loads the state_dict of a PyTorch object from a specified path.

    This is a more robust version of the of the PyTorch way in the sense that
    the device mapping is automatically handled.

    Args:
        module (object): Any PyTorch object that has a state_dict
        path (pathlib.Path): The path to folder containing the state_dict file
        alternative_name (str or None): A fallback name for the file if the
            module object does not have a name attribute. Default: ``None``

    Raises:
        RuntimeError: If no :attr:`alternative_name` is provided and the module
            does not have a name.

    .. note::
        If you already know the file name, set :attr:`alternative_name` to that.

        This is just a convinience method that assumes that the file name
        will be the same as the name of the module (if there is one).
    """
    name = alternative_name if not hasattr(module, 'name') else module.name
    if name is None: raise RuntimeError('Module Name is None!')

    filepath = path / (name + '.pt')

    device = 'cuda:0' if mag.device.type == 'cuda' else 'cpu' # Needed patch
    if filepath.exists(): module.load_state_dict(torch.load(filepath, map_location=device))

def save_state(module, path, alternative_name=None):
    r"""Saves the state_dict of a PyTorch object to a specified path.

    Args:
        module (object): Any PyTorch object that has a state_dict
        path (pathlib.Path): The path to a folder to save the state_dict to
        alternative_name (str or None): A fallback name for the file if the
            module object does not have a name attribute. Default: ``None``

    Raises:
        RuntimeError: If no :attr:`alternative_name` is provided and the module
            does not have a name.
    """
    name = alternative_name if not hasattr(module, 'name') else module.name
    if name is None: raise RuntimeError('Module Name is None!')

    path.mkdir(parents=True, exist_ok=True)
    filepath = path / (name + '.pt')

    torch.save(module.state_dict(), filepath)

def load_object(path, **kwargs):
    r"""A convinience method to unpickle a file.

    Args:
        path (pathlib.Path): The path to the pickle file

    Keyword Args:
        default (object): A default value to be returned
            if the file does not exist. Default: ``None``

    Raises:
        RuntimeError: If a default keyword argument is not provided and the
            file is not found.
    """
    if path.exists():
        with open(path, 'rb') as f: return pickle.load(f)
    elif 'default' in kwargs.keys():
        return kwargs['default']
    else:
        raise RuntimeError(f'The path {path} does not exist. No default provided either.')

def save_object(obj, path):
    r"""A convinience method to pickle an object.

    Args:
        obj (object): The object to pickle
        path (pathlib.Path): The path to save to

    .. note::
        If the :attr:`path` does not exists, it is created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f: pickle.dump(obj, f)