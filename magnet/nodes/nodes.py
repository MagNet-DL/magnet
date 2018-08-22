import torch
import magnet as mag

from torch import nn
from torch.nn import functional as F

from magnet.utils.misc import caller_locals, get_function_name

class Node(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._parse_args()
        self._built = False

    def build(self, *args, **kwargs):
        self._built = True
        self.to(mag.device)

    def __call__(self, *args, **kwargs):
        if not (self._built and mag.build_lock): self.build(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def _check_parameters(self, return_val):
        import warnings
        if not self._built: raise RuntimeError(f'Node {self.name} not built yet')
        if not mag.build_lock: warnings.warn('Build-lock disabled. The node may be re-built', RuntimeWarning)

        return return_val

    def parameters(self):
        return self._check_parameters(super().parameters())

    def named_parameters(self, memo=None, prefix=''):
        return self._check_parameters(super().named_parameters())

    def _parse_args(self):
        args = caller_locals(ancestor=True)
        args.update(args.pop('kwargs', {}))

        self.name = args.pop('name', self.__class__.__name__)

        self._args = args

    def get_args(self):
        return ', '.join(str(k) + '=' + str(v) for k, v in self._args.items())

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        try: self.device = next(self.parameters())[0].device
        except StopIteration: pass
        return self

    def load_state_dict(self, f):
        from pathlib import Path
        if isinstance(f, (str, Path)):
            device = self.device.type
            if device == 'cuda': device = 'cuda:0'

            return super().load_state_dict(torch.load(f, map_location=device))
        else:
            return super().load_state_dict(f)

    def  _mul_int(self, n):
        return [self] + [self.__class__(**self._args) for _ in range(n - 1)]

    def  _mul_list(self, n):
        raise NotImplementedError

    def __mul__(self, n):
        if type(n) is int or (type(n) is float and n.is_integer()):
            return self._mul_int(n)

        if type(n) is tuple or type(n) is list:
            return self._mul_list(n)