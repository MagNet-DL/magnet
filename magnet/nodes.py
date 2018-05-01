from torch import nn

from ._utils import caller_locals

class Conv(nn.Module):
    def __init__(self, *args, **kwargs):
        self._parse_params()
        super().__init__()
        
    def build(self, in_shape):
        shape_dict = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        ndim = len(in_shape) - 2
        module = shape_dict[ndim - 1]
        
        self._set_padding(in_shape)
        
        kw_map = {'k': 'kernel_size', 'c': 'out_channels','s': 'stride',
                 'p': 'padding', 'd': 'dilation', 'g': 'groups', 'b': 'bias'}
        kwargs = {kw_map[k]: v for k, v in self._args.items()}
        kwargs['in_channels'] = in_shape[1]
        self.conv = module(**kwargs)
        
    def forward(self, x):
        return self.conv(x)
    
    def _set_padding(self, in_shape):
        p = self._args['p']
        if p == 'half': f = 0.5
        elif p == 'same': f = 1
        else: return
        
        s = 1 / f
        if not s.is_integer(): 
            raise RuntimeError("Padding value won't hold for all vector sizes")
            
        self._args['d'] = 1
        self._args['s'] = int(s)
        self._args['p'] = int(self._args['k'] // 2)
        if self._args['c'] is None: 
            self._args['c'] = self._args['s'] * in_shape[1]
        
    def _parse_params(self):
        args = caller_locals()
        
        default_param_list = list(self._default_params.items())

        for i, arg_val in enumerate(args['args']):
            param_name = default_param_list[i][0]
            args[param_name] = arg_val
        args.pop('args')
        
        for param_name, default in default_param_list:
            if param_name in args['kwargs'].keys():
                args[param_name] = args['kwargs'][param_name]
            elif param_name not in args.keys():
                args[param_name] = default
        args.pop('kwargs')
        
        self._args = args
        
    @property
    def _default_params(self):
        return {'c': None, 'k': 3, 'p': 'half', 's': 1, 'd': 1, 'g': 1, 'b': True}