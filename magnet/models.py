import torch

from torch import nn

from .nodes import Node

class Sequential(nn.Sequential, Node):
    def __init__(self, *nodes, **kwargs):
        from ._utils import get_output_shape, to_node

        nodes = list(nodes)

        if type(nodes[-1]) in [list, tuple] and all(type(i) is int for i in nodes[-1]) or hasattr(nodes[-1], 'shape'):
            input_shape = nodes.pop()
        elif 'x' in kwargs.keys(): input_shape = kwargs.pop('x')
        else: raise RuntimeError('Need to provide an input shape or tensor, x')

        if hasattr(input_shape, 'shape'): input_shape = tuple(input_shape.shape)

        if len(nodes) == 1 and type(nodes[0]) in [list, tuple]: nodes = nodes[0]

        self._shape_sequence = [input_shape]
        name_dict = {}
        for i, node in enumerate(nodes):
            node = nodes[i] = to_node(node, input_shape)
            if isinstance(node, Node): node.build(torch.randn(input_shape))

            if node.name in name_dict.keys():
                name_dict[node.name] += 1
                node.name = node.name + str(name_dict[node.name])
            else: name_dict[node.name] = 1

            input_shape = get_output_shape(node, input_shape)
            self._shape_sequence.append(input_shape)

        super().__init__(*nodes)
        super().build()

    def forward(self, x=None):
        if x is None: x = torch.randn(1, *self._shape_sequence[0][1:])
        return super().forward(x)

    def summary(self, parameters='trainable', arguments=False, batch=False, max_width=120):
        from beautifultable import BeautifulTable
        from ._utils import num_params
        from .nodes import Node

        def _handle_parameter_output(mode, node=None):
            str_dict = {'trainable': 'Trainable', 'non-trainable': 'NON-Trainable', 'all': '', True: '(Trainable, NON-Trainable)'}
            if mode == 'col': return str_dict[parameters] + ' Parameters'

            def _get_num_params(module):
                n = num_params(module) if module is not None else (0, 0)
                n_dict = {'trainable': n[0], 'non-trainable': n[1], 'all': sum(n), True: n}
                n = n_dict[parameters]
                return ', '.join(['{:,}'] * len(n)).format(*n) if type(n) is tuple else '{:,}'.format(n)
            
            if mode == 'row': return _get_num_params(node)

            print('Total ' + str_dict[parameters] + ' Parameters:', _get_num_params(self))

        _start_idx = 0 if batch else 1
        shape_sequence = [', '.join(str(i) for i in s[_start_idx:]) for s in self._shape_sequence]


        table = BeautifulTable(max_width=max_width)
        column_headers = ['Node', 'Shape']
        if parameters is not False: column_headers.append(_handle_parameter_output('col'))

        if arguments: column_headers.append('Arguments')
        table.column_headers = column_headers
        
        row = ['input', shape_sequence[0]]
        if parameters is not False: row.append(_handle_parameter_output('row'))

        if arguments: row.append('')
        table.append_row(row)

        for node, shape in zip(self.children(), shape_sequence[1:]):
            name = node.name

            row = [name, shape]
            if parameters is not False: row.append(_handle_parameter_output('row', node))

            if arguments: 
                if isinstance(node, Node):row.append(node.get_args())
                else: row.append('')
            table.append_row(row)
            
        print(table)

        if parameters is not False: _handle_parameter_output('total')

class MonoSequential(Sequential):
    def __init__(self, node, *args, **kwargs):
        self._node = node

        sizes, args = self._find_sizes(*args)

        layers = self._make_layers(*sizes, **kwargs)

        args = layers + args

        super().__init__(*args, **kwargs)

    def _find_sizes(self, *args):
        i = [i for i, s in enumerate(args) if type(s) is int][-1] + 1
        return args[:i], list(args[i:])

    def _make_layers(self, *sizes, **kwargs):
        node = self._node

        layers = node(**kwargs) * sizes[:-1]
        kwargs.pop('act', None)
        layers.append(node(sizes[-1], act=None, **kwargs))

        return layers

class DNN(MonoSequential):
    def __init__(self, *sizes, **kwargs):
        from .nodes import Linear

        super().__init__(Linear, *sizes, **kwargs)

class CNN(MonoSequential):
    def __init__(self, *sizes, **kwargs):
        from .nodes import Conv

        super().__init__(Conv, *sizes, **kwargs)

    def _make_layers(self, *sizes, **kwargs):
        from .functional import global_avg_pool
        return super()._make_layers(*sizes, **kwargs) + [global_avg_pool]
