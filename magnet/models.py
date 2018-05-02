from torch import nn

class Sequential(nn.Sequential):
    def __init__(self, *nodes, input_shape):
        from .nodes import Node, Lambda
        from inspect import isfunction

        self._shape_sequence = [input_shape]
        nodes = list(nodes)
        for i, node in enumerate(nodes):
            if isfunction(node): 
                nodes[i] = Lambda(node)
                node = nodes[i]

            if isinstance(node, Node):
                node.build(input_shape)
                input_shape = node.get_output_shape(input_shape)

            self._shape_sequence.append(input_shape)

        super().__init__(*nodes)

    def summary(self, show_parameters='trainable', show_args=False):
        from beautifultable import BeautifulTable
        from ._utils import num_params

        table = BeautifulTable()
        column_headers = ['Node', 'Shape']
        if show_parameters == 'trainable': column_headers.append('Trainable Parameters')
        elif show_parameters == 'non-trainable': column_headers.append('NON-Trainable Parameters')
        elif show_parameters == 'all': column_headers.append('Parameters')
        elif show_parameters: column_headers.append('Parameters (Trainable, NON-Trainable)')

        if show_args: column_headers.append('Arguments')
        table.column_headers = column_headers
        
        row = ['input', self._shape_sequence[0]]
        if show_parameters == 'trainable' or show_parameters == 'non-trainable' or show_parameters == 'all': row.append(0)
        elif show_parameters: row.append((0, 0))

        if show_args: row.append('')
        table.append_row(row)
        for node, shape in zip(self.children(), self._shape_sequence[1:]):
            name = str(node).split('(')[0]
            row = [name, shape]
            if show_parameters == 'trainable': row.append('{:,}'.format(num_params(node)[0]))
            elif show_parameters == 'non-trainable': row.append('{:,}'.format(num_params(node)[1]))
            elif show_parameters == 'all': row.append('{:,}'.format(sum(num_params(node))))
            elif show_parameters: row.append('{:,}, {:,}'.format(*num_params(node)))

            if show_args: row.append(node.get_args())
            table.append_row(row)
            
        print(table)

        if show_parameters == 'trainable': print('Total Trainable Parameters:', '{:,}'.format(num_params(self)[0]))
        elif show_parameters == 'non-trainable': print('Total NON-Trainable Parameters:', '{:,}'.format(num_params(self)[1]))
        elif show_parameters == 'all': print('Total Parameters:', '{:,}'.format(sum(num_params(self))))
        elif show_parameters: print('Total Parameters (Trainable, NON-Trainable):', '{:,}, {:,}'.format(*num_params(node)))