from torch import nn

class Sequential(nn.Sequential):
    def __init__(self, *layers, input_shape):
        from .nodes import Node, Lambda
        from inspect import isfunction

        self._shape_sequence = [input_shape]
        layers = list(layers)
        for i, layer in enumerate(layers):
            if isfunction(layer): 
                layers[i] = Lambda(layer)
                layer = layers[i]

            if isinstance(layer, Node):
                layer.build(input_shape)
                input_shape = layer.get_output_shape(input_shape)

            self._shape_sequence.append(input_shape)

        super().__init__(*layers)

    def summary(self, show_parameters='trainable'):
        from beautifultable import BeautifulTable
        from ._utils import num_params

        table = BeautifulTable()
        column_headers = ['Layer', 'Shape']
        if show_parameters == 'trainable': column_headers.append('Trainable Parameters')
        elif show_parameters == 'non-trainable': column_headers.append('NON-Trainable Parameters')
        elif show_parameters == 'all': column_headers.append('Parameters')
        elif show_parameters: column_headers.append('Parameters (Trainable, NON-Trainable)')
        table.column_headers = column_headers
        
        row = ['input', self._shape_sequence[0]]
        if show_parameters == 'trainable' or show_parameters == 'non-trainable' or show_parameters == 'all': row.append(0)
        elif show_parameters: row.append((0, 0))
        table.append_row(row)
        for layer, shape in zip(self.children(), self._shape_sequence[1:]):
            name = str(layer).split('(')[0]
            row = [name, shape]
            if show_parameters == 'trainable': row.append(num_params(layer)[0])
            elif show_parameters == 'non-trainable': row.append(num_params(layer)[1])
            elif show_parameters == 'all': row.append(sum(num_params(layer)))
            elif show_parameters: row.append(num_params(layer))
            table.append_row(row)
            
        print(table)

        if show_parameters == 'trainable': print('Total Trainable Parameters:', num_params(self)[0])
        elif show_parameters == 'non-trainable': print('Total NON-Trainable Parameters:', num_params(self)[1])
        elif show_parameters == 'all': print('Total Parameters:', sum(num_params(self)))
        elif show_parameters: print('Total Parameters (Trainable, NON-Trainable):', num_params(layer))