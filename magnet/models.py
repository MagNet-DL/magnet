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

    def summary(self):
        from beautifultable import BeautifulTable
        table = BeautifulTable()
        table.column_headers = ['Layer', 'Shape']
        
        table.append_row(['input', self._shape_sequence[0]])
        for layer, shape in zip(self.children(), self._shape_sequence[1:]):
            name = str(layer).split('(')[0]
            table.append_row([name, shape])
            
        print(table)