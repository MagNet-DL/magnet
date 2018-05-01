from torch import nn

class Sequential(nn.Sequential):
    def __init__(self, *layers, input_shape):
        from .nodes import Node, Lambda
        from inspect import isfunction

        layers = list(layers)
        for i, layer in enumerate(layers):
            if isfunction(layer): layers[i] = Lambda(layer)

            if isinstance(layer, Node):
                layer.build(input_shape)
                input_shape = layer.get_output_shape(input_shape)

        super().__init__(*layers)