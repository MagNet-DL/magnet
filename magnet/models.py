import torch

from torch import nn

from .nodes import Node

class Sequential(nn.Sequential):
    def __init__(self, *layers, input_shape):
        layers = list(layers)
        for layer in layers:
            if isinstance(layer, Node):
                layer.build(input_shape)
                input_shape = layer.get_output_shape(input_shape)

        super().__init__(*layers)