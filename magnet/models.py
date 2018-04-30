from torch import nn

class Sequential(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)