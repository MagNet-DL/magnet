def Sequential(*layers):
    from torch import nn
    
    return nn.Sequential(*layers)