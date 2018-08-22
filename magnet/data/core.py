from . import data
from .transforms import image_transforms

def MNIST(val_split=0.2, path=data.DIR_DATA, **kwargs):
    from torchvision.datasets import mnist

    dataset = {mode: mnist.MNIST(path, train=(mode == 'train'), download=True)
                        for mode in ('train', 'test')}
    transforms = kwargs.pop('transforms', image_transforms())
    return data.Data.make(**dataset, val_split=val_split, transforms=transforms)