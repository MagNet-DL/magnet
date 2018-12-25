from . import data
from .transforms import image_transforms

def MNIST(val_split=0.2, path=None, **kwargs):
    r"""The MNIST Dataset.

    Args:
        val_split (float): The fraction of training data to hold out
            as validation if validation set is not given. Default: ``0.2``
        path (pathlib.Path or str): The path to save the dataset to.
            Default: Magnet Datapath

    Keyword Args:
        (): See ``Data`` for more details.
    """
    from torchvision.datasets import mnist

    if path is None: path = data.DIR_DATA

    dataset = {mode: mnist.MNIST(path, train=(mode == 'train'), download=True)
                        for mode in ('train', 'test')}
    transforms = kwargs.pop('transforms', image_transforms())
    return data.Data(**dataset, val_split=val_split, transforms=transforms)


def CIFAR10(val_split=0.2, path=None, **kwargs):
    r"""The CIFAR10 Dataset.

    Args:
        val_split (float): The fraction of training data to hold out
            as validation if validation set is not given. Default: ``0.2``
        path (pathlib.Path or str): The path to save the dataset to.
            Default: Magnet Datapath

    Keyword Args:
        (): See ``Data`` for more details.
    """
    from torchvision.datasets import cifar

    if path is None: path = data.DIR_DATA

    dataset = {mode: cifar.CIFAR10(path, train=(mode == 'train'), download=True)
                        for mode in ('train', 'test')}
    transforms = kwargs.pop('transform', image_transforms())
    
    return data.Data(**dataset, val_split=0.2, transforms=transforms)


def CIFAR100(val_split=0.2, path=None, **kwargs):
    r"""The CIFAR100 Dataset.

    Args:
        val_split (float): The fraction of training data to hold out
            as validation if validation set is not given. Default: ``0.2``
        path (pathlib.Path or str): The path to save the dataset to.
            Default: Magnet Datapath

    Keyword Args:
        (): See ``Data`` for more details.
    """
    from torchvision.datasets import cifar

    if path is None: path = data.DIR_DATA

    dataset = {mode: cifar.CIFAR100(path, train=(mode == 'train'), download=True)
                        for mode in ('train', 'test')}
    transforms = kwargs.pop('transform', image_transforms())
    
    return data.Data(**dataset, val_split=0.2, transforms=transforms)
