def augmented_image_transforms(d=0, t=0, s=0, sh=0, ph=0, pv=0, resample=2):
    r"""Returns a list of augmented transforms to be applied to natural images.

    Args:
        d (sequence or float or int): Range of degrees to select from.
            Default: ``0``
        t (tuple): Tuple of maximum absolute fraction for horizontal
            and vertical translations. Default: ``0``
        s (tuple, optional): Scaling factor interval. Default: ``0``
        sh (sequence or float or int, optional): Range of shear. Default: ``0``
        ph (float): The probability of flipping the image horizontally.
            Default: ``0``
        pv (float): The probability of flipping the image vertically.
            Default: ``0``
        resample (int): An optional resampling filter. Default: ``2``

    See :py:class:`torchvision.transforms` for more details.
    """
    from torchvision import transforms

    degrees = d
    translate = None if t == 0 else (t, t)
    scale = None if s == 0 else (1 - s, 1 + s)
    shear = None if sh == 0 else sh
    return transforms.Compose([transforms.RandomAffine(degrees, translate, scale, shear, resample),
                             transforms.RandomHorizontalFlip(ph),
                             transforms.RandomVerticalFlip(pv),
                             transforms.ToTensor(),
                             transforms.Normalize(*[[0.5] * 3] * 2)])

def image_transforms(augmentation=0, direction='horizontal'):
    r"""Returns a list of transforms to be applied to natural images.

    Args:
        augmentation (float): The percentage of augmentation to be applied.
            Default: ``0``
        direction (str): The direction to flip the image at random.
            Default: ``'horizontal'``
    """
    x = augmentation
    if direction == 'horizontal': ph = 0.25 * x; pv = 0
    elif direction == 'vertical': ph = 0; pv = 0.25 * x
    elif direction == 'both': ph = 0.25 * x; pv = 0.25 * x
    return augmented_image_transforms(d=45 * x, t=0.25 * x, s=0.13 * x, sh=6 * x, ph=ph, pv=pv)