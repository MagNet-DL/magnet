import arghandle

import numpy as np, matplotlib.pyplot as plt

from arghandle.handlers import typecheck
from pathlib import Path
from torch import is_tensor

def show_images(images, titles=None, pixel_range='auto', cmap='gray', shape='square',
                resize='smean', merge=True, retain=False, savepath=None, **kwargs):
    images = __handle_image_type(images)
    images = __handle_image_dimensions(images)

    n_images = len(images)

    if titles is None and not merge: titles = [None] * n_images

    typecheck(pixel_range=pixel_range, include=(str, tuple, list, np.ndarray))
    if pixel_range == 'auto':
        pixel_range = (min(image.min() for image in images), max(image.max() for image in images))
    elif isinstance(pixel_range, str):
        raise ValueError(f"pixel_range should be either a (min, max) tuple or 'auto'\nGot {pixel_range}")

    shape = __handle_shape(n_images, shape)

    if isinstance(resize, str): resize = __handle_resize(images, resize)

    typecheck(savepath=savepath, include=(Path, str, None))

    return arghandle.args()

def _show_image(image, title=None, cmap='gray', ax=None, pixel_range='auto', retain=False):
    if image.shape[-1] == 1: image = image.squeeze(-1)
    if ax is None: ax = plt.subplots()[1]

    return arghandle.args()

def __handle_shape(n_images, shape):
    typecheck(shape=shape, include=(str, tuple, list))

    # String shapes
    if isinstance(shape, str): return __handle_string_shape(n_images, shape)

    # Shapes have to be positive integers
    if not all(isinstance(s, int) and s > 0 for s in shape):
        raise ValueError('All shape elements need to be positive integers')

    # Shape mismatch with number of images
    n_shape = np.prod(shape)
    if n_shape != n_images:
        if n_images == 1: error_msg = f'is just one image!'
        else: error_msg = f'are {n_images} images!'
        error_msg = f"""The shape {shape} has {n_shape} cells. But there """ + error_msg + """
                    \nLet it be the default ('square') if you're unsure."""
        raise ValueError(error_msg)

    return shape

def __handle_string_shape(n_images, shape):
    if shape == 'row': return 1, n_images
    if shape == 'column': return n_images, 1
    if shape == 'square': return __square_factors(n_images)

    raise ValueError(f"`shape` needs to be one of 'square', 'row' or 'column'. Got {shape}")

def __handle_resize(images, size='smean'):
    shapes = np.array([image.shape[:-1] for image in images])

    # Make all the shapes square
    if size[0] == 's':
        shapes = np.array([[int(np.sqrt(np.prod(s)))] * 2 for s in shapes])
        size = size[1:]

    if size == 'min': size = shapes.min(0)
    elif size == 'max': size = shapes.max(0)
    elif size == 'mean': size = shapes.mean(0)

    size = size.astype(np.uint)

    return size

def __square_factors(x):
    if x == 1: return 1, 1
    if x == 2: return 1, 2

    factors = [i for i in range(2, int(np.sqrt(x)) + 1) if x % i == 0]

    # x is prime
    if len(factors) == 0: return 1, x

    return factors[-1], x // factors[-1]

def __handle_image_dimensions(images, stacked=True):
    if isinstance(images, (list, tuple)):
        return [__handle_image_dimensions(image, stacked=False) for image in images]

    if stacked and isinstance(images, np.ndarray): return images

    if len(images.shape) == 2: return np.repeat(np.expand_dims(images, -1), 3, -1)

    error_msg = f'Incorrect image dimensions.\nGot {images.shape}'
    if len(images.shape) in (3, 4):
        if images.shape[-1] == 1: return np.repeat(images, 3, -1)
        elif images.shape[-1] != 3: raise ValueError(error_msg)
        return images

    raise ValueError(error_msg)

def __handle_image_type(image):
    if __is_generator(image): image = list(image)
    if isinstance(image, (list, tuple)): return [__handle_image_type(img) for img in image]

    if isinstance(image, np.ndarray): return image

    if isinstance(image, str): image = Path(str)

    if isinstance(image, Path):
        if not image.exists(): raise RuntimeError(f'No such file exists: {image}')
        return plt.imread(image)

    if is_tensor(image):
        if len(image.shape) == 4:
            return image.permute(0, 2, 3, 1).detach().cpu().numpy()
        return image.permute(1, 2, 0).detach().cpu().numpy()

def __is_generator(iterable):
    return hasattr(iterable,'__iter__') and not hasattr(iterable,'__len__')