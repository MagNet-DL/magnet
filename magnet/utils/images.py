import numpy as np
import itertools
import matplotlib.pyplot as plt
import os

from skimage.transform import resize as imresize

from torch import is_tensor

def show_images(images, **kwargs):
    r"""A nifty helper function to show images represented by tensors.

    Args:
        images (list or numpy.ndarray or str or torch.Tensor): The images
            to show

    * :attr:`images` can be anything which from you could conceivable harvest
      an image.

      If it's a :py:class:`torch.Tensor`, it is converted to
      a :py:class:`numpy.ndarray`.

      The first dimension of the tensor is treated as a batch dimension.

      If it's a ``str``, it is treated as a glob path from which all images
      are extracted.

      More commonly, a list of numpy arrays can be given.

    Keyword Arguments:
        pixel_range (tuple or ``'auto'``): The range of pixel values
            to be expected. Default: ``'auto'``
        cmap (str or None): The color map for the plots. Default: ``None``
        merge (bool): If ``True``, all images are merged into one giant image.
            Default: ``True``
        titles (list or None): The titles for each image. Default: ``None``
        shape (str): The shape of the merge tile.
            Default: ``'square'``
        resize (str): The common shape to which images are resized.
            Default: `smean`
        retain (bool): If ``True``, the plot is retained. Default: ``False``
        savepath (str or None): If given, the image is saved to this path.
            Default: ``None``

    * :attr:`pixel_range` default to the range in the image.

    * :attr:`cmap` is set to ``'gray'`` if the images are B/W.

    * :attr:`titles` should only be given if :attr:`merge` is ``True``.

    .. note::
        The merge shape is controlled by :attr:`shape` which can be either
        ``'square'``, ``'row'``, ``'column'`` or a ``tuple`` which explicitly
        specifies this shape.

        ``'square'`` automatically finds a shape with least difference between
        the number of rows and columns. This is aesthetically pleasing.

        In the explicit case, the product of the tuple needs to equal the
        number of images.
    """
    titles = kwargs.pop('titles', None)
    pixel_range = kwargs.pop('pixel_range', 'auto')
    cmap = kwargs.pop('cmap', None)
    shape = kwargs.pop('shape', 'square')
    resize = kwargs.pop('resize', 'smean')
    merge = kwargs.pop('merge', True)
    retain = kwargs.pop('retain', False)
    savepath = kwargs.pop('savepath', None)

    def _handle_args():
        nonlocal images, pixel_range, shape, resize
        if not isinstance(images, (list, tuple, np.ndarray, str)) and not is_tensor(images):
            raise TypeError('images needs to be a list, tuple, string or numpy array. Got {}'.format(type(images)))
        if is_tensor(images):
            images = list(images.permute(0, 2, 3, 1).detach().cpu().numpy())
        elif type(images) is str:
            from glob import glob
            images = [plt.imread(f) for f in glob(images, recursive=True)]
        elif type(images) is np.ndarray:
            images = list(images)
        elif type(images) in (list, tuple):
            if any(type(image) is not np.ndarray for image in images):
                raise TypeError('All images need to be numpy arrays')

        if type(pixel_range) not in [tuple, list, np.ndarray]:
            if type(pixel_range) is str:
                if pixel_range == 'auto':
                    pixel_range = (min([image.min() for image in images]), max([image.max() for image in images]))
                else:
                    raise ValueError('pixel_range should be auto. Found {}'.format(pixel_range))
            else:
                raise TypeError("pixel_range needs to be a tuple, list, numpy array or 'auto'."
                                " Found {}".format(type(pixel_range)))
        elif len(pixel_range) != 2:
            raise ValueError('pixel_range needs to be of size 2 - (min, max). Found size {}'.format(len(pixel_range)))

        resize = 'min' if len(images) == 1 else resize
        resize = kwargs.pop('resize', resize)

        if type(merge) is not bool:
            raise TypeError('merge needs to be either true or false. Got {}'.format(type(merge)))

        if titles is not None and type(titles) not in (list, tuple, str):
            raise TypeError('title needs to be a string or a list or tuple of strings. Got {}'.format(type(titles)))
        elif type(titles) is str:
            if not merge:
                raise TypeError('Given a single title, merge should be True.'
                                '\nElse give a list of titles.')
        elif titles is not None:
            if merge:
                raise TypeError('Given a list of titles, merge should be False.'
                                '\nElse give a single title or leave it None.')

        if savepath is not None:
            if type(savepath) is not str:
                raise TypeError('savepath needs to be a string')
            os.makedirs(os.path.split(savepath)[0], exist_ok=True)

        shape = _resolve_merge_shape(len(images), shape)

    err_arg = _handle_args()
    if err_arg is not None:
        return err_arg

    if cmap is None:
        if len(images[0].shape) == 2 or (len(images[0].shape)  == 3 and images[0].shape[-1] == 1):
            cmap = 'gray'

    images = _colorize_images(images)
    images = [(image - pixel_range[0]) / (pixel_range[1] - pixel_range[0]) * 2 - 1 for image in images]
    images = _resize_images(images, shape=resize)
    if not merge:
        fig, axes = plt.subplots(shape[0], shape[1])
        for i, ax in enumerate(axes.flat):
            _show_image(images[i], title=titles[i], cmap=cmap, ax=ax, retain=True)
        fig.tight_layout()
    else:
        _show_image(_merge_images(images, shape), title=titles, cmap=cmap, retain=True)

    if not retain:
        plt.show()

    if savepath is not None:
        plt.savefig(savepath, dpi=400, bbox_inches='tight')


def _colorize_images(images):
    def _handle_args():
        if type(images) is np.ndarray:
            if len(images.shape) > 2:
                return _colorize_images(list(images))

    err_arg = _handle_args()
    if err_arg is not None:
        return err_arg

    color_images = []
    for i, image in enumerate(images):
        if len(image.shape) == 2:
            color_images.append(np.repeat(np.expand_dims(image, -1), 3, -1))
        elif len(image.shape) == 3:
            if image.shape[-1] == 3:
                color_images.append(image)
            elif image.shape[-1] == 1:
                color_images.append(np.repeat(image, 3, -1))
            else:
                raise ValueError('Incorrect image dimensions for image at {}'.format(i))
        else:
            raise ValueError('Incorrect image dimensions for image at {}'.format(i))

    return color_images


def _resize_images(images, shape='smean', interp=1, mode='constant'):
    def _resolve_shape():
        nonlocal shape
        shapes = np.array([image.shape[:-1] for image in images])

        # Make all the shapes square
        if shape[0] == 's':
            shapes = np.array([[int(np.sqrt(np.prod(s)))] * 2 for s in shapes])
            shape = shape[1:]

        if shape == 'min':
            shape = shapes.min(0)
        elif shape == 'max':
            shape = shapes.max(0)
        elif shape == 'mean':
            shape = shapes.mean(0)

        shape = shape.astype(np.uint)

    def _handle_args():
        nonlocal shape
        if any(len(image.shape) != 3 for image in images):
            raise ValueError('All images must have 3 dimensions.')
        if type(shape) not in (tuple, list):
            if type(shape) is not str:
                raise TypeError('shape must be either a tuple, list or string.')
            if shape not in ['min', 'max', 'mean', 'smin', 'smax', 'smean']:
                raise ValueError("shape must be one of ('min', 'max', 'mean', 'smin', 'smax', 'smean')")

            _resolve_shape()
        elif any(type(s) is not int or s <= 0 for s in shape):
            raise ValueError('shape must have positive integer elements')

    err_arg = _handle_args()
    if err_arg is not None:
        return err_arg

    shape = list(shape) + [3]
    return [imresize(image, shape, interp, mode, anti_aliasing=False) for image in images]


def _show_image(image, **kwargs):
    title = kwargs.pop('title', None)
    cmap = kwargs.pop('cmap', None)
    ax = kwargs.pop('ax', None)
    retain = kwargs.pop('retain', False)

    def _handle_args():
        nonlocal image, ax
        if type(image) is not np.ndarray:
            raise TypeError('image needs to be a numpy array. Found {}'.format(type(image)))
        if len(image.shape) not in (2, 3):
            raise ValueError('invalid image dimensions. Needs to be 2 or 3-D. Found {}'.format(len(image.shape)))
        elif len(image.shape) == 3 and np.all(image[:, :, 0] == image[:, :, 1]) and\
                np.all(image[:, :, 1] == image[:, :, 2]):
            image = image[:, :, 0]

        if title is not None and type(title) is not str:
            raise TypeError('title needs to be None or a valid string. Found {}'.format(title))

        if ax is None:
            ax = plt.subplots()[1]

        if type(retain) is not bool:
            raise TypeError('retain must be either True or False')

    err_arg = _handle_args()
    if err_arg is not None:
        return err_arg

    ax.imshow(image, cmap, vmin=-1, vmax=1, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    if not retain:
        plt.show()


def _merge_images(images, shape='square'):
    def _handle_args():
        nonlocal shape

        if type(images) not in (list, tuple):
            if type(images) is np.ndarray:
                if len(images.shape) != 4:
                    raise ValueError('images needs to be a 4-D array. Got {} dimensions'.format(len(images.shape)))

                return _merge_images(list(images), shape)
            else:
                raise TypeError('images needs to be a list, tuple or numpy array. Got {}'.format(type(images)))

        shape = _resolve_merge_shape(len(images), shape)

        if len(images) != np.prod(shape):
            raise ValueError('Shape mismatch. Got shape {} but images are {} long'.format(shape, len(images)))

        if any(np.std([image.shape for image in images], 0) != 0):
            raise ValueError('All images need to be of the same shape')

    err_arg = _handle_args()
    if err_arg is not None:
        return err_arg

    img_shape = np.array(images[0].shape[:-1])
    merged_image = np.zeros(np.append(img_shape * np.array(shape), 3))

    # noinspection PyTypeChecker
    for idx, (row, column) in enumerate(list(itertools.product(range(shape[0]), range(shape[1])))):
        merged_image[row * img_shape[0]:(row + 1) * img_shape[0],
                     column * img_shape[1]:(column + 1) * img_shape[1], :] = images[idx]

    return merged_image


def _resolve_merge_shape(num_images, shape):
    if type(shape) not in [str, tuple, list]:
        raise TypeError('shape needs to be a string, tuple or list. Got {}'.format(type(shape)))
    elif type(shape) is str:
        if shape not in ['square', 'row', 'column']:
            raise ValueError("shape needs to be one of 'square', 'row' or 'column'. Got {}".format(shape))
        else:
            if shape is 'square':
                def _square_factors(x):
                    if x == 1:
                        return 1, 1
                    if x == 2:
                        return 1, 2
                    factors = [i for i in range(2, int(np.sqrt(x)) + 1) if x % i == 0]
                    if len(factors) == 0:
                        return 1, x

                    return factors[-1], x // factors[-1]

                return _square_factors(num_images)
            elif shape is 'row':
                return 1, num_images
            elif shape is 'column':
                return num_images, 1
    else:
        if any(type(s) is not int or s <= 0 for s in shape):
            raise ValueError('All shape elements need to be positive integers')

    return shape
