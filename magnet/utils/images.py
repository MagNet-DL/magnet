import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize as imresize
from arghandle import arghandle

@arghandle
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
        cmap (str or None): The color map for the plots. Default: ``'gray'``
        merge (bool): If ``True``, all images are merged into one giant image.
            Default: ``True``
        titles (list or None): The titles for each image. Default: ``None``
        shape (str): The shape of the merge tile.
            Default: ``'square'``
        resize (str): The common shape to which images are resized.
            Default: ``'smean'``
        retain (bool): If ``True``, the plot is retained. Default: ``False``
        savepath (str or None): If given, the image is saved to this path.
            Default: ``None``
    * :attr:`pixel_range` default to the range in the image.
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
    cmap = kwargs.pop('cmap', 'gray')
    shape = kwargs.pop('shape', 'square')
    resize = kwargs.pop('resize', 'smean')
    merge = kwargs.pop('merge', True)
    retain = kwargs.pop('retain', False)
    savepath = kwargs.pop('savepath', None)

    images = _resize(images, resize)

    if merge:
        _show_image(_merge(images, shape), titles, cmap, None, pixel_range, retain)
        return

    fig, axes = plt.subplots(shape[0], shape[1])
    for ax, title, image in zip(axes.flat, titles, images):
        _show_image(image, title, cmap, ax, pixel_range, retain=True)

    fig.tight_layout()

    if not retain: plt.show()

    if savepath is not None: plt.savefig(Path(savepath), dpi=400, bbox_inches='tight')

def _resize(images, size='smean'):
    return np.stack([imresize(image, size, order=1, mode='constant',
                     anti_aliasing=False, preserve_range=True)
                     for image in images])

def _merge(images, shape):
    images = images.reshape((*shape, *images.shape[1:]))
    for _ in range(2): images = np.concatenate([img for img in images], axis=1)
    return images

@arghandle
def _show_image(image, title=None, cmap='gray', ax=None, pixel_range='auto', retain=False):
    image = (image - pixel_range[0]) * 255 / (pixel_range[1] - pixel_range[0])
    ax.imshow(image.astype(np.uint8), cmap)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    if title is not None: ax.set_title(title)
    if not retain: plt.show()