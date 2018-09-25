import pytest
import numpy as np
import matplotlib
matplotlib.use('agg')
import torch

from magnet.utils.images import show_images

class TestShowImages:
    def test_pass_numpy_array(self):
        show_images(np.random.randn(64, 28, 28))
        show_images(np.random.randn(64, 28, 28, 1))
        show_images(np.random.randn(16, 28, 28, 3))

    def test_torch_tensor(self):
        show_images(torch.randn(4, 1, 28, 28))

    def test_cannot_mix_inputs(self):
        with pytest.raises(TypeError):
            show_images([np.random.randn(28, 28), './images/'])

    def test_pixel_range_string(self):
        with pytest.raises(ValueError):
            show_images([np.random.randn(28, 28)], pixel_range='min')

    def test_pixel_range_integer(self):
        with pytest.raises(TypeError):
            show_images([np.random.randn(28, 28)], pixel_range=2)

    def test_bad_merge_shape(self):
        with pytest.raises(ValueError):
            show_images([np.random.randn(28, 28)], shape=(2, 1))

    def test_shape_negative(self):
        with pytest.raises(ValueError):
            show_images(np.random.randn(3, 28, 28), shape=(-4, 2))

    def test_shape_row_column(self):
        for shape in ('row', 'column'):
            show_images([np.random.randn(28, 28)], shape=shape)

    def test_shape_dict(self):
        with pytest.raises(TypeError):
            show_images([np.random.randn(28, 28)], shape={})

    def test_shape_evil(self):
        with pytest.raises(ValueError):
            show_images([np.random.randn(28, 28)], shape='churchill')

    def test_resize_negative(self):
        with pytest.raises(ValueError):
            show_images(np.random.randn(3, 28, 28), resize=(-4, 2))

    def test_savepath_not_string(self):
        with pytest.raises(TypeError):
            show_images([np.random.randn(28, 28)], savepath=True)

    def test_can_plot_seperately(self):
        show_images([np.random.randn(28, 28, 3) for _ in range(4)], merge=False)