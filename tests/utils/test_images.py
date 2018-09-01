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

    def test_glob_no_images(self):
        with pytest.raises(RuntimeError):
            show_images('./images/')

    def test_cannot_mix_inputs(self):
        with pytest.raises(TypeError):
            show_images([np.random.randn(28, 28), './images/'])

    def test_pixel_range_string(self):
        with pytest.raises(ValueError):
            show_images([np.random.randn(28, 28)], pixel_range='min')

    def test_pixel_range_integer(self):
        with pytest.raises(TypeError):
            show_images([np.random.randn(28, 28)], pixel_range=2)

    def test_pixel_range_size(self):
        with pytest.raises(ValueError):
            show_images([np.random.randn(28, 28)], pixel_range=(2, 3, 4))

    def test_merge_string(self):
        with pytest.raises(TypeError):
            show_images([np.random.randn(28, 28)], merge='yes')

    def test_bad_merge_shape(self):
        with pytest.raises(ValueError):
            show_images([np.random.randn(28, 28)], shape=(2, 1))

    def test_shape_negative(self):
        with pytest.raises(ValueError):
            show_images(np.random.randn(3, 28, 28), shape=(-4, 2))

    def test_shape_row_column(self):
        for shape in ('row', 'column'):
            show_images([np.random.randn(28, 28)], shape=shape)

    def test_titles_dict(self):
        with pytest.raises(TypeError):
            show_images([np.random.randn(28, 28)], titles={})

    def test_title_given_merge(self):
        with pytest.raises(TypeError):
            show_images([np.random.randn(28, 28)], titles=['hey'])

    def test_no_merge_titles_string(self):
        with pytest.raises(TypeError):
            show_images([np.random.randn(28, 28)], titles='hey', merge=False)

    def test_shape_dict(self):
        with pytest.raises(TypeError):
            show_images([np.random.randn(28, 28)], shape={})

    def test_shape_evil(self):
        with pytest.raises(ValueError):
            show_images([np.random.randn(28, 28)], shape='churchill')

    def test_resize_dict(self):
        with pytest.raises(TypeError):
            show_images(np.random.randn(3, 28, 28), resize={})

    def test_resize_evil(self):
        with pytest.raises(ValueError):
            show_images(np.random.randn(3, 28, 28), resize='hitler')

    def test_resize_negative(self):
        with pytest.raises(ValueError):
            show_images(np.random.randn(3, 28, 28), resize=(-4, 2))

    def test_savepath_not_string(self):
        with pytest.raises(TypeError):
            show_images([np.random.randn(28, 28)], savepath=True)

    def test_retain_string(self):
        with pytest.raises(TypeError):
            show_images([np.random.randn(28, 28)], retain='please')

    def test_can_plot_seperately(self):
        show_images([np.random.randn(28, 28, 3) for _ in range(4)], merge=False)

    def test_cannot_pass_5d_arrays(self):
        with pytest.raises(ValueError):
            show_images(np.random.randn(4, 3, 2, 4, 5))

    def test_cannot_pass_incorrect_channels(self):
        with pytest.raises(ValueError):
            show_images(np.random.randn(4, 5, 28, 28))

    def test_no_dict_input(self):
        with pytest.raises(TypeError):
            show_images({'a': 1})