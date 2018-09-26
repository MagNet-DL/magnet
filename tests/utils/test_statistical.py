import numpy as np
import pytest
import matplotlib
matplotlib.use('agg')

from magnet.utils.statistical import find_outliers, smoothen


class TestRemoveOutlier:
    def test_data_can_be_1d(self):
        find_outliers(np.zeros(5))

    def test_data_can_be_linear(self):
        find_outliers(np.linspace(0, 5, 100))

    def test_cannot_send_none(self):
        with pytest.raises(Exception):
            find_outliers(None)

    def test_cannot_send_empty(self):
        with pytest.raises(ValueError):
            find_outliers([])

        with pytest.raises(ValueError):
            find_outliers(np.array([]))

    def test_threshold_not_negative(self):
        with pytest.raises(ValueError):
            find_outliers(np.zeros(5), -1)

    def test_threshold_not_none(self):
        with pytest.raises(TypeError):
            find_outliers(np.zeros(5), None)

    def test_threshold_not_inf(self):
        with pytest.raises(ValueError):
            find_outliers(np.zeros(5), np.inf)

    def test_window_fraction_is_fraction(self):
        window_fraction = 2

        with pytest.raises(ValueError):
            find_outliers(np.zeros(5), window_fraction=window_fraction)


class TestSmoothen:
    def test_cannot_send_none(self):
        with pytest.raises(TypeError):
            smoothen(None)

        with pytest.raises(TypeError):
            smoothen(np.zeros(5), None)

        with pytest.raises(TypeError):
            smoothen(np.zeros(5), order=None)

    def test_cannot_send_empty(self):
        with pytest.raises(ValueError):
            smoothen([])

        with pytest.raises(ValueError):
            smoothen(np.array([]))

    def test_cannot_send_2d(self):
        with pytest.raises(ValueError):
            smoothen(np.ones((4, 3)))

    def test_cannot_send_illegal(self):
        data = np.ones(100)
        with pytest.raises(ValueError):
            data[np.random.randint(0, len(data))] = np.nan
            smoothen(data)

        data = np.ones(10)
        with pytest.raises(ValueError):
            data[np.random.randint(0, len(data))] = np.inf
            smoothen(data)

    def test_window_fraction_is_fraction(self):
        window_fraction = 2.0

        with pytest.raises(ValueError):
            smoothen(np.zeros(5), window_fraction=window_fraction)

    def test_window_fraction_too_low_warning(self):
        with pytest.warns(RuntimeWarning):
            smoothen(np.zeros(5), window_fraction=0.01)

    def test_order_cannot_be_negative(self):
        with pytest.raises(ValueError):
            smoothen(np.zeros(5), order=-1)

    def test_interpolate_fn_cannot_be_none(self):
        with pytest.raises(ValueError):
            smoothen(np.zeros(5), interpolate_fn=None)

    def test_returns_same_shape(self):
        data = np.ones(3)
        window_fraction = 0.3

        window_length = int(len(data) * window_fraction)
        if window_length % 2 == 0:
            window_length = max(1, window_length - 1)

        order = 1

        assert len(smoothen(data, window_fraction, order=order)) == len(data)