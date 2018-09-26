import numpy as np

from magnet.utils.plot import smooth_plot


class TestSmoothenPlot:
    def test_get_gained_points_back(self):
        x = np.linspace(0, 1, 100)
        y = np.linspace(1, 2, 100)
        gain = 10

        smooth_lines = smooth_plot(x, y, gain=gain)

        assert int(len(x) * gain) >= len(smooth_lines[0].get_xdata())