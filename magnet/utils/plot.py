import numpy as np
import matplotlib.pyplot as plt

from .statistical import smoothen, _spline_interpolate, find_outliers


def smooth_plot(*args, **kwargs):
    ax = kwargs.pop('ax', None)
    window_fraction = kwargs.pop('window_fraction', 0.3)
    gain = kwargs.pop('gain', 10)
    replace_outliers = kwargs.pop('replace_outliers', True)

    lines = plt.plot(*args, **kwargs) if ax is None else ax.plot(*args, **kwargs)

    def _smoothen_line(line):
        x, y = line.get_data()
        x_new = np.linspace(x.min(), x.max(), int(gain * len(x)))

        if replace_outliers:
            outlier_mask = find_outliers(y)
            y = y[~outlier_mask]
            x = x[~outlier_mask]

        y = smoothen(y, window_fraction, outlier_mask=None)

        if len(x) > 1:
            y_new = _spline_interpolate(x, y, x_new)
            line.set_data(x_new, y_new)
        else:
            line.set_data(x, y)

    for l in lines:
        _smoothen_line(l)

    return lines