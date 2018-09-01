import warnings

import numpy as np
from scipy.signal import savgol_filter
from scipy import interpolate

def find_outliers(data, threshold=3.5, window_fraction=0.15):
    """Based on http://www.itl.nist.gov/div898/handbook/eda/section3
    /eda35h.htm """

    def _handle_args():
        if type(data) is not np.ndarray and type(data) is not list:
            raise TypeError('data needs to be a list or numpy array. Got {}'.format(type(data)))
        if len(data) == 0:
            raise ValueError('data is empty!')
        if len(data.shape) == 1:
            return find_outliers(np.expand_dims(data, -1), threshold, window_fraction)

        if window_fraction < 0 or window_fraction > 1:
            raise ValueError('window_fraction should be a fraction (duh!). But got {}'.format(window_fraction))
        if np.isinf(window_fraction) or np.isnan(window_fraction):
            raise ValueError('window_fraction should be a finite number but got {}'.format(window_fraction))

        if threshold < 0:
            raise ValueError(
                'threshold should be non negative but got {}'.format(
                    threshold))
        elif np.isinf(threshold) or np.isnan(threshold):
            raise ValueError(
                'threshold should be a finite number but got {}'.format(
                    threshold))

    arg_err = _handle_args()
    if arg_err is not None:
        return arg_err

    # Subdivide data into small windows
    window_length = max(int(len(data) * window_fraction), 1)

    if len(data) - window_length >= 1:
        split_data = np.stack([data[i:i + window_length] for i in range(len(data) - window_length + 1)])
    else:
        split_data = np.expand_dims(data, 0)

    def _find_outliers(x):
        outlier_factor = 0.6745

        median = np.median(x, axis=0)
        distances = np.linalg.norm(x - median, axis=-1)
        median_deviation = np.median(distances)

        # No deviation. All values are same. No outlier
        if median_deviation == 0:
            return np.array([False] * len(x))
        modified_z_scores = outlier_factor * distances / median_deviation

        outlier_mask = modified_z_scores > threshold

        return outlier_mask

    outlier_idx = np.concatenate([np.arange(i, i + window_length)[_find_outliers(d)] for i, d in enumerate(split_data)])
    return np.array([i in np.unique(outlier_idx) for i in range(len(data))])


def smoothen(data, window_fraction=0.3, **kwargs):
    order = kwargs.pop('order', 3)
    outlier_mask = kwargs.pop('outlier_mask', find_outliers)
    interpolate_fn = kwargs.pop('interpolate_fn', _spline_interpolate)

    def _handle_args():
        nonlocal data
        if type(data) is not np.ndarray and type(data) is not list:
            raise TypeError('data needs to be a list or numpy array. Got {}'.format(type(data)))
        if len(data) == 0:
            raise ValueError('data is empty!')
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError('some of the data is either nan or inf')
        if len(data.shape) > 1:
            raise ValueError('data needs to be 1-dimensional for now')

        if type(window_fraction) is not float:
            raise TypeError('window_fraction should be a fraction (duh!). But got {}'.format(type(window_fraction)))
        if window_fraction < 0 or window_fraction > 1:
            raise ValueError('window_fraction should be a fraction (duh!). But got {}'.format(window_fraction))
        if np.isinf(window_fraction) or np.isnan(window_fraction):
            raise ValueError('window_fraction should be a finite number but got {}'.format(window_fraction))

        if type(order) is not int:
            raise TypeError('order needs to be a non-negative integer but got {}'.format(type(order)))
        if order < 0:
            raise ValueError('order needs to be a non-negative integer but got {}'.format(order))

        # Replace Outliers
        if outlier_mask is not None:
            if interpolate_fn is None:
                raise ValueError('if outlier_mask is not None, need interpolate_fn')

            outliers = outlier_mask(data)
            new_data = data.copy()
            if len(np.where(outliers)[0]) != 0 and len(np.where(~outliers)[0]) > 1:
                new_data[outliers] = interpolate_fn(np.where(~outliers)[0], data[~outliers], np.where(outliers)[0])
                data = new_data

    arg_err = _handle_args()
    if arg_err is not None:
        return arg_err

    window_length = int(len(data) * window_fraction)
    # savgol_filter needs an odd window_length
    if window_length % 2 == 0:
        window_length = max(window_length - 1, 1)

    if window_length <= order:
        warnings.warn('window_fraction ({}) too low for order ({}) and length ({}) of data'
                      '\nReturning raw data'.format(window_fraction, order, len(data)),
                      RuntimeWarning)
        return data

    return savgol_filter(data, window_length, order)


def _spline_interpolate(x, y, x_new, **kwargs):
    s = kwargs.pop('s', 0)
    k = kwargs.pop('k', 3)
    extrapolate = kwargs.pop('extrapolate', True)

    def _handle_args():
        nonlocal x, y

        # Sort the data in ascending order
        order_idx = np.argsort(x)
        x = x[order_idx]
        y = y[order_idx]

    err_arg = _handle_args()
    if err_arg is not None:
        return err_arg

    t, c, k = interpolate.splrep(x, y, s=s, k=k)
    spline = interpolate.BSpline(t, c, k, extrapolate=extrapolate)

    return spline(x_new)