import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from hypothesis.extra import numpy as nph

from magnet.utils.plot import smooth_plot


class TestSmoothenPlot:
    @given(nph.arrays(nph.floating_dtypes(), nph.array_shapes(max_dims=1, min_side=4), unique=True),
           st.data(),
           st.floats(1, 10))
    def test_get_gained_points_back(self, x, y, gain):
        y = y.draw(nph.arrays(nph.floating_dtypes(), x.shape))
        if np.any(np.isinf(x)) or np.any(np.isnan(x)) or np.any(np.isinf(y)) or np.any(np.isnan(y)):
            return

        smooth_lines = smooth_plot(x, y, gain=gain)

        assert int(len(x) * gain) >= len(smooth_lines[0].get_xdata())