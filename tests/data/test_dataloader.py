import pytest

from magnet.data import Data

class TestDataLoader:
    def test_batch_size_cannnot_be_too_high(self):
        data = Data.get('mnist')
        with pytest.raises(RuntimeError):
            data(batch_size=10000000)
