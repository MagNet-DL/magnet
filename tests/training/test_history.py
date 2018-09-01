from time import time

from magnet.training.history import History

class TestHistory:
    def test_can_show(self):
        history = History()
        for _ in range(5):
            for i in range(100): history.append('loss', i, buffer_size=10)
            history.flush(time=time())

        history.show()