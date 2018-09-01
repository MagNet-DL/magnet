import pytest
import matplotlib
matplotlib.use('agg')
import torch

from torch import nn

import magnet as mag
import magnet.nodes as mn
import magnet.debug as mdb

from magnet.data.core import MNIST
from magnet.training import SupervisedTrainer, callbacks

def test_overfit():
    data, model, trainer = get_obj()
    mdb.overfit(trainer, data, batch_size=64)

class TestFlow:
    def test_ok(self):
        data, model, trainer = get_obj()
        mdb.check_flow(trainer, data)

    def test_broken(self):
        data, model, trainer = get_obj(broken=True)
        with pytest.raises(RuntimeError):
            mdb.check_flow(trainer, data)

class TestBabysitter:
    def test_accumulating(self):
        data, model, trainer = get_obj()

        trainer.train(data(), callbacks=[mdb.Babysitter()])

        history = trainer.callbacks[0].history

        assert len(history['layer.weight']) == 10

def get_obj(broken=False):
    data = MNIST(val_split=0.99)
    if broken:
        model = BrokenModel()
    else:
        model = mn.Linear(10, act=None)
    with mag.eval(model): model(next(data())[0])
    trainer = SupervisedTrainer(model)

    return data, model, trainer

class BrokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = mn.Linear()
        self.fc2 = mn.Linear(10, act=None)

    def forward(self, x):
        x = self.fc1(x).detach()
        x = self.fc2(x)
        return x

    def sample(self, x):
        x = self.fc1(x)
        return x