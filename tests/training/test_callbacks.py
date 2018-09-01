import torch
import pytest
import matplotlib
matplotlib.use('agg')

from torch.optim.lr_scheduler import ExponentialLR
from pathlib import Path
from shutil import rmtree

import magnet as mag
import magnet.nodes as mn
import magnet.debug as mdb

from magnet.data import Data
from magnet.training import SupervisedTrainer, callbacks

class TestCheckpoint:
    @staticmethod
    def get_callbacks(data, trainer, path):
        return [callbacks.Validate(data(mode='val'), trainer.validate),
                callbacks.Monitor(),
                callbacks.LRScheduler(ExponentialLR(trainer.optimizers[0],
                                                    gamma=0.1)),
                mdb.Babysitter(),
                callbacks.Checkpoint(path)]

    def test_start_from_checkpoint(self):
        data, model, trainer = get_obj()
        save_path = Path.cwd() / '.mock_trainer'

        trainer.train(data(sample_space=0.01),
                      callbacks=self.get_callbacks(data, trainer, save_path))

        weight_before = copy_tensor(model.layer.weight.data.detach())

        data, model, trainer = get_obj()

        trainer.train(data(sample_space=0.01), iterations=0,
                      callbacks=self.get_callbacks(data, trainer, save_path))

        assert torch.all(model.layer.weight.data.detach() == weight_before)

        rmtree(save_path)

class TestColdStart:
    def test_not_trained(self):
        data, model, trainer = get_obj()
        model.eval()

        weight_before = copy_tensor(model.layer.weight.data.detach())

        trainer.train(data(), epochs=0.1, callbacks=[callbacks.ColdStart()])

        assert torch.all(model.layer.weight.data.detach() == weight_before)

class TestLRScheduler:
    def test_lr_decay(self):
        data, _, trainer = get_obj()

        scheduler = ExponentialLR(trainer.optimizers[0], gamma=0.1)

        trainer.train(data(sample_space=0.01),
                      callbacks=[callbacks.LRScheduler(scheduler)])

        assert trainer.optimizers[0].param_groups[0]['lr'] == 1e-3

class TestCallbackQueue:
    def test_exists(self):
        queue = callbacks.CallbackQueue([callbacks.Monitor()])
        assert queue.exists('monitor')
        assert not queue.exists('einstein')

    def test_multiple_callbacks_error(self):
        queue = callbacks.CallbackQueue([callbacks.Monitor() for _ in range(2)])
        with pytest.raises(RuntimeError):
            queue.exists('monitor')

    def test_cannot_add_same_name(self):
        queue = callbacks.CallbackQueue([callbacks.Monitor()])
        queue.append(callbacks.Monitor())
        assert len(queue) == 1

        queue.extend([callbacks.Monitor()])
        assert len(queue) == 1

def get_obj():
    data = Data.get('mnist')
    model = mn.Linear(10, act=None)
    with mag.eval(model): model(next(data())[0])
    trainer = SupervisedTrainer(model)

    return data, model, trainer

def copy_tensor(x):
    return torch.zeros_like(x).copy_(x)