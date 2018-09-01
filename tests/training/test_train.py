import torch
import pytest

from torch import optim
from pathlib import Path

import magnet as mag
import magnet.nodes as mn

from magnet.data import Data
from magnet.training import (Trainer, SupervisedTrainer,
                             callbacks, finish_training)

class TestTrainer:
    def test_cannot_call(self):
        data, model, _ = get_obj()
        trainer = Trainer([model], [optim.Adam(model.parameters())])

        with pytest.raises(NotImplementedError):
            trainer.train(data(), iterations=1)

class TestSupervisedTrainer:
    def test_iterations(self):
        data, model, trainer = get_obj()
        trainer.train(data(), iterations=100)

        assert trainer.iterations == 99

    def test_epochs(self):
        data, model, trainer = get_obj()
        trainer.train(data(), epochs=0.01)

        assert trainer.iterations == int(len(data) * 0.01) - 1

    def test_epoch_start(self):
        data, model, trainer = get_obj()
        trainer.train(data(), iterations=0)

        assert trainer.epochs('start')

    def test_epoch_end(self):
        data, model, trainer = get_obj()
        trainer.train(data(sample_space=0.01), iterations=int(len(data) * 0.01))

        assert trainer.epochs('end')

    def test_less_loss(self):
        data, model, trainer = get_obj()
        cbacks = [callbacks.Validate(data(mode='val', sample_space=0.01),
                                     SupervisedTrainer.validate),
                  callbacks.Monitor()]
        trainer.train(data(sample_space=0.01), epochs=0.3, callbacks=cbacks)

        losses = trainer.callbacks[1].history['loss']
        assert losses[0] > losses[1]

        val_losses = trainer.callbacks[1].history['val_loss']
        assert losses[0] > losses[1]

    def test_not_training_when_eval(self):
        data, model, trainer = get_obj()
        model.eval()

        weight_before = copy_tensor(model.layer.weight.data.detach())

        trainer.train(data(), iterations=10)

        assert torch.all(model.layer.weight.data.detach() == weight_before)

    def test_mocking(self):
        data, model, trainer = get_obj()
        weight_before = copy_tensor(model.layer.weight.data.detach())

        with trainer.mock(): trainer.train(data(), iterations=10)

        assert torch.all(model.layer.weight.data.detach() == weight_before)

def test_finish_training():
    from shutil import rmtree
    data, model, trainer = get_obj()

    save_path = Path.cwd() / '.mock_trainer' / 'trainer'

    trainer.train(data(), iterations=10,
                  callbacks=[callbacks.Checkpoint(save_path)])

    finish_training(save_path, names=['my_model'])

    files = list(save_path.parent.glob('*'))
    assert len(files) == 1
    assert files[0].name == 'my_model.pt'

    rmtree(save_path.parent)

def get_obj():
    data = Data.get('mnist')
    model = mn.Linear(10, act=None)
    with mag.eval(model): model(next(data())[0])
    trainer = SupervisedTrainer(model)

    return data, model, trainer

def copy_tensor(x):
    return torch.zeros_like(x).copy_(x)