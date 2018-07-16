import sys, inspect, torch
from contextlib import contextmanager

def overfit(trainer, data, batch_size, epochs=1, metric='loss', sample_space=None, ax=None):
    from matplotlib import pyplot as plt

    if sample_space is None:
        _, ax = plt.subplots()
        epochs *= 100

        for sample_space in (1, 2, 4, 8, 16):
            overfit(trainer, data, batch_size=1, epochs=epochs,
                    metric=metric, sample_space=sample_space, ax=ax)

        overfit(trainer, data, batch_size=16, epochs=epochs, metric=metric, sample_space=16, ax=ax)

        bs = min(batch_size, 16)
        if bs > 16:
            overfit(trainer, data, bs, epochs, metric, sample_space=16, ax=ax)

        sample_length = int(len(data) * 0.01)
        bs = min(batch_size, sample_length)
        if sample_length > 16:
            overfit(trainer, data, bs, epochs, metric, sample_length, ax)

        plt.show()
        return

    from magnet.training.callbacks import Monitor

    with trainer.mock():
        trainer.train(data(batch_size, sample_space=sample_space), epochs, callbacks=[Monitor(frequency=10 / epochs)])
        trainer.callbacks[0].history.show(metric, x_key='epochs',
                                validation=False, ax=ax, log=True,
                                legend=f'{batch_size}, {sample_space}')
def check_flow(trainer, data):
    broken_parameters = {}
    def callback(trainer, signal, **kwargs):
        if signal == 'gradient':
            broken_parameters.update(*[{name for name, p in model.named_parameters() if p.requires_grad and p.grad is None} for model in kwargs.pop('models')])
    callback.name = 'check_flow'

    with trainer.mock(): trainer.train(data(), callbacks=[callback], iterations=1)

    if len(broken_parameters) == 0:
        print('No breakage detected')
    else:
        raise RuntimeError('Breaks in the following parameters: ' + ', '.join(broken_weights))

class Babysitter:
    def __init__(self, frequency=10, **kwargs):
        from magnet.training.history import History

        self.name = kwargs.pop('name', 'babysitter')
        self.frequency = frequency

        self.history = History()

    def __call__(self, trainer, signal, **kwargs):
        if signal == 'gradient':
            batches_per_epoch = len(trainer.dataloader)
            if trainer.iterations % int(batches_per_epoch // self.frequency): return

            self.append(trainer, kwargs.pop('models'))

        elif signal == 'load':
            self.load(kwargs.pop('path'))

        elif signal == 'save':
            self.save(kwargs.pop('path'))

    def append(self, trainer, models):
        stamps = {'iterations': trainer.iterations, 'epochs': trainer.epochs()}

        for model in models:
            for name, p in model.named_parameters():
                v = torch.abs(p.grad / p)
                v[v != v] = 0
                self.history.append(name, v.mean().item(), **stamps)

    def load(self, path):
        from magnet.training.utils import load_object
        self.history = load_object(path / self.name / 'history.p', default=self.history)

    def save(self, path):
        from magnet.training.utils import save_object
        save_object(self.history, path / self.name / 'history.p')

@contextmanager
def shape(debug=True):
    with _SetTrace(_Monitor(debug)): yield

class _SetTrace(object):
    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sys.settrace(self.func)
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        sys.settrace(None)

class _Monitor:
    def __init__(self, names=True):
        self.names = names

    def init(self, frameinfo):
        self.filename = frameinfo.filename.strip()
        self.stackdepth = len(inspect.stack())
        self.lineno = frameinfo.lineno

    def is_same(self, frame):
        frameinfo = inspect.getframeinfo(frame)
        return frameinfo.function.strip() == 'forward' and len(inspect.stack()) == self.stackdepth

    def __call__(self, frame, event, arg):
        frameinfo = inspect.getframeinfo(frame)
        if not hasattr(self, 'filename') and frameinfo.function.strip() == 'forward':
            self.init(frameinfo)

        if not self.is_same(frame): return

        if event not in ("line", "return"): return self.__call__

        if self.names is True:
            shape_dict = {n: tuple(v.shape)
                   for n, v in frame.f_locals.items() if torch.is_tensor(v)}
        elif isinstance(self.names, (tuple, list)):
            shape_dict = {n: tuple(v.shape)
                   for n, v in frame.f_locals.items() if torch.is_tensor(v) and n in self.names}

        if len(shape_dict) != 0:
            print(shape_dict)
            lineno = frameinfo.lineno - self.lineno if event != "return" else 'return'
            print(f'{lineno}.', frameinfo.code_context[0].strip(), '\n')
        return self.__call__