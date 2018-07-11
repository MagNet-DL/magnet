import sys, inspect, torch
from contextlib import contextmanager

def overfit(trainer, batch_size, epochs=1, metric='loss', sample_space=None, ax=None):
    from matplotlib import pyplot as plt
    if sample_space is None:
        _, ax = plt.subplots()
        epochs *= 100
        for sample_space in (1, 2, 4, 8, 16):
            overfit(trainer, batch_size=1, epochs=epochs,
                    metric=metric, sample_space=sample_space, ax=ax)
        overfit(trainer, batch_size=16, epochs=epochs, metric=metric, sample_space=16, ax=ax)
        bs = min(batch_size, 16)
        if bs > 16:
            overfit(trainer, bs, epochs, metric, sample_space=16, ax=ax)
        sample_length = int(len(trainer.data['train']) * 0.01)
        bs = min(batch_size, sample_length)
        if sample_length > 16:
            overfit(trainer, bs, epochs, metric, sample_length, ax)

        plt.show()
        return

    data = trainer.data
    with trainer.mock():
        trainer.data = (data(batch_size, sample_space=sample_space),
                        data(mode='val', sample_space=1))
        trainer.train(epochs, monitor_freq=10 / epochs, save_interval=None)
        trainer.history.show(metric, x_key='epochs', xlabel='epochs',
                             validation=False, ax=ax, log=True,
                             label=f'{batch_size}, {sample_space}')

    trainer.data = data

def breakage(trainer, iterations=100, frac_sample=0.01):
    from types import MethodType

    broken_weights = []
    _prev_grad_callback = trainer._gradient_callback
    def gradient_callback(self, batch):
        _prev_grad_callback(batch)
        for model in self.models:
            for name, p in model.named_parameters():
                if p.grad is None and name not in broken_weights:
                    broken_weights.append(name)

    trainer._gradient_callback = MethodType(gradient_callback, trainer)

    data = trainer.data
    trainer.data = (data(sample_space=frac_sample), data(mode='val'))

    with trainer.mock(): trainer.train(iterations=iterations, save_interval=None)

    trainer.data = data
    trainer._gradient_callback = _prev_grad_callback

    if len(broken_weights) == 0:
        print('No breakage detected')
    else:
        raise RuntimeError('Breaks in the following parameters: ' + ', '.join(broken_weights))

class SetTrace(object):
    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sys.settrace(self.func)
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        sys.settrace(None)

class Monitor:
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

@contextmanager
def shape(debug=True):
    with SetTrace(Monitor(debug)): yield

class Babysitter:
    def __init__(self, monitor_freq=10):
        from magnet.training.history import History

        self.history = History()

        self.monitor_freq = monitor_freq

    def append(self, models, **stamps):
        if stamps['batches'] == 0: return
        if stamps['batches'] % int((stamps['batches'] / stamps['epochs']) // self.monitor_freq): return

        for model in models:
            for name, p in model.named_parameters():
                v = torch.abs(p.grad / p)
                v[v != v] = 0
                self.history.append(name, v.mean().item(), **stamps)

    def flush(self, **stamps):
        self.history.flush(**stamps)

    def save(self, save_path):
        with open(save_path / ('babysitter.p'), 'wb') as f: pickle.dump(self.history, f)

    def load(self, save_path):
        path = save_path / ('babysitter.p')
        if path.exists():
            with open(path, 'rb') as f: self.history = pickle.load(f)
