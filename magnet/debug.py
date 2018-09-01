import sys, inspect, torch
from contextlib import contextmanager

def overfit(trainer, data, batch_size, epochs=1, metric='loss', **kwargs):
    r"""Runs training on small samples of the dataset in order to overfit.

    If you can't overfit a small sample, you can't model the data well.

    This debugger tries to overfit on multple small samples of the data.
    The sample size and batch sizes are varied and the training is done for
    a fixed number of epochs.

    This usually gives an insight on what to expect from the actual training.

    Args:
        trainer (magnet.training.Trainer): The Trainer object
        data (magnet.data.Data): The data object used for training
        batch_size (int): The intended batch size
        epochs (float): The expected epochs for convergence for 1% of the data.
            Default: ``1``
        metric (str): The metric to plot.
            Default: ``'loss'``

    .. note::
        The maximum sample size is 1% of the size of the dataset.

    Examples::

        >>> import magnet as mag
        >>> import magnet.nodes as mn
        >>> import magnet.debug as mdb

        >>> from magnet.data import Data
        >>> from magnet.training import SupervisedTrainer

        >>> data = Data.get('mnist')

        >>> model = mn.Linear(10)
        >>> with mag.eval(model): model(next(data())[0])

        >>> trainer = SupervisedTrainer(model)

        >>> mdb.overfit(trainer, data, batch_size=64)

    .. image:: _static/img/overfit-fail.png

    ::

        >>> # Oops! Looks like there was something wrong.
        >>> # Loss does not considerable decrease for samples sizes >= 4.
        >>> # Of course, the activation was 'relu'.
        >>> model = mn.Linear(10, act=None)
        >>> with mag.eval(model): model(next(data())[0])

        >>> trainer = SupervisedTrainer(model)

        >>> mdb.overfit(trainer, data, batch_size=64)
        >>> # Should be much better now.

    .. image:: _static/img/overfit-pass.png
    """
    from matplotlib import pyplot as plt
    from magnet.training.callbacks import Monitor

    sample_space = kwargs.pop('sample_space', None)
    ax = kwargs.pop('ax', None)

    if sample_space is None:
        _, ax = plt.subplots()
        epochs *= 100

        # First fit for small samples with batch size 1
        for sample_space in (1, 2, 4, 8, 16):
            overfit(trainer, data, batch_size=1, epochs=epochs,
                    metric=metric, sample_space=sample_space, ax=ax)

        # Fit with sample size 16 and batch size 16
        if batch_size >= 16:
            overfit(trainer, data, batch_size=16, epochs=epochs, metric=metric, sample_space=16, ax=ax)

        # Fit with 1% of the data at given batch size
        sample_length = int(len(data) * 0.01)
        bs = min(batch_size, sample_length)
        if sample_length > 16:
            overfit(trainer, data, bs, epochs, metric, sample_space=sample_length, ax=ax)

        plt.show()

        return

    with trainer.mock():
        trainer.train(data(batch_size, sample_space=sample_space, shuffle=True), epochs, callbacks=[Monitor(frequency=10 / epochs)])
        trainer.callbacks[0].history.show(metric, x_key='epochs', validation=False, ax=ax, log=True,
                                          legend=f'{batch_size}, {sample_space}', smoothen=False)

def check_flow(trainer, data):
    r"""Checks if any trainable parameter is not receiving gradients.

    Super useful for large architectures that use the :py:meth:`detach` function

    Args:
        trainer (magnet.trainer.Trainer): The Trainer object
        data (magnet.data.Data): The data object used for training
    """
    broken_parameters = set()
    def callback(trainer, signal, **kwargs):
        # This callback reacts to the 'gradient' signal and logs any parameters
        # that require gradient but have not accumulated any.
        if signal == 'gradient':
            for model in kwargs.pop('models'):
                broken_parameters.update(set(name for name, p in model.named_parameters(prefix=model.__class__.__name__) if p.requires_grad and p.grad is None))

    callback.name = 'check_flow'

    # Run the trainer for one iteration with the callback.
    with trainer.mock(): trainer.train(data(), callbacks=[callback], iterations=1)

    if len(broken_parameters) == 0:
        print('No breakage detected')
    else:
        raise RuntimeError('Breaks in the following parameters: ' + ', '.join(broken_parameters))

class Babysitter:
    r"""A callback which monitors the mean relative gradients
    for all parameters.

    Args:
        frequency (int): Then number of times per epoch to monitor.
            Default: :math:`10`

    Keyword Args:
        name (str): Name of this callback. Default: ``'babysitter'``
    """
    def __init__(self, frequency=10, **kwargs):
        from magnet.training.history import History

        self.name = kwargs.pop('name', 'babysitter')
        self.frequency = frequency

        self.history = History()

    def __call__(self, trainer, signal, **kwargs):
        r"""
        Responds to the following signals:

        * ``'gradient'``: Called after gradients have accumulated.
          Logs the mean relative gradients for each parameter.

        * ``'load'``: Loads the state of this callback from :attr:`path`.

        * ``'save'``: Saves the state of this callback to :attr:`path`.
        """
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
                v[v != v] = 0 # Ignore NaN
                self.history.append(name, v.mean().item(), **stamps)

    def load(self, path):
        from magnet.training.utils import load_object
        self.history = load_object(path / self.name / 'history.p', default=self.history)

    def save(self, path):
        from magnet.training.utils import save_object
        save_object(self.history, path / self.name / 'history.p')

@contextmanager
def shape(debug=True):
    r"""The shapes of every tensor is printed out if a module is called
    within this context manager.

    Useful for debugging the flow of tensors through layers and finding
    the values of various hyperparameters.

    Args:
        debug (bool or str): If ``str``, only the tensor with this name
            is tracked. If ``True``, all tensors are tracked.
            Else, nothing is tracked.
    """
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
        if isinstance(names, str): names = (names, )

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