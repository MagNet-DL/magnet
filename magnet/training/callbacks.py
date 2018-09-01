import magnet as mag
import torch

from time import time

class Monitor:
    r"""Allows easy monitoring of the training process.

    Stores any metric / quantity broadcast using the ``'write_stats'`` signal.

    Also adds a nice progress bar!

    Args:
        frequency (int): Then number of times per epoch to flush the buffer.
            Default: 10
        show_progress (bool): If ``True``, adds a progress bar.
            Default: ``True``

    Keyword Args:
        name (str): Name of this callback. Default: ``'monitor'``

    * :attr:`frequency` is useful only if there are buffered metrics.

    Examples::

        >>> import torch

        >>> import magnet as mag
        >>> import magnet.nodes as mn

        >>> from magnet.training import callbacks, SupervisedTrainer

        >>> model = mn.Linear(10, act=None)
        >>> with mag.eval(model): model(torch.randn(4, 1, 28, 28))

        >>> trainer = SupervisedTrainer(model)

        >>> callbacks = callbacks.CallbackQueue([callbacks.Monitor()])
        >>> callbacks(signal='write_stats', trainer=trainer, key='loss', value=0.1)

        >>> callbacks[0].history
        {'loss': [{'val': 0.1}]}
    """
    def __init__(self, frequency=10, show_progress=True, **kwargs):
        from magnet.training.history import History

        self.name = kwargs.pop('name', 'monitor')
        self.frequency = frequency
        self.show_progress = show_progress

        self.history = History()

    def __call__(self, trainer, signal, **kwargs):
        r"""
        Responds to the following signals:

        * ``'write_stats'``: Any keyword arguments will be passed to the
          :py:meth:`History.append` method.

        * ``'on_training_start'``: To be called before start of training.
          Initializes the progress bar.

        * ``'on_batch_start'``: Called before the training loop.
          Updates the progress bar.

        * ``'on_batch_end'``: Called after the training loop.
          Flushes the history buffer if needed and
          sets the progress bar description.

        * ``'on_training_end'``: To be called after training.
          Closes the progress bar.

        * ``'load_state'``: Loads the state of this callback from :attr:`path`.

        * ``'save_state'``: Saves the state of this callback to :attr:`path`.
        """
        if signal == 'on_training_start':
            from magnet.utils.misc import get_tqdm; tqdm = get_tqdm()

            if self.show_progress:
                self.progress_bar = tqdm(total=kwargs.pop('total_iterations'), unit_scale=True,
                                        unit_divisor=len(trainer.dataloader), leave=False)

        elif signal == 'on_batch_start':
            if self.show_progress:
                self.progress_bar.update()
                self.progress_bar.refresh()

        elif signal == 'write_stats':
            self.history.append(**kwargs)

        elif signal == 'on_batch_end' and trainer.iterations != 0:
            batches_per_epoch = len(trainer.dataloader)
            if trainer.iterations % int(batches_per_epoch // self.frequency): return

            self.history.flush(iterations=trainer.iterations, epochs=trainer.epochs())

            if not self.show_progress or 'loss' not in self.history.keys(): return

            if 'val_loss' in self.history.keys():
                description = f"{self.history['loss'][-1]:.2f}, {self.history['val_loss'][-1]:.2f}"
            else:
                description = f"{self.history['loss'][-1]:.2f}"
            self.progress_bar.set_description(description, refresh=False)

        elif signal == 'on_training_end' and self.show_progress:
            self.progress_bar.close()
            self.progress_bar = None

        elif signal == 'load_state':
            self.load_state(kwargs.pop('path'))

        elif signal == 'save_state':
            self.save_state(kwargs.pop('path'))

    def show(self, metric=None, log=False, x_key='epochs', **kwargs):
        r"""Calls the corresponding :py:meth:`History.show` method.
        """
        self.history.show(metric, log, x_key, **kwargs)

    def __repr__(self):
        self.show()
        return ''

    def load_state(self, path):
        from magnet.training.utils import load_object
        self.history = load_object(path / self.name / 'history.p', default=self.history)

    def save_state(self, path):
        from magnet.training.utils import save_object
        save_object(self.history, path / self.name / 'history.p')

class Validate:
    r"""Runs a validation function over a dataset during the course of training.

    Most Machine Learning research uses a held out validation set as a proxy
    for the test set / real-life data. Hyperparameters are usually tuned
    on the validation set.

    Often, this is done during training in order to view the simultaneous
    learning on the validation set and catch any overfitting / underfitting.

    This callback enables you to run a custom ``validate`` function
    over a :attr:`dataloader`.

    Args:
        dataloader (``DataLoader``): DataLoader containing the validation set
        validate (bool): A callable that does the validation
        frequency (int): Then number of times per epoch to run the function.
            Default: :math:`10`
        batches (int or None): The number of times / batches to call the validate
            function in each run. Default: ``None``
        drop_last (bool): If ``True``, the last batch is not run.
            Default: ``False``

    Keyword Args:
        name (str): Name of this callback. Default: ``'validate'``

    * :attr:`validate` is a function which takes two arguments:
      (trainer, dataloader).

    * :attr:`batches` defaults to a value which ensures that an epoch of the
      validation set matches an epoch of the training set.

      For instance, if the training set has :math:`80` datapoints and the
      validation set has :math:`20` and the batch size is :math:`1` for both,
      an epoch consists of :math:`80` iterations for the training set and
      :math:`20` for the validation set.

      If the validate function is run :math:`10` times(:attr:`frequency`)
      per epoch of the training set, then :attr:`batches` must be :math:`2`.
    """
    def __init__(self, dataloader, validate, frequency=10, batches=None, drop_last=False, **kwargs):
        self.name = kwargs.pop('name', 'validate')
        self.dataloader = dataloader
        self.frequency = frequency
        self.batches = batches
        self.drop_last = drop_last
        self.validate = validate

    def __call__(self, trainer, signal, **kwargs):
        r"""
        Responds to the following signals:

        * ``'on_training_start'``: To be called before start of training.
          Automatically finds the number of batches per run.

        * ``'on_batch_end'``: Called after the training loop.
          Calls the :attr:`validate` function.

        * ``'on_training_end'``: To be called after training.
          If :attr:`drop_last`, calls the :attr:`validate` function.

        * ``'load_state'``: Loads the state of this callback from :attr:`path`.

        * ``'save_state'``: Saves the state of this callback to :attr:`path`.
        """
        if signal == 'on_training_start':
            if self.batches is None: self.batches = int(len(self.dataloader) // self.frequency)

        elif signal == 'on_batch_end' and trainer.iterations != 0:
            batches_per_epoch = len(trainer.dataloader)
            if not trainer.iterations % int(batches_per_epoch // self.frequency): self.validate_batch(trainer)

        elif signal == 'on_training_end':
            if not self.drop_last: self.validate_batch(trainer)

        elif signal == 'load_state':
            self.load_state(kwargs.pop('path'))

        elif signal == 'save_state':
            self.save_state(kwargs.pop('path'))

    def validate_batch(self, trainer):
        with mag.eval(*trainer.models):
            for _ in range(self.batches): self.validate(trainer, self.dataloader)

    def load_state(self, path):
        from magnet.training.utils import load_object
        state_dict = load_object(path / self.name / 'dataloader.p', default=None)
        if state_dict is not None: self.dataloader.load_state_dict(state_dict)

    def save_state(self, path):
        from magnet.training.utils import save_object
        save_object(self.dataloader.state_dict(), path / self.name / 'dataloader.p')

class Checkpoint:
    r"""Serializes stateful objects during the training process.

    For many practical Deep Learning projects,
    training takes many hours, even days.

    As such, it is only natural that you'd want to save the progress every
    once in a while.

    This callback saves the models, optimizers, schedulers and the trainer
    itself periodically and automatically loads from those states if found.

    Args:
        path (pathlib.Path): The root path to save to
        interval (str): The time between checkpoints. Default: '5 m'

    Keyword Args:
        name (str): Name of this callback. Default: ``'checkpoint'``

    * :attr:`interval` should be a string of the form ``'{duration} {unit}'``.
      Valid units are: ``'us'`` (microseconds), ``'ms'`` (milliseconds),
      ``'s'`` (seconds), ``'m'`` (minutes)', ``'h'`` (hours), ``'d'`` (days).
    """
    def __init__(self, path, interval='5 m', **kwargs):
        self.name = kwargs.pop('name', 'checkpoint')
        self.path = path
        self.interval = self.parse_duration(interval)

    @staticmethod
    def parse_duration(interval):
        interval, multiplier = interval.split(' ')
        interval = float(interval); multiplier = multiplier.lower()
        multiplier_dict = {'m': 60, 's': 1, 'h': 3600, 'ms': 1e-3, 'us': 1e-6, 'd': 24 * 3600}
        multiplier = multiplier_dict[multiplier]
        return interval * multiplier

    def __call__(self, trainer, signal, **kwargs):
        r"""
        Responds to the following signals:

        * ``'on_training_start'``: To be called before start of training.
          Creates the path if it doesn't exist and loads from it if it does.
          Also sets the starting time.

        * ``'on_batch_end'``: Called after the training loop.
          Checkpoints if the interval is crossed and resets the clock.

        * ``'on_training_end'``: To be called after training.
          Checkpoints one last time.

        * ``'load_state'``: Loads the state of this callback from :attr:`path`.

        * ``'save_state'``: Saves the state of this callback to :attr:`path`.
        """
        if signal == 'on_training_start':
            self.path.mkdir(parents=True, exist_ok=True)
            trainer.load_state(self.path)
            self.start_time = time()

        elif signal == 'on_batch_end' and trainer.iterations != 0 and time() - self.start_time > self.interval:
            trainer.save_state(self.path)
            self.start_time = time()

        elif signal == 'on_training_end':
            trainer.save_state(self.path)

        elif signal == 'load_state':
            self.load_state(trainer, kwargs.pop('path'))

        elif signal == 'save_state':
            self.save_state(trainer, kwargs.pop('path'))

    def clear_state(self):
        from shutil import rmtree
        rmtree(self.path)

    def load_state(self, trainer, path):
        from magnet.training.utils import load_object
        state_dict = load_object(path / self.name / 'dataloader.p', default=None)
        if state_dict is not None: trainer.dataloader.load_state_dict(state_dict)

    def save_state(self, trainer, path):
        from magnet.training.utils import save_object
        save_object(trainer.dataloader.state_dict(), path / self.name / 'dataloader.p')

class ColdStart:
    r"""Starts the trainer in ``eval`` mode for a few iterations.

    Sometimes, you may want to find out how the model performs
    prior to any training. This callback freezes the training initially.

    Args:
        epochs (float): The number of epochs to freeze the trainer.
            Default: :math:`0.1`

    Keyword Args:
        name (str): Name of this callback. Default: ``'coldstart'``
    """
    def __init__(self, epochs=0.1, **kwargs):
        self.name = kwargs.pop('name', 'coldstart')
        self.epochs = epochs
        self.iterations = kwargs.pop('iterations', None)

    def __call__(self, trainer, signal, **kwargs):
        r"""
        Responds to the following signals:

        * ``'on_training_start'``: To be called before start of training.
          Sets the models in ``eval`` mode.

        * ``'on_batch_end'``: Called after the training loop.
          If the :attr:`epochs` is exhausted, unfreezes the trainer and
          removes this callback from the queue.
        """
        if signal == 'on_training_start':
            torch.no_grad()
            for model in trainer.models: model.eval()

            if self.iterations is None: self.iterations = int(self.epochs * len(trainer.dataloader))

        elif signal == 'on_batch_end' and trainer.iterations == self.iterations - 1:
            torch.enable_grad()
            for model in trainer.models: model.train()
            trainer.callbacks.remove(self)

class LRScheduler:
    r"""A helper callback to add in optimizer schedulers.

    Args:
        scheduler (``LRScheduler``): The scheduler.

    Keyword Args:
        name (str): Name of this callback. Default: ``'lr_scheduler'``
    """
    def __init__(self, scheduler, **kwargs):
        self.name = kwargs.pop('name', 'lr_scheduler')
        self.scheduler = scheduler

    def __call__(self, trainer, signal, **kwargs):
        r"""
        Responds to the following signals:

        * ``'on_batch_start'``: Called before the training loop.
          If it is the start of an epoch, steps the scheduler.
        """
        if signal == 'on_batch_start' and trainer.epochs('start'): self.scheduler.step()

        elif signal == 'load_state':
            self.load_state(kwargs.pop('path'))

        elif signal == 'save_state':
            self.save_state(kwargs.pop('path'))

    def load_state(self, path):
        from magnet.training.utils import load_state
        load_state(self.scheduler, path / self.name, alternative_name='scheduler')

    def save_state(self, path):
        from magnet.training.utils import save_state
        save_state(self.scheduler, path / self.name, alternative_name='scheduler')

class CallbackQueue(list):
    r"""A container for multiple callbacks that can be called in parallel.

    If multiple callbacks need to be called together (as intended), they
    can be registered via this class.

    Since callbacks need to be unique (by their name), this class also ensures
    that there are no duplicates.
    """
    def append(self, callback):
        if not self.exists(callback.name): super().append(callback)

    def extend(self, callbacks):
        super().extend([callback for callback in callbacks if not self.exists(callback.name)])

    def find(self, name):
        r"""Scans through the registered list and
        finds the callback with :attr:`name`.

        If not found, returns None.

        Raises:
            RuntimeError: If multiple callbacks are found.
        """
        callbacks = [callback for callback in self if callback.name == name]
        if len(callbacks) == 0: return None
        if len(callbacks) == 1: return callbacks[0]
        raise RuntimeError('Multiple callbacks with the same name found!')

    def exists(self, name):
        return self.find(name) is not None

    def __call__(self, signal, *args, **kwargs):
        r"""Broadcasts a signal to all registered callbacks along with
        payload arguments.

        Args:
            signal (object): Any object that is broadcast as a signal.

        .. note::
            Any other arguments will be sent as-is to the callbacks.
        """
        for callback in self: callback(*args, **kwargs, signal=signal)