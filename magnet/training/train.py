from torch import optim
from contextlib import contextmanager

class Trainer:
    r"""Abstract base class for training models.

    The Trainer class makes it incredibly simple and convinient to train,
    monitor, debug and checkpoint entire Deep Learning projects.

    Simply define your training loop by
    implementing the :py:meth:`optimize` method.

    Args:
        models (list of :py:class:`nn.Module`): All the models that need
            to be trained
        optimizers (list of :py:class:`optim.Optimizer`): Any optimizers that
            are used

    .. note::
        If any model is in eval() model, the trainer is *set off*.
        This means that as per protocol, *all* models will not train.

    Attributes:
        callbacks (list): A list of callbacks attached to the trainer.

    Take a look at :py:class:`SupervisedTrainer` for an idea on how to extend this class.
    """
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers

        self.parameters = set()
        self.register_parameter('iterations', 0)

    def optimize(self):
        r""" Defines the core optimization loop.
        This method is called on each iteration.

        Two quick protocols that one needs to follow are:

        1. **Do NOT** actually backpropagate or step() the optimizers if the
        trainer is not training. Use the :py:meth:`is_training` method
        to find out.
        This is essential since this will ensure that the trainer behaves
        as expected when :py:meth:`is_training` is ``False``.
        Useful, for example, in cases like :py:class:`callbacks.ColdStart`

        2. Send a callback the signal ``'gradient'`` with a keyword argument
        ``'models'`` that is the list of models that accumulate a gradient.
        Usually, it's all the modules (``self.modules``).

        Any callbacks that listen to this signal are interested in the gradient
        information (eg. ``callbacks.Babysitter``).
        """
        raise NotImplementedError

    def train(self, dataloader, epochs=1, callbacks=None, **kwargs):
        r"""Starts the training process.

        Args:
            dataloader (``DataLoader``): The MagNet dataloader that iterates
                over the training set
            epochs (float or int): The number of epochs to train for.
                Default: ``1``
            callbacks (list): Any callbacks to be attached. Default: ``None``

        Keyword Args:
            iterations (int): The number of iterations to train for.
                Overrides :attr:`epochs`.

        .. note::
            PyTorch ``DataLoader`` s are not supported.

            Ideally, encapsulate your dataset in the ``Data`` class.
        """
        from magnet.training.callbacks import CallbackQueue
        self.dataloader = dataloader

        if callbacks is None: callbacks = []
        self.callbacks = CallbackQueue(callbacks)

        total_iterations = kwargs.get('iterations', int(epochs * len(dataloader)))

        self.callbacks('on_training_start', trainer=self, total_iterations=total_iterations)
        for self.iterations in range(self.iterations, self.iterations + total_iterations): next(self)
        self.callbacks('on_training_end', trainer=self)

    def __iter__(self):
        return self

    def __next__(self):
        self.callbacks('on_batch_start', trainer=self)
        self.optimize()
        self.callbacks('on_batch_end', trainer=self)

    @contextmanager
    def mock(self, path=None):
        r"""A context manager that creates a temporary *'safe'* scope for training.

        All impact to stateful objects (models, optimizers and the
        trainer itself) are forgotten once out of this scope.

        This is very useful if you need to try out *what-if experiments*.

        Args:
            path (pathlib.Path): The path to save temporary states into
                Default: ``{System temp directory}/.mock_trainer``
        """
        from shutil import rmtree

        if path is None:
            from pathlib import Path
            from tempfile import gettempdir
            path = Path(gettempdir()) / '.mock_trainer'

        rmtree(path, ignore_errors=True) # Remove any existing directory
        self.save_state(path)
        try:
            yield
        finally:
            self.load_state(path)
            rmtree(path)

    def epochs(self, mode=None):
        r"""The number of epochs completed.

        Args:
            mode (str or None): If the mode is ``'start'`` or ``'end'``, a
                boolean is returned signalling if it's the start or end of an epoch
        """
        if mode is None:
            return self.iterations / len(self.dataloader)
        if mode == 'start':
            return (self.iterations / len(self.dataloader)).is_integer()
        if mode == 'end':
            return ((self.iterations + 1) / len(self.dataloader)).is_integer()

    def is_training(self):
        return all(model.training for model in self.models)

    def load_state(self, path):
        from magnet.training.utils import load_state, load_object

        for i, model in enumerate(self.models): load_state(model, path / 'models', alternative_name=str(i))
        for i, optimizer in enumerate(self.optimizers): load_state(optimizer, path / 'optimizers', alternative_name=str(i))

        state_dict = load_object(path / 'state.p', default={})
        for attr, val in state_dict.items(): self.register_parameter(attr, val)

        try: self.callbacks('load_state', trainer=self, path=path / 'callbacks')
        except AttributeError: pass

        try: self.dataloader.load_state_dict(path / 'dataloader.p')
        except AttributeError: pass

    def save_state(self, path):
        from magnet.training.utils import save_state, save_object

        for i, model in enumerate(self.models): save_state(model, path / 'models', alternative_name=str(i))
        for i, optimizer in enumerate(self.optimizers): save_state(optimizer, path / 'optimizers', alternative_name=str(i))

        state_dict = {attr: getattr(self, attr) for attr in self.parameters}
        save_object(state_dict, path / 'state.p')

        try: self.callbacks('save_state', trainer=self, path=path / 'callbacks')
        except AttributeError: pass

        try: self.dataloader.save_state_dict(path / 'dataloader.p')
        except AttributeError: pass

    def register_parameter(self, name, value):
        r"""Use this to register *'stateful'* parameters that are serialized
        """
        setattr(self, name, value)
        self.parameters.add(name)

class SupervisedTrainer(Trainer):
    r"""A simple trainer that implements a supervised approach where a simple
    model :math:`\hat{y} = f(x)` is trained to map :math:`\hat{y}` to
    ground-truth :math:`y` according to some specified loss.

    This is the training routine that most high-level deep learning frameworks
    implement.

    Args:
        model (``nn.Module``): The model that needs to be trained
        optimizer (str or optim.Optimzer): The optimizer used to train
            the model. Default: ``'adam'``
        loss (str or ``callable``): A loss function that gives the objective
            to be minimized. Default: ``'cross_entropy'``
        metrics (list): Any other metrics that need to be monitored.
            Default: ``None``

    * :attr:`optimizer` can be an actual ``optim.Optimizer`` instance or the
      name of a popular optimzizer (eg. ``'adam'``).

    * :attr:`loss` can be a function or the name of a popular
      loss function (eg. ``'cross_entropy'``).
      It should accept 2 arguments (:math:`\hat{y}`, :math:`y`).

    * :attr:`metrics` should contain a list of functions which accept
      2 arguments (:math:`\hat{y}`, :math:`y`), like the loss function.

    .. note::
        A static :py:meth:`validate` function is provided for the
        validation callback

    .. note::
        The :attr:`metrics` is of no use unless there is some
        callback (eg.``callbacks.Monitor``) to receive the metrics

    Examples::

        >>> import magnet as mag
        >>> import magnet.nodes as mn

        >>> from magnet.data import Data
        >>> from magnet.training import callbacks, SupervisedTrainer

        >>> data = Data.get('mnist')

        >>> model = mn.Linear(10, act=None)
        >>> model.build(x=next(data())[0])

        >>> trainer = SupervisedTrainer(model)
        >>> callbacks=[callbacks.Monitor(),
                       callbacks.Validate(data(64, mode='val'), SupervisedTrainer.validate)]
        >>> trainer.train(data(64, shuffle=True), 1, callbacks)
    """
    def __init__(self, model, optimizer='adam', loss='cross_entropy', metrics=None):
        from magnet.nodes.functional import wiki

        if isinstance(optimizer, str): optimizer = optimizer_wiki[optimizer.lower()](model.parameters())
        if isinstance(loss, str): loss = wiki['losses'][loss.lower()]

        if metrics is None: metrics = []
        if not isinstance(metrics, (tuple, list)): metrics = [metrics]
        for i, metric in enumerate(metrics):
            if isinstance(metric, str): metrics[i] = (metric, wiki['metrics'][metric.lower()])

        super().__init__([model], [optimizer])

        self.loss = loss
        self.metrics = metrics

    def optimize(self):
        optimizer = self.optimizers[0]

        loss = self.get_loss(self.dataloader)

        # Protocol 1: Backprop and step() only if trainer is training
        if self.is_training():
            loss.backward()

            # Protocol 2: Broadcast the models that accumulate the gradient
            # using signal 'gradient' before clearing them.
            self.callbacks('gradient', trainer=self, models=self.models)

            optimizer.step()
            optimizer.zero_grad()

    @staticmethod
    def validate(trainer, dataloader):
        r"""Static helper method to validate models in :attr:`trainer` against
        data in :attr:`dataloader`.

        Can be passed to ``callbacks.Validate()``.
        """
        trainer.get_loss(dataloader, validation=True)

    def get_loss(self, dataloader, validation=False):
        r"""Utility function that returns the loss and broadcasts metrics.
        """
        def write_stats(key, value):
            self.callbacks('write_stats', trainer=self, key=key, value=value, validation=validation, buffer_size=len(dataloader))

        model = self.models[0]

        x, y = next(dataloader)
        y_pred = model(x)

        loss = self.loss(y_pred, y)

        # Broadcast the loss and any other metrics using the 'write_stats' signal.
        write_stats('loss', loss.item())
        for metric in self.metrics: write_stats(metric[0], metric[1](y_pred, y).item())

        return loss

def finish_training(path, names=None):
    r""" A helper function for cleaning up the training logs and other
    checkpoints and retaining only the state_dicts of the trained models.

    Args:
        path (pathlib.Path): The path where the trainer was checkpointed
        names (list): The names of the models in the order given to the trainer.
            Default: ``None``

    * :attr:`names` can be used if the models themselves did not have names
      prior to training.
      The checkpoints default to an ordered naming scheme.
      If passed, the files are additionally renamed to these names.

    .. note::
        Does nothing / fails silently if the path does not exist.

    Example::

        >>> # Assume that we've defined two models - encoder and decoder,
        >>> # and a suitable trainer. The models do not have a 'name' attribute.

        >>> trainer.save_state(checkpoint_path / 'my-trainer')

        >>> # Suppose the checkpoint directory contains the following files:
        >>> # my-trainer/
        >>> #     models/
        >>> #         0.pt
        >>> #         1.pt
        >>> #     callbacks/
        >>> #         monitor/
        >>> #         babysitter/
        >>> #     state.p

        >>> finish_training(path, names=['encoder', 'decoder'])

        >>> # Now the directory contains these files:
        >>> # encoder.pt
        >>> # decoder.pt
    """
    if not path.exists(): return

    import shutil

    if isinstance(names, str): names = [names]
    filenames = list((path / 'models').glob('*.pt'))
    if names is None: names = [filename.stem for filename in filenames]

    for name, filename in zip(names, filenames):
        shutil.move(filename, path.parent / (name + '.pt'))

    shutil.rmtree(path)

optimizer_wiki = {'adam': optim.Adam}