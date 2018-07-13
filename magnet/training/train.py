import magnet as mag
from contextlib import contextmanager

class Trainer:
	def __init__(self, models, losses=None, optimizers=['adam'], metrics=None):
		from magnet.training import metrics as metrics_module

		from torch import optim

		self.models = models
		self.losses = losses
		if optimizers[0] == 'adam': optimizers = [optim.Adam(model.parameters(), amsgrad=True) for model in models]
		self._optimizers = optimizers
		self._metrics = {metrics: getattr(metrics_module, metrics.lower())} if metrics is not None else {}

		self.babysitter = None
		self.iterations = 0

	def optimize(self):
		raise NotImplementedError

	def validate(self, dataloader):
		pass

	def train(self, dataloader, epochs=1, callbacks=[], **kwargs):
		from magnet.training.callbacks import CallbackQueue

		self.dataloader = dataloader
		self.callbacks = CallbackQueue(callbacks)

		cold_start = kwargs.get('cold_start', False) and not hasattr(self, '_iterations')
		self.training = kwargs.get('training', True)
		self.babysitter = kwargs.pop('babysitter', None)

		batches_per_epoch = len(self.dataloader)

		total_iterations = kwargs.get('iterations', int(epochs * batches_per_epoch))

		if cold_start:
			_kwargs = {k: v for k, v in kwargs.items() if k not in ('iterations', 'cold_start', 'training', 'monitor_finally')}
			with mag.eval(*self.models):
				self.train(epochs, batch_size, shuffle, iterations=int(batches_per_epoch // monitor_freq) + 1,
							cold_start=False, training=False, monitor_finally=False, **_kwargs)

		start_iteration = self.iterations

		self.callbacks('on_training_start', trainer=self, total_iterations=total_iterations)
		for self.iterations in range(start_iteration, self.iterations + total_iterations):
			self.callbacks('on_batch_start', trainer=self)
			self.optimize()
			self.callbacks('on_batch_end', trainer=self)
		self.callbacks('on_training_end', trainer=self)

	@contextmanager
	def mock(self):
		from pathlib import Path
		save_path = Path('.mock_trainer')

		self._save(save_path=save_path)
		yield
		self._load(save_path=save_path)
		self._clear_checkpoints(save_path=save_path)

	def show_history(self, keys=None, vs=None, log=None):
		xlabel = None

		if vs is None:
			vs = 'epochs' if self._iterations > self._batches_per_epoch else 'batches'
		vs = vs.lower()
		if vs == 'epochs':
			xlabel = 'epochs'
		elif vs in ('batches', 'iterations'):
			vs = 'batches'
			xlabel = 'iterations'

		if keys is None:
			keys = [k for k in self.history.keys() if k not in ('batches', ) and k[:4] != 'val_']

		for k in keys:
			if 'loss' in k:
				_log = log if log is not None else True
				self.history.show(k, log=_log, x_key=vs, xlabel=xlabel)
			else:
				_log = log if log is not None else False
				self.history.show(k, log=_log, x_key=vs, xlabel=xlabel)

	def epochs(self, mode=None):
		if mode is None:
			return self.iterations / len(self.dataloader)
		if mode == 'start':
			return (self.iterations / len(self.dataloader)).is_integer()
		if mode == 'end':
			return ((self.iterations + 1) / len(self.dataloader)).is_integer()

	def _gradient_callback(self, batch):
		if self.babysitter is not None: self.babysitter.append(self.models, batches=batch, epochs=batch / self._batches_per_epoch)

	def load(self, path=None):
		from magnet.training.utils import load_state, load_object

		for i, model in enumerate(self.models): load_state(model, path / 'models', alternative_name=str(i))
		for i, optimizer in enumerate(self._optimizers): load_state(optimizer, path / 'optimizers', alternative_name=str(i))

		state_dict = load_object(path / 'state.p', default={})
		for attr, val in state_dict.items(): setattr(self, attr, val)

		self.callbacks('load', trainer=self, path=path)

		if self.babysitter is not None: self.babysitter.load(save_path)

	def save(self, path=None):
		from magnet.training.utils import save_state, save_object

		for i, model in enumerate(self.models): save_state(model, path / 'models', alternative_name=str(i))
		for i, optimizer in enumerate(self._optimizers): save_state(optimizer, path / 'optimizers', alternative_name=str(i))

		state_dict = {attr: getattr(self, attr) for attr in ('iterations', ) if hasattr(self, attr)}
		save_object(state_dict, path / 'state.p')

		self.callbacks('save', trainer=self, path=path)

		if self.babysitter is not None: self.babysitter.save(save_path)

class SupervisedTrainer(Trainer):
	def __init__(self, model, loss=None, optimizer='adam', metrics=None):
		super().__init__([model], [loss], [optimizer], metrics)

	def optimize(self):
		model = self.models[0]; optimizer = self._optimizers[0]

		loss = self._get_loss(self.dataloader)

		if self.training:
			loss.backward()
			self.callbacks('gradient', trainer=self, models=[model])
			optimizer.step()
			optimizer.zero_grad()

	def validate(self, dataloader):
		self._get_loss(dataloader, validation=True)

	def _get_loss(self, dataloader, validation=False):
		model = self.models[0]; loss_fn = self.losses[0]

		mode = 'val' if validation else 'train'

		x, y = next(dataloader)
		y_pred = model(x)

		loss = loss_fn(y_pred, y)

		self.callbacks('before_optimization', trainer=self, key='loss', value=loss.item(), validation=validation, buffer=True)
		for k in self._metrics.keys():
			self.callbacks('before_optimization', trainer=self, key=k, value=self._metrics[k](y_pred, y).item(), validation=validation, buffer=True)

		return loss

class ClassifierTrainer(SupervisedTrainer):
	def __init__(self, model, optimizer='adam'):
		from torch import nn

		super().__init__(model, nn.CrossEntropyLoss(), optimizer, metrics='accuracy')