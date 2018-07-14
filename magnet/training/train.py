import magnet as mag
from contextlib import contextmanager

class Trainer:
	def __init__(self, models, optimizers=['adam']):
		from torch import optim
		from magnet.training.callbacks import CallbackQueue

		self.models = models
		if optimizers[0] == 'adam': optimizers = [optim.Adam(model.parameters(), amsgrad=True) for model in models]
		self.optimizers = optimizers

		self.iterations = 0
		self.callbacks = CallbackQueue([])

	def optimize(self):
		raise NotImplementedError

	def train(self, dataloader, epochs=1, callbacks=[], **kwargs):
		self.dataloader = dataloader
		self.callbacks.extend(callbacks)

		cold_start = kwargs.get('cold_start', False) and not hasattr(self, '_iterations')
		self.training = kwargs.get('training', True)

		batches_per_epoch = len(self.dataloader)

		total_iterations = kwargs.get('iterations', int(epochs * batches_per_epoch))

		if cold_start:
			_kwargs = {k: v for k, v in kwargs.items() if k not in ('iterations', 'cold_start', 'training', 'monitor_finally')}
			with mag.eval(*self.models):
				self.train(epochs, batch_size, shuffle, iterations=int(batches_per_epoch // monitor_freq) + 1,
							cold_start=False, training=False, monitor_finally=False, **_kwargs)

		start_iteration = self.iterations

		self.callbacks('on_training_start', trainer=self, total_iterations=total_iterations)
		for self.iterations in range(start_iteration, self.iterations + total_iterations): next(self)
		self.callbacks('on_training_end', trainer=self)

	def __iter__(self):
		return self

	def __next__(self):
		self.callbacks('on_batch_start', trainer=self)
		self.optimize()
		self.callbacks('on_batch_end', trainer=self)
		self.iterations += 1

	@contextmanager
	def mock(self, path=None):
		from shutil import rmtree

		if path is None:
			from pathlib import Path
			path = Path('.mock_trainer')

		self.save_state(path)
		yield
		self.load_state(path)

		rmtree(path)

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

	def load_state(self, path=None):
		from magnet.training.utils import load_state, load_object

		for i, model in enumerate(self.models): load_state(model, path / 'models', alternative_name=str(i))
		for i, optimizer in enumerate(self.optimizers): load_state(optimizer, path / 'optimizers', alternative_name=str(i))

		state_dict = load_object(path / 'state.p', default={})
		for attr, val in state_dict.items(): setattr(self, attr, val)

		self.callbacks('load_state', trainer=self, path=path / 'callbacks')

	def save_state(self, path=None):
		from magnet.training.utils import save_state, save_object

		for i, model in enumerate(self.models): save_state(model, path / 'models', alternative_name=str(i))
		for i, optimizer in enumerate(self.optimizers): save_state(optimizer, path / 'optimizers', alternative_name=str(i))

		state_dict = {attr: getattr(self, attr) for attr in ('iterations', ) if hasattr(self, attr)}
		save_object(state_dict, path / 'state.p')

		self.callbacks('save_state', trainer=self, path=path / 'callbacks')

class SupervisedTrainer(Trainer):
	def __init__(self, model, optimizer='adam', loss='cross_entropy', metric=None):
		from magnet.training import metrics as metrics_module
		from torch.nn import functional as F

		super().__init__([model], [optimizer])
		loss_wiki = {'cross_entropy': F.cross_entropy}
		self.loss = loss_wiki[loss]
		self.metric = (metric, getattr(metrics_module, metric.lower())) if metric is not None else None

	def optimize(self):
		model = self.models[0]; optimizer = self.optimizers[0]

		loss = self.get_loss(self.dataloader)

		if self.training:
			loss.backward()
			self.callbacks('gradient', trainer=self, models=[model])
			optimizer.step()
			optimizer.zero_grad()

	@staticmethod
	def validate(trainer, dataloader):
		trainer.get_loss(dataloader, validation=True)

	def get_loss(self, dataloader, validation=False):
		model = self.models[0]

		mode = 'val' if validation else 'train'

		x, y = next(dataloader)
		y_pred = model(x)

		loss = self.loss(y_pred, y)

		self.callbacks('write_metrics', trainer=self, key='loss', value=loss.item(), validation=validation, buffer=True)
		if self.metric is not None:
			self.callbacks('write_metrics', trainer=self, key=self.metric[0], value=self.metric[1](y_pred, y).item(), validation=validation, buffer=True)

		return loss

class ClassifierTrainer(SupervisedTrainer):
	def __init__(self, model, optimizer='adam'):
		super().__init__(model, optimizer, metric='accuracy')