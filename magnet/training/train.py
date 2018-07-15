import magnet as mag

from torch import optim
from contextlib import contextmanager

class Trainer:
	def __init__(self, models, optimizers):
		self.models = models
		self.optimizers = optimizers

		self.iterations = 0

	def optimize(self):
		raise NotImplementedError

	def train(self, dataloader, epochs=1, callbacks=[], **kwargs):
		from magnet.training.callbacks import CallbackQueue

		self.dataloader = dataloader
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
		from shutil import rmtree

		if path is None:
			from pathlib import Path
			path = Path('.mock_trainer')

		self.save_state(path)
		yield
		self.load_state(path)

		rmtree(path)

	def epochs(self, mode=None):
		if mode is None:
			return self.iterations / len(self.dataloader)
		if mode == 'start':
			return (self.iterations / len(self.dataloader)).is_integer()
		if mode == 'end':
			return ((self.iterations + 1) / len(self.dataloader)).is_integer()

	def is_training(self):
		return all(model.training for model in self.models)

	def load_state(self, path=None):
		from magnet.training.utils import load_state, load_object

		for i, model in enumerate(self.models): load_state(model, path / 'models', alternative_name=str(i))
		for i, optimizer in enumerate(self.optimizers): load_state(optimizer, path / 'optimizers', alternative_name=str(i))

		state_dict = load_object(path / 'state.p', default={})
		for attr, val in state_dict.items(): setattr(self, attr, val)

		try: self.callbacks('load_state', trainer=self, path=path / 'callbacks')
		except AttributeError: pass

	def save_state(self, path=None):
		from magnet.training.utils import save_state, save_object

		for i, model in enumerate(self.models): save_state(model, path / 'models', alternative_name=str(i))
		for i, optimizer in enumerate(self.optimizers): save_state(optimizer, path / 'optimizers', alternative_name=str(i))

		state_dict = {attr: getattr(self, attr) for attr in ('iterations', ) if hasattr(self, attr)}
		save_object(state_dict, path / 'state.p')

		try: self.callbacks('save_state', trainer=self, path=path / 'callbacks')
		except AttributeError: pass

class SupervisedTrainer(Trainer):
	def __init__(self, model, optimizer='adam', loss='cross_entropy', metric=None):
		from magnet.functional import wiki
		from torch.nn import functional as F

		if isinstance(optimizer, str): optimizer = optimizer_wiki[optimizer](model.parameters())

		super().__init__([model], [optimizer])

		self.loss = wiki['losses'][loss]
		self.metric = (metric, wiki['metrics'][metric.lower()]) if metric is not None else None

	def optimize(self):
		optimizer = self.optimizers[0]

		loss = self.get_loss(self.dataloader)

		if self.is_training():
			loss.backward()
			self.callbacks('gradient', trainer=self, models=self.models)
			optimizer.step()
			optimizer.zero_grad()

	@staticmethod
	def validate(trainer, dataloader):
		trainer.get_loss(dataloader, validation=True)

	def get_loss(self, dataloader, validation=False):
		model = self.models[0]

		x, y = next(dataloader)
		y_pred = model(x)

		loss = self.loss(y_pred, y)

		self.callbacks('write_stats', trainer=self, key='loss', value=loss.item(), validation=validation, buffer_size=len(dataloader))
		if self.metric is not None:
			self.callbacks('write_stats', trainer=self, key=self.metric[0], value=self.metric[1](y_pred, y).item(), validation=validation, buffer_size=len(dataloader))

		return loss

class ClassifierTrainer(SupervisedTrainer):
	def __init__(self, model, optimizer='adam'):
		super().__init__(model, optimizer, metric='accuracy')

optimizer_wiki = {'adam': optim.Adam}