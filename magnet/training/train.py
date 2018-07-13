import magnet as mag
from contextlib import contextmanager

class Trainer:
	def __init__(self, models, data, losses=None, optimizers=['adam'], metrics=None, save_path=None):
		from magnet.training import metrics as metrics_module
		from magnet.training.history import History
		from magnet.training.callbacks import CallbackQueue, Monitor

		from pathlib import Path
		from torch import optim

		self.models = models
		self.data = data
		self._losses = losses
		if optimizers[0] == 'adam': optimizers = [optim.Adam(model.parameters(), amsgrad=True) for model in models]
		self._optimizers = optimizers
		self._metrics = {metrics: getattr(metrics_module, metrics.lower())} if metrics is not None else {}

		self.history = History()

		self.save_path = save_path

		self.babysitter = None
		self.iterations = 0
		self.callbacks = CallbackQueue([Monitor()])

		if save_path is not None:
			self.save_path = Path(save_path)
			self.save_path.mkdir(parents=True, exist_ok=True)
			self._load()

	def _optimize(self, dataloader, batch):
		raise NotImplementedError

	def _validate(self, dataloader, validation_batches):
		pass

	def train(self, epochs=1, batch_size=1, shuffle=True, **kwargs):
		import torch
		from magnet._utils import get_tqdm; tqdm = get_tqdm()
		from magnet import data as data_module
		from time import time

		start_time = time()

		save_interval = kwargs.get('save_interval', '5 m')
		monitor_freq = kwargs.get('monitor_freq', 10)
		validate_freq = kwargs.get('validate_freq', monitor_freq)
		batch_size_val = kwargs.get('batch_size_val', batch_size)
		shuffle_val = kwargs.get('shuffle_val', False)
		cold_start = kwargs.get('cold_start', False) and not hasattr(self, '_iterations')
		training = kwargs.get('training', True)
		monitor_finally = kwargs.get('monitor_finally', True)
		self.babysitter = kwargs.pop('babysitter', None)

		if isinstance(self.data, data_module.Data):
			dataloader = {'train': self.data(batch_size, shuffle)}

			if batch_size_val < 0: batch_size_val = batch_size
			dataloader['val'] = self.data(batch_size_val, shuffle=shuffle_val, mode='val')
		else:
			dataloader = {'train': self.data[0], 'val': self.data[1]}
		if self.save_path is not None:
			dataloader['train'].load_state_dict(self.save_path / 'dl_train.p')
			dataloader['val'].load_state_dict(self.save_path / 'dl_val.p')

		for k, v in dataloader.items():
			if hasattr(self, 'dataloader'):
				if k in self.dataloader.keys() and v.compatible_with(self.dataloader[k]):
					dataloader[k] = self.dataloader[k]
				else:
					self.dataloader[k] = v
			else:
				self.dataloader = {k: v}

		validation_batches=kwargs.get('validation_batches', int(len(dataloader['val']) // validate_freq))

		batches_per_epoch = len(dataloader['train'])

		total_iterations = kwargs.get('iterations', int(epochs * batches_per_epoch))

		if cold_start:
			_kwargs = {k: v for k, v in kwargs.items() if k not in ('iterations', 'cold_start', 'training', 'monitor_finally')}
			with mag.eval(*self.models):
				self.train(epochs, batch_size, shuffle, iterations=int(batches_per_epoch // monitor_freq) + 1,
							cold_start=False, training=False, monitor_finally=False, **_kwargs)

		self.callbacks('on_training_start', trainer=self, total_iterations=total_iterations)

		if save_interval is not None:
			save_interval, _save_multiplier = save_interval.split(' ')
			save_interval = float(save_interval); _save_multiplier = _save_multiplier.lower()
			_save_multiplier_dict = {'m': 60, 's': 1, 'h': 3600, 'ms': 1e-3, 'us': 1e-6, 'd': 24 * 3600}
			_save_multiplier = _save_multiplier_dict[_save_multiplier]
			save_interval *= _save_multiplier

		start_iteration = self.iterations
		for self.iterations in range(start_iteration, self.iterations + total_iterations):
			is_last_batch = (self.iterations == start_iteration + total_iterations - 1)

			try:
				if not self.iterations % self._batches_per_epoch:
					self.callbacks('on_epoch_start', trainer=self)
			except AttributeError: pass

			self.callbacks('on_batch_start', trainer=self)

			self._optimize(dataloader['train'], self.iterations, training)

			if (is_last_batch and monitor_finally) or (not self.iterations % int(batches_per_epoch // validate_freq) and self.iterations != 0):
				with mag.eval(*self.models): self._validate(dataloader['val'], validation_batches)

			self.callbacks('on_batch_end', trainer=self)

			try:
				if not (self.iterations + 1) % self._batches_per_epoch:
					self.callbacks('on_epoch_end', trainer=self)
			except AttributeError: pass

			if save_interval is not None and (is_last_batch or ((time() - start_time > save_interval) and self.iterations != 0)):
				self._save(dataloader)
				start_time = time()

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

	@property
	def epochs(self):
		return self.iterations / len(self.dataloader['train'])

	def _gradient_callback(self, batch):
		if self.babysitter is not None: self.babysitter.append(self.models, batches=batch, epochs=batch / self._batches_per_epoch)

	def _load(self, save_path=None):
		if save_path is None: save_path = self.save_path
		if save_path is None: return
		import torch, pickle

		def _load_module(module, subpath, name_alternative):
			name = name_alternative if not hasattr(module, 'name') else module.name
			filepath = save_path / subpath / (name + '.pt')
			if filepath.exists(): module.load_state_dict(torch.load(filepath))

		def _load_obj(name, default=None):
			filepath = save_path / (name + '.p')
			if filepath.exists():
				with open(filepath, 'rb') as f: return pickle.load(f)
			else:
				return default

		for i, model in enumerate(self.models): _load_module(model, 'models', str(i))

		for i, optimizer in enumerate(self._optimizers): _load_module(optimizer, 'optimizers', str(i))

		history = _load_obj('history')
		if history is not None: self.history = history

		state_dict = _load_obj('state', {})
		for attr, val in state_dict.items(): setattr(self, attr, val)

		if self.babysitter is not None: self.babysitter.load(save_path)

	def _save(self, dataloader=None, save_path=None):
		if save_path is None: save_path = self.save_path
		if save_path is None: return

		import torch, pickle

		subpaths = ['models', 'optimizers']
		for subpath in subpaths: (save_path / subpath).mkdir(parents=True, exist_ok=True)

		def _save_module(module, subpath, name_alternative):
			name = name_alternative if not hasattr(module, 'name') else module.name
			filepath = save_path / subpath / (name + '.pt')
			torch.save(module.state_dict(), filepath)

		def _save_obj(obj, name):
			with open(save_path / (name + '.p'), 'wb') as f: pickle.dump(obj, f)

		for i, model in enumerate(self.models): _save_module(model, 'models', str(i))

		for i, optimizer in enumerate(self._optimizers): _save_module(optimizer, 'optimizers', str(i))

		_save_obj(self.history, 'history')

		state_dict = {attr: getattr(self, attr) for attr in ('_batches_per_epoch', '_iterations') if hasattr(self, attr)}
		_save_obj(state_dict, 'state')

		if dataloader is not None:
			dataloader['train'].save_state_dict(save_path / 'dl_train.p')
			dataloader['val'].save_state_dict(save_path / 'dl_val.p')

		if self.babysitter is not None: self.babysitter.save(save_path)

	def _clear_checkpoints(self, save_path=None):
		if save_path is None: save_path = self.save_path
		if save_path is None: return

		if save_path.exists():
			import shutil
			shutil.rmtree(save_path)

class SupervisedTrainer(Trainer):
	def __init__(self, model, data, loss=None, optimizer='adam', metrics=None, save_path=None):
		super().__init__([model], data, [loss], [optimizer], metrics, save_path)

	def _optimize(self, dataloader, batch, training):
		model = self.models[0]; optimizer = self._optimizers[0]

		loss = self._get_loss(dataloader)

		if training:
			loss.backward()
			self.callbacks('gradient', trainer=self, models=[model])
			optimizer.step()
			optimizer.zero_grad()

	def _validate(self, dataloader, validation_batches):
		for _ in range(validation_batches): self._get_loss(dataloader, validation=True)

	def _get_loss(self, dataloader, validation=False):
		model = self.models[0]; loss_fn = self._losses[0]

		x, y = next(dataloader)
		y_pred = model(x)

		loss = loss_fn(y_pred, y)

		self.callbacks('before_optimization', trainer=self, key='loss', value=loss.item(), validation=validation, buffer=True)
		for k in self._metrics.keys():
			self.callbacks('before_optimization', trainer=self, key=k, value=self._metrics[k](y_pred, y).item(), validation=validation, buffer=True)

		return loss

class ClassifierTrainer(SupervisedTrainer):
	def __init__(self, model, data, optimizer='adam', save_path=None):
		from torch import nn

		super().__init__(model, data, nn.CrossEntropyLoss(), optimizer, metrics='accuracy', save_path=save_path)