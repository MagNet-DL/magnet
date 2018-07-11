import magnet as mag
from contextlib import contextmanager

class Trainer:
	def __init__(self, models, data, losses=None, optimizers=['adam'], metrics=None, save_path=None):
		from magnet.training import metrics as metrics_module
		from magnet.training.history import History
		from pathlib import Path
		from torch import optim

		self._models = models
		self.data = data
		self._losses = losses
		if optimizers[0] == 'adam': optimizers = [optim.Adam(model.parameters(), amsgrad=True) for model in models]
		self._optimizers = optimizers
		self._metrics = {metrics: getattr(metrics_module, metrics.lower())} if metrics is not None else {}

		self.history = History()

		self._save_path = save_path

		self._iterations = -1
		if save_path is not None:
			self._save_path = Path(save_path)
			self._save_path.mkdir(parents=True, exist_ok=True)
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

		if isinstance(self.data, data_module.Data):
			dataloader = {'train': self.data(batch_size, shuffle)}

			if batch_size_val < 0: batch_size_val = batch_size
			dataloader['val'] = self.data(batch_size_val, shuffle=shuffle_val, mode='val')
		else:
			dataloader = {'train': self.data[0], 'val': self.data[1]}
		if self._save_path is not None:
			dataloader['train'].load_state_dict(self._save_path / 'dl_train.p')
			dataloader['val'].load_state_dict(self._save_path / 'dl_val.p')

		for k, v in dataloader.items():
			if hasattr(self, '_dataloader'):
				if k in self._dataloader.keys() and v.compatible_with(self._dataloader[k]):
					dataloader[k] = self._dataloader[k]
				else:
					self._dataloader[k] = v
			else:
				self._dataloader = {k: v}

		validation_batches=kwargs.get('validation_batches', int(len(dataloader['val']) // validate_freq))

		self._batches_per_epoch = len(dataloader['train'])

		iterations = kwargs.get('iterations', int(epochs * self._batches_per_epoch))

		if cold_start:
			_kwargs = {k: v for k, v in kwargs.items() if k not in ('iterations', 'cold_start', 'training', 'monitor_finally')}
			with mag.eval(*self._models):
				self.train(epochs, batch_size, shuffle, iterations=int(self._batches_per_epoch // monitor_freq) + 1,
							cold_start=False, training=False, monitor_finally=False, **_kwargs)

		self.history.buffer_size = kwargs.get('buffer_size', self._batches_per_epoch)
		self.history.val_buffer_size = kwargs.get('val_buffer_size', len(dataloader['val']))

		try: start_iteration = self._iterations + 1
		except AttributeError: start_iteration = self._iterations = 0

		progress_bar = tqdm(range(start_iteration, start_iteration + iterations), unit_scale=True,
							unit_divisor=self._batches_per_epoch, leave=False)

		self._on_training_start()

		save_interval, _save_multiplier = save_interval.split(' ')
		save_interval = float(save_interval); _save_multiplier = _save_multiplier.lower()
		_save_multiplier_dict = {'m': 60, 's': 1, 'h': 3600, 'ms': 1e-3, 'us': 1e-6, 'd': 24 * 3600}
		_save_multiplier = _save_multiplier_dict[_save_multiplier]
		save_interval *= _save_multiplier

		for batch in progress_bar:
			is_last_batch = (batch == start_iteration + iterations - 1)

			try:
				if not batch % self._batches_per_epoch:
					self._on_epoch_start(int(batch * self._batches_per_epoch))
			except AttributeError: pass

			self._on_batch_start(batch)

			self._optimize(dataloader['train'], batch, training)

			self._on_batch_end(batch)

			if (is_last_batch and monitor_finally) or (not batch % int(self._batches_per_epoch // validate_freq) and batch != 0):
				with mag.eval(*self._models): self._validate(dataloader['val'], validation_batches)

			if not batch % int(self._batches_per_epoch // monitor_freq) and batch != 0:
				self._monitor(batch, progress_bar=progress_bar)

			try:
				if not (batch + 1) % self._batches_per_epoch:
					self._on_epoch_end(int(batch // self._batches_per_epoch))
			except AttributeError: pass

			if is_last_batch or ((time() - start_time > save_interval) and batch != 0):
				self._save(dataloader)
				start_time = time()

		self._on_training_end()

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

	def _on_training_start(self):
		pass

	def _on_epoch_start(self, epoch):
		pass

	def _on_batch_start(self, batch):
		pass

	def _on_batch_end(self, batch):
		pass

	def _monitor(self, batch, **kwargs):
		self._iterations = batch
		self.history.flush(batches=batch, epochs=batch / self._batches_per_epoch)

	def _on_epoch_end(self, epoch):
		pass

	def _on_training_end(self):
		pass

	def _load(self, save_path=None):
		if save_path is None: save_path = self._save_path
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

		for i, model in enumerate(self._models): _load_module(model, 'models', str(i))

		for i, optimizer in enumerate(self._optimizers): _load_module(optimizer, 'optimizers', str(i))

		history = _load_obj('history')
		if history is not None: self.history = history

		state_dict = _load_obj('state', {})
		for attr, val in state_dict.items(): setattr(self, attr, val)

	def _save(self, dataloader=None, save_path=None):
		if save_path is None: save_path = self._save_path
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

		for i, model in enumerate(self._models): _save_module(model, 'models', str(i))

		for i, optimizer in enumerate(self._optimizers): _save_module(optimizer, 'optimizers', str(i))

		_save_obj(self.history, 'history')

		state_dict = {attr: getattr(self, attr) for attr in ('_batches_per_epoch', '_iterations') if hasattr(self, attr)}
		_save_obj(state_dict, 'state')

		if dataloader is not None:
			dataloader['train'].save_state_dict(save_path / 'dl_train.p')
			dataloader['val'].save_state_dict(save_path / 'dl_val.p')

	def _clear_checkpoints(self, save_path=None):
		if save_path is None: save_path = self._save_path
		if save_path is None: return

		if save_path.exists():
			import shutil
			shutil.rmtree(save_path)

class SupervisedTrainer(Trainer):
	def __init__(self, model, data, loss=None, optimizer='adam', metrics=None, save_path=None):
		super().__init__([model], data, [loss], [optimizer], metrics, save_path)

	def _optimize(self, dataloader, batch, training):
		model = self._models[0]; optimizer = self._optimizers[0]

		loss = self._get_loss(dataloader)

		if training:
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

	def _validate(self, dataloader, validation_batches):
		for _ in range(validation_batches): self._get_loss(dataloader, validation=True)

	def _get_loss(self, dataloader, validation=False):
		model = self._models[0]; loss_fn = self._losses[0]

		x, y = next(dataloader)
		y_pred = model(x)

		loss = loss_fn(y_pred, y)

		self.history.append('loss', loss.item(), validation=validation, buffer=True)
		for k in self._metrics.keys():
			self.history.append(k, self._metrics[k](y_pred, y).item(), validation=validation, buffer=True)

		return loss

	def _monitor(self, batch, **kwargs):
		super()._monitor(batch, **kwargs)
		progress_bar = kwargs.pop('progress_bar')

		loss = self.history['loss'][-1];
		try:
			val_loss = self.history['val_loss'][-1]
			progress_bar.set_description(f'{loss:.2f}, {val_loss:.2f}', refresh=False)
		except KeyError:
			progress_bar.set_description(f'{loss:.2f}', refresh=False)

class ClassifierTrainer(SupervisedTrainer):
	def __init__(self, model, data, optimizer='adam', save_path=None):
		from torch import nn

		super().__init__(model, data, nn.CrossEntropyLoss(), optimizer, metrics='accuracy', save_path=save_path)
