import magnet as mag

class Trainer:
	def __init__(self, models, data, optimizers, save_path=None):
		from pathlib import Path

		self._models = models
		self._data = data
		self._optimizers = optimizers

		from magnet.training.history import History
		self._history = History()

		self._save_path = save_path
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
		cold_start = kwargs.get('cold_start', False) and 'batches' not in self._history.keys()
		monitor_finally = kwargs.get('monitor_finally', True)

		if isinstance(self._data, data_module.Data):
			dataloader = {'train': self._data(batch_size, shuffle)}

			if batch_size_val < 0: batch_size_val = batch_size
			dataloader['val'] = self._data(batch_size_val, shuffle=shuffle_val, mode='val')
		else:
			dataloader = {'train': self._data[0], 'val': self._data[1]}

		if self._save_path is not None:
			dataloader['train'].load_state_dict(self._save_path / 'dl_train.p')
			dataloader['val'].load_state_dict(self._save_path / 'dl_val.p')

		validation_batches=kwargs.get('validation_batches', int(len(dataloader['val']) // validate_freq))

		self._batches_per_epoch = len(dataloader['train'])

		iterations = kwargs.get('iterations', int(epochs * self._batches_per_epoch))

		if cold_start:
			kwargs.pop('monitor_finally', None); kwargs.pop('iterations', None); kwargs.pop('cold_start', None)
			with mag.eval(*self._models):
				self.train(epochs, batch_size, shuffle, monitor_finally=False, iterations=int(self._batches_per_epoch // monitor_freq), cold_start=False, **kwargs)

		try: start_iteration = self._history['batches'][-1]
		except KeyError: start_iteration = 0

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

			self._optimize(dataloader['train'], batch)

			self._on_batch_end(batch)

			if (is_last_batch and monitor_finally) or (not batch % int(self._batches_per_epoch // validate_freq) and batch != 0):
				with mag.eval(*self._models): self._validate(dataloader['val'], validation_batches)

			if (is_last_batch and monitor_finally) or (not batch % int(self._batches_per_epoch // monitor_freq) and batch != 0):
				self._monitor(batch, progress_bar=progress_bar)
				"""if cold_start:
					cold_start = False
					torch.enable_grad()
					for model, training in zip(self._models, cold_start_model_training):
						if training: model.train()"""

			try:
				if not (batch + 1) % self._batches_per_epoch:
					self._on_epoch_end(int(batch // self._batches_per_epoch))
			except AttributeError: pass

			if is_last_batch or ((time() - start_time > save_interval) and batch != 0):
				self._save(dataloader)
				start_time = time()

		self._on_training_end()

	def show_history(self, keys=None, vs=None):
		xlabel = None

		if vs is None:
			vs = 'epochs' if self._history['batches'][-1] > self._batches_per_epoch else 'batches'
		vs = vs.lower()
		if vs == 'epochs':
			vs = [b / self._batches_per_epoch for b in self._history['batches']]
			xlabel = 'epochs'
		elif vs in ('batches', 'iterations'):
			vs = 'batches'
			xlabel = 'iterations'

		if keys is None:
			keys = [k for k in self._history.keys() if k not in ('batches', ) and k[:4] != 'val_']

		for k in keys:
			if 'loss' in k:
				self._history.show(k, log=True, x_key=vs, xlabel=xlabel)
			else:
				self._history.show(k, x_key=vs, xlabel=xlabel)

	def _on_training_start(self):
		pass

	def _on_epoch_start(self, epoch):
		pass

	def _on_batch_start(self, batch):
		pass

	def _on_batch_end(self, batch):
		pass

	def _monitor(self, batch, **kwargs):
		self._history.append('batches', batch)
		self._history.flush()

	def _on_epoch_end(self, epoch):
		pass

	def _on_training_end(self):
		pass

	def _load(self):
		if self._save_path is None: return
		import torch, pickle

		def _load_module(module, subpath, name_alternative):
			device = module.device.type
			if device == 'cuda': device = 'cuda:0'

			name = name_alternative if not hasattr(module, 'name') else optimizer.name
			filepath = self._save_path / subpath / (name + '.pt')
			if filepath.exists(): module.load_state_dict(torch.load(filepath, map_location=device))

		def _load_obj(name, default):
			filepath = self._save_path / (name + '.p')
			if filepath.exists():
				with open(filepath, 'rb') as f: return pickle.load(f)
			else:
				return None

		for i, model in enumerate(self._models): _load_module(model, 'models', str(i))

		for i, optimizer in enumerate(self._optimizers): _load_module(optimizer, 'optimizers', str(i))

		history = _load_obj('history')
		if history is not None: self._history = history

		state_dict = _load_obj('state', {})
		for attr, val in state_dict.items(): setattr(self, attr, val)

	def _save(self, dataloader):
		if self._save_path is None: return
		import torch, pickle

		subpaths = ['models', 'optimizers']
		for subpath in subpaths: (self._save_path / subpath).mkdir(parents=True, exist_ok=True)

		def _save_module(module, subpath, name_alternative):
			name = name_alternative if not hasattr(module, 'name') else optimizer.name
			filepath = self._save_path / subpath / (name + '.pt')
			torch.save(module.state_dict(), filepath)

		def _save_obj(object, name):
			with open(self._save_path / (name + '.p'), 'wb') as f: pickle.dump(obj, f)

		for i, model in enumerate(self._models): _save_module(model, 'models', str(i))

		for i, optimizer in enumerate(self._optimizers): _save_module(optimizer, 'optimizers', str(i))

		_save_obj(self._history, 'history')

		state_dict = {attr: getattr(self, attr) for attr in ('_batches_per_epoch', )}
		_save_obj(state_dict, 'state')

		dataloader['train'].save_state_dict(self._save_path / 'dl_train.p')
		dataloader['val'].save_state_dict(self._save_path / 'dl_val.p')

class SupervisedTrainer(Trainer):
	def __init__(self, model, data, loss, optimizer='adam', metrics=None, save_path=None):
		optimizers = [self._get_optimizer(model, optimizer)]
		self._loss = loss
		self._set_metrics(metrics)
		super().__init__([model], data, optimizers, save_path)

	def _optimize(self, dataloader, batch):
		model = self._models[0]; optimizer = self._optimizers[0]

		loss = self._get_loss(dataloader)

		if model.training:
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

	def _validate(self, dataloader, validation_batches):
		for _ in range(validation_batches): self._get_loss(dataloader, validation=True)

	def _get_loss(self, dataloader, validation=False):
		model = self._models[0]; loss_fn = self._loss

		x, y = next(dataloader)
		y_pred = model(x)

		loss = loss_fn(y_pred, y)

		self._history.append('loss', loss.item(), validation=validation, buffer=True)
		for k in self._metrics.keys():
			self._history.append(k, self._metrics[k](y_pred, y).item(), validation=validation, buffer=True)

		return loss

	def _get_optimizer(self, model, optimizer):
		from torch import optim

		if optimizer == 'adam':
			return optim.Adam(model.parameters(), amsgrad=True)

	def _set_metrics(self, metrics):
		from magnet.training import metrics as metrics_module
		if isinstance(metrics, str): self._metrics = {metrics: getattr(metrics_module, metrics.lower())}
		elif isinstance(metrics, (tuple, list)): self._metrics = {m: getattr(metrics_module, m.lower()) for m in metrics}
		elif isinstance(metrics, dict): self._metrics = metrics

	def _monitor(self, batch, **kwargs):
		super()._monitor(batch, **kwargs)
		progress_bar = kwargs.pop('progress_bar')

		loss = self._history['loss'][-1];
		try:
			val_loss = self._history['val_loss'][-1]
			progress_bar.set_description(f'{loss:.2f}, {val_loss:.2f}', refresh=False)
		except KeyError:
			progress_bar.set_description(f'{loss:.2f}', refresh=False)

class ClassifierTrainer(SupervisedTrainer):
	def __init__(self, model, data, optimizer='adam', save_path=None):
		from torch import nn

		super().__init__(model, data, nn.CrossEntropyLoss(), optimizer, metrics='accuracy', save_path=save_path)
