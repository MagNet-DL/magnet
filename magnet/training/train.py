import magnet as mag

class Trainer:
	def __init__(self, models, data, optimizers):
		self._models = models
		self._data = data
		self._optimizers = optimizers

		from magnet.training.history import History
		self._history = History()

	def _optimize(self, dataloader, batch):
		raise NotImplementedError

	def _validate(self, dataloader):
		pass

	def train(self, iterations=1, monitor_freq=1):
		dataloader = {'train': iter(self._data())}
		dataloader['val'] = iter(self._data(mode='val'))

		self._batches_per_epoch = len(dataloader['train'])

		self._on_training_start()

		for batch in range(iterations):
			try:
				if not batch % self._batches_per_epoch:
					self._on_epoch_start(int(batch // self._batches_per_epoch))
			except AttributeError: pass

			self._on_batch_start(batch)

			self._optimize(dataloader['train'], batch)

			self._on_batch_end(batch)

			if not batch % monitor_freq:
				self._history.append('batches', batch)
				with mag.eval(*self._models): self._validate(dataloader['val'])
				self._history.flush()

			try:
				if not (batch + 1) % self._batches_per_epoch:
					self._on_epoch_end(int(batch // self._batches_per_epoch))
			except AttributeError: pass

		self._on_training_end()

	def show_history(self, vs='batches'):
		xlabel = None

		vs = vs.lower()
		if vs == 'epochs':
			vs = [b / self._batches_per_epoch for b in self._history['batches']]
			xlabel = 'epochs'

		self._history.show('loss', log=True, x_key=vs, xlabel=xlabel)
		for k in self._metrics.keys(): self._history.show(k, x_key=vs, xlabel=xlabel)

	def _on_training_start(self):
		pass

	def _on_epoch_start(self, epoch):
		pass

	def _on_batch_start(self, batch):
		pass

	def _on_batch_end(self, batch):
		pass

	def _on_epoch_end(self, epoch):
		pass

	def _on_training_end(self):
		pass

class SupervisedTrainer(Trainer):
	def __init__(self, model, data, loss, optimizer='adam', metrics=None):
		super().__init__([model], data, optimizers=None)

		self._optimizers = [self._get_optimizer(optimizer)]
		self._loss = loss
		self._set_metrics(metrics)

	def _optimize(self, dataloader, batch):
		model = self._models[0]; loss_fn = self._loss; optimizer = self._optimizers[0]

		x, y = next(dataloader)
		y_pred = model(x)

		loss = loss_fn(y_pred, y)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		self._history.append('loss', loss.item(), buffer=True)
		for k in self._metrics.keys():
			self._history.append(k, self._metrics[k](y_pred, y).item(), buffer=True)

	def _validate(self, dataloader):
		model = self._models[0]; loss_fn = self._loss; optimizer = self._optimizers[0]

		x, y = next(dataloader)
		y_pred = model(x)

		loss = loss_fn(y_pred, y)

		self._history.append('loss', loss.item(), validation=True, buffer=True)
		for k in self._metrics.keys():
			self._history.append(k, self._metrics[k](y_pred, y).item(), validation=True, buffer=True)

	def _get_optimizer(self, optimizer):
		from torch import optim

		if optimizer == 'adam':
			return optim.Adam(self._models[0].parameters(), amsgrad=True)

	def _set_metrics(self, metrics):
		from magnet.training import metrics as metrics_module
		if isinstance(metrics, str): self._metrics = {metrics: getattr(metrics_module, metrics.lower())}
		elif isinstance(metrics, (tuple, list)): self._metrics = {m: getattr(metrics_module, m.lower()) for m in metrics}
		elif isinstance(metrics, dict): self._metrics = metrics

class ClassifierTrainer(SupervisedTrainer):
	def __init__(self, model, data, optimizer='adam'):
		from torch import nn

		super().__init__(model, data, nn.CrossEntropyLoss(), optimizer, metrics='accuracy')
