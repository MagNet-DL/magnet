class Trainer:
	def __init__(self):
		self._history = History(batch=0)

	def train(self, iterations=1, monitor_freq=1):
		self._on_training_start()

		for batch in range(iterations):
			try:
				if not batch % self._batches_per_epoch:
					self._on_epoch_start(int(batch // self._batches_per_epoch))
			except AttributeError: pass

			self._in_batch(batch)

			if not batch % monitor_freq: self._history.free_buffers()

			try:
				if not (batch + 1) % self._batches_per_epoch:
					self._on_epoch_end(int(batch // self._batches_per_epoch))
			except AttributeError: pass
			self._history['batch'] += 1

		self._on_training_end()

	def _on_training_start(self):
		pass

	def _on_epoch_start(self, epoch):
		pass

	def _in_batch(self, batch):
		pass

	def _on_epoch_end(self, epoch):
		pass

	def _on_training_end(self):
		pass

class SimpleTrainer(Trainer):
	def __init__(self, model, data, loss, optimizer='adam'):
		super().__init__()

		self._model = model
		self._data = data
		self._loss = loss
		self._optimizer = self._get_optimizer(optimizer)

		self._history['loss'] = []

	def show_history(self):
		self._history.show('loss', log=True)

	def _on_training_start(self):
		self._dl = iter(self._data())
		self._batches_per_epoch = len(self._dl)
		self._history['buffer'] = {'loss': []}

	def _in_batch(self, batch):
		loss = self._get_loss(self._dl)
		loss.backward()
		self._optimizer.step()
		self._optimizer.zero_grad()

		self._history['buffer']['loss'].append(loss.item())

	def _get_loss(self, dataloader):
		x, y = next(self._dl)
		return self._loss(self._model(x), y)

	def _get_optimizer(self, optimizer):
		from torch import optim

		if optimizer == 'adam':
			return optim.Adam(self._model.parameters(), amsgrad=True)

class History(dict):
	def show(self, key, log=False):
		from matplotlib import pyplot as plt

		x = self[key]
		if len(x) == 0: return
		if len(x) == 1: print(key, '=', x)

		plt.plot(x)
		if log: plt.yscale('log')
		plt.title(key.title())
		plt.show()

	def free_buffers(self):
		mean = lambda x: sum(x) / len(x)

		try:
			for k, v in self['buffer'].items():
				if type(v) is not list: continue

				try:
					self[k].append(mean(v))
					self['buffer'][k] = []
				except KeyError: pass
		except KeyError: pass
