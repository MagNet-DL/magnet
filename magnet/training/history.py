from magnet.utils.plot import smooth_plot

class History(dict):
	def find(self, key):
		return {k: self[k] for k in self.keys() if key in k}

	def append(self, key, value, validation=False, buffer_size=None, **stamps):
		if validation: key = 'val_' + key

		try:
			self[key].append(value, buffer=(buffer_size is not None), **stamps)
		except KeyError:
			self[key] = SnapShot(buffer_size)
			self[key].append(value, buffer=(buffer_size is not None), **stamps)

	def show(self, key=None, log=False, x_key=None, validation=True, legend=None, **kwargs):
		from matplotlib import pyplot as plt

		ax = kwargs.pop('ax', None)

		if key is None:
			for k in self.keys():
				if 'val_' in k: continue
				self.show(k, 'loss' in k, x_key, validation, legend=k, **kwargs)
				plt.show()
			return

		if ax is None: fig, ax = plt.subplots()
		label = 'training' if legend is None else legend
		self[key].show(ax, x_key, label=label, **kwargs)

		if validation:
			try:
				label = 'validation' if legend is None else legend
				self['val_' + key].show(ax, x_key, label=label)
			except KeyError: pass

		if log: plt.yscale('log')

		plt.ylabel(key.title())
		if isinstance(x_key, str):
			plt.xlabel(x_key)
			plt.title(f'{key.title()} vs {x_key.title()}')
		elif isinstance(x_key, str):
			plt.xlabel(x_key)
			plt.title(f'{key.title()} vs {x_key.title()}')
		else: plt.title(key.title())

		plt.legend()

	def flush(self, key=None, **stamps):
		if key is None:
			for k in self.keys(): self.flush(k, **stamps)
			return

		self[key].flush(**stamps)

class SnapShot:
	def __init__(self, buffer_size=-1):
		self._snaps = []
		if buffer_size is not None:
			self._buffer_size = buffer_size
			self._buffer = SnapShot(buffer_size=None)

	def append(self, value, buffer=False, **stamps):
		if buffer:
			self._buffer.append(value, **stamps)
			if self._buffer_size > 0 and len(self._buffer) > self._buffer_size: self._buffer._pop(0)
			return

		self._snaps.append(dict(val=value, **stamps))

	def flush(self, **stamps):
		if not hasattr(self, '_buffer') or len(self._buffer) == 0: return

		values = self._buffer._retrieve()
		value = sum(values) / len(values)

		self.append(value, **stamps)

		if self._buffer_size < 0: self._buffer._clear()

	def _retrieve(self, key='val', stamp=None):
		if stamp is None: return [snap[key] for snap in self._snaps]
		return list(zip(*[(snap[stamp], snap[key]) for snap in self._snaps if stamp in snap.keys()]))

	def _pop(self, index):
		self._snaps.pop(index)

	def _clear(self):
		self._snaps = []

	def __len__(self):
		return len(self._snaps)

	def __getitem__(self, index):
		return self._snaps[index]['val']

	def show(self, ax, x=None, label=None, **kwargs):
		if x is None:
			x = list(range(len(self)))
			y = self._retrieve()
		else:
			x, y = self._retrieve(stamp=x)

		if len(x) != 0: 
			line, = ax.plot(x, y, alpha=0.3)
			smooth_plot(x, y, label=label, ax=ax, c=line.get_color(), **kwargs)