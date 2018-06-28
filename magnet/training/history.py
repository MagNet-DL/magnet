class History(dict):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.buffer = {}

	def append(self, key, value):
		try:
			self.buffer[key].append(value)
		except KeyError:
			self.buffer[key] = [value]

	def show(self, key, log=False):
		from matplotlib import pyplot as plt

		x = self[key]
		if len(x) == 0: return
		if len(x) == 1: print(key, '=', x)

		plt.plot(x)
		if log: plt.yscale('log')
		plt.title(key.title())
		plt.show()

	def flush(self, key=None):
		if key is None:
			for k in self.buffer.keys(): self.flush(k)
			return

		value = self.buffer[key]
		value = sum(value) / len(value) # Mean

		try: self[key].append(value)
		except KeyError: self[key] = [value]
		finally: self.buffer[key] = []