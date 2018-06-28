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