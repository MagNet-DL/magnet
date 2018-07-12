class Callbacks:
	def __init__(self):
		self.callbacks = {}

	def add(self, name, callback, phases):
		if not isinstance(phases, (tuple, list)): phases = (phases, )

		if self.exists(name): return

		for phase in phases:
			try:
				self.callbacks[phase][name] = callback
			except KeyError:
				self.callbacks[phase] = {name: callback}

	def find(self, name):
		found_callback = None
		phases = []

		for phase, callbacks in self.callbacks.items():
			if name in callbacks.keys():
				found_callback = callbacks[name]
				phases.append(phase)

		return found_callback, phases

	def exists(self, name):
		return self.find(name)[0] is not None

	def __call__(self, phase, *args, **kwargs):
		try: callbacks = self.callbacks[phase]
		except KeyError: return

		for callback in callbacks: callback(*args, **kwargs)