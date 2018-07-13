import magnet as mag

from time import time

class Monitor:
	def __init__(self, frequency=10, show_progress=True, **kwargs):
		from magnet.training.history import History

		self.name = kwargs.pop('name', 'monitor')
		self.frequency = frequency
		self.show_progress = show_progress

		self.history = History()

	def __call__(self, trainer, signal, **kwargs):
		if signal == 'on_training_start':
			from magnet._utils import get_tqdm; tqdm = get_tqdm()

			self.history.buffer_size = trainer.dataloader.buffer_size

			self.progress_bar = tqdm(total=kwargs.pop('total_iterations'), unit_scale=True, unit_divisor=len(trainer.dataloader), leave=False) if self.show_progress else None

		elif signal == 'on_batch_start':
			if self.show_progress:
				self.progress_bar.update()
				self.progress_bar.refresh()

		elif signal == 'before_optimization':
			self.history.append(**kwargs)

		elif signal == 'on_batch_end':
			if trainer.iterations == 0: return

			batches_per_epoch = len(trainer.dataloader)
			if trainer.iterations % int(batches_per_epoch // self.frequency): return

			self.history.flush(iterations=trainer.iterations, epochs=trainer.epochs())
			if self.show_progress:
				self.progress_bar.set_description(f"{self.history['loss'][-1]:.2f},"
			 									f"{self.history['val_loss'][-1]:.2f}", refresh=False)

		elif signal == 'on_training_end':
			if self.show_progress:
				self.progress_bar.close()
				self.progress_bar = None

		elif signal == 'load':
			from magnet.training.utils import load_object
			self.history = load_object(kwargs.pop('path') / self.name / 'history.p', default=self.history)

		elif signal == 'save':
			from magnet.training.utils import save_object
			save_object(self.history, kwargs.pop('path') / self.name / 'history.p')

class Validate:
	def __init__(self, dataloader, frequency=10, batches=None, drop_last=False, **kwargs):
		self.name = kwargs.pop('name', 'validate')
		self.dataloader = dataloader
		self.frequency = frequency
		self.batches = batches
		self.drop_last = drop_last

	def __call__(self, trainer, signal, **kwargs):
		if signal == 'on_training_start':
			self.start_iteration = trainer.iterations
			self.total_iterations = kwargs.pop('total_iterations')
			if self.batches is None: self.batches = int(len(self.dataloader) // self.frequency)

		elif signal == 'on_batch_end':
			batches_per_epoch = len(trainer.dataloader)
			not_last_iteration = trainer.iterations != self.start_iteration + self.total_iterations - 1

			if trainer.iterations == 0: return
			if trainer.iterations % int(batches_per_epoch // self.frequency): return
			if not_last_iteration and self.drop_last: return

			with mag.eval(*trainer.models):
				for _ in range(self.batches): trainer.validate(self.dataloader)

		elif signal == 'load':
			from magnet.training.utils import load_object
			state_dict = load_object(kwargs.pop('path') / self.name / 'dataloader.p', default=None)
			if state_dict is not None: self.dataloader.load_state_dict(state_dict)

		elif signal == 'save':
			from magnet.training.utils import save_object
			save_object(self.dataloader.state_dict(), kwargs.pop('path') / self.name / 'dataloader.p')

class Checkpoint:
	def __init__(self, path, interval='5 m', **kwargs):
		self.name = kwargs.pop('name', 'validate')
		self.path = path
		self.interval = self.parse_duration(interval)

	def parse_duration(self, interval):
		interval, multiplier = interval.split(' ')
		interval = float(interval); multiplier = multiplier.lower()
		multiplier_dict = {'m': 60, 's': 1, 'h': 3600, 'ms': 1e-3, 'us': 1e-6, 'd': 24 * 3600}
		multiplier = multiplier_dict[multiplier]
		return interval * multiplier

	def __call__(self, trainer, signal, **kwargs):
		if signal == 'on_training_start':
			self.path.mkdir(parents=True, exist_ok=True)

			trainer.load(self.path)

			self.start_iteration = trainer.iterations
			self.total_iterations = kwargs.pop('total_iterations')
			self.start_time = time()

		elif signal == 'on_batch_end':
			batches_per_epoch = len(trainer.dataloader)
			not_last_iteration = trainer.iterations != self.start_iteration + self.total_iterations - 1

			if trainer.iterations == 0: return
			if not_last_iteration and (time() - self.start_time < self.interval): return

			trainer.save(self.path)
			self.start_time = time()

		elif signal == 'load':
			from magnet.training.utils import load_object
			state_dict = load_object(kwargs.pop('path') / self.name / 'dataloader.p', default=None)
			if state_dict is not None: trainer.dataloader.load_state_dict(state_dict)

		elif signal == 'save':
			from magnet.training.utils import save_object
			save_object(trainer.dataloader.state_dict(), kwargs.pop('path') / self.name / 'dataloader.p')

	def clear(self):
		from shutil import rmtree
		rmtree(self.path)

class CallbackQueue(list):
	def append(self, callback):
		if not self.exists(callback.name): super().append(callback)

	def find(self, name):
		callbacks = [callback for callback in self if callback.name == name]
		if len(callbacks) == 0: return None
		if len(callbacks) == 1: return callbacks[0]
		raise RuntimeError('Multiple callbacks with the same name found!')

	def exists(self, name):
		return self.find(name) is not None

	def __call__(self, signal, *args, **kwargs):
		for callback in self: callback(*args, **kwargs, signal=signal)