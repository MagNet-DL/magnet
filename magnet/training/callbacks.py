import magnet as mag
import torch

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
			from magnet.utils.misc import get_tqdm; tqdm = get_tqdm()

			if self.show_progress:
				self.progress_bar = tqdm(total=kwargs.pop('total_iterations'), unit_scale=True,
										unit_divisor=len(trainer.dataloader), leave=False)

		elif signal == 'on_batch_start':
			if self.show_progress:
				self.progress_bar.update()
				self.progress_bar.refresh()

		elif signal == 'write_stats':
			self.history.append(**kwargs)

		elif signal == 'on_batch_end' and trainer.iterations != 0:
			batches_per_epoch = len(trainer.dataloader)
			if trainer.iterations % int(batches_per_epoch // self.frequency): return

			self.history.flush(iterations=trainer.iterations, epochs=trainer.epochs())

			if not self.show_progress or 'loss' not in self.history.keys(): return

			if 'val_loss' in self.history.keys():
				description = f"{self.history['loss'][-1]:.2f}, {self.history['val_loss'][-1]:.2f}"
			else:
				description = f"{self.history['loss'][-1]:.2f}"
			self.progress_bar.set_description(description, refresh=False)

		elif signal == 'on_training_end' and self.show_progress:
			self.progress_bar.close()
			self.progress_bar = None

		elif signal == 'load_state':
			self.load_state(kwargs.pop('path'))

		elif signal == 'save_state':
			self.save_state(kwargs.pop('path'))

	def load_state(self, path):
		from magnet.training.utils import load_object
		self.history = load_object(path / self.name / 'history.p', default=self.history)

	def save_state(self, path):
		from magnet.training.utils import save_object
		save_object(self.history, path / self.name / 'history.p')

class Validate:
	def __init__(self, dataloader, validate, frequency=10, batches=None, drop_last=False, **kwargs):
		self.name = kwargs.pop('name', 'validate')
		self.dataloader = dataloader
		self.frequency = frequency
		self.batches = batches
		self.drop_last = drop_last
		self.validate = validate

	def __call__(self, trainer, signal, **kwargs):
		if signal == 'on_training_start':
			if self.batches is None: self.batches = int(len(self.dataloader) // self.frequency)

		elif signal == 'on_batch_end' and trainer.iterations != 0:
			batches_per_epoch = len(trainer.dataloader)
			if not trainer.iterations % int(batches_per_epoch // self.frequency): self.validate_batch(trainer)

		elif signal == 'on_training_end':
			if not self.drop_last: self.validate_batch(trainer)

		elif signal == 'load_state':
			self.load_state(kwargs.pop('path'))

		elif signal == 'save_state':
			self.save_state(kwargs.pop('path'))

	def validate_batch(self, trainer):
		with mag.eval(*trainer.models):
			for _ in range(self.batches): self.validate(trainer, self.dataloader)

	def load_state(self, path):
		from magnet.training.utils import load_object
		state_dict = load_object(path / self.name / 'dataloader.p', default=None)
		if state_dict is not None: self.dataloader.load_state_dict(state_dict)

	def save_state(self, path):
		from magnet.training.utils import save_object
		save_object(self.dataloader.state_dict(), path / self.name / 'dataloader.p')

class Checkpoint:
	def __init__(self, path, interval='5 m', **kwargs):
		self.name = kwargs.pop('name', 'checkpoint')
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
			trainer.load_state(self.path)
			self.start_time = time()

		elif signal == 'on_batch_end' and trainer.iterations != 0 and time() - self.start_time > self.interval:
			trainer.save_state(self.path)
			self.start_time = time()

		elif signal == 'on_training_end':
			trainer.save_state(self.path)

		elif signal == 'load_state':
			self.load_state(trainer, kwargs.pop('path'))

		elif signal == 'save_state':
			self.save_state(trainer, kwargs.pop('path'))

	def clear_state(self):
		from shutil import rmtree
		rmtree(self.path)

	def load_state(self, trainer, path):
		from magnet.training.utils import load_object
		state_dict = load_object(path / self.name / 'dataloader.p', default=None)
		if state_dict is not None: trainer.dataloader.load_state_dict(state_dict)

	def save_state(self, trainer, path):
		from magnet.training.utils import save_object
		save_object(trainer.dataloader.state_dict(), path / self.name / 'dataloader.p')

class ColdStart:
	def __init__(self, epochs=0.1, **kwargs):
		self.name = kwargs.pop('name', 'checkpoint')
		self.epochs = epochs
		self.iterations = kwargs.pop('iterations', None)

	def __call__(self, trainer, signal, **kwargs):
		if signal == 'on_training_start':
			torch.no_grad()
			for model in trainer.models: model.eval()

			if self.iterations is None: self.iterations = int(self.epochs * len(trainer.dataloader))

		elif signal == 'on_batch_end' and trainer.iterations == self.iterations - 1:
			torch.enable_grad()
			for model in trainer.models: model.train()
			trainer.callbacks.remove(self)

class LRScheduler:
	def __init__(self, scheduler, **kwargs):
		self.name = kwargs.pop('name', 'lr_scheduler')
		self.scheduler = scheduler

	def __call__(self, trainer, signal, **kwargs):
		if signal == 'on_batch_start' and trainer.epochs('start'): self.scheduler.step()

	def load_state(self, path):
		from magnet.training.utils import load_state
		load_state(self.scheduler, path / self.name, alternative_name='scheduler')

	def save_state(self, path):
		from magnet.training.utils import save_state
		save_state(self.scheduler, path / self.name, alternative_name='scheduler')

class CallbackQueue(list):
	def append(self, callback):
		if not self.exists(callback.name): super().append(callback)

	def extend(self, callbacks):
		super().extend([callback for callback in callbacks if not self.exists(callback.name)])

	def find(self, name):
		callbacks = [callback for callback in self if callback.name == name]
		if len(callbacks) == 0: return None
		if len(callbacks) == 1: return callbacks[0]
		raise RuntimeError('Multiple callbacks with the same name found!')

	def exists(self, name):
		return self.find(name) is not None

	def __call__(self, signal, *args, **kwargs):
		for callback in self: callback(*args, **kwargs, signal=signal)