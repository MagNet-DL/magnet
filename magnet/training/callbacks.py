import magnet as mag

class Monitor:
	def __init__(self, name='monitor', frequency=10, show_progress=True):
		from magnet.training.history import History

		self.name = name
		self.frequency = frequency
		self.show_progress = show_progress

		self.history = History()

	def __call__(self, trainer, signal, **kwargs):
		if signal == 'on_training_start':
			from magnet._utils import get_tqdm; tqdm = get_tqdm()

			self.history.buffer_size = trainer.dataloader['train'].buffer_size
			self.history.val_buffer_size = trainer.dataloader['val'].buffer_size

			self.progress_bar = tqdm(total=kwargs.pop('total_iterations'), unit_scale=True, unit_divisor=len(trainer.dataloader['train']), leave=False) if self.show_progress else None

		elif signal == 'on_batch_start':
			if self.show_progress:
				self.progress_bar.update()
				self.progress_bar.refresh()

		elif signal == 'before_optimization':
			self.history.append(**kwargs)

		elif signal == 'on_batch_end':
			if trainer.iterations == 0: return

			batches_per_epoch = len(trainer.dataloader['train'])
			if trainer.iterations % int(batches_per_epoch // self.frequency): return

			self.history.flush(iterations=trainer.iterations, epochs=trainer.epochs())
			if self.show_progress:
				self.progress_bar.set_description(f"{self.history['loss'][-1]:.2f},"
			 									f"{self.history['val_loss'][-1]:.2f}", refresh=False)

		elif signal == 'on_training_end':
			if self.show_progress:
				self.progress_bar.close()
				self.progress_bar = None

class Validate:
	def __init__(self, name='validate', frequency=10, batches=None, drop_last=False):
		self.name = name
		self.frequency = frequency
		self.batches = batches
		self.drop_last = drop_last

	def __call__(self, trainer, signal, **kwargs):
		if signal == 'on_training_start':
			self.start_iteration = trainer.iterations
			self.total_iterations = kwargs.pop('total_iterations')
			if self.batches is None: self.batches = int(len(trainer.dataloader['val']) // self.frequency)

		if signal == 'on_batch_end':
			batches_per_epoch = len(trainer.dataloader['train'])
			not_last_iteration = trainer.iterations != self.start_iteration + self.total_iterations - 1

			if trainer.iterations == 0: return
			if trainer.iterations % int(batches_per_epoch // self.frequency): return
			if not_last_iteration and self.drop_last: return

			with mag.eval(*trainer.models):
				for _ in range(self.batches): trainer.validate()

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