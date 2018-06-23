def _get_data_dir():
	import os
	from pathlib import Path

	DIR_DATA = os.environ.get('MAGNET_DATAPATH')
	if DIR_DATA is None:
		raise RuntimeError('You need to have an environment variable called MAGNET_DATAPATH. Add this to your .bashrc file:\nexport MAGNET_DATAPATH=<path>\n'
							'Where <path> is the desired path where all MagNet datasets are stored by default.')

	return Path(DIR_DATA)

DIR_DATA = _get_data_dir()

class Data:
	def __init__(self, path=None):
		if not hasattr(self, '_name'): self._name = self.__class__.__name__

		if path is None: path = DIR_DATA

		self._path = path / self._name

		self._path.mkdir(parents=True, exist_ok=True)

		self._download()
		self._preprocess()

	def _is_downloaded(self):
		return True

	def _download(self):
		if self._is_downloaded(): return
		pass

	def _preprocess(self):
		pass

	def __call__(self, mode='train'):
		raise NotImplementedError()

class MNIST(Data):
	def __init__(self, path=None, val_split=None):
		super().__init__(path)

		from torchvision.datasets import mnist

		self._dataset = {mode: mnist.MNIST(self._path, train=(mode == 'train'), download=True)
						for mode in ('train', 'test')}

		if val_split is not None: self._split_val(val_split)

	def _split_val(self, val_split):
		if isinstance(val_split, int):
			len_val = val_split
			dataset_len = len(self._dataset['train'])
			val_ids = list(range(dataset_len - len_val, dataset_len))
			return self._split_val(val_ids)
		elif isinstance(val_split, float) and val_split >= 0 and val_split < 1:
			num_val = int(val_split * len(self._dataset['train']))
			return self._split_val(num_val)

		from torch.utils.data.dataset import Subset

		val_ids = set(val_split)
		if len(val_ids) != len(val_split):
			raise ValueError("The indices in val_split should be unique. If you're not super"
							 "pushy, pass in a fraction to split the dataset.")

		total_ids = set(range(len(self._dataset['train'])))
		train_ids = list(total_ids - val_ids)

		self._dataset['val'] = Subset(self._dataset['train'], val_split)
		self._dataset['train'] = Subset(self._dataset['train'], train_ids)

	def __call__(self, mode='train'):
		return self._dataset[mode]