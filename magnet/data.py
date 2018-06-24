import numpy as np

from torch.utils.data.dataloader import DataLoader as DataLoaderPyTorch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

def _get_data_dir():
	import os
	from pathlib import Path

	DIR_DATA = os.environ.get('MAGNET_DATAPATH')
	if DIR_DATA is None:
		raise RuntimeError('You need to have an environment variable called MAGNET_DATAPATH. Add this to your .bashrc file:\nexport MAGNET_DATAPATH=<path>\n'
							'Where <path> is the desired path where all MagNet datasets are stored by default.')

	return Path(DIR_DATA)

DIR_DATA = _get_data_dir()

class TransformedDataset(Dataset):
	def __init__(self, dataset, transforms=None):
		self.dataset = dataset
		self.transforms = transforms

	def __getitem__(self, index):
		x = list(self.dataset[index])
		return self._apply_transforms(x)

	def __len__(self):
		return len(self.dataset)

	def _apply_transforms(self, x):
		transforms = self.transforms
		if transforms is None: return x

		if not isinstance(transforms, (tuple, list)): transforms = [transforms]

		if len(transforms) > len(x): raise ValueError('Provide a single transform for the first datapoint or a'
														'tuple with each transform applied to the respective datapoint.')

		for i, transform in enumerate(transforms):
			if not isinstance(transform, (tuple, list)):
				x[i] = transform(x[i])
				continue

			x_i = x[i]
			for t in transform: x_i = t(x_i)
			x[i] = x_i

		return x

class OmniSampler(Sampler):
	def __init__(self, dataset, shuffle=False, replace=False, probabilities=None):
		self.dataset = dataset
		self.shuffle = shuffle
		self.replace = replace
		self.probabilities = probabilities

		self._begin()

	def _begin(self):
		self.indices = list(range(len(self)))
		
		if self.shuffle: 
			self.indices = np.random.choice(self.indices, len(self),
											self.replace, self.probabilities)
		self.pos = -1

	def __next__(self):
		self.pos += 1
		if self.pos >= len(self): self._begin()
			
		return self.indices[self.pos]
		
	def __iter__(self):
		return self

	def __len__(self):
		return len(self.dataset)

class DataLoader(DataLoaderPyTorch):
	def state_dict(self):
		sampler = self.sampler
		if sampler.shuffle and sampler.replace: return None

		return {'indices': sampler.indices, 'pos': sampler.pos}

	def load_state_dict(self, state_dict):
		self.sampler.indices = state_dict['indices']
		self.sampler.pos = state_dict['pos']

class Data:
	def __init__(self, path=None):
		if not hasattr(self, '_name'): self._name = self.__class__.__name__

		if path is None: path = DIR_DATA

		self._path = path / self._name

		self._path.mkdir(parents=True, exist_ok=True)

		self._download()
		self._preprocess()

		self._transforms = None

	def _is_downloaded(self):
		return True

	def _download(self):
		if self._is_downloaded(): return
		pass

	def _preprocess(self):
		pass

	def __getitem__(self, args):
		if isinstance(args, int): return self['train'][args]
		elif isinstance(args, str):
			try: return self._dataset[args]
			except KeyError as err:
				if args == 'val': err_msg = "This dataset has no validation set held out! If the constructor has a val_split attribute, consider setting that."
				elif args == 'test': err_msg = 'This dataset has no test set.'
				else: err_msg = "The only keys are 'train', 'val', and 'test'."
				raise KeyError(err_msg) from err

		mode = args[1]
		index = args[0]

		return self[mode][index]

	def __setitem__(self, mode, dataset):
		self._dataset[mode] = dataset

	def __len__(self):
		return len(self['train'])

	def _split_val(self, val_split):
		if isinstance(val_split, int):
			len_val = val_split
			dataset_len = len(self)
			val_ids = list(range(dataset_len - len_val, dataset_len))
			return self._split_val(val_ids)
		elif isinstance(val_split, float) and val_split >= 0 and val_split < 1:
			num_val = int(val_split * len(self))
			return self._split_val(num_val)

		from torch.utils.data.dataset import Subset

		val_ids = set(val_split)
		if len(val_ids) != len(val_split):
			raise ValueError("The indices in val_split should be unique. If you're not super"
							 "pushy, pass in a fraction to split the dataset.")

		total_ids = set(range(len(self['train'])))
		train_ids = list(total_ids - val_ids)

		self['val'] = Subset(self['train'], val_split)
		self['train'] = Subset(self['train'], train_ids)

	def __call__(self, batch_size=1, shuffle=False, replace=False, probabilities=None, num_workers=0, collate_fn=default_collate,
				pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, transforms=None, mode='train'):
		if transforms is None: transforms = self._transforms

		dataset = TransformedDataset(self._dataset[mode], transforms) 
		sampler = OmniSampler(dataset, shuffle, replace, probabilities)
		shuffle = False
		batch_sampler = None
		
		return DataLoader(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
							collate_fn, pin_memory, drop_last, timeout, worker_init_fn)

class MNIST(Data):
	def __init__(self, val_split=0.2, path=None):
		from torchvision.datasets import mnist
		from torchvision.transforms import ToTensor

		super().__init__(path)

		self._dataset = {mode: mnist.MNIST(self._path, train=(mode == 'train'), download=True)
						for mode in ('train', 'test')}

		if val_split is not None: self._split_val(val_split)

		self._transforms = ToTensor()

_data_wiki = {'mnist': MNIST}

def get_data(name):
	try:
		return _data_wiki[name.lower()]()
	except KeyError as err:
		raise KeyError('No such dataset.') from err