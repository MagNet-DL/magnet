import numpy as np

from torch.utils.data.dataloader import DataLoader as DataLoaderPyTorch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset as DatasetPyTorch
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

class Dataset(DatasetPyTorch):
    def __init__(self, dataset, transforms=None, get_fn=None):
        self.dataset = dataset
        self.transforms = transforms
        self.get_fn = get_fn

    def __getitem__(self, index):
        x = list(self.dataset[index])
        x = self._apply_transforms(x)
        if self.get_fn is not None: x = self.get_fn(x)
        return x

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
    def __init__(self, dataset, shuffle=False, replace=False, probabilities=None, sample_space=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.replace = replace
        self.probabilities = probabilities
        self.sample_space = sample_space

        self._begin(-1)

    def _begin(self, pos):
        if self.sample_space is None:
            self.indices = list(range(len(self.dataset)))
        elif isinstance(self.sample_space, (list, tuple)):
            self.indices = self.sample_space
        elif isinstance(self.sample_space, int):
            self.indices = list(range(self.sample_space))
        elif isinstance(self.sample_space, float):
            self.indices = list(range(int(self.sample_space * len(self.dataset))))

        if self.shuffle:
            self.indices = np.random.choice(self.indices, len(self),
                                            self.replace, self.probabilities)
        self.pos = pos

    def __next__(self):
        self.pos += 1
        if self.pos >= len(self): self._begin(0)

        return self.indices[self.pos]

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.indices)

class DataLoader(DataLoaderPyTorch):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0, collate_fn=default_collate,
                pin_memory=False, drop_last=False, timeout=0,
                worker_init_fn=None, buffer_size='full'):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,
                        pin_memory, drop_last, timeout, worker_init_fn)

        if buffer_size == 'full': buffer_size = len(self)
        self.buffer_size = buffer_size

        if len(self) == 0:
          raise RuntimeError(f"Batch size too high. Either need more data / sample space or less batch size.\nMaximum allowed batch size is {len(self.sampler)}")

    def state_dict(self):
        sampler = self.sampler
        if sampler.shuffle and sampler.replace: return None

        return {'indices': sampler.indices, 'pos': sampler.pos}

    def save_state_dict(self, path):
        import pickle
        from pathlib import Path
        with open(Path(path), 'wb') as f: pickle.dump(self.state_dict(), f)

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            import pickle
            from pathlib import Path
            path = Path(state_dict)
            if path.exists():
                with open(path, 'rb') as f: state_dict = pickle.load(f)
            else: return

        self.sampler.indices = state_dict['indices']
        self.sampler.pos = state_dict['pos']

    def compatible_with(self, dataloader):
        return self.batch_size == dataloader.batch_size and self.sampler.shuffle == dataloader.sampler.shuffle and self.sampler.replace == dataloader.sampler.replace and self.sampler.sample_space == dataloader.sampler.sample_space

    def __next__(self):
        return next(iter(self))

    def __len__(self):
        return int(len(self.sampler) // self.batch_size)

class Data:
    def __init__(self, path=None, **kwargs):
        if not hasattr(self, '_name'): self._name = self.__class__.__name__

        if path is None: path = DIR_DATA
        self._path = path / self._name
        self._path.mkdir(parents=True, exist_ok=True)
        self._num_workers = kwargs.pop('num_workers', 0)
        self._collate_fn = kwargs.pop('collate_fn', default_collate)
        self._pin_memory = kwargs.pop('pin_memory', False)
        self._timeout = kwargs.pop('timeout', 0)
        self._worker_init_fn = kwargs.pop('worker_init_fn', None)
        self._transforms = kwargs.pop('transforms', None)
        self._get_fn = kwargs.pop('get_fn', None)

        self._download()
        self._preprocess()

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

    def __call__(self, batch_size=1, shuffle=False, replace=False, probabilities=None, sample_space=None, mode='train'):
        dataset = Dataset(self._dataset[mode], self._transforms, self._get_fn)
        sampler = OmniSampler(dataset, shuffle, replace, probabilities, sample_space)
        shuffle = False
        batch_sampler = None
        drop_last = False

        num_workers = self._num_workers
        collate_fn = self._collate_fn
        pin_memory = self._pin_memory
        timeout = self._timeout
        worker_init_fn = self._worker_init_fn

        return DataLoader(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
                            collate_fn, pin_memory, drop_last, timeout, worker_init_fn)

    @staticmethod
    def make(train, val=None, test=None, val_split=0.2, **kwargs):
        return _from_dataset(train, val, test, val_split, **kwargs)

def _from_dataset(train, val=None, test=None, val_split=0.2, **kwargs):
    data = Data(**kwargs)
    data._dataset = {'train': train, 'val': val, 'test': test}
    data._dataset = {k: v for k, v in data._dataset.items() if v is not None}

    if 'val' not in data._dataset.keys(): data._split_val(val_split)

    return data

def MNIST(val_split=0.2, path=DIR_DATA, **kwargs):
    from torchvision import transforms
    from torchvision.datasets import mnist

    dataset = {mode: mnist.MNIST(path, train=(mode == 'train'), download=True)
                        for mode in ('train', 'test')}
    transforms = kwargs.pop('transforms', transforms.Compose([transforms.ToTensor(), transforms.Normalize(*[[0.5] * 3] * 2)]))
    return Data.make(**dataset, val_split=val_split, transforms=transforms)

wiki = {'mnist': MNIST}

def get_data(name):
    try:
        return wiki[name.lower()]()
    except KeyError as err:
        raise KeyError('No such dataset.') from err