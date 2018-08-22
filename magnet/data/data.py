import numpy as np

import magnet as mag

def _get_data_dir():
    import os
    from pathlib import Path

    DIR_DATA = os.environ.get('MAGNET_DATAPATH')
    if DIR_DATA is None:
        raise RuntimeError('You need to have an environment variable called MAGNET_DATAPATH. Add this to your .bashrc file:\nexport MAGNET_DATAPATH=<path>\n'
                            'Where <path> is the desired path where all MagNet datasets are stored by default.')

    return Path(DIR_DATA)

DIR_DATA = _get_data_dir()

from . import core
wiki = {'mnist': core.MNIST}

class Data:
    def __init__(self, **kwargs):
        from .dataloader import pack_collate
        if not hasattr(self, '_name'): self._name = self.__class__.__name__

        self.num_workers = kwargs.pop('num_workers', 0)
        self.collate_fn = kwargs.pop('collate_fn', pack_collate)
        self.pin_memory = kwargs.pop('pin_memory', False)
        self.timeout = kwargs.pop('timeout', 0)
        self.worker_init_fn = kwargs.pop('worker_init_fn', None)
        self.transforms = kwargs.pop('transforms', None)
        self.fetch_fn = kwargs.pop('fetch_fn', None)

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
        from .dataloader import TransformedDataset, DataLoader
        from .sampler import OmniSampler

        dataset = TransformedDataset(self._dataset[mode], self.transforms, self.fetch_fn)
        sampler = OmniSampler(dataset, shuffle, replace, probabilities, sample_space)
        shuffle = False
        batch_sampler = None
        drop_last = False

        num_workers = self.num_workers
        collate_fn = self.collate_fn
        pin_memory = self.pin_memory
        timeout = self.timeout
        worker_init_fn = self.worker_init_fn

        return DataLoader(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
                            collate_fn, pin_memory, drop_last, timeout, worker_init_fn)

    @staticmethod
    def make(train, val=None, test=None, val_split=0.2, **kwargs):
        return _from_dataset(train, val, test, val_split, **kwargs)

    @staticmethod
    def get(name):
        try:
            return wiki[name.lower()]()
        except KeyError as err:
            raise KeyError('No such dataset.') from err

def _from_dataset(train, val=None, test=None, val_split=0.2, **kwargs):
    data = Data(**kwargs)
    data._dataset = {'train': train, 'val': val, 'test': test}
    data._dataset = {k: v for k, v in data._dataset.items() if v is not None}

    if 'val' not in data._dataset.keys(): data._split_val(val_split)

    return data