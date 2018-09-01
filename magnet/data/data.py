def _get_data_dir():
    import os, warnings
    from pathlib import Path

    DIR_DATA = os.environ.get('MAGNET_DATAPATH', '~/.data')
    if DIR_DATA is None:
        warnings.warn('You need to have an environment variable called MAGNET_DATAPATH. Add this to your .bashrc file:\nexport MAGNET_DATAPATH=<path>\n'
                            'Where <path> is the desired path where all MagNet datasets are stored by default.', RuntimeError)

    DIR_DATA = Path(DIR_DATA).expanduser()
    DIR_DATA.mkdir(parents=True, exist_ok=True)
    return DIR_DATA

DIR_DATA = _get_data_dir()

from . import core
wiki = {'mnist': core.MNIST}

class Data:
    r"""A container which holds the Training, Validation
    and Test Sets and provides DataLoaders on call.

    This is a convenient abstraction which is used
    downstream with the Trainer and various debuggers.

    It works in tandem with the custom Dataset, DataLoader and Sampler
    sub-classes that MagNet defines.

    Args:
        train (``Dataset``): The training set
        val (``Dataset``): The validation set. Default: ``None``
        test (``Dataset``): The test set. Default: ``None``
        val_split (float): The fraction of training data to hold out
            as validation if validation set is not given. Default: ``0.2``

    Keyword Args:
        num_workers (int): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: ``0``
        collate_fn (callable): merges a list of samples to form a mini-batch
            Default: :py:meth:`pack_collate`
        pin_memory (bool): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them. Default: ``False``
        timeout (numeric): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. Default: ``0``
        worker_init_fn (callable): If not ``None``, this will be called on each
            worker subprocess with the worker id
            (an int in ``[0, num_workers - 1]``) as input, after seeding and
            before data loading. Default: ``None``
        transforms (list or callable): A list of transforms to be applied to
            each datapoint. Default: ``None``
        fetch_fn (callable): A function which is applied to each datapoint
            before collating. Default: ``None``
    """
    def __init__(self, train, val=None, test=None, val_split=0.2, **kwargs):
        from .dataloader import pack_collate
        if not hasattr(self, '_name'): self._name = self.__class__.__name__

        self._dataset = {'train': train, 'val': val, 'test': test}
        self._dataset = {k: v for k, v in self._dataset.items() if v is not None}

        if 'val' not in self._dataset.keys(): self._split_val(val_split)

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
        r"""Returns a MagNet DataLoader that iterates over the dataset.

        Args:
            batch_size (int): How many samples per batch to load. Default: ``1``
            shuffle (bool): Set to ``True`` to have the data reshuffled
                at every epoch. Default: ``False``
            replace (bool): If ``True`` every datapoint can be resampled per
                epoch. Default: ``False``
            probabilities (list or numpy.ndarray): An array of probabilities
                of drawing each member of the dataset. Default: ``None``
            sample_space (float or int or list): The fraction / length / indices
                of the sample to draw from. Default: ``None``
            mode (str): One of [``'train'``, ``'val'``, ``'test'``].
                Default: ``'train'``
        """
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
    def get(name):
        try:
            return wiki[name.lower()]()
        except KeyError as err:
            raise KeyError('No such dataset.') from err