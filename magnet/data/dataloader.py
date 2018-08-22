import torch, collections

from torch.utils.data.dataloader import DataLoader as DataLoaderPyTorch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset

import magnet as mag
from magnet.utils.varseq import pack

class TransformedDataset(Dataset):
    def __init__(self, dataset, transforms=None, fetch_fn=None):
        self.dataset = dataset
        self.transforms = transforms
        self.fetch_fn = fetch_fn

    def __getitem__(self, index):
        x = list(self.dataset[index])
        x = self._apply_transforms(x)
        if self.fetch_fn is not None: x = self.fetch_fn(x)
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

def pack_collate(batch, pack_dims=None):
    def len_tensor(tensor):
        if len(tensor.shape) == 0: return 1
        return len(tensor)

    if torch.is_tensor(batch[0]):
        if pack_dims is True:
            batch, order = pack(batch)
            return batch.to(mag.device), order

        return default_collate(batch).to(mag.device)

    if pack_dims == 'all': pack_dims = list(range(len(batch[0])))
    elif pack_dims is None: pack_dims = []

    if isinstance(batch[0], collections.Mapping):
        return {key: pack_collate([d[key] for d in batch], i in pack_dims) for i, key in enumerate(batch[0])}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [pack_collate(samples, i in pack_dims) for i, samples in enumerate(transposed)]

    return default_collate(batch)