import numpy as np

from torch.utils.data.sampler import Sampler

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