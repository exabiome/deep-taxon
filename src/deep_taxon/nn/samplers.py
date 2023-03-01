import numpy as np
import torch
from torch.utils.data import Sampler, SequentialSampler


def check_rng(rng):
    if rng is None:
        return np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        return np.random.default_rng(rng)
    return rng


class WORHelper:

    def __init__(self, length, rng=None):
        self.period = length
        self.rng = check_rng(rng)
        self.indices = np.arange(length)
        self.curr_max = length - 1

    def sample(self):
        idx = self.rng.integers(self.curr_max + 1)
        ret = self.indices[idx]
        end = self.curr_max
        self.indices[end], self.indices[idx] = self.indices[idx], self.indices[end]
        self.curr_max = (self.curr_max - 1) % self.period
        return ret


class WORSampler(Sampler):
    """Without Replacement Sampler"""

    def __init__(self, length, rng=None, n_partitions=1, part_smplr_rng=None, max_samples=None):
        """
        Args:
            n_partitions :          number of deterministic partitions to break up dataset into
            part_smplr_rng :        Partition Sampler RNG
        """
        super().__init__(None)
        self.rng = check_rng(rng)
        dtype = np.uint32
        if length > (2**32 - 1):
            dtype = np.uint64


        # trim will clip extra samples (i.e. length % size) so that each
        # rank has the same number of samples.
        # Use this later if we decide we don't want to trim tail.
        self.indices = np.arange(length, dtype=dtype)

        self.part_sampler = WORHelper(n_partitions, rng=part_smplr_rng)

        # set an initial value for curr_part (i.e. current partition)
        # so we can calculate lenght in __len__
        self.curr_part = self.part_sampler.sample()

        self.__len = (len(self.indices) - self.curr_part - 1) // self.part_sampler.period + 1
        if max_samples is not None:
            self.__len = min(self.__len, max_samples)
        self.max_samples = max_samples
        self.i = self.__len

        self.curr_len = 0

    def __iter__(self):
        self.curr_len = (len(self.indices) - self.curr_part - 1) // self.part_sampler.period + 1 # len(self)
        if self.max_samples is not None:
            self.i = 0
        return self

    def __len__(self):
        return self.__len

    def __next__(self):
        if self.max_samples is not None:
            if self.i < self.__len:
                self.i += 1
            else:
                raise StopIteration

        if self.curr_len == 0:
            self.curr_part = self.part_sampler.sample()
            raise StopIteration

        idx = self.rng.integers(self.curr_len) * self.part_sampler.period + self.curr_part
        end = (self.curr_len - 1) * self.part_sampler.period + self.curr_part
        ret = self.indices[idx]
        self.indices[end], self.indices[idx] = self.indices[idx], self.indices[end]
        self.curr_len -= 1
        return ret


class DSSampler(SequentialSampler):
    """Distributed Sequential Sampler"""

    def __init__(self, length, n_partitions=1, part_smplr_rng=None, max_samples=None):
        """
        Args:
            n_partitions :          number of deterministic partitions to break up dataset into
            part_smplr_rng :        Partition Sampler RNG
        """
        super().__init__(None)
        self.part_sampler = WORHelper(n_partitions, rng=part_smplr_rng)

        # set an initial value for curr_part (i.e. current partition)
        # so we can calculate lenght in __len__
        self.curr_part = self.part_sampler.sample()
        self.start = 0
        self.end = length
        self.__len = (self.end - self.start - self.curr_part - 1) // self.part_sampler.period + 1
        if max_samples is not None:
            self.__len = min(self.__len, max_samples)
        self.max_samples = max_samples
        self.i = self.__len


    def __iter__(self):
        self.__it = iter(range(self.start + self.curr_part, self.end, self.part_sampler.period))
        if self.max_samples is not None:
            self.i = 0
        return self

    def __len__(self):
        return self.__len

    def __next__(self):
        try:
            if self.max_samples is not None:
                if self.i < self.__len:
                    self.i += 1
                else:
                    raise StopIteration
            return next(self.__it)
        except StopIteration:
            self.curr_part = self.part_sampler.sample()
            raise StopIteration
