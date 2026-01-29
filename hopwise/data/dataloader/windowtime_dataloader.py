# @Time   : 2025/12/22
# @Author : Giacomo Medda
# @Email  : giacomo.medda@unica.it

"""hopwise.data.dataloader.windowedtime_dataloader
################################################
"""

# ruff: noqa: PLW0602 PLW0603

from collections.abc import Iterator

import torch

from hopwise.data.utils import construct_transform
from hopwise.utils import ModelType

start_iter = False


class WindowedTimeSampler(torch.utils.data.SequentialSampler):
    """A sampler that samples indices based on a fixed time window and yields them batched.
    It is assumed that the data source is sorted by timestamp in ascending order.

    Args:
        data_source (Dataset): The dataset to sample from.
        window_size (int): The size of the time window.

    Example:
        >>> data_source_format = ["timestamp"]
        >>> data_source = [2, 3, 4, 5, 6]
        >>> list(WindowedTimeSampler(data_source=data_source, window_size=3))
        >>> [[0, 1, 2], [3, 4]]
        >>> list(WindowedTimeSampler(data_source=data_source, window_size=2))
        >>> [[0, 1], [2, 3]], [[4]]
    """

    def __init__(self, data_source, window_size):
        if not isinstance(window_size, int) and not isinstance(window_size, float) or window_size <= 0:
            raise ValueError(f"window_size should be a positive number, but got window_size={window_size}")

        self.data_source = data_source
        self.window_size = float(window_size)

    def __iter__(self) -> Iterator[list[int]]:
        source_iter = iter(self.data_source)

        class window_takewhile:
            def __init__(self, source_size):
                self._last = None
                self.iter_indexer = iter(range(source_size))

            def __call__(self, predicate, iterable):
                for i, x in zip(self.iter_indexer, iterable):
                    if predicate(x):
                        yield i
                    else:
                        self._last = i
                        return

            @property
            def last(self):
                if self._last is None:
                    raise AttributeError("No last element stored.")
                ret = self._last
                self._last = None
                return ret

        _takewhile = window_takewhile(len(self.data_source))

        start_window = next(source_iter)
        batch = [start_window, *_takewhile(lambda x: x - start_window < self.window_size, source_iter)]
        while batch:
            yield batch
            try:
                start_window = _takewhile.last
                batch = [start_window, *_takewhile(lambda x: x - start_window < self.window_size, source_iter)]
            except StopIteration:
                batch = []

    def __len__(self) -> int:
        # This is an approximation since we cannot know the exact number of windows
        # without iterating through the entire data source.
        return (len(self.data_source) + self.window_size - 1) // self.window_size


class DistributedWindowedTimeSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self):
        raise NotImplementedError("Distributed sampling is not implemented for WindowedTimeSampler yet.")


class WindowedTimeDataLoader(torch.utils.data.DataLoader):
    """:class:`WindowedTimeDataLoader` is a dataloader that preserves the temporal order of interactions
    within each batch. It samples a window of interactions based on a fixed time window.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
    """

    def __init__(self, config, dataset, sampler):
        self.config = config
        self._dataset = dataset
        self._sampler = sampler
        self.sample_size = len(dataset)
        index_sampler = WindowedTimeSampler(list(range(self.sample_size)), window_size=self.config["time_window_size"])

        self.transform = construct_transform(config)
        self.is_sequential = config["MODEL_TYPE"] == ModelType.SEQUENTIAL
        if not config["single_spec"]:
            # TODO: implement distributed sampling for windowed time dataloader
            # time windows should be split among workers such that the same user's interactions
            # are not split across different workers to preserve temporal order
            raise NotImplementedError("Distributed sampling is not implemented for WindowedTimeDataLoader yet.")
            index_sampler = torch.utils.data.distributed.DistributedSampler(
                list(range(self.sample_size)), drop_last=False
            )
            self.step = max(1, self.step // config["world_size"])
        super().__init__(
            dataset=list(range(self.sample_size)),
            collate_fn=self.collate_fn,
            num_workers=config["worker"],
            shuffle=False,
            batch_sampler=index_sampler,
        )

    def update_config(self, config):
        """Update configure of dataloader, such as :attr:`batch_size`, :attr:`step` etc.

        Args:
            config (Config): The new config of dataloader.
        """
        self.config = config

    def collate_fn(self, index):
        index = torch.tensor(index)
        data = self._dataset[index]
        transformed_data = self.transform(self._dataset, data)
        return transformed_data  # self._neg_sampling(transformed_data)

    def __iter__(self):
        global start_iter
        start_iter = True
        res = super().__iter__()
        start_iter = False
        return res

    def __getattribute__(self, __name: str):
        global start_iter
        if not start_iter and __name == "dataset":
            __name = "_dataset"
        return super().__getattribute__(__name)
