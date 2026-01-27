# @Time   : 2025/12/22
# @Author : Giacomo Medda
# @Email  : giacomo.medda@unica.it

"""hopwise.data.dataloader.windowedtime_dataloader
################################################
"""

import itertools
from collections.abc import Iterator

import torch

from hopwise.data.utils import construct_transform
from hopwise.utils import ModelType

start_iter = False


class WindowedTimeSampler(torch.utils.data.SequentialSampler):
    """A sampler that samples indices based on a fixed time window and yields them batched.

    Args:
        data_source (Dataset): The dataset to sample from.
        window_size (int): The size of the time window.

    Example:
        >>> data_source_format = ["user_id", "item_id", "timestamp"]
        >>> data_source = [[0, 1, 2], [0, 2, 3], [1, 2, 4], [1, 3, 5], [0, 3, 6]]
        >>> list(WindowedTimeSampler(data_source=data_source, window_size=3))
        >>> [[0, 1, 2], [0, 2, 3], [1, 2, 4]], [[1, 3, 5], [0, 3, 6]]
        >>> list(WindowedTimeSampler(data_source=data_source, window_size=2))
        >>> [[0, 1, 2], [0, 2, 3]], [[1, 2, 4], [1, 3, 5]], [[0, 3, 6]]
    """

    def __init__(self, data_source, window_size):
        self.data_source = data_source
        self.window_size = window_size
        self.indices = self._generate_indices()

    def _generate_indices(self):
        pass

    # def __init__(
    #     self,
    #     sampler: Union[Sampler[int], Iterable[int]],
    #     batch_size: int,
    #     drop_last: bool,
    # ) -> None:
    #     # Since collections.abc.Iterable does not check for `__getitem__`, which
    #     # is one way for an object to be an iterable, we don't do an `isinstance`
    #     # check here.
    #     if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
    #         raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
    #     if not isinstance(drop_last, bool):
    #         raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
    #     self.sampler = sampler
    #     self.batch_size = batch_size
    #     self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        sampler_iter = iter(self.sampler)
        if self.drop_last:
            # Create multiple references to the same iterator
            args = [sampler_iter] * self.batch_size
            for batch_droplast in zip(*args):
                yield [*batch_droplast]
        else:
            batch = [*itertools.islice(sampler_iter, self.batch_size)]
            while batch:
                yield batch
                batch = [*itertools.islice(sampler_iter, self.batch_size)]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


class DistributedWindowedTimeSampler(torch.utils.data.Sampler):
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
        self._init_batch_size_and_step()
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
        self._init_batch_size_and_step()

    def collate_fn(self):
        """Collect the sampled index, and apply neg_sampling or other methods to get the final data."""
        raise NotImplementedError("Method [collate_fn] must be implemented.")

    def __iter__(self):
        # global start_iter
        # start_iter = True
        res = super().__iter__()
        # start_iter = False
        return res

    def __getattribute__(self, __name: str):
        # global start_iter
        if not start_iter and __name == "dataset":
            __name = "_dataset"
        return super().__getattribute__(__name)
