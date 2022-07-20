import torch

from typing import Optional
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler


def dataloader(dataset: Dataset, is_train: Optional[bool] = False, **kwargs):
    """
    Create the :class:`torch.utils.data.DataLoader` with given parameters.

    Args:
        dataset (Dataset): The dataset to iterate through.
        is_train (bool , optional): A boolean to check whether the training mode is activated or not.
        **kwargs: The dataloader parameters.
        
    Returns:
        An iterable over the given dataset.
        
    """
    shuffle = kwargs.get("shuffle", None)

    if not is_train or shuffle is None or not shuffle:
        sampler = SequentialSampler(data_source=dataset)
    else:
        generator = kwargs.get("generator", None)
        if generator is None:
            generator = torch.Generator(device=torch.device("cpu"))
        sampler = RandomSampler(dataset, generator=generator)

    batch_size = kwargs["batch_size"]
    drop_last = kwargs["drop_last"]
    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=drop_last)

    # Once we set the following arguments below, we must remove them. Otherwise, we will get a ValueError:
    # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
    kwargs.pop("shuffle")
    kwargs.pop("batch_size")
    kwargs.pop("drop_last")

    kwargs["batch_sampler"] = batch_sampler
    dataloader = DataLoader(dataset=dataset, **kwargs)

    return dataloader
