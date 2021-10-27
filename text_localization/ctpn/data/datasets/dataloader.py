from text_localization.ctpn.samplers import IterationBatchSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler


def create_dataloader(dataset, is_train: bool = False, start_iteration=1, max_iterations=None, **kwargs):
    """
    Create the :class:`torch.utils.data.DataLoader` with given parameters.

    Args:
        dataset: The dataset to iterate through.
        is_train:
        start_iteration: The starting iteration of this dataloader.
        max_iterations: The end iteration of this dataloader.
        **kwargs: The dataloader parameters.
        
    Returns:
        An iterable over the given dataset.
        
    """
    if is_train and kwargs["shuffle"]:
        sampler = RandomSampler(dataset, generator=kwargs["generator"])
    else:
        sampler = SequentialSampler(data_source=dataset)
        # Normally, one does not need this line below if it were set False in the yaml configs,
        # but just in case. Otherwise the dataloader will complain about.
        kwargs["shuffle"] = False
    
    batch_size = kwargs["batch_size"]
    drop_last = kwargs["drop_last"]
    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
    
    # Once we set the following arguments below, we must remove them.
    # Otherwise, we will get a ValueError: batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
    kwargs.pop("shuffle")
    kwargs.pop("batch_size")
    kwargs.pop("drop_last")
    if max_iterations is not None:
        batch_sampler = IterationBatchSampler(batch_sampler=batch_sampler,
                                              start_iteration=start_iteration,
                                              max_iterations=max_iterations)
    kwargs["batch_sampler"] = batch_sampler
    dataloader = DataLoader(dataset=dataset, **kwargs)
    return dataloader
