import datetime as dt


def get_process_time(start_time: float, elapsed_time: float, current_epoch: int, max_epochs: int):
    """
    Calculate the remaining and ETA times.

    Args:
        start_time (float): The starting time.
        elapsed_time (float): The elapsed time.
        current_epoch (int): The current epoch.
        max_epochs (int): The maximum number of epochs.
    Returns:
        A tuple containing the remaining and the ETA times.

    """

    estimated_time = (elapsed_time / current_epoch) * max_epochs

    remaining_time = estimated_time - elapsed_time

    finish_time = str(dt.datetime.fromtimestamp(start_time + estimated_time).strftime("%Y/%m/%d at %H:%M:%S"))

    times = (int(remaining_time), finish_time)

    return times
