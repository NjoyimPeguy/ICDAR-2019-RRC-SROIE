import sys
import logging
import os.path as osp

from typing import Optional, Union


def logger(name: str, output_dir: Union[str, None] = None, log_filename: Optional[str] = "logs"):
    """
    Create a logger that will help to track events.

    Args:
        name (string): The logger's name.
        output_dir (string, none): The output directory where a txt path_to_file will be saved.
        log_filename (string, optional): The txt path_to_file name.

    Returns:
        A logger.
    """

    # logging settings
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)-5.5s: %(message)s")
    root_logger = logging.getLogger(name)
    root_logger.setLevel(logging.DEBUG)

    # path_to_file handler
    if output_dir is not None:
        fh = logging.FileHandler(osp.join(output_dir, "".join([log_filename, ".txt"])))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    # stream handler (stdout)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    return root_logger
