import sys
import logging
import os.path as osp


def create_logger(name, output_dir=None, log_filename="logs"):
    """
    Create a logger that will help to track events.
    
    Args:
        name: The logger's name.
        output_dir: The output directory where a txt file will be saved.
        log_filename: The txt file name.

    Returns:

    """
    # logging settings
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)-5.5s: %(message)s")
    root_logger = logging.getLogger(name)
    root_logger.setLevel(logging.DEBUG)
    
    # file handler
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
