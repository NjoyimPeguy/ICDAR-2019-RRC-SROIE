import os
import torch
import logging


class Checkpointer:
    def __init__(self, output_dir, logger=None):
        """
        An abstract implementation of a save which contains information about the current state of your experiment
        so that you can resume from this point later on.

        Args:
            output_dir: The directory of the save.
            logger: An event tracker.

        """
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.output_dir = output_dir
        self.extension = ".pth"
    
    def save(self, name: str = "current_checkpoint", is_best: bool = False, data: dict = None):
        """
        Save a checkpoint by name.
        
        Args:
            name: The name of the checkpoint.
            is_best: A boolean indicating whether it is the best checkpoint or not.
            data: The data to save.

        """
        if data is None or len(data) == 0:
            self.logger.info("Cannot save empty data!")
        else:
            if not is_best:
                path_to_last_checkpoint = os.path.join(self.output_dir, "".join([name, self.extension]))
                torch.save(data, path_to_last_checkpoint)
                self.logger.info("Saving a checkpoint to {}...\n\n".format(path_to_last_checkpoint))
            else:
                path_to_best_checkpoint = os.path.join(self.output_dir, "".join([name, self.extension]))
                torch.save(data, path_to_best_checkpoint)
                self.logger.info("Saving the best checkpoint to {}...\n\n".format(path_to_best_checkpoint))
    
    def load(self, file_to_load, map_location):
        """
        Load a specific checkpoint.
        
        Args:
            file_to_load: The path to the checkpoint file.
            map_location: A string or a dict specifying how to remap storage locations.

        Returns:
            A dictionary containing the saved objects.
            
        """
        self.logger.info("Loading checkpoint from {}".format(file_to_load))
        
        saved_checkpoint = torch.load(file_to_load, map_location=map_location)
        
        return saved_checkpoint
