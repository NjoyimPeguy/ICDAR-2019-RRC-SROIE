import os
import torch

from typing import Optional


class Checkpointer:
    def __init__(self, logger, output_dir=None):
        """
        An abstract implementation of a save which contains information about the current state of your experiment
        so that you can resume from this point later on.

        Args:
            logger: An event tracker.
            output_dir (string or None): The directory of the saving.

        """
        self.logger = logger
        self.output_dir = output_dir
        self.extension = ".pth"

    def save(self,
             name: Optional[str] = "current_checkpoint",
             is_best: Optional[bool] = False,
             data=None):
        """
        Save a checkpoint by name.

        Args:
            name (string, optional): The name of the checkpoint.
            is_best (bool, optional): A boolean indicating whether it is the best checkpoint or not.
            data (dict, optional): The data to save.

        """
        if data is None or len(data) == 0:
            self.logger.info("Cannot save empty data!")
        else:
            if not is_best:
                path_to_last_checkpoint = os.path.join(self.output_dir, "".join([name, self.extension]))
                torch.save(data, path_to_last_checkpoint)
                self.logger.info("Saving a checkpoint to {0}...\n".format(path_to_last_checkpoint))
            else:
                path_to_best_checkpoint = os.path.join(self.output_dir, "".join([name, self.extension]))
                torch.save(data, path_to_best_checkpoint, _use_new_zipfile_serialization=False)
                self.logger.info("Saving the best checkpoint to {0}...\n".format(path_to_best_checkpoint))

    def load(self, file_to_load: str, map_location: torch.device):
        """
        Load a specific checkpoint.

        Args:
            file_to_load (string): The path to the checkpoint path_to_file.
            map_location (device): A string or a dict specifying how to remap storage locations.

        Returns:
            A dictionary containing the saved objects.

        """
        self.logger.info("Loading checkpoint from {}".format(file_to_load))

        saved_checkpoint = torch.load(file_to_load, map_location=map_location)

        return saved_checkpoint
