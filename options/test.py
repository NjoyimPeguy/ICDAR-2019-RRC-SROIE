from .base import BaseArgs


class TestArgs(BaseArgs):

    def __init__(self, description: str):
        super(TestArgs, self).__init__(description)

        self.parser.add_argument("--model-checkpoint", action="store",
                                 help="The path to the trained model state dict file")
        self.parser.add_argument("--image-folder", default="", type=str,
                                 help="The path to the images for testing.")
        self.parser.add_argument("--output-folder", default="", type=str,
                                 help="The directory to save prediction results")
