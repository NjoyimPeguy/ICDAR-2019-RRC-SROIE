import argparse


class BaseArgs:

    def __init__(self, description: str):
        super(BaseArgs, self).__init__()

        self.parser = argparse.ArgumentParser(description=description,
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument("--gpu-device", default=0, type=int,
                                 help="Specify the GPU ID to use. By default, it is the ID 0")
        self.parser.add_argument("--use-cuda", action="store_true", help="enable/disable cuda training")
        self.parser.add_argument("--use-amp", action="store_true",
                                 help="enable/disable automatic mixed precision. By default it is disable."
                                      "For further info, check those following links:"
                                      "https://pytorch.org/docs/stable/amp.html"
                                      "https://pytorch.org/docs/stable/notes/amp_examples.html"
                                      "https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html")

    def get_args(self):
        args, _ = self.parser.parse_known_args()
        return args

    def get_parser(self):
        return self.parser
