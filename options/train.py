from .base import BaseArgs


class TrainArgs(BaseArgs):

    def __init__(self, description: str):
        super(TrainArgs, self).__init__(description)

        self.parser.add_argument("--save-steps", action="store", type=int, metavar="N",
                                 help="After a certain number of iterations, a checkpoint is saved")
        self.parser.add_argument("--plot-steps", action="store", type=int,
                                 help="After a certain number of iterations, "
                                      "the loss is updated and can be seen on the visualizer.")
        self.parser.add_argument("--log-steps", action="store", type=int, help="Print logs every log steps")
        self.parser.add_argument("--resume", action="store", type=str,
                                 help="Resume training from a given checkpoint."
                                      "If None, then the training will be resumed"
                                      "from the latest checkpoint.")
        self.parser.add_argument("--use-visdom", action="store_true",
                                 help="A class to visualise data during training, "
                                      "i.e., in real time using Visdom"
                                      "and for each iteration.")
