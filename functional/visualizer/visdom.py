import torch

from visdom import Visdom
from typing import List, Optional


# Taken from https://github.com/noagarcia/visdom-tutorial/blob/master/utils.py
# For further info : https://github.com/facebookresearch/visdom
class VisdomVisualizer(object):

    def __init__(self, port: int, env_name: Optional[str] = "SROIE2019"):
        """
        A flexible tool for creating, organizing, and sharing visualizations of live, rich data.

        Args:
            port (int): The port number
            env_name (string, optional): The environment name.
        """
        self.__viz: Visdom = Visdom(port=port)

        self.__env: str = env_name

    def createPlot(self, xLabel: str, yLabel: str, title_name: str, legend_names: List[str]):
        """
        Create a given plot/window to visualize.

        Args:
            xLabel: The x-axis label.
            yLabel: The y-axis label.
            title_name: The title name that is drawn at the upper center of the plot/window.
            legend_names: The legend names that are drawn at the upper right of the plot/window.

        Returns:
            A plot/window to update/visualize to.

        """
        window = self.__viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, len(legend_names))).cpu(),
            env=self.__env,
            opts=dict(
                legend=legend_names,
                title=title_name,
                xlabel=xLabel,
                ylabel=yLabel
            )
        )

        return window

    def update_plot(self, window, data_x: int, data_y: List[float]):
        """
        Update a given plot/window.

        Args:
            window: The window to update to.
            data_x (int): The x-axis data.
            data_y (int): The y-axis data.

        """
        self.__viz.line(X=torch.ones((1, len(data_y))).cpu() * data_x,
                        Y=torch.tensor(data_y).unsqueeze(0).cpu(),
                        env=self.__env, win=window, update="append")

    def save(self):
        """
        Save the environment that contains all the plots/windows.
        """
        self.__viz.save([self.__env])
