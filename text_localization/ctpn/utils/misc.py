import time
import torch
import datetime as dt

from visdom import Visdom


# Taken from https://github.com/noagarcia/visdom-tutorial/blob/master/utils.py
# For further info : https://github.com/facebookresearch/visdom
class Visualizer(object):
    
    def __init__(self, port: int, env_name="SROIE-2019-Text-Localisation"):
        """
        A flexible tool for creating, organizing, and sharing visualizations of live, rich data.
        
        Args:
            port: The port number
            env_name: The environment name.
        """
        self.__viz = Visdom(port=port)
        self.__env = env_name
    
    def createPlot(self, xLabel: str, yLabel: str, title_name: str, legend_names: list):
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
    
    def update_plot(self, window, data_x: int, data_y: list):
        """
        Update a given plot/window.
        
        Args:
            window: The window to update to.
            data_x: The x-axis data.
            data_y: The y-axis data.
            
        """
        self.__viz.line(X=torch.ones((1, len(data_y))).cpu() * data_x,
                        Y=torch.tensor(data_y).unsqueeze(0).cpu(),
                        env=self.__env, win=window, update="append")
    
    def save(self):
        """
        Save the environment that contains all the plots/windows.
        """
        self.__viz.save([self.__env])


def get_process_time(start_time, current_iteration, max_iterations):
    """
    Calculate the elapsed, remaining and ETA times.
    
    Args:
        start_time: The starting time.
        current_iteration: The current iteration.
        max_iterations: The maximum number of iterations.

    Returns:
        A tuple containing the elapsed, remaining and the ETA times.
        
    """
    elapsed_time = time.time() - start_time
    
    estimated_time = (elapsed_time / current_iteration) * max_iterations
    remaining_time = estimated_time - elapsed_time  # in seconds
    
    finishtime = str(dt.datetime.fromtimestamp(start_time + estimated_time).strftime("%Y/%m/%d at %H:%M:%S"))
    
    times = (int(elapsed_time), int(remaining_time), finishtime)
    
    return times


# Adapted from https://github.com/pytorch/examples/blob/adc5bb40f1fa5ebae690787b474af4619df170b8/imagenet/main.py#L363
class AverageMeter(object):
    
    def __init__(self, fmt=":f"):
        """
        Computes and stores the average and current value.
        
        Args:
            fmt: The string format.
            
        """
        self.fmt = fmt
        self.count = 0
        self.value = 0.0
        self.total = 0.0
        self.global_avg = 0.0
    
    def update(self, value, n=1):
        """
        Update the current values.
        
        Args:
            value: The value to update with.
            n: A multiplier.
            
        """
        self.value = value
        self.total += (value * n)
        self.count += n
        self.global_avg = self.total / self.count
    
    def __str__(self):
        fmtstr = '{value' + self.fmt + '} ({global_avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
