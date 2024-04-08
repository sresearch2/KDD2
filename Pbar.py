import time
from IPython.display import clear_output
from matplotlib import pyplot as plt


class ProgressBar:
    """
    Class to create a progress bar with optional features like displaying a string, updating a graph, etc.
    """

    def __init__(self, timegap, length=40, char1="-", char2=">"):
        """
        Initializes the progress bar with the given parameters.

        Args:
            timegap (int): The time interval (in seconds) between updates.
            length (int): The length of the progress bar.
            char1 (str): The character representing completed progress.
            char2 (str): The character representing remaining progress.
        """
        self.start_time = time.time()
        self.timegap = timegap
        self.time_ellapsed = 0
        self.length = length
        self.char1 = char1
        self.char2 = char2
        self.past = 0
        self.tim1 = 0
        self.past_error = 0

    def print(
        self,
        current,
        max_val,
        string="",
        clear=True,
        graph=False,
        error=0,
        graph_length=10,
        smoothing=0.2,
    ):
        """
        Prints the progress bar and updates it at specified intervals.

        Args:
            current (int): The current progress value.
            max_val (int): The maximum progress value.
            string (str, optional): Additional string to display.
            clear (bool, optional): Whether to clear the previous output. Defaults to True.
            graph (bool, optional): Whether to update the graph. Defaults to False.
            error (float, optional): Error value for graph smoothing. Defaults to 0.
            graph_length (int, optional): Length of the graph data to display. Defaults to 10.
            smoothing (float, optional): Smoothing factor for error. Defaults to 0.2.
        """
        assert smoothing < 1
        progress = round(self.length * current / max_val)
        if progress == 0:
            self.past_error = error
        curr_time = time.time()
        if curr_time - self.start_time > self.timegap:
            self.time_spent = time.time() - self.start_time
            self.time_ellapsed += self.timegap
            self._update_progress(progress, string, clear)
            if graph:
                self._update_graph(error, graph_length, smoothing)

    def _update_progress(self, progress, string, clear):
        """
        Updates and prints the progress bar.

        Args:
            progress (int): Current progress value.
            string (str): Additional string to display.
            clear (bool): Whether to clear the previous output.
        """
        if clear:
            clear_output(wait=True)
            if progress != self.past:
                self.tim1 = round(
                    (self.length - progress) / (progress / self.time_spent), 2
                )
            print(
                "["
                + self.char1 * int(progress)
                + self.char2
                + " " * (self.length - progress)
                + "]",
                "\n",
                string,
                f"\ntime {self.tim1}s",
                f"\ntime_spent {self.time_spent}s",
                flush=True,
            )
            self.past = progress

    def _update_graph(self, error, graph_length, smoothing):
        """
        Updates and displays the graph.

        Args:
            error (float): Error value for graph smoothing.
            graph_length (int): Length of the graph data to display.
            smoothing (float): Smoothing factor for error.
        """
        if len(self.data) > graph_length:
            self.data.pop(0)
        error = smoothing * error + (1 - smoothing) * self.past_error
        self.past_error = error
        self.data.append([self.time_ellapsed, error])
        plt.plot([x[0] for x in self.data], [x[1] for x in self.data])
        plt.show()
