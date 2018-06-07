"""
Logic to track results of neural nets.
"""

# System Imports.
from matplotlib import patches, pyplot
import numpy

# User Class Imports.
from resources import logging


# Initialize logging.
logger = logging.get_logger(__name__)


class ResultTracker():
    """
    Tracks statistics and progress of Neural Net.
    """
    def __init__(self, min_iterations):
        self.iterations = []
        self.best_iteration_index = 0
        self.min_iterations = min_iterations

    def add_iteration(self, weights, accuracy):
        """
        Adds a new set of results to track.
        :param weights: Weights of current iteration.
        :param accuracy: Accuracy of current iteration.
        """
        new_iteration = [weights, accuracy]
        self.iterations.append(new_iteration)
        # logger.info('Iteration {0}: {1}'.format(len(self.iterations) - 1, new_iteration))
        #
        # logger.info('Previous Best: {0}   New Value: {1}'
        #             .format(self.iterations[self.best_iteration_index][1], error_margin))

        # Calculate best iteration thus far. Based on total error margin.
        if accuracy > self.iterations[self.best_iteration_index][1]:
            # logger.info('Setting new best. Cur: {0}   Prev: {1}'.format(accuracy, self.iterations[self.best_iteration_index][1]))
            self.best_iteration_index = len(self.iterations) - 1


    def continue_training_check(self, max_epochs=None):
        """
        Determines if Neural Net should continue training.
        :param max_epochs: Optional arg to set a max epoch count on training iterations.
        :return: True on continued training. False on training complete.
        """
        exit_training = False
        total_iterations = len(self.iterations)

        # Make Neural Net iterate a minimum set of times.
        if total_iterations <= self.min_iterations:
            exit_training = True

        # Check if Neural Net is still improving.
        if max_epochs is None:
            # Continue if progress has made in last x iterations.
            if self.best_iteration_index > (total_iterations - self.min_iterations):
                exit_training = True
        else:
            # Check if under epoch count.
            if total_iterations < max_epochs:
                # Continue if progress has been made in last x iterations.
                if self.best_iteration_index > (total_iterations - self.min_iterations):
                    exit_training = True

        return exit_training

    def plot_results(self, results, color='b', best_fit_line=False, mean_line=False, line_label=None, label_list=None):
        """
        Appends results to pyplot graph.
        Can take in multiple sets of results per graph. Simply call function multiple times.
        To actually display results, run display_plot method.
        :param results: Numpy array in format of [error, epoch/index]
        :param best_fit_line: Bool to display best fit line.
        :param mean_line: Bool to display mean line.
        """
        label_list = label_list or []

        # Plot accuracy of training, over time.
        x = []
        y = []

        for result in results:
            pyplot.scatter(result[1], result[0], alpha=0.75, c=color)
            x.append(result[1])
            y.append(result[0])

        # Create best-fit line.
        if best_fit_line:
            # Borrowed from https://stackoverflow.com/questions/22239691/code-for-line-of-best-fit-of-a-scatter-plot-in-python.
            pyplot.plot(numpy.unique(x), numpy.poly1d(numpy.polyfit(x, y, 1))(numpy.unique(x)), color, alpha=0.75)
            if line_label is None:
                label_list.append(patches.Patch(color=color, label='Best Fit Line'))
            else:
                label_list.append(patches.Patch(color=color, label=line_label))

        # Create average line.
        if mean_line:
            y_mean = [numpy.mean(y) for i in y]
            pyplot.plot(x, y_mean, color)
            if line_label is None:
                label_list.append(patches.Patch(color=color, label='Mean'))
            else:
                label_list.append(patches.Patch(color=color, label=line_label))

        if label_list is not []:
            pyplot.legend(handles=label_list)

        return label_list

    def display_plot(self, title, x_label, y_label):
        """
        Displays any created pyplot graphs.
        Seperate method in case one wishes to place multiple datasets ontop of one another.
        :param title: Title of graph.
        :param x_label: X coordinate label of graph.
        :param y_label: Y coordinate label of graph.
        :return:
        """
        # Plot labels.
        pyplot.title(title)
        pyplot.xlabel(x_label)
        pyplot.ylabel(y_label)
        pyplot.show()

