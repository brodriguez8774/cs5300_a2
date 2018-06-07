"""
Examples of image-recognition neural nets.
"""

# System Imports.


# User Class Imports.
from resources import logging
import conv_mnist_net, mnist_net, result_tracker


# Initiate logging.
logger = logging.get_logger(__name__)


# Set Variables.
epochs = 25
epoch_num = 0
basic_mnist_train_data = []
basic_mnist_test_data = []
conv_mnist_train_data = []
conv_mnist_test_data = []
basic_mnist_tracker = result_tracker.ResultTracker(epochs)
conv_mnist_tracker = result_tracker.ResultTracker(epochs)


# Run basic Mnist tensor net.
mnist_set = mnist_net.BasicMnist()

while basic_mnist_tracker.continue_training_check(max_epochs=epochs):
    basic_mnist_accuracy = mnist_set.train()
    basic_mnist_train_data.append([basic_mnist_accuracy[0], epoch_num])
    basic_mnist_test_data.append([basic_mnist_accuracy[1], epoch_num])
    basic_mnist_tracker.add_iteration(0, basic_mnist_accuracy[1])
    logger.info('Epoch {0}   Train Accuracy: {1}   Test Accuracy: {2}'
                .format(epoch_num, basic_mnist_accuracy[0], basic_mnist_accuracy[1]))
    epoch_num += 1
mnist_set = None

logger.info('')
logger.info('')
logger.info('')
logger.info('')


# Run convolutational Mnist tensor net.
epoch_num = 0
mnist_set = conv_mnist_net.ConvMnist()
while conv_mnist_tracker.continue_training_check(max_epochs=epochs):
    conv_mnist_accuracy = mnist_set.train()
    conv_mnist_train_data.append([conv_mnist_accuracy[0], epoch_num])
    conv_mnist_test_data.append([conv_mnist_accuracy[1], epoch_num])
    conv_mnist_tracker.add_iteration(0, conv_mnist_accuracy)
    logger.info('Epoch {0}   Train Accuracy: {1}   Test Accuracy: {2}'
                .format(epoch_num, conv_mnist_accuracy[0], conv_mnist_accuracy[1]))
    epoch_num += 1
mnist_set = None


# Display all basic mnist results.
l_list = basic_mnist_tracker.plot_results(basic_mnist_train_data, color='b', best_fit_line=True, line_label='Training Best Fit')
basic_mnist_tracker.plot_results(basic_mnist_test_data, color='r', best_fit_line=True, line_label='Testing Best Fit', label_list=l_list)
basic_mnist_tracker.display_plot('Basic Mnist Accuracy Results', 'Epoch Number', 'Accuracy')


# Display all conv mnist results.
l_list = conv_mnist_tracker.plot_results(conv_mnist_train_data, color='b', best_fit_line=True, line_label='Training Best Fit')
conv_mnist_tracker.plot_results(conv_mnist_test_data, color='r', best_fit_line=True, line_label='Testing Best Fit', label_list=l_list)
conv_mnist_tracker.display_plot('Conv Mnist Error Results', 'Epoch Number', 'Accuracy')


# Compare training of both basic and conv mnist sets.
l_list = basic_mnist_tracker.plot_results(basic_mnist_test_data, color='b', best_fit_line=True, line_label='Basic Mnist Fit')
basic_mnist_tracker.plot_results(conv_mnist_train_data, color='r', best_fit_line=True, line_label='Conv Mnist Fit', label_list=l_list)
basic_mnist_tracker.display_plot('Mnist Training Accuracy', 'Epoch Number', 'Accuracy')


# Compare testing of both basic and conv mnist sets.
l_list = conv_mnist_tracker.plot_results(conv_mnist_train_data, color='b', best_fit_line=True, line_label='Basic Mnist Fit')
conv_mnist_tracker.plot_results(conv_mnist_test_data, color='r', best_fit_line=True, line_label='Conv Mnist Fit', label_list=l_list)
conv_mnist_tracker.display_plot('Mnist Testing Accuracy', 'Epoch Number', 'Accuracy')


# cifar_set = cifar_net.BasicCifar()
# cifar_set.train()
# cifar_set = None


logger.info('Closing program.')
