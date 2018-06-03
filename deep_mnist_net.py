"""
A more advanced image recognition neural net based off of the mnist dataset.
"""

# System Imports.
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow

# User Class Imports.
from resources import logging


# Initiate logging.
logger = logging.get_logger(__name__)


class DeepMnist():
    def __init__(self):
        logger.info('Starting Deep MNIST Tensor Net.')

        self.mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.input_matrix = None
        self.output_matrix = None
        self.delta_matrix = None
        self.cross_entropy = self.create_model()

    def __del__(self):
        logger.info('Deep MNIST Tensor Net finished.')

    def train(self):
        """
        Train tensor net.
        """
        # Define a training step, using delta and Gradient Descent. Learning rate of 0.5.
        train_step = tensorflow.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)

        # Initialize tensorflow session.
        tensor_session = tensorflow.InteractiveSession()
        tensorflow.global_variables_initializer().run()

        # Actually step through and train on data.
        for index in range(1000):
            features, targets = self.mnist_data.train.next_batch(100)
            tensor_session.run(
                train_step,
                feed_dict={self.input_matrix: features, self.delta_matrix: targets}
            )

        # Evaluate training results and print out.
        correct_prediction = tensorflow.equal(tensorflow.argmax(self.output_matrix, 1), tensorflow.argmax(self.delta_matrix, 1))
        accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

        logger.info(
            tensor_session.run(
                accuracy,
                feed_dict={self.input_matrix: self.mnist_data.test.images, self.delta_matrix: self.mnist_data.test.labels}
            )
        )

    def create_model(self):
        """
        Create various "tensors" (multi-dimensional matrixes) to manipulate data.
        A shape value of "None" means it can take in any arbitrary number of items in that dimension.
        784 is the total number of pixels per input image.
        :return: Returns "cross_entropy" function, which is used to hold error.
        """
        # Define initial matrix setups.
        self.input_matrix = tensorflow.placeholder(tensorflow.float32, shape=[None, 784])
        weight_matrix = tensorflow.Variable(tensorflow.zeros([784, 10]))
        bias_matrix = tensorflow.Variable(tensorflow.zeros([10]))

        # Define output and loss matrix setups.
        self.output_matrix = tensorflow.matmul(self.input_matrix, weight_matrix) + bias_matrix
        self.delta_matrix = tensorflow.placeholder(tensorflow.float32, [None, 10])
        cross_entropy = tensorflow.reduce_mean(
            tensorflow.nn.softmax_cross_entropy_with_logits(labels=self.delta_matrix, logits=self.output_matrix))

        return cross_entropy
