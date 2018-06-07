"""
An image recognition neural net based off of the mnist dataset.
"""

# System Imports.
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow

# User Class Imports.
from resources import logging


# Initiate logging.
logger = logging.get_logger(__name__)


class BasicMnist():
    def __init__(self):
        logger.info('Starting Basic MNIST Tensor Net.')

        self.tensor_session = None
        self.mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
        logger.info('MNIST Data {0}'.format(self.mnist_data))
        self.input_matrix = None
        self.output_matrix = None
        self.delta_matrix = None
        self.cross_entropy = self.create_model()

    def __del__(self):
        # Close tensorflow session.
        if self.tensor_session:
            self.tensor_session.close()
        logger.info('Basic MNIST Tensor Net finished.')

    def create_model(self):
        """
        Create various "tensors" (multi-dimensional matrixes) to manipulate data.
        A shape value of "None" means it can take in any arbitrary number of items in that dimension.
        784 is the total number of pixels per input image.
        :return: Returns "cross_entropy" function, which is used to hold error.
        """
        # Define initial matrix setups.
        self.input_matrix = tensorflow.placeholder(tensorflow.float32, shape=[None, 784])
        weight_matrix = tensorflow.Variable(tensorflow.truncated_normal([784, 10], stddev=0.1))
        bias_matrix = tensorflow.Variable(tensorflow.constant(0.1, shape=[10]))

        # Define output and loss matrix setups.
        self.output_matrix = tensorflow.matmul(self.input_matrix, weight_matrix) + bias_matrix
        self.delta_matrix = tensorflow.placeholder(tensorflow.float32, [None, 10])
        cross_entropy = tensorflow.reduce_mean(
            tensorflow.nn.softmax_cross_entropy_with_logits_v2(labels=self.delta_matrix, logits=self.output_matrix)
        )

        # Initialize tensorflow session.
        self.tensor_session = tensorflow.InteractiveSession()
        tensorflow.global_variables_initializer().run()

        return cross_entropy

    def train(self):
        """
        Train tensor net.
        """
        # Define a training step, using delta and Gradient Descent. Learning rate of 0.5.
        train_step = tensorflow.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)

        # Create structures to calculate accuracy.
        correct_prediction = tensorflow.equal(
            tensorflow.argmax(self.output_matrix, 1),
            tensorflow.argmax(self.delta_matrix, 1)
        )
        accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))
        total_accuracy = 0
        highest_accuracy = 0

        # Step through and train on data.
        for index in range(1000):
            features, targets = self.mnist_data.train.next_batch(50)

            # Only print out every 100 values.
            if index % 100 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={
                        self.input_matrix: features,
                        self.delta_matrix: targets
                    }
                )
                if train_accuracy > highest_accuracy:
                    highest_accuracy = train_accuracy
                # logger.info(
                #     'Step: {0} | Cur Accuracy: {1} | Best Accuracy: {2}'.format(index, train_accuracy, highest_accuracy)
                # )
                total_accuracy += train_accuracy

            # Run a training step.
            self.tensor_session.run(
                train_step,
                feed_dict={
                    self.input_matrix: features,
                    self.delta_matrix: targets
                }
            )

        test_accuracy = accuracy.eval(
            feed_dict={
                self.input_matrix: self.mnist_data.test.images,
                self.delta_matrix: self.mnist_data.test.labels
            }
        )
        # Evaluate training results and print out.
        # logger.info('Testing Accuracy: {0}   Best Training Accuracy: {1}'.format(test_accuracy, highest_accuracy))

        return [(total_accuracy / 10), test_accuracy]
