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


class ConvMnist():
    def __init__(self):
        logger.info('Starting Conv MNIST Tensor Net.')

        self.tensor_session = None
        self.mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.input_matrix = None
        self.output_matrix = None
        self.delta_matrix = None
        self.keep_prob = tensorflow.placeholder(tensorflow.float32)
        self.cross_entropy = self.create_model()

    def __del__(self):
        # Close tensorflow session.
        if self.tensor_session:
            self.tensor_session.close()
        logger.info('Conv MNIST Tensor Net finished.')

    def create_model(self):
        """
        Create various "tensors" (multi-dimensional matrixes) to manipulate data.
        A shape value of "None" means it can take in any arbitrary number of items in that dimension.
        784 is the total number of pixels per input image.
        :return: Returns "cross_entropy" function, which is used to hold error.
        """
        # Define input matrix setup.
        self.input_matrix = tensorflow.placeholder(tensorflow.float32, shape=[None, 784])

        # Define first convolutional layer.
        # Weights have 32 features per 5x5 patch.
        conv_1_weights = self.create_weights([5, 5, 1, 32])
        conv_1_biases = self.create_bias([32])

        # Reshape image by width, height, and number of color channels. Not sure what first dimension is.
        x_image = tensorflow.reshape(self.input_matrix, [-1, 28, 28, 1])

        hidden_conv_1 = tensorflow.nn.relu(
            tensorflow.nn.conv2d(x_image, conv_1_weights, strides=[1, 1, 1, 1], padding='SAME') + conv_1_biases
        )
        hidden_pool_1 = tensorflow.nn.max_pool(hidden_conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Define second convolutional layer.
        # Weights have 64 features per 5x5 patch.
        conv_2_weights = self.create_weights([5, 5, 32, 64])
        conv_2_biases = self.create_bias([64])
        hidden_conv_2 = tensorflow.nn.relu(
            tensorflow.nn.conv2d(hidden_pool_1, conv_2_weights, strides=[1, 1, 1, 1], padding='SAME') + conv_2_biases
        )
        hidden_pool_2 = tensorflow.nn.max_pool(hidden_conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Define dense layer.
        # Images are now 7x7 with 64 features.
        dense_1_weights = tensorflow.Variable(tensorflow.truncated_normal([7 * 7 * 64, 1024], mean=0.0, stddev=0.01))
        dense_1_biases = tensorflow.Variable(tensorflow.zeros([1024]))

        # What does this do??
        flat_pool = tensorflow.reshape(hidden_pool_2, [-1, 7 * 7 * 64])

        # Define output and loss matrix setups.
        conv_output = tensorflow.nn.relu(tensorflow.matmul(flat_pool, dense_1_weights) + dense_1_biases)
        self.delta_matrix = tensorflow.placeholder(tensorflow.float32, [None, 10])

        # "Dropout" layer used during training to prevent overfitting.
        # Note: Syntax does not seem to be supported by tensorflow 1.5. Excluding dropout functionality.
        # self.keep_prob = tensorflow.placeholder(tensorflow.float32)
        hidden_drop = tensorflow.nn.dropout(conv_output, self.keep_prob)

        # Create final dense layer.
        dense_2_weights = tensorflow.Variable(tensorflow.truncated_normal([1024, 10], mean=0.0, stddev=0.01))
        dense_2_bias = tensorflow.Variable(tensorflow.truncated_normal([10], mean=0.0, stddev=0.01))
        self.output_matrix = tensorflow.matmul(hidden_drop, dense_2_weights) + dense_2_bias

        # Determine cross entropy.
        cross_entropy = tensorflow.reduce_mean(
            tensorflow.nn.softmax_cross_entropy_with_logits_v2(labels=self.delta_matrix, logits=self.output_matrix)
        )

        return cross_entropy

    def create_weights(self, shape):
        """
        Create weights of the given shape.
        :param shape: Number of values to contain.
        :return: Weight matrix variable.
        """
        weights = tensorflow.truncated_normal(shape, stddev=0.1)
        return tensorflow.Variable(weights)

    def create_bias(self, shape):
        """
        Create bias of the given shape.
        :param shape: Number of values to contain.
        :return: Weight matrix variable.
        """
        bias = tensorflow.constant(0.1, shape=shape)
        return tensorflow.Variable(bias)

    def train(self):
        """
        Train tensor net.
        """
        # Define a training step, using delta and Gradient Descent. Learning rate of 0.5.
        train_step = tensorflow.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)

        # Initialize tensorflow session.
        self.tensor_session = tensorflow.InteractiveSession()
        tensorflow.global_variables_initializer().run()

        # Create structures to calculate accuracy.
        correct_prediction = tensorflow.equal(
            tensorflow.argmax(self.output_matrix, 1),
            tensorflow.argmax(self.delta_matrix, 1)
        )
        accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))
        highest_accuracy = 0
        logger.info('')
        logger.info('')

        # Actually step through and train on data.
        for index in range(1000):
            features, targets = self.mnist_data.train.next_batch(50)

            # Only print out every 100 values.
            if index % 100 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={self.input_matrix: features, self.delta_matrix: targets, self.keep_prob: 1.0}
                )
                if train_accuracy > highest_accuracy:
                    highest_accuracy = train_accuracy
                logger.info('Step: {0} | Cur Accuracy: {1} | Best Accuracy: {2}'.format(index, train_accuracy, highest_accuracy))

            self.tensor_session.run(
                train_step,
                feed_dict={self.input_matrix: features, self.delta_matrix: targets, self.keep_prob: 0.5}
            )

        # Evaluate training results and print out.
        logger.info('Testing Accuracy: {0}   Best Training Accuracy: {1}'.format(
            self.tensor_session.run(
                accuracy,
                feed_dict={self.input_matrix: self.mnist_data.test.images, self.delta_matrix: self.mnist_data.test.labels, self.keep_prob: 1.0}
            ),
            highest_accuracy
        ))
