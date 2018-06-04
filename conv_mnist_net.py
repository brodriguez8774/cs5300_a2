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

        # Reshape image by width, height, and number of color channels. Not sure what first dimension is.
        x_image = tensorflow.reshape(self.input_matrix, [-1, 28, 28, 1])

        # Define first convolutional layer.
        conv_layer_1 = self.create_conv_layer(x_image, 1, 32)

        # Define second convolutional layer.
        conv_layer_2 = self.create_conv_layer(conv_layer_1, 32, 64)

        # Downsample further for simplification of dense layers.
        flattening_pool = tensorflow.reshape(conv_layer_2, [-1, 7*7*64])

        # Define first dense layer. Also applies relu for gradient learning function.
        conv_output = tensorflow.nn.relu(self.create_dense_layer(flattening_pool, 7 * 7 * 64, 1024))

        # Define delta matrix.
        self.delta_matrix = tensorflow.placeholder(tensorflow.float32, [None, 10])

        # "Dropout" layer used during training to prevent overfitting.
        dropout_layer = tensorflow.nn.dropout(conv_output, self.keep_prob)

        # Create final layer/second dense layer.
        self.output_matrix = self.create_dense_layer(dropout_layer, 1024, 10)

        # Determine cross entropy. Used to actually train net gradient.
        cross_entropy = tensorflow.reduce_mean(
            tensorflow.nn.softmax_cross_entropy_with_logits_v2(labels=self.delta_matrix, logits=self.output_matrix)
        )

        return cross_entropy

    def create_conv_layer(self, x_inputs, input_dimension, output_dimension):
        """
        Define a given convolutional layer.
        See https://www.tensorflow.org/tutorials/layers.
        :param x_inputs: The input values for layer to manipulate.
        :param input_dimension: Input dimension of features.
        :param output_dimension: Output dimension of features.
        :return: Full convolutional layer (or at least the necessary hook into it).
        """
        # Weights will have [output_dimension] features per 5x5 patch.
        conv_weights = self.create_conv_weights([5, 5, input_dimension, output_dimension])
        conv_biases = self.create_conv_bias([output_dimension])

        # Apply convolutional filters. Stride first and last value should always be 1.
        # Inner values are height and width, respectively.
        conv_filter = tensorflow.nn.relu(
            tensorflow.nn.conv2d(x_inputs, conv_weights, strides=[1, 1, 1, 1], padding='SAME') + conv_biases
        )

        # Pooling downsamples image data to a smaller pixel size, as filters seem to greatly expand it.
        pooling_layer = tensorflow.nn.max_pool(conv_filter, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pooling_layer

    def create_conv_weights(self, shape):
        """
        Create weights of the given shape.
        :param shape: Dimensions of weights, in array format.
        :return: Weight matrix variable.
        """
        weights = tensorflow.truncated_normal(shape, stddev=0.1)
        return tensorflow.Variable(weights)

    def create_conv_bias(self, shape):
        """
        Create bias of the given shape.
        :param shape: Dimensions of bias, in array format.
        :return: Bias matrix variable.
        """
        bias = tensorflow.constant(0.1, shape=shape)
        return tensorflow.Variable(bias)

    def create_dense_layer(self, x_inputs, input_dimension, output_dimension):
        """
        Define a given dense node layer.
        :param x_inputs: The input values for layer to manipulate.
        :param input_dimension: Input dimension of features.
        :param output_dimension: Output dimension of features.
        :return: Full dense node layer (or at least the necessary hook into it).
        """
        dense_weights = self.create_dense_weights([input_dimension, output_dimension])
        dense_biases = self.create_dense_bias([output_dimension])

        # Apply matrix layer logic.
        dense_layer = tensorflow.matmul(x_inputs, dense_weights) + dense_biases
        return dense_layer

    def create_dense_weights(self, shape):
        """
        Create weights of the given shape.
        :param shape: Dimensions of weights, in array format.
        :return: Weight matrix variable.
        """
        weights = tensorflow.truncated_normal(shape, mean=0.0, stddev=0.01)
        return tensorflow.Variable(weights)

    def create_dense_bias(self, shape):
        """
        Create bias of the given shape.
        :param shape: Dimensions of bias, in array format.
        :return: Bias matrix variable.
        """
        bias = tensorflow.truncated_normal(shape, mean=0.0, stddev=0.01)
        return tensorflow.Variable(bias)

    def train(self):
        """
        Train tensor net.
        """
        # Define a training step, using entropy and Gradient Descent. Learning rate of 0.5.
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

            # Only print out every 100 values. Displays step number and accuracy info.
            if index % 100 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={
                        self.input_matrix: features,
                        self.delta_matrix: targets,
                        self.keep_prob: 1.0
                    }
                )
                if train_accuracy > highest_accuracy:
                    highest_accuracy = train_accuracy
                logger.info(
                    'Step: {0} | Cur Accuracy: {1} | Best Accuracy: {2}'.format(index, train_accuracy, highest_accuracy)
                )

            # Run a given training step.
            self.tensor_session.run(
                train_step,
                feed_dict={
                    self.input_matrix: features,
                    self.delta_matrix: targets,
                    self.keep_prob: 0.5
                }
            )

        # Evaluate results and print out.
        logger.info('Testing Accuracy: {0}   Best Training Accuracy: {1}'.format(
            self.tensor_session.run(
                accuracy,
                feed_dict={
                    self.input_matrix: self.mnist_data.test.images,
                    self.delta_matrix: self.mnist_data.test.labels,
                    self.keep_prob: 1.0
                }
            ),
            highest_accuracy
        ))
