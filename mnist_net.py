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


logger.info('Starting MNIST Tensor Net.')

# Grab MNIST data.
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create TensorFlow data structs, called "Tensors" (aka multi-dimensional matrixes).
# None means it can take in any length. 784 is the number of pixels per image.
input_matrix = tensorflow.placeholder(tensorflow.float32, [None, 784])
weights = tensorflow.Variable(tensorflow.zeros([784, 10]))
biases = tensorflow.Variable(tensorflow.zeros([10]))

# Implement the learning model, using above structs.
output_matrix = tensorflow.nn.softmax(tensorflow.matmul(input_matrix, weights) + biases)

# Create struct to hold error, using "Cross-Entropy".
delta = tensorflow.placeholder(tensorflow.float32, [None, 10])

# Implement Cross-Entropy function.
cross_entropy = tensorflow.reduce_mean(
    -tensorflow.reduce_sum(delta * tensorflow.log(output_matrix), reduction_indices=[1],
))

# Tell it to train with Gradient Descent. Minimize errors using cross_entropy. Learning rate of 0.5.
train_step = tensorflow.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Launch tensorflow session now that structure defined.
tensor_session = tensorflow.InteractiveSession()

# Initialize defined variables (weights, biases, etc).
tensorflow.global_variables_initializer().run()

# Run training step 1000 times. Each step grabs a random batch of 100 training values to use.
for index in range(1000):
    features, targets = mnist_data.train.next_batch(100)
    tensor_session.run(train_step, feed_dict={input_matrix: features, delta: targets})

# Evaluate tensor net.
correct_prediction = tensorflow.equal(tensorflow.argmax(output_matrix, 1), tensorflow.argmax(delta, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

# Finally, print out accuracy in a human-readible format.
logger.info(tensor_session.run(
    accuracy,
    feed_dict={input_matrix: mnist_data.test.images, delta: mnist_data.test.labels}
))

logger.info('MNIST Tensor Net finished.')
