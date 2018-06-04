"""
Examples of image-recognition neural nets.
"""

# System Imports.


# User Class Imports.
from resources import logging
import deep_mnist_net, mnist_net


# Initiate logging.
logger = logging.get_logger(__name__)

# Run basic Mnist tensor net.
# mnist_set = mnist_net.BasicMnist()
# mnist_set.train()
# mnist_set = None

# Run advanced Mnist tensor net.
mnist_set = deep_mnist_net.DeepMnist()
mnist_set.train()
mnist_set = None

logger.info('Closing program.')
