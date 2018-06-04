"""
Examples of image-recognition neural nets.
"""

# System Imports.


# User Class Imports.
from resources import logging
import conv_mnist_net, mnist_net


# Initiate logging.
logger = logging.get_logger(__name__)

# Run basic Mnist tensor net.
mnist_set = mnist_net.BasicMnist()
mnist_set.train()
mnist_set = None

logger.info('')
logger.info('')
logger.info('')
logger.info('')

# Run convolutational Mnist tensor net.
mnist_set = conv_mnist_net.ConvMnist()
mnist_set.train()
mnist_set = None

logger.info('Closing program.')
