"""
Examples of image-recognition neural nets.
"""

# System Imports.


# User Class Imports.
from resources import logging
import mnist_net


# Initiate logging.
logger = logging.get_logger(__name__)


mnist_set = mnist_net.DeepMnist()
mnist_set.train()

logger.info('Closing program.')
