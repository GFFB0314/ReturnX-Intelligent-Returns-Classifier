"""
Shared logging configuration for the ReturnX project.
"""

import logging
import sys


def setup_logger(name: str):
    """Configures and returns a logger with a standard format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(name)
