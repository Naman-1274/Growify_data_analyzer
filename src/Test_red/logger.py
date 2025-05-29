# logger.py

import logging
import sys

# Create a logger for the entire app
logger = logging.getLogger("julius_clone")
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)  # change to DEBUG for more verbose logs
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Example usage:
# from app_backend.logger import logger
# logger.info("This is an informational message.")
# logger.error("This is an error message.")
