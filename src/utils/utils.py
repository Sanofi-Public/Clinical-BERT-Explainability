"""
General utility functions for the project.
"""

import logging
import sys
import yaml


def parse_yaml_args(config_file):
    """
    Given a yaml file, open it and parse it into nested dictionariers and lists.
    """
    logger.info(f"Using config: {config_file}")
    with open(config_file) as f:
        config = yaml.safe_load(f)
        logger.info(config)
    return config


def setup_logger(logger_name=None, level=logging.INFO):
    """
    Sets up the logger configuration.

    :param logger_name: Name of the logger to set up. If None, sets up the root logger.
    :param level: Logging level.
    :param stream: Stream to log to, default is sys.stdout.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    logger.handlers = []

    # Add the new handler
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    return logger


def split_s3_file(s3_filename):
    """
    Given a s3_filename string of the form s3://{bucket_name}/{this/is/the/key}
    return the bucket and key as a tuple: ("bucket_name", "this/is/the/key")
    """
    s3_filename_split = s3_filename.replace("s3://", "").split("/")
    bucket = s3_filename_split[0]
    key = "/".join(s3_filename_split[1:])
    return bucket, key


logger = setup_logger()
