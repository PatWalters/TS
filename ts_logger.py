import logging
from typing import Optional


def get_logger(name: str = None, level: str = "INFO", filename: Optional[str] = None) -> logging.Logger:
    """
    Basic logger for this repo.
    :param name: usually the file name which can be passed to the get_logger function like this get_logger(__name__)
    :param level: logging level
    :param filename: Filename to write logging to. If None, logging will print to screen.
    :return: logger
    """
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(levelname)-8s %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d:%H:%M:%S",
        filename=filename
    )
    if name is None:
        name = "TSLogger"
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
