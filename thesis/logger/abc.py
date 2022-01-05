"""TODO."""
from abc import ABC, abstractmethod

from thesis.logger.core import add, set_logger


class ABCLogger(ABC):
    """TODO."""

    name: str = ""

    def __init__(self, name=""):
        """TODO."""
        self.name = name
        self.name and add(self)

    @abstractmethod
    def log(self, *args, **kwargs):
        """TODO."""
        pass

    @abstractmethod
    def add_scalars(self, who, what, when):
        """TODO."""
        pass

    def __enter__(self):
        """TODO."""
        self.__PREV_LOGGER = set_logger(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """TODO."""
        set_logger(self.__PREV_LOGGER)
        return False
