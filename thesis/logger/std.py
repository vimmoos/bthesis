"""TODO."""
from thesis.logger.abc import ABCLogger
from thesis.logger.core import DEFAULT_LOGGER, NO_OP_LOGGER, set_logger


class StdLogger(ABCLogger):
    """TODO."""

    def log(self, *args, **kwargs):
        """TODO."""
        print(*args, **kwargs)

    def add_scalars(self, who, what, when):
        """TODO."""
        print(f"{when}:{who}->{what}")


set_logger(StdLogger(name=DEFAULT_LOGGER))


class NoOpLogger(ABCLogger):
    """TODO."""

    def log(self, *args, **kwargs):
        """TODO."""

    def add_scalars(self, who, what, when):
        """TODO."""


NoOpLogger(name=NO_OP_LOGGER)


class NamedLogger(ABCLogger):
    """TODO."""

    def __init__(self, name):
        """TODO."""
        super().__init__(name)

    def log(self, *args, **kwargs):
        """TODO."""
        print(f"{self.name}\t", *args, **kwargs)

    def add_scalars(self, who, what, when):
        """TODO."""
        print(f"{self.name}\t{when}:{who}->{what}")
