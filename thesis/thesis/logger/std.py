"""TODO."""
from sys import stdout
from typing import List

from thesis.logger.abc import ABCLogger
from thesis.logger.core import DEFAULT_LOGGER, NO_OP_LOGGER, logger, set_logger


def WITH_NO_ABS(cls):
    """Becarefull."""
    if not hasattr(cls, "__abstractmethods__"):
        return cls
    cls.__abstractmethods__ = frozenset(set())
    return cls


@WITH_NO_ABS
class Loggers(ABCLogger):
    """TODO."""

    loggers: List[ABCLogger]

    def __init__(self, *loggers, name: str = "", best_effort: bool = False):
        """TODO."""
        super().__init__(name)
        self.loggers = [
            logger(logr, best_effort) if isinstance(logr, str) else logr
            for logr in loggers
        ]
        for method in ABCLogger.__abstractmethods__:
            methods = {getattr(logr, method) for logr in self.loggers}
            setattr(
                self,
                method,
                lambda *args, methods=methods, **kwargs: [
                    m(*args, **kwargs) for m in methods
                ],
            )

    def __enter__(self):
        """TODO."""
        super().__enter__()
        return self.loggers

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """TODO."""
        super().__exit__(exc_type, exc_value, exc_traceback)
        return all(
            [
                getattr(logger, "__exit__")(exc_type, exc_value, exc_traceback)
                for logger in self.loggers
            ]
        )


class PrinterLogger:
    """TODO."""

    @staticmethod
    def log(*args, **kwargs):
        """TODO."""
        print(*args, **kwargs)

    @staticmethod
    def flush():
        """TODO."""
        stdout.flush()


class StdLogger(PrinterLogger, ABCLogger):
    """TODO."""

    def add_scalars(self, who, what, when):
        """TODO."""
        self.log(f"{when}:{who}->{what}")


set_logger(StdLogger(name=DEFAULT_LOGGER))


class NoOpLogger(ABCLogger):
    """TODO."""

    def log(self, *args, **kwargs):
        """TODO."""

    def add_scalars(self, who, what, when):
        """TODO."""

    def flush(self):
        """TODO."""


NoOpLogger(name=NO_OP_LOGGER)


class NamedLogger(StdLogger):
    """TODO."""

    def __init__(self, name):
        """TODO."""
        super().__init__(name)

    def log(self, *args, **kwargs):
        """TODO."""
        super().log(f"{self.name}\t", *args, **kwargs)

    def add_scalars(self, who, what, when):
        """TODO."""
        super().add_scalars(who, what, f"{self.name}\t{when}")
