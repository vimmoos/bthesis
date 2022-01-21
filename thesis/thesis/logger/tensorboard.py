"""TODO."""
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from thesis.logger.abc import ABCLogger
from thesis.logger.std import PrinterLogger


class TBLogger(PrinterLogger, ABCLogger):
    """TODO."""

    writer: SummaryWriter

    def __init__(
        self,
        where,
        name=datetime.now().strftime("%Y%m%d_%H%M%S"),
        **kwargs,
    ):
        """TODO."""
        super().__init__(name=name, **kwargs)
        self.writer = SummaryWriter(f"runs/{where}/{name}")

    def add_scalars(self, who, what, when):
        """TODO."""
        self.writer.add_scalars(who, what, when)

    def flush(self):
        """TODO."""
        self.writer.flush()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """TODO."""
        self.flush()
        return super().__exit__(exc_type, exc_value, exc_traceback)
