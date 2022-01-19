"""TODO."""
from typing import Any, Dict

from thesis.logger import add_scalars, flush
from thesis.losses import MLoss


class Stats:
    """TODO."""

    current_losses: Dict[str, float]
    epoch: int = 0
    batch_size: int = 1
    episode: int = 1
    model_name: str

    def __init__(
        self,
        hparams: Dict[str, Any],
        model_name: str = "test",
    ):
        """Initialize class."""
        self.model_name = model_name
        self.current_losses = {}

    def add_losses(self, losses):
        """TODO."""
        self.current_losses.update(
            (k, v.item() + self.current_losses.get(k, 0)) for k, v in losses.items()
        )

        if self.episode % 100 == 0:
            add_scalars(
                self.model_name,
                {
                    "prefix": self.prefix,
                    **{
                        f"loss_{k}": self.current_losses[k]
                        / (self.episode * self.batch_size)
                        for k in self.current_losses
                    },
                    "loss_fun": self.loss_name,
                },
                (self.epoch * self.data_size + (self.episode * self.batch_size))
                / self.data_size,
            )

        self.episode += 1

    def __call__(self, prefix: str, data, loss: MLoss):
        """TODO."""
        self.batch_size = data.batch_size
        self.data_size = len(data.dataset)
        self.prefix = prefix
        self.loss_name = loss.name
        return self

    def __enter__(
        self,
    ):
        """Reset all the losses.

        Usually used after a full loop over all the data.
        """
        self.current_losses = {}
        self.episode = 1
        return self.add_losses

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Print all losses."""
        flush()
        return False
