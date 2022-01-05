"""TODO."""
import math
from typing import Any, Dict, List, Optional

from thesis.logger import add_scalars


def create_args_backward(**kwargs):
    """TODO."""
    return {k: v.get("backward", {}) for k, v in kwargs.items()}


def create_order(**kwargs):
    """TODO."""
    return [
        x[0] for x in sorted(kwargs.items(), key=lambda x: x[1].get("order", math.inf))
    ]


class MLosses:
    """TODO."""

    losses_arg: Dict[str, Dict[str, Any]]
    losses_ord: Optional[List[str]]
    current_losses: Dict[str, float]
    model_name: str
    epoch: int = 0
    batch_size: int = 1
    episode: int = 1

    def __init__(self, model_name="test", **kwargs):
        """Initialize class."""
        self.model_name = model_name
        self.losses_arg = create_args_backward(**kwargs)
        self.losses_ord = create_order(**kwargs)
        self.current_losses = {k: 0.0 for k in self.losses_ord}

    def compute_backward(self, losses):
        """Compute all the gradients using .backward and adds the losses values."""
        for k in self.losses_ord:
            losses[k].backward(**self.losses_arg[k])

    def add_losses(self, losses):
        """TODO."""
        for k in self.losses_ord:
            self.current_losses[k] += losses[k].item()
            if self.episode % 100 == 0:
                add_scalars(
                    f"{self.model_name}/loss/{k}",
                    {
                        self.prefix: self.current_losses[k]
                        / (self.episode * self.batch_size)
                    },
                    (self.epoch * self.data_size + self.episode) / self.data_size,
                )

        self.episode += 1

    def __call__(self, prefix: str, data):
        """TODO."""
        self.batch_size = data.batch_size
        self.data_size = len(data.dataset)
        self.prefix = prefix
        return self

    def __enter__(
        self,
    ):
        """Reset all the losses.

        Usually used after a full loop over all the data.
        """
        self.current_losses = {k: 0.0 for k, _ in self.current_losses.items()}
        self.episode = 1
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Print all losses."""
        for k, v in self.current_losses.items():
            add_scalars(
                f"{self.model_name}/lossAvg/{k}",
                {self.prefix: v / (self.batch_size * self.episode)},
                self.epoch,
            )
        return False
