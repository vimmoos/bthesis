"""TODO."""
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def ELBOLoss(out: Tensor, mu: Tensor, var: Tensor, x: Tensor) -> Dict:
    """Compute the ELBO loss.

    *NOTE* it uses the binary cross entropy to compare out to x

    Returns a map containing the following keys:
       + out -> the ELBO loss
    """
    KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
    # normalize by Batchsize * inputsize
    KLD /= x.shape[0] * x.shape[1]
    return {"ae": F.binary_cross_entropy(out, x) + KLD}


__mseloss = torch.nn.MSELoss()


def mse(out, x):
    return {"ae": __mseloss(out, x)}


def ELBOWithDiscLoss(
    out: Tensor, logprior: Tensor, logpost: Tensor, x: Tensor
) -> Dict:
    """Compute the ELBO loss and the discriminator loss.

    *NOTE* it uses the binary cross entropy for both the *disc* and *ae* losses

    Returns a map containing the following keys:
      + disc -> the discriminator loss
      + ae   -> the autoencoder loss
    """
    disc_loss = torch.mean(
        F.binary_cross_entropy_with_logits(logprior, torch.zeros_like(logprior))
        + F.binary_cross_entropy_with_logits(logpost, torch.ones_like(logpost))
    )
    recon_lh = -F.binary_cross_entropy(out, x) * x.data.shape[0]
    return {"disc": disc_loss, "ae": torch.mean(logpost) - torch.mean(recon_lh)}


def create_args_backward(**kwargs):
    """TODO."""
    return {
        k: v["backward"] if "backward" in v.keys() else {} for k, v in kwargs.items()
    }


def create_order(**kwargs):
    """TODO. there is a bug here when two or more keys do not have an order key"""
    order = {v.get("order", math.inf): k for k, v in kwargs.items()}
    return [order[x] for x in sorted(order.keys())]


class MLosses:
    """TODO."""

    losses_arg: Dict[str, Dict[str, Any]]
    losses_ord: Optional[List[str]]
    current_losses: Dict[str, float]
    data_size: int = 1

    def __init__(self, **kwargs):
        """Initialize class."""
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
            # if self.cnt % 200 == 0:
            #     print(
            #         f"current loss {losses[k].item()}, total : {self.current_losses[k]}"
            #     )
            self.current_losses[k] += losses[k].item()

    def reset_losses(self):
        """Reset all the losses.

        Usually used after a full loop over all the data.
        """
        self.current_losses = {k: 0.0 for k, _ in self.current_losses.items()}

    def print_losses(
        self,
    ):
        """Print all losses."""
        for k, v in self.current_losses.items():
            print(f"{self.prefix}->Loss {k}: {v/self.data_size}")

    def __call__(self, prefix: str, data_size: int):
        """TODO."""
        self.data_size = data_size
        self.prefix = prefix
        return self

    def __enter__(
        self,
    ):
        self.reset_losses()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.print_losses()
        return False
