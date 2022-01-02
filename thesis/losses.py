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
    """TODO."""
    order = {
        v["order"] if "order" in v.keys() else math.inf: k for k, v in kwargs.items()
    }
    sort = sorted(order.keys())
    return [order[x] for x in sort]


class MLosses:
    """TODO."""

    losses_arg: Dict[str, Dict[str, Any]]
    losses_ord: Optional[List[str]]
    current_losses: Dict[str, float]
    cnt: int = 0

    def __init__(self, **kwargs):
        """Initialize class."""
        self.losses_arg = create_args_backward(**kwargs)
        self.losses_ord = create_order(**kwargs)
        self.current_losses = {k: 0.0 for k in self.losses_ord}

    def compute_backward(self, losses):
        """Compute all the gradients using .backward and adds the losses values."""
        self.cnt += 1
        for k in self.losses_ord:
            losses[k].backward(**self.losses_arg[k])
            if self.cnt % 200 == 0:
                print(
                    f"current loss {losses[k].item()}, total : {self.current_losses[k]}"
                )
            self.current_losses[k] += losses[k].item()

    def reset_losses(self):
        """Reset all the losses.

        Usually used after a full loop over all the data.
        """
        print("reset")
        self.cnt = 0
        self.current_losses = {k: 0.0 for k, _ in self.current_losses.items()}

    def print_losses(
        self,
    ):
        """Print all losses."""
        for k, v in self.current_losses.items():
            print(f"Loss {k}: {v/self.cnt}")
