"""TODO."""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# from torch.distributions import Normal
# from torch.nn import KLDivLoss


class ELBOLoss(nn.Module):
    """TODO."""

    def forward(
        self, out: Tensor, mu: Tensor, var: Tensor, target: Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute the ELBO loss.

        *NOTE* it uses the binary cross entropy to compare out to x

        Returns a map containing the following keys:
        + ae -> the ELBO loss
        """
        KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        # normalize by Batchsize * inputsize
        KLD /= target.shape[0] * target.shape[1]
        BCE = F.binary_cross_entropy(out, target)
        return {"ae": BCE + KLD}


class MSELoss(nn.Module):
    """TODO."""

    def __init__(self):
        """TODO."""
        super().__init__()
        self.fun = torch.nn.MSELoss()

    def forward(self, target, out):
        """TODO."""
        return {"ae": self.fun(out, target)}


class ELBOWithDiscLoss(nn.Module):
    """TODO."""

    def forward(
        self, out: Tensor, logprior: Tensor, logpost: Tensor, target: Tensor
    ) -> Dict[str, torch.Tensor]:
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
        recon_lh = -F.binary_cross_entropy(out, target) * target.data.shape[0]
        return {"disc": disc_loss, "ae": torch.mean(logpost) - torch.mean(recon_lh)}
