"""Vanilla Autoencoder."""
from typing import Dict, List, Tuple, Type

import torch
from torch import nn

import thesis.arch.utils as ua
import thesis.utils as u


@u.with_hparams
class VanillaAutoencoder(nn.Module):
    """Vanilla AE.

    Uses a Sequential model for both encoding and decoding. It
    implements the standard Autoencoder.

    At each step it returns a dict containing the following keys:
      + *out* -> the output of decode(encode(x))
    """

    def __init__(
        self,
        enc_layers: List[Tuple[int, int]],
        dec_layers: List[Tuple[int, int]],
        enc_inter_act: Type[nn.Module] = nn.ReLU,
        enc_final_act: Type[nn.Module] = nn.Identity,
        dec_inter_act: Type[nn.Module] = nn.ReLU,
        dec_final_act: Type[nn.Module] = nn.Sigmoid,
    ):
        """Initialize the class."""
        kwargs = locals()
        super().__init__()
        self.encoder = torch.jit.script(
            u.sapply(ua.Sequential, kwargs, "enc_"),
        )
        self.decoder = torch.jit.script(
            u.sapply(ua.Sequential, kwargs, "dec_"),
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """Make a single step of the AE.

        Returns a dict containing the following keys:
           + *out* -> the output of decode(encode(x))
        """
        latent = self.encoder(x)["out"]
        return {"out": self.decoder(latent)["out"], "latent": latent}
