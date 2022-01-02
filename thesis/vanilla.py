"""Vanilla Autoencoder."""
from typing import Dict, List, Tuple, Type

from torch import nn

import thesis.utils as u


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
        self.encoder = u.Sequential(**u.sel_and_rm(kwargs, "enc_"))
        self.decoder = u.Sequential(**u.sel_and_rm(kwargs, "dec_"))

    def forward(self, x) -> Dict:
        """Make a single step of the AE.

        Returns a dict containing the following keys:
           + *out* -> the output of decode(encode(x))
        """
        return self.decoder(self.encoder(x)["out"])
