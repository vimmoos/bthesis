"""VAE."""

from typing import Dict, List, Tuple, Type

from torch import nn
from torch.autograd import Variable

import thesis.encoders as enc
import thesis.utils as u


def reparameterize(training: bool, mu: Variable, logvar: Variable) -> Variable:
    """Perform the reparameterisation trick (needed for differentiating the sampling operations).

    Stochastic if *training* is True else returns the mean (*mu*)
    """
    if training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    # Potentially take a random sample? ofc mu has the highest
    # chance of beeing selected
    return mu


class VAE(nn.Module):
    """Variational AutoEncoder Module.

    Uses a *VariationalEncoder* as encoder (module encoders) and a
    Sequential for the decoder part. It implements the standard
    Variational AutoEncoder.

    Each step of it returns a dict containing the following keys:
      + *out* ->  the output of the decoder
      + *mu* and *var*  -> see *VariationalEncoder* for info
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
        self.encoder = enc.VariationalEncoder(**u.sel_and_rm(kwargs, "enc_"))
        self.decoder = u.Sequential(**u.sel_and_rm(kwargs, "dec_"))

    def forward(self, x) -> Dict:
        """Perform a single step of the VAE (x -> encoder(x) -> z -> decode(z) -> x').

        Returns a map containing the following keys:
           + out -> the output of the step
           + mu  -> the mean of the latent dist
           + var -> the log variance of the lantent dist
        """
        latent = self.encoder(x)
        z = reparameterize(self.training, latent["mu"], latent["var"])
        return {"out": self.decoder(z), **latent}
