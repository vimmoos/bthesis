"""TODO."""
from collections.abc import Mapping
from typing import Callable, Dict, Type

import torch.optim as optim
from torch import nn

import thesis.ednconf.core as ec
import thesis.utils as u


def create_optims(
    **kwargs,
) -> Callable[[nn.Module], Dict[str, Type[optim.Optimizer]]]:
    """Given some kwargs it returns a unary-function which instantiate the optimizers.

    The kwargs should be of the form:
        {name:
          {"optim":
            {"name": optim,
             "args": [] or {}}
             or  function-pointer}}

    All extra keys will be ignored beside in the first layers where
    the keys should indicate some name for the optimizer.

    *NOTE* the value of each of the "optim" key will be passed to
    *utils.to_unary* function. For more info see its doc.

    The unary-function returned will create a dict with keys the name
    of the optimizers(defined in *kwargs*) and as value the
    instantiation of these optimizer (as instructed by the *kwargs*)
    """
    partial_optim = {k: v["optim"] for k, v in kwargs.items()}

    has_multi = len(kwargs.keys()) >= 2

    if len(kwargs.keys()) == 0:
        raise Exception("No Optimazer supplied!")

    def _create_optim(model: nn.Module) -> Dict[str, Type[optim.Optimizer]]:
        """Return a dict from str to Optimizer."""
        params = model.parameters()

        if not isinstance(params, Mapping):
            if has_multi:
                raise Exception(
                    f"Cannot create optimizers there are too many keys compared\
                     to params group! keys: {kwargs.keys()} params: {params}\n"
                )
            return {k: v(params) for k, v in partial_optim.items()}
        if len(params.keys() - kwargs.keys()) > 0:
            print(
                f"=============WARNING!==================\
                  Params has more keys (or different) than the ones supplied to create the optimazer.\
                  Most likely something is wrong! Procede with CAUTION!\
                  params keys {params.keys()} supplied keys {kwargs.keys()}"
            )
        return {k: v(params[k]) for k, v in partial_optim.items()}

    return _create_optim


class MOptims:
    """Manager of Optimizers.

    It 'sincronize' a dictionary of optimizers. Generally do not call manually
    the optimizers. If some particular behavior is needed
    from only part/one of them then procede with caution.
    """

    _ignite: Callable
    optims: Dict[str, Type[optim.Optimizer]]

    def __init__(self, **kwargs):
        """Initialize class."""
        self._ignite = create_optims(**kwargs)

    def ignite(self, model):
        """Given a model initialize the optimizers."""
        self.optims = self._ignite(model)

    def zero_grad(self):
        """Perform zero_grad on all optimizer."""
        for k, v in self.optims.items():
            v.zero_grad()

    def step(self):
        """Perform step on all optimizer."""
        for k, v in self.optims.items():
            v.step()

    def __enter__(
        self,
    ):
        self.zero_grad()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.step()
        return False
