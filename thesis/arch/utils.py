"""TODO."""
from typing import Dict, List, Tuple, Type

from torch import nn

import thesis.utils as u


def gen_layers_number(
    input: int,
    output: int,
    nlayers: int,
) -> List[Tuple[int, int]]:
    """Generate a list of (dimensions) layers.

    Given the input dimension, the output dimension and the number of wanted
    layers produce a list of tuple with the dimension for each layer
    given the following formula:
    n_i = output*(input/output)^(nlayers-i / nlayers -1)
    for reference see readme.
    """
    nlayers += 1
    inputs = [
        round(output * (input / output) ** ((nlayers - n) / (nlayers - 1)))
        for n in range(1, nlayers + 1)
    ]
    return list(zip(inputs, inputs[1:]))


def make_linear_seq(
    layers: List[Tuple[int, int]],
    inter_act: Type[nn.Module],
    final_act: Type[nn.Module],
) -> nn.Sequential:
    """Create a Sequential module.

    This function takes a list of tuple where each tuple indicate a
    layer. The first int indicates the input dimension and the second
    one indicates the output dimension. Then it takes two activation
    function. The first one will be interleaved between each layer and
    the second one will be used on the last output layer.
    """
    linears = [
        x
        for t in zip(
            [nn.Linear(i, o) for (i, o) in layers], [inter_act() for _ in layers]
        )
        for x in t
    ][:-1]
    if final_act:
        linears.append(final_act())
    return nn.Sequential(*linears)


class Sequential(nn.Module):
    """Create a Sequential Module.

    Basically the same as a nn.Sequential with the main difference it
    returns map from the forward call.

    Takes a list of tuple where each tuple indicate a layer input-output
    dimensions, an intermediate activation function which will be
    interleaved between the layers and lastly a 'closing' activation
    function which will be called on the output of this sequential
    Module.

    Return a dict containing only *out*.
    """

    def __init__(
        self,
        layers: List[Tuple[int, int]],
        inter_act: Type[nn.Module],
        final_act: Type[nn.Module],
    ):
        """Initialize the class."""
        super().__init__()
        self.seq = make_linear_seq(**u.sel_args(locals(), make_linear_seq))

    def forward(self, x) -> Dict:
        """Make a single step.

        Return a dict containing only *out*.
        """
        return {"out": self.seq(x)}
