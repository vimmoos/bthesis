"""Utils module, contains misc utility functions."""

import inspect as i
import statistics as stats
from collections.abc import Mapping
from operator import itemgetter
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
from torchvision import datasets, transforms


def sel_and_rm(kw: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Select only the entries which starts with the prefix and remove the prefix."""
    lpre = len(prefix)
    return {k[lpre:]: v for k, v in kw.items() if prefix == k[0:lpre]}


def sel_args(kw: Dict[str, Any], fun: Callable) -> Dict[str, Any]:
    """Select only the args needed by the function from the kw."""
    return {k: v for k, v in kw.items() if k in list(i.signature(fun).parameters)}


def to_unary(
    dict_or_fun: Union[Dict[str, Any], Callable],
):
    """Given a dict or a function returns a unary-function.

    The resulting unary function will have *x* as parameter
    name. *NOTE* x will be used as positional argument in the
    partialed function.

    The map should have at least *name* and *args* (all the other keys
    will be ignored):
      + *name* must be a pointer to function
      + *args* can be either a list or a dict of arguments which will be partialed.

    If a function is given then this fuction should be able to behave
    as a unary-function.
    """
    if isinstance(dict_or_fun, Mapping):
        f, args = itemgetter("name", "args")(dict_or_fun)
        return (
            (lambda x: f(x, **args))
            if isinstance(args, Mapping)
            else (lambda x: f(x, *args))
        )
    if not callable(dict_or_fun):
        raise Exception(
            "The argument must be a Callable or a dict defining \
             how to create such Callable! current argument: {dict_or_fun}"
        )
    return lambda x: dict_or_fun(x)


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
        self.seq = make_linear_seq(**sel_args(locals(), make_linear_seq))

    def forward(self, x) -> Dict:
        """Make a single step.

        Return a dict containing only *out*.
        """
        return {"out": self.seq(x)}


def load_data():
    """TODO."""
    tensor_transform = transforms.ToTensor()

    train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Lambda(
            lambda x: tensor_transform(x).reshape(-1, 28 * 28)
        ),
    )
    test = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Lambda(
            lambda x: tensor_transform(x).reshape(-1, 28 * 28)
        ),
    )

    return (
        torch.utils.data.DataLoader(dataset=train, batch_size=32, shuffle=True),
        torch.utils.data.DataLoader(dataset=test, batch_size=32, shuffle=True),
    )


def msd(arr):
    """TODO."""
    return (stats.mean(arr), stats.stdev(arr))
