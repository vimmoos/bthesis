import torch.nn as nn
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from more_itertools import peekable
from collections import Iterable
import torch


def identity(x):
    return x


# class ABCAutoencoder(ABC, nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(args, kwargs)

#     @abstractmethod
#     def forward(self, x):
#         pass

#     @abstractmethod
#     def learn(self, data, epoch):
#         pass

# x = {
#     "name": "gna",
#     "layers": {
#         "linear1": [nn.Linear, "inp", "out"],
#         "linear2": [nn.Linear, "out", "mid"]
#     },
#     "forward": [("linear1", identity, "relu"),
#                 ("linear2", identity, "sigmoid")]
# }


@dataclass
class ti():
    x: int = lambda s: int(s)


def nonmi(arg1):
    print(arg1)

    def mi(cl):
        def wrapper(*args, **kwargs):
            print(dir(cl))
            return cl

        return wrapper

    return mi


def dio(x):
    return x


@nonmi(["x", 2])
class gna():
    x = 0


def tes(*args):
    gargs = peekable(args)
    return gargs.peek(), next(gargs), next(gargs)


{
    "ti": ("Linear", "input_size", "hidden"),
    "mi": {"Linear", "hidden", "output"}
}


def getarg(*args, **kwargs):
    known = {}
    gargs = (x for x in args)

    def call(who):
        nonlocal known
        if who in known.keys():
            return known.get(who)
        try:
            known[who] = kwargs.get(who) if who in kwargs.keys() else next(
                gargs)
        except StopIteration:
            raise Exception(f"Missing argument {who}")
        return known[who]

    return call


def porco(*args, **kwargs):
    llist = getarg(*args, **kwargs)
    for x in ['x', 'x', 'z', 'g']:
        print(llist(x))


def setLayer(self, llist, name, torchfun, *args):
    if isinstance(torchfun, str):
        torchfun = getattr(nn, torchfun)
    if not callable(torchfun):
        raise Exception(
            f"Function for creating layer not callable.\n {torchfun}")
    self.__setattr__(
        name,
        torchfun(*[
            sum(llist(x) for x in arg) if isinstance(arg, list) else llist(arg)
            for arg in args
        ]))


helper_fun = {'relu': torch.nn.ReLU(), 'sigmoid': torch.nn.Sigmoid()}


def mapsetter(setter, m):
    for k, vs in m.items():
        setter(k, *vs)


def varsetter(setter, *args):
    for i, vs in enumerate(args):
        setter(f"layer{i}", *vs)


def network(*nargs):
    def make_init(old_init, fun):
        def init(self, *args, **kwargs):
            super(type(self), self).__init__()
            llist = getarg(*args, **kwargs)
            setlayer = lambda *args, **kwargs: setLayer(
                self, llist, *args, **kwargs)
            fun(setlayer, *nargs)
            old_init(self)

        return init

    def inner(cl):

        setattr(
            cl, '__init__',
            make_init(getattr(cl, '__init__'),
                      mapsetter if type(nargs[0]) is dict else varsetter))
        for k, v in helper_fun.items():
            setattr(cl, k, v)
        return cl

    return inner


@network({
    'ti1': ('Linear', ['input_size', "latent"], 'latent'),
    'ti2': ('Linear', 'latent', 'output')
})
class madonna(torch.nn.Module):
    pass


@network(('Linear', 'input_size', 'latent'), ('Linear', 'latent', 'output'))
class madonna():
    pass
