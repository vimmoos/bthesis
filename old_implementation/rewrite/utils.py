import torch.nn as nn
import torch
import sys
from numbers import Number
from functools import reduce

identity = lambda x: x

normal = torch.distributions.Normal(0, 1)

car = lambda x: x[0]


def getarg(*args, **kwargs):
    known = {}
    gargs = (x for x in args)

    def call(who):
        nonlocal known
        if who in known.keys():
            return known.get(who)
        if isinstance(who, Number):
            return who
        try:
            known[who] = kwargs.get(who) if who in kwargs.keys() else next(
                gargs)
        except StopIteration:
            raise Exception(f"Missing argument {who}")
        return known[who]

    return call


def flatten(x):
    if not isinstance(x, list):
        return [x]
    return reduce(
        lambda acc, x: [*flatten(x), *acc]
        if isinstance(x, list) else [x, *acc], x, [])
