from utils import identity, normal, getarg, flatten
import torch.nn as nn
import torch
import sys
from numbers import Number
from functools import reduce

make_helper_fun = {'relu': torch.nn.ReLU, 'sigmoid': torch.nn.Sigmoid}


def setter(self, llist, name, fun, *args):
    if isinstance(fun, str):
        fun = getattr(nn, fun)
    if not callable(fun):
        raise Exception(f"Function for creating layer not callable.\n {fun}")
    self.__setattr__(
        name,
        fun(*[
            sum(llist(x) for x in arg) if isinstance(arg, list) else llist(arg)
            for arg in args
        ]))


def mapsetter(setter, m):
    for k, vs in m.items():
        setter(k, *vs)


def varsetter(setter, *args):
    for i, vs in enumerate(args):
        setter(f"layer{i}", *vs)


def getorder(*args):
    return sorted([
        x for x in list(
            set(
                flatten([list(vs[1:])
                         for _, vs in args[0].items()] if isinstance(
                             args[0], dict) else [list(vs[1:])
                                                  for vs in args])))
        if not isinstance(x, Number)
    ])


def network(*nargs):
    def make_init(fun, args_order):
        def init(self, *args, **kwargs):
            super(type(self), self).__init__()
            llist = getarg(*args, **kwargs)
            for x in args_order:
                llist(x)
            setlayer = lambda *args, **kwargs: setter(self, llist, *args, **
                                                      kwargs)
            fun(setlayer, *nargs)

        return init

    args_order = getorder(*nargs)
    doc = reduce(lambda acc, x: acc + ' ' + str(x) + ',', args_order,
                 '').strip()

    def inner(fun):
        fun.__doc__ = fun.__doc__ if fun.__doc__ is not None else " "
        cl = type(
            fun.__name__, (nn.Module, ), {
                '__args_order__':
                doc,
                '__doc__':
                ":args -> " + doc + "\n\n" + fun.__doc__,
                '__module__':
                fun.__module__,
                '__init__':
                make_init(mapsetter if type(nargs[0]) is dict else varsetter,
                          args_order),
                'forward':
                fun
            })
        setattr(sys.modules[fun.__module__], fun.__name__, cl)
        for k, v in make_helper_fun.items():
            setattr(cl, k, v())
        return cl

    return inner


def KW(fun, *args, **kwargs):
    if not all(map(lambda x: isinstance(x, str), args)):
        raise Exception(
            f"cannot convert one of the args: {args} into kwargs.\n all the args must be string."
        )
    return (fun, {**kwargs, **{x: x for x in args}})


# def network2(**kwfields):
#     def init(self, *args, **kwargs):
#         super(type(self), self).__init__()
#         llist = getarg(*args, **kwargs)
#         # setlayer = lambda *args, **kwargs: setter(self, llist, *args, **kwargs)
#         for k, vs in kwfields.items():
#             val =
#             if isinstance (vs,tuple):
#                 val =
#             self.__setattr__(k,)

#         return init

#     def inner(fun):
#         fun.__doc__ = fun.__doc__ if fun.__doc__ is not None else " "
#         cl = type(
#             fun.__name__, (nn.Module, ), {
#                 '__doc__': fun.__doc__,
#                 '__module__': fun.__module__,
#                 '__init__': init,
#                 'forward': fun
#             })
#         setattr(sys.modules[fun.__module__], fun.__name__, cl)
#         for k, v in make_helper_fun.items():
#             setattr(cl, k, v())
#         return cl

#     return inner


@network(input=(Linear, 'input', 'hidden'),
         hidden=(Linear, 'hidden', 'latent'))
def Encoder(self, x):
    x = torch.flatten(x, start_dim=1)
    x = self.relu(self.input(x))
    return self.hidden(x)
