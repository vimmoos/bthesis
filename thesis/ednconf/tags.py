"""TODO."""
from collections.abc import Mapping
from operator import itemgetter
from typing import Any, Callable, Dict, Union

import edn_format as edn

import thesis.ednconf.eval as ev


def _to(d: Dict[str, Any], dict_or_fun: Union[Dict[str, Callable], Callable]):
    """Given a dict or a function returns a ?-function.

    The map should have at least *name* and *args* (all the other keys
    will be ignored):
      + *name* must be a pointer to function
      + *args* can be either a list or a dict of arguments which will be partialed.

    If a function is given then this fuction should be able to behave
    as a unary-function.
    """
    if isinstance(dict_or_fun, Mapping):
        edict = ev.clojure_eval(dict_or_fun)
        f, args = itemgetter("name", "args")(edict)
        return d["dict"](f, args) if isinstance(args, Mapping) else d["list"](f, args)
    if isinstance(dict_or_fun, edn.Keyword):
        return d["keyword"](dict_or_fun)
    if not callable(dict_or_fun):
        raise Exception(
            "The argument must be a Callable or a dict defining \
             how to create such Callable! current argument: {dict_or_fun}"
        )
    return d["fun"](dict_or_fun)


def to_unary(
    dict_or_fun: Union[Dict[str, Any], Callable],
):
    """Given a dict or a function returns a unary-function."""
    return _to(
        {
            "list": lambda f, args: lambda x: f(x, *args),
            "dict": lambda f, args: lambda x: f(x, **args),
            "fun": lambda f: lambda x: f(x),
            "keyword": lambda f: lambda x: ev.clojure_eval(f)(x),
        },
        dict_or_fun,
    )


def to_partialed(
    dict_or_fun: Union[Dict[str, Any], Callable],
):
    """Given a dict or a function returns a partialed-function."""
    return _to(
        {
            "list": lambda f, args: lambda *args1, **kwargs1: f(
                *args, *args1, **kwargs1
            ),
            "dict": lambda f, kwargs: lambda *args1, **kwargs1: f(
                *args1, **kwargs, **kwargs1
            ),
            "fun": lambda f: lambda *args1, **kwargs1: f(*args1, **kwargs1),
            "keyword": lambda f: lambda *args1, **kwargs1: ev.clojure_eval(f)(
                *args1, **kwargs1
            ),
        },
        dict_or_fun,
    )


def to_called(
    dict_or_fun: Union[Dict[str, Any], Callable],
):
    """Given a definition/function returns the calling that function/definition."""
    return _to(
        {
            "list": lambda f, args: f(*args),
            "dict": lambda f, args: f(**args),
            "fun": lambda f: f(),
            "keyword": lambda f: ev.clojure_eval(f)(),
        },
        dict_or_fun,
    )


def rcompress(d):
    """Flatten a dictionary (while merging keys)."""
    while all([isinstance(v, Mapping) for v in d.values()]):
        d = {f"{k}_{k1}": v1 for k, v in d.items() for k1, v1 in v.items()}
    return d


def to_called_compressed(
    dict_or_fun: Union[Dict[str, Any], Callable],
):
    """Given a definition/function returns the calling that function/definition."""
    return _to(
        {
            "list": lambda f, args: f(*args),
            "dict": lambda f, args: f(**rcompress(args)),
            "fun": lambda f: f(),
            "keyword": lambda f: ev.clojure_eval(f)(),
        },
        dict_or_fun,
    )
