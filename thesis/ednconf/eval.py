"""TODO."""
import importlib as imp
import operator as op
import sys
from collections.abc import Mapping
from numbers import Number

import edn_format as edn

OPS = {
    "+": op.add,
    "-": op.sub,
    "*": op.mul,
    "/": op.truediv,
    "%": op.mod,
    "^": op.xor,
}


def mini_clojure_eval(sexp):
    """TODO."""
    car, *rest = sexp
    if isinstance(car, Number):
        return sexp
    if car.name not in OPS:
        raise Exception(f"Unable to evaluate the following expression: {sexp}")
    ret = None
    for el in rest:
        el = clojure_eval(el)
        ret = OPS[car.name](ret, el) if ret else el
    return ret


def clojure_eval(v):
    """TODO."""
    if isinstance(v, tuple):
        return mini_clojure_eval(v)
    if isinstance(v, list):
        return [mini_clojure_eval(x) for x in v]
    if isinstance(v, Number) or isinstance(v, str):
        return v
    if not isinstance(v, edn.Keyword):
        return v
    if not v.namespace:
        raise Exception(
            f"Unable to resolve symbol! \
        Probably missing namespace symbol: {v}"
        )
    imp.import_module(v.namespace)
    return getattr(sys.modules[v.namespace], v.with_namespace("").name)


def clojure_dict_eval(di: edn.immutable_dict):
    """TODO."""
    return {
        getattr(k, "name", k): clojure_dict_eval(v)
        if isinstance(v, Mapping)
        else clojure_eval(v)
        for k, v in di.items()
    }
