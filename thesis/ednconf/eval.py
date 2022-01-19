"""TODO."""
import importlib as imp
import operator as op
import sys
from numbers import Number
from typing import Any, Dict, List, Union

from edn_format import ImmutableDict, ImmutableList, Keyword

OPS = {
    "+": op.add,
    "-": op.sub,
    "*": op.mul,
    "/": op.truediv,
    "%": op.mod,
    "^": op.xor,
}


def eval_sexp(sexp):
    """Trivially eval a clojure sexp.

    If the car of the sexp starts with something different then a
    keyword it returns directly the sexp with doing anything.
    otherwise it tries to eval the sexp. The only operations supported
    are (*NOTE* they are all applied in a variadic manner):
       + "+"
       + "-"
       + "*"
       + "/"
       + "%"
       + "^"
    """
    car, *rest = sexp
    if isinstance(car, Number):
        return sexp
    if not hasattr(car, "name"):
        return sexp
    if car.name not in OPS:
        raise Exception(f"Unable to evaluate the following expression: {sexp}")
    ret = None
    for el in rest:
        el = clojure_eval(el)
        ret = OPS[car.name](ret, el) if ret else el
    return ret


def eval_keyword(keyword: Keyword):
    """Eval a keyword.

    If the keyword does not have namespace it returns a string
    otherwise it tries to resolve the namespace and returns the
    appropriate function. *NOTE* it also imports the namespace in the
    python environment
    """
    if not keyword.namespace:
        return keyword.name
    imp.import_module(keyword.namespace)
    return getattr(sys.modules[keyword.namespace], keyword.with_namespace("").name)


def eval_dict(dict_: Union[Dict[Any, Any], ImmutableDict]):
    """Eval a dict.

    Applies for every k,v the clojure_eval function
    """
    return {clojure_eval(k): clojure_eval(v) for k, v in dict_.items()}


def eval_list(list_: Union[List[Any], ImmutableList]):
    """Eval a list.

    Applies for every element the clojure_eval function
    """
    return [clojure_eval(el) for el in list_]


def clojure_eval(v):
    """Eval a clojure expression."""
    return {
        Keyword: eval_keyword,
        tuple: eval_sexp,
        list: eval_list,
        ImmutableList: eval_list,
        dict: eval_dict,
        ImmutableDict: eval_dict,
    }.get(type(v), lambda x: x)(v)
