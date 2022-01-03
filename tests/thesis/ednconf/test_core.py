import inspect as i
import unittest

import thesis.ednconf.core as c


def pippo(a, b, c):
    return a, b, c


# c.to_unary()
# print(i.signature(lambda x: x))

# core.to_unary
# core.clojure_dict_eval
# core.clojure_eval
# core.mini_clojure_eval
def with_doc(func):
    def wrapper(*args, **kwargs):
        print(func.__doc__)
        return func(*args, **kwargs)

    return wrapper


class Test_to_unary(unittest.TestCase):
    @property
    def get_fun(self):
        def pippo(a, b, c):
            return a, b, c

        return pippo

    @with_doc
    def test_signature_with_func(self):
        """TEST signature"""
        res = c.to_unary(lambda x: x)
        params = i.signature(res).parameters
        self.assertEqual(len(params), 1, "a")

    @with_doc
    def test_signature_with_map(self):
        """TEST signature"""

        res = c.to_unary({"name": self.get_fun, "args": {"b": 1, "c": 2}})
        params = i.signature(res).parameters
        self.assertEqual(len(params), 1, "a")
