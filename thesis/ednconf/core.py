"""TODO."""

import inspect as i

import edn_format as edn

import thesis.ednconf.eval as e
import thesis.ednconf.tags as t

tags = [
    ("call", t.to_called),
    ("unary", t.to_unary),
    ("p", t.to_partialed),
    ("ccall", t.to_called_compressed),
]

for name, f in tags:
    edn.add_tag(name, f)


def load_and_resolve(filename: str, basefolder: str = "resources"):
    """TODO."""
    conf = ""
    with open(f"./{basefolder}/{filename}.edn", "r") as f:
        conf = "".join(f.readlines())
    return e.clojure_dict_eval(edn.loads(conf))
