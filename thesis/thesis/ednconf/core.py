"""TODO."""

import edn_format as edn

import thesis.ednconf.eval as e


def load(filename: str, basefolder: str):
    """Load an edn file with the reader macro defined in tags.edn."""
    with open("./thesis/ednconf/tags.edn", "r") as f:
        tags = e.clojure_eval(edn.loads("".join(f.readlines())))
        for name, f in tags.items():
            edn.add_tag(name, f)
    with open(f"./{basefolder}/{filename}.edn", "r") as f:
        return edn.loads("".join(f.readlines()))


def load_and_resolve(filename: str, basefolder: str = "resources"):
    """Load and resolve an edn file."""
    return e.clojure_eval(load(filename, basefolder))
