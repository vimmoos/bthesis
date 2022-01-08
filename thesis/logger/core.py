"""TODO."""
from typing import Dict, List

LOGGER = None
LOGGERS_MAP: Dict = {}
DEFAULT_LOGGER: str = "DEFAULT"
NO_OP_LOGGER: str = "NOOP"
# AVAilable Functions


def log(*args, **kwargs):
    """TODO."""
    global LOGGER
    return LOGGER.log(*args, **kwargs)


def add_scalars(who, what, when):
    """TODO."""
    global LOGGER
    return LOGGER.add_scalars(who=who, what=what, when=when)


def flush():
    """TODO."""
    global LOGGER
    return LOGGER.flush()


# UTILS


def add(logger):
    """TODO."""
    global LOGGERS_MAP
    LOGGERS_MAP[logger.name] = logger
    return True


def detach(name):
    """TODO."""
    global LOGGERS_MAP
    try:
        del LOGGERS_MAP[name]
    except KeyError:
        raise Exception(f"Cannot remove Logger with name: {logger.name}")


def logger(name, best_effort=False):
    """TODO."""
    global LOGGERS_MAP
    try:
        return LOGGERS_MAP[name]
    except KeyError:
        if not best_effort:
            raise Exception(f"Cannot find Logger with name: {name}")
        log("FAILED TO LOAD LOGGER, default logger will be used")
        return LOGGERS_MAP[DEFAULT_LOGGER]


def set_logger(logger):
    """TODO."""
    global LOGGER
    __PREV_LOGGER = LOGGER
    LOGGER = logger
    return __PREV_LOGGER


def no_logger():
    """TODO."""
    global NO_OP_LOGGER
    return logger(NO_OP_LOGGER)
