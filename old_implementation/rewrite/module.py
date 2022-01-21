import sys


class Keyword:
    def __getattribute__(self, name):
        return name


sys.modules[__name__] = Keyword()
