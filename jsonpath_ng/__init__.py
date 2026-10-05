from .jsonpath import *  # noqa
from .parser import parse  # noqa


def parse_rfc9535(path):
    from .rfc9535 import parse as rfc9535_parse

    return rfc9535_parse(path)


# Current package version
__version__ = '1.8.0'
