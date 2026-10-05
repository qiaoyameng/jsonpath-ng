from .jsonpath import *  # noqa
from .parser import parse  # noqa


# Current package version
__version__ = '1.8.0'

from .sqlite import (  # noqa
    CompiledJSONPath,
    JSONPathError,
    JSONPathNotSupported,
    JSONPathSyntaxError,
    Match,
    compile_jsonpath,
    find as find_jsonpath_sqlite,
)
