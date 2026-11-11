"""Compile JSONPath ASTs to SQLite JSON1 SQL.

The :func:`compile_jsonpath` entry point turns a supported JSONPath
expression into a SQL fragment, bind parameters, and documentation of the
supported scope. The generated SQL evaluates the query entirely inside the
database using only the SQLite JSON1 extension, so no ORM or database
service is required.
"""

from jsonpath_ng.ext.sqlite.compiler import JSON1Compilation
from jsonpath_ng.ext.sqlite.compiler import SUPPORTED_SCOPE
from jsonpath_ng.ext.sqlite.compiler import compile_jsonpath

__all__ = ["JSON1Compilation", "SUPPORTED_SCOPE", "compile_jsonpath"]
