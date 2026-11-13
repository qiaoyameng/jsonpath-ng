#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""Parameterized JSONPath queries.

A template is parsed exactly once, then executed any number of times with
different parameter dictionaries. Parameters are declared in the query with
the ``:name`` (or ``:'name'``) placeholder syntax and may only appear where a
filter comparison expects a literal value, so they can never inject field
names, operators, or path segments.
"""

from __future__ import annotations

from contextvars import Token
import typing

from ..exceptions import DuplicateParameterError
from ..exceptions import MissingParameterError
from ..exceptions import UnsupportedParameterTypeError
from ..jsonpath import DatumInContext
from ..jsonpath import JSONPath
from .filter import _current_bindings
from .filter import Parameter
from .parser import ExtendedJsonPathParser

#: The JSON scalar types a parameter value is allowed to have.
#: ``bool`` is listed explicitly because it is a subclass of ``int``.
_ALLOWED_PARAMETER_TYPES = (type(None), bool, int, float, str)


def _collect_parameters(path: JSONPath) -> list[str]:
    """Walk the AST and return parameter names in first-seen order."""

    names: list[str] = []
    seen: set[str] = set()
    pending: list[JSONPath] = [path]
    while pending:
        node = pending.pop()
        if isinstance(node, Parameter):
            if node.name in seen:
                raise DuplicateParameterError(
                    f"Parameter {node.name!r} is declared more than once"
                )
            seen.add(node.name)
            names.append(node.name)
            continue
        if isinstance(node, (list, tuple)):
            pending.extend(node)
            continue
        for attribute in (
            "left",
            "right",
            "target",
            "value",
            "expressions",
            "operands",
        ):
            child = getattr(node, attribute, None)
            if child is None:
                continue
            if isinstance(child, JSONPath):
                pending.append(child)
            elif isinstance(child, (list, tuple)):
                pending.extend(item for item in child if isinstance(item, JSONPath))
    return names


class ParameterizedQuery:
    """A parsed template that binds parameters anew on every execution."""

    def __init__(
        self,
        template: str,
        parser: ExtendedJsonPathParser | None = None,
    ) -> None:
        parser = parser or ExtendedJsonPathParser()
        self.template = template
        self.path: JSONPath = parser.parse(template)
        self.parameters: tuple[str, ...] = tuple(_collect_parameters(self.path))

    def _bind(
        self, params: dict[str, typing.Any]
    ) -> Token[dict[str, typing.Any] | None]:
        if not isinstance(params, dict):
            raise TypeError(
                f"Parameter bindings must be a dict, got {type(params).__name__}"
            )

        # Every problem with the bindings is reported before the document is
        # touched, so a failed execution never partially evaluates the data.
        missing = [name for name in self.parameters if name not in params]
        if missing:
            raise MissingParameterError(
                "Missing values for parameter(s): "
                + ", ".join(repr(name) for name in missing)
            )

        for name in self.parameters:
            value = params[name]
            if not isinstance(value, _ALLOWED_PARAMETER_TYPES):
                raise UnsupportedParameterTypeError(
                    f"Parameter {name!r} has an unsupported type "
                    f"{type(value).__name__}; allowed types are null, bool, "
                    "int, float, and str"
                )

        return _current_bindings.set({name: params[name] for name in self.parameters})

    def find(
        self, data: typing.Any, params: dict[str, typing.Any]
    ) -> list[DatumInContext]:
        token = self._bind(params)
        try:
            return self.path.find(data)
        finally:
            _current_bindings.reset(token)

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.template!r})"


def parse_template(
    template: str, parser: ExtendedJsonPathParser | None = None
) -> ParameterizedQuery:
    """Parse *template* once into a reusable `ParameterizedQuery`."""

    return ParameterizedQuery(template, parser=parser)
