import warnings

from ._deprecations import _DEPRECATED_CLASS_NAMES
from ._deprecations import _deprecated_class_message
from .jsonpath import *  # noqa
from .parser import parse  # noqa
from .update_plan import ConflictUpdatePlanError  # noqa
from .update_plan import StaleUpdatePlanError  # noqa
from .update_plan import UpdateConflict  # noqa
from .update_plan import UpdateOperation  # noqa
from .update_plan import UpdatePlan  # noqa
from .update_plan import UpdatePlanConflict  # noqa
from .update_plan import UpdatePlanConflictError  # noqa
from .update_plan import UpdatePlanError  # noqa
from .update_plan import UpdatePlanStale  # noqa
from .update_plan import UpdatePlanStaleError  # noqa
from .update_plan import UpdatePlanner  # noqa
from .update_plan import UpdateRule  # noqa
from .update_plan import UnsupportedUpdateDataError  # noqa
from .update_plan import apply_update_plan  # noqa
from .update_plan import build_update_plan  # noqa
from .update_plan import create_update_plan  # noqa
from .update_plan import make_update_plan  # noqa
from .update_plan import plan_update  # noqa
from .update_plan import update_plan  # noqa


def __getattr__(name: str) -> object:
    if name in _DEPRECATED_CLASS_NAMES:
        from . import jsonpath

        warnings.warn(_deprecated_class_message(name), DeprecationWarning, stacklevel=2)
        return getattr(jsonpath, f"_{name}")
    if name == "__version__":
        msg = (
            "The `__version__` attribute is deprecated. "
            "It will be removed in version 2.0.0. "
            'Use `importlib.metadata.version("jsonpath-ng")` instead.'
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        import importlib.metadata

        return importlib.metadata.version("jsonpath-ng")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
