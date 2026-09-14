"""Loads user-defined Python functions referenced from a schema as ``custom:...``.

Same philosophy as ``~/franka-policy-runner/policy_runner/policies/
callable_adapter.py``'s ``CallableAdapter``: rather than pre-building a
generic pipeline for every possible preprocessing step (which is impossible
-- e.g. a language-instruction embedding needs a model the user chooses),
give one small, honest extension point. The user writes a plain function in
a local file and references it by name from the schema editor; no plugin
API to subclass, no registration boilerplate.

Plugin functions receive a ``SourceContext`` (schema/sources.py: ``.cameras``,
``.robot_state``) and return whatever value that request field needs, e.g.:

    # ~/franka-deploy/user_plugins.py
    import numpy as np

    def embed_instruction(ctx):
        # load your own text encoder however you like; this is just an
        # example shape the schema can reference as "custom:user_plugins.embed_instruction"
        return np.zeros((1, 512), dtype=np.float32)
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable

# Directory searched for a plugin module named in a "custom:<module>.<fn>"
# source, before falling back to a regular importable package. Override with
# FRANKA_DEPLOY_PLUGIN_DIR if the project lives somewhere else.
PLUGIN_DIR = Path(os.environ.get("FRANKA_DEPLOY_PLUGIN_DIR", Path(__file__).resolve().parents[2]))

_cache: dict[str, Callable] = {}


def load_plugin(dotted_path: str) -> Callable:
    """Resolve "<module>.<function>" (or bare "<function>" from the default
    ``user_plugins.py``) to a callable. Cached after first load."""
    if dotted_path in _cache:
        return _cache[dotted_path]

    if "." in dotted_path:
        module_name, fn_name = dotted_path.rsplit(".", 1)
    else:
        module_name, fn_name = "user_plugins", dotted_path

    module = _load_module(module_name)
    if not hasattr(module, fn_name):
        raise AttributeError(f"plugin module {module_name!r} has no function {fn_name!r}")
    fn = getattr(module, fn_name)
    _cache[dotted_path] = fn
    return fn


def _load_module(module_name: str):
    candidate = PLUGIN_DIR / f"{module_name}.py"
    if candidate.exists():
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    # Fall back to a normally-importable module (e.g. a real package).
    return importlib.import_module(module_name)


def clear_cache() -> None:
    """Used by tests / the config-reload endpoint after editing user_plugins.py."""
    _cache.clear()
