"""
_ecosystem.py -- load functions FROM the ecosystem, never re-implement them.

WHY THIS EXISTS
    The project's rule is to reuse the mature JEV code and adapt it, not to rewrite it. That rule has
    a sharp edge: a copied function is a fork. The moment two copies exist, one of them is edited and
    the pair diverges, and the divergence is invisible until a number disagrees -- and by then the
    number has already been used in a keep decision.

    So this module LOADS the ecosystem's functions from the external checkout at runtime. There is
    exactly one copy of load_tau; this project simply calls it. If the ecosystem changes it, this
    project gets the change.

    Loading by file path rather than by package import is necessary and not a workaround: the
    ecosystem is a separate git clone on another volume, not an installed package, and its scripts
    directory is not a package (no __init__.py). importlib.util.spec_from_file_location is the
    supported way to execute a module from an explicit path.

WHAT IS AND IS NOT LOADED
    Only pure, side-effect-free helpers. The ecosystem agent's main(), its git operations and its
    training subprocess launches are deliberately NOT reachable from here: a module that resets git
    or launches a 2.11 h training run is not a function to import casually, and a gate that could
    trigger one would be a gate nobody runs.

    If a future adaptation needs a richer helper, it goes here with the same rule: load, do not copy.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

# Project root, derived from this file's location rather than hardcoded: a literal machine path makes
# the repository uncloneable and unrunnable elsewhere (Rule 1 of docs/PROJECT_RULES.md).
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import require, subject                                              # noqa: E402

# Pure helpers only. Anything that mutates git, writes into the subject repo, or launches training
# stays out of this list on purpose.
PURE_HELPERS = ("load_tau", "parse_metrics", "get_baseline_accuracy", "run_cmd")

_MODULE_CACHE: dict[str, object] = {}


def ecosystem_module():
    """Import agent-jev's autoresearch_agent.py by path, once, and cache it.

    Cached because the module reads nothing at import time but re-executing it on every call would
    make a gate's cost depend on how many times it is called, which is the sort of thing that makes
    people cache the gate instead.
    """
    if "agent" in _MODULE_CACHE:
        return _MODULE_CACHE["agent"]

    subj = require(subject(), "the agent-jev subject repository (JEVRSI_SUBJECT)")
    path = subj / "scripts" / "autoresearch_agent.py"
    if not path.exists():
        raise FileNotFoundError(
            f"ecosystem agent not found at {path}. This project drives the agent-jev loop rather "
            f"than reimplementing it, so the subject repository must be present.")

    spec = importlib.util.spec_from_file_location("jevrsi_ecosystem_agent", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path} as a module")
    mod = importlib.util.module_from_spec(spec)
    # Registering before exec_module matters: the ecosystem module has no package of its own, and
    # dataclasses/typing inside it resolve __module__ through sys.modules during execution.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE["agent"] = mod
    return mod


def _delegate(name: str):
    if name not in PURE_HELPERS:
        raise AttributeError(
            f"{name!r} is not exposed from the ecosystem. Only {PURE_HELPERS} are loadable, because "
            f"they are side-effect free. Anything that mutates git or launches training is called "
            f"by the loop itself, not imported by a gate.")

    def wrapper(*args, **kwargs):
        return getattr(ecosystem_module(), name)(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = f"ecosystem.{name}"
    wrapper.__doc__ = (f"The ecosystem's own {name}(), loaded from agent-jev at call time.\n\n"
                       f"Delegated rather than copied so there is one implementation. See "
                       f"adaptation/_ecosystem.py for why.")
    return wrapper


load_tau = _delegate("load_tau")
parse_metrics = _delegate("parse_metrics")
get_baseline_accuracy = _delegate("get_baseline_accuracy")
run_cmd = _delegate("run_cmd")


if __name__ == "__main__":
    print("=" * 78)
    print("ecosystem bridge -- functions are LOADED from agent-jev, not copied here")
    print("=" * 78)
    try:
        mod = ecosystem_module()
    except Exception as e:                                      # noqa: BLE001
        raise SystemExit(f"[FAIL] {type(e).__name__}: {e}")
    print(f"  loaded from  {getattr(mod, '__file__', '?')}")
    print(f"  exposed      {', '.join(PURE_HELPERS)}")
    missing = [n for n in PURE_HELPERS if not hasattr(mod, n)]
    if missing:
        raise SystemExit(f"[FAIL] the ecosystem module does not define: {missing}. "
                         f"It may have been refactored; re-check this bridge.")
    print("  [OK] every exposed helper resolves in the ecosystem module")
