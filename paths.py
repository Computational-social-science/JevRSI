"""paths.py -- the single place that resolves where things live.

WHY THIS EXISTS

Rule 1 of docs/PROJECT_RULES.md forbids absolute paths in code, because a repository containing
a hardcoded drive-letter path cannot be cloned and run by anyone else. The obvious migration -- replace each literal
with a path derived from `__file__` -- was applied on 2026-09-29 and immediately produced a real defect:
twelve modules each guessed that the external subject clone lives at `ROOT.parent / "agent-jev"`, and it
does not. The pure-relative path was wrong in a way no guard could see, because a guard can only tell that
a literal is absent, not that a derived default points at the wrong place.

So the locations that genuinely cannot be derived -- the external `agent-jev` clone, the downloaded base
model, the borrowed autoresearch harness -- are declared ONCE in config/paths.json, relative to the
project root, and read here. Three properties follow:

  * one place to change when a layout differs, instead of twelve
  * the file itself is relative, so it survives being cloned to another machine
  * an unresolvable path RAISES, rather than falling back to something plausible, because a run that
    proceeds with the wrong path produces numbers attributed to a checkpoint it never loaded -- and that
    is the failure mode Rule 3 of the scientific standard exists to prevent.

Precedence: environment variable, then config/paths.json, then raise.

Usage:
    from paths import subject, backbone
    rows = ev.load_split("dev")           # the evaluator resolves its own paths relatively
"""

import json
import os
import pathlib

# Project root, derived from this file's location rather than hardcoded: a literal machine path makes
# the repository uncloneable and unrunnable elsewhere (Rule 1 of docs/PROJECT_RULES.md).
ROOT = pathlib.Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "paths.json"


def _cfg() -> dict:
    if not CONFIG.exists():
        raise SystemExit(f"[fatal] {CONFIG} is missing. It is the single place that declares the "
                         f"project's external locations; without it no module can resolve them.")
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _resolve(key: str, env_key: str | None = None) -> pathlib.Path:
    """env var, else config, then raise. Relative entries resolve against the project root.

    Existence is NOT checked here. That check belongs in the caller, because a location may legitimately
    be produced later in a run (a bundle that has not been generated yet), and a resolver that demanded
    every path exist at import time would make the ordering of unrelated steps load-bearing. The
    `subject()` / `backbone()` / `harness()` wrappers below do the check, because those three are inputs
    that must be present before any work starts.
    """
    c = _cfg()
    env_key = env_key or c.get(f"{key}_env") or f"JEVRSI_{key.upper()}"
    raw = os.environ.get(env_key) or c.get(key)
    if not raw:
        raise SystemExit(f"[fatal] no path configured for {key!r}. Set {env_key} or add {key!r} to "
                         f"config/paths.json.")
    p = pathlib.Path(raw).expanduser()
    return (p if p.is_absolute() else (ROOT / p)).resolve()


def require(path: pathlib.Path, what: str) -> pathlib.Path:
    """Fail loudly when a configured path is absent.

    A run that proceeds with a missing path does not crash -- it produces a number, and the number gets
    attributed to whatever was loaded last. That is the quietest and most expensive way to be wrong.
    """
    if not path.exists():
        raise SystemExit(f"[fatal] {what} not found at {path}. "
                         f"Override its location in config/paths.json or via the environment.")
    return path


def subject() -> pathlib.Path:
    """The agent-jev clone -- an EXTERNAL repository, not part of this one.

    Existence is verified here. A negative control caught the alternative: with no environment override,
    this used to return a config fallback pointing at a directory that does not exist, and the run
    proceeded to produce numbers attributed to a checkpoint it had never loaded. That is the quietest
    and most expensive way to be wrong, so the check is unconditional.
    """
    return require(_resolve("subject"), "the agent-jev subject repository (JEVRSI_SUBJECT)")


def backbone() -> pathlib.Path:
    """The downloaded Qwen3 base model (JEVRSI_BACKBONE)."""
    return require(_resolve("backbone"), "the Qwen3 base model (JEVRSI_BACKBONE)")


def harness() -> pathlib.Path:
    """The borrowed autoresearch harness, jsegov/autoresearch-win-rtx (JEVRSI_HARNESS)."""
    return require(_resolve("harness"), "the autoresearch harness (JEVRSI_HARNESS)")


if __name__ == "__main__":
    for name, fn in (("subject", subject), ("backbone", backbone), ("harness", harness)):
        try:
            p = fn()
            print(f"  {name:9s} {p}  {'[ok]' if p.exists() else '[MISSING]'}")
        except SystemExit as e:
            print(f"  {name:9s} {e}")

