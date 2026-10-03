"""
paths.py -- the single place a machine-specific location is written down.

WHY THIS EXISTS
    The reference repository, the code environment, the backbone and the corpus are on a different
    volume from this repository. A path like E:/2026-AI4S/v1_env cannot be written relative to the
    project root and cannot be committed without making the tree uncloneable on another machine. So
    each location is declared once in config/paths.json, and an environment variable may override it.

WHY A LOUD FAILURE INSTEAD OF A DEFAULT
    A default that is wrong on this host is worse than no default: it produces a plausible run
    against the wrong data, and the error surfaces hours later as a number that does not reproduce.
    An unset override raises with the variable name in the message. Every path returned here is one
    that exists.

WHY THERE IS NO `subject`
    There was one, pointing at a subject repository that belonged to an earlier objective. Its
    presence was not inert: paths.py resolved it for the edit guard, so a guard ran against another
    project's files while reporting on this one's. The subject here is the arms plus the v1.0 code
    environment, both declared directly.

USAGE
    from paths import arms, backbone, require
    BACKBONE = require(backbone(), "the backbone weights")
"""
from __future__ import annotations

import json
import os
import pathlib

CONFIG = pathlib.Path(__file__).resolve().parent / "config" / "paths.json"

# key in config/paths.json -> environment variable that overrides it
ENV_FOR = {
    "backbone": "JEVRSI_BACKBONE",
    "synth_corpus": "JEVRSI_SYNTH_CORPUS",
    "arms": "JEVRSI_ARMS",
    "reference_repo": "JEVRSI_REFERENCE_REPO",
    "published_release": "JEVRSI_PUBLISHED_RELEASE",
    "v1_env": "JEVRSI_V1_ENV",
    "bootloops": "JEVRSI_BOOTLOOPS",
}


def _config() -> dict:
    if not CONFIG.is_file():
        raise SystemExit(f"[fatal] {CONFIG.name} is missing. Every machine-specific location is "
                         f"declared there; nothing falls back to a built-in default.")
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def arms() -> pathlib.Path:
    """Where an arm's run directory lives: its spec, its checkpoints, its records."""
    return _resolve("arms")


def backbone() -> pathlib.Path:
    return _resolve("backbone")


def synth_corpus() -> pathlib.Path:
    """RSI-Jev's own build_synth_corpus.py output -- the corpus every arm trains on."""
    return _resolve("synth_corpus")


def reference_repo() -> pathlib.Path:
    return _resolve("reference_repo")


def published_release() -> pathlib.Path:
    """Their v1.0 checkpoint's directory: meta.json (the authoritative spec), verify.json, code/.

    Read rather than reconstructed. This project's first spec was rebuilt from the checkout's
    DEFAULTs plus the release notes, and it was wrong in two load-bearing ways, because the
    checkout's HEAD is not the revision that produced v1.0.
    """
    return _resolve("published_release")


def bootloops() -> pathlib.Path:
    """The BootLoops clone: the toolkit, with the protocols repo beside it as its README requires.

    The protocols are the project's working discipline. The toolkit packages are mathematical
    physics instruments and do not apply to this project -- see docs/BOOTLOOPS.md.
    """
    return _resolve("bootloops")


def v1_env() -> pathlib.Path:
    """The v1.0 code environment: their release's modules, driven by the checkout's harness.

    HEAD is not the code that trained v1.0. Build with scripts/build_v1_env.py.
    """
    return _resolve("v1_env")


def _resolve(key: str) -> pathlib.Path:
    env = ENV_FOR.get(key, "")
    raw = os.environ.get(env) if env else None
    if raw:
        return pathlib.Path(raw)
    val = _config().get(key)
    if not val:
        raise SystemExit(f"[fatal] no path for {key!r} in {CONFIG.name} and ${env} is unset.")
    return pathlib.Path(val)


def require(path, what: str) -> pathlib.Path:
    """Return an existing path, or explain precisely what is missing.

    The message names the environment variable too, because on this host locations are usually set
    from the shell rather than from the config, and a failure that mentions only the config sends
    you to edit a file that was never the problem. The variable is found by asking each declared
    key whether it currently resolves to THIS path -- config value or environment value alike --
    rather than by inspecting the call site, which would only ever know about one key.
    """
    p = pathlib.Path(path)
    if p.exists():
        return p
    cfg = _config()
    names = []
    for key, env in ENV_FOR.items():
        candidate = os.environ.get(env) or cfg.get(key)
        if candidate and pathlib.Path(candidate) == p:
            names.append(f"${env}" if os.environ.get(env) else f"{CONFIG.name}:{key}")
    hint = f" -- set {' or '.join(names)}" if names else ""
    raise SystemExit(f"[fatal] {what} not found at {p}{hint}")
