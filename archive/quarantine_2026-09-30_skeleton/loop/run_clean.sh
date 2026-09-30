#!/usr/bin/env bash
# run_clean.sh -- run a python command with Hermes' PYTHONPATH removed.
#
# WHY THIS EXISTS. On this host the Hermes agent process exports
#
#     PYTHONPATH=<hermes kernel tmp>;<hermes tools venv>/Lib/site-packages;...
#
# and that tools venv contains tokenizers 0.23.1, while the project environment (user site-packages)
# has tokenizers 0.22.2, which is what transformers 5.3.0 requires (`tokenizers<=0.23.0,>=0.22.0`).
# Because PYTHONPATH precedes site-packages on sys.path, EVERY python process started from this shell
# inherits the wrong tokenizers and dies at import time with
#
#     ImportError: tokenizers>=0.22.0,<=0.23.0 is required ... but found tokenizers==0.23.1
#
# The failure looks like a broken project environment. It is not: the packages are correct and were
# never touched. It is a PATH collision created by the agent runtime.
#
# This was found the hard way. A baseline evaluation had already run successfully earlier in the day
# (38.9 s, dev accuracy 0.8150), and then every subsequent invocation failed -- which reads exactly like
# "the environment broke" and invites a reinstall that would change pinned versions. It had not. The
# fix is one variable.
#
# Usage:
#   bash loop/run_clean.sh python scripts/evaluate_split.py --split dev
#   bash loop/run_clean.sh python loop/test_state.py
set -euo pipefail
unset PYTHONPATH
exec "$@"
