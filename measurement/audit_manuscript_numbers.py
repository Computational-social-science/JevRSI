"""audit_manuscript_numbers.py -- check every quantitative claim in manuscript.html against its artifact.

WHY. A manuscript written from a summary is exactly where a mis-quoted number survives review: the prose
looks authoritative and nobody re-derives it. This script re-derives each figure from the artifact that is
supposed to be its source, and reports a mismatch instead of trusting the transcription.

It enforces three things a naive presence-check cannot:

  FORBIDDEN  values retracted by a documented decision must not reappear, so a silent revert fails
             rather than passing alongside the correct value;
  FIGURES    every <img> must resolve to a real file and carry a caption.

WHAT CHANGED ON 2026-09-29, and why it matters. The previous version audited a manuscript written against
the laya-multilingual substrate and read floor_a_cluster.json / floor_b_laya.json / tau_calibration.json /
learning_curve.json. Those artifacts belong to a RETIRED objective and their checkpoints are archived, so
every one of those checks had become vacuous: the guard was green because it was looking for numbers in a
document that no longer existed. A guard that silently checks nothing is worse than no guard, because it
appears in the gate list. It now audits the CURRENT manuscript, whose numbers come from
INSTRUMENT_CALIBRATION.json, lora_feasibility.json, and the recorded baseline log.

A note on provenance strength: the baseline numbers come from the frozen evaluator's stdout, transcribed into
docs/DAY1_PROGRESS.md, not from a machine-readable artifact. That is weaker than the other two sources and
the script says so in its output. Promoting them is outstanding work, not something to pretend is done.

Run: python measurement/audit_manuscript_numbers.py
Exit 0 = every claim traces to an artifact. Exit 1 = at least one does not.
"""
from __future__ import annotations

import pathlib
i