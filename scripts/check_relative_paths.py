#!/usr/bin/env python3
"""check_relative_paths.py -- no machine-specific absolute paths in the live tree.

WHY THIS IS A HARD RULE

A repository containing `D:/2026-AI4S/...` cannot be cloned and run by anyone else. It is not portable,
so it is not published work, and the reproducibility requirement (a third party can rebuild the results
from the repository alone) is unsatisfiable by construction. The same string is also wrong on the machine
that wrote it, the moment the project moves, a volume is remounted, or work is split across drives.

This project has concrete evidence for the second half: the subject repository lives on a removable
volume that disconnected twice in a single day, and work was lost both times.

WHAT COUNTS AS A VIOLATION

    D:/2026-AI4S/...      C:\\Users\\...      /home/someone/...      E:/...
    D:\\somewhere          /mnt/c/...

WHAT DOES NOT

  * `pathlib.Path(__file__).resolve().parents[N]` -- the correct way to locate a project file
  * a CLI flag or an environment variable with a default -- the one legitimate source of an absolute
    path, because the user supplies it
  * `archive/` -- history, kept as written. Rewriting a record of what was true to satisfy a present
    rule would be falsification, not tidying
  * binary sources (`.docx`, `.pdf`, `.svg`) -- the original brief and cited papers
  * markdown table cells that quote a log line verbatim from such a file, and the lines documenting
    this rule itself

THE EXEMPTION LIST IS PRINTED ON EVERY RUN

Same design as the language and terminology guards: an exemption that can grow silently is
indistinguishable from no check. Every entry states its reason at the point of enforcement.

Usage:
    python scripts/check_relative_paths.py
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# A drive letter followed by a separator, or a POSIX absolute path that is NOT inside the project.
# Anchored on the drive letter so that a relative path containing a colon ("metrics:accuracy") is not
# caught, and so a doc line like "no `D:/...` allowed" is still caught -- which is why the docs that
# *document* the rule are exempt below rather than pattern-matched loosely.
DRIVE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z]:[\\/])")
POSIX_ABS = re.compile(r"(?<![\w.])/(?:home|Users|mnt|Volumes|media|opt|srv|var)/")

SKIP_DIRS = ("archive/", ".git/", "__pycache__", ".cache", ".pytest_cache", "node_modules",
             "literature/")
SCAN_SUFFIXES = {".py", ".md", ".json", ".sh", ".yaml", ".yml", ".txt", ".toml", ".cfg", ".html"}

EXEMPT: dict[str, str] = {
    "scripts/check_relative_paths.py": "this guard must be able to name the pattern it forbids",
    "docs/PROJECT_RULES.md": "Rule 1 states the rule and quotes the forbidden form",
    "docs/BACKUP_agent_jev_worktree.md":
        "RESTORE INSTRUCTIONS, not a code path. The file records the tar taken before the first "
        "autoresearch_agent.py run -- that script calls `git reset --hard` on both the crash and the "
        "discard path, so the working tree needed protecting first. The absolute path is the -C "
        "argument of `tar -xf`; rewriting it relative would make the restore command wrong, and a "
        "backup whose restore command does not work is not a backup.",
    "docs/TERMINOLOGY.md": "quotes a source-repository URL path, not a local filesystem path",
    "CURRENT_OBJECT.md": "carries the rule so a fresh session loading the SSOT cannot miss it",
    # Artifacts that RECORD a path rather than USE one. A run stamp that says where the weights were
    # loaded from is a historical fact; rewriting it to a relative path would make the record false
    # and would break the very provenance check Rule 1 exists to support.
    "measurement/hillclimb/": "run stamps: the recorded init_from is a historical fact, not a code path",
    "measurement/head_offline.json": "records the checkpoint path that was actually evaluated",
    "measurement/lora_feasibility.json": "records the script path that was actually run",
    "measurement/features/manifest_final.json": "records the checkpoint and backbone actually used",
    "measurement/day1_baseline_metrics.json": "records the source logits file actually consumed",
    "docs/incident_2026-09-25_tree_loss.md":
        "a verbatim incident record: it quotes the machine paths that were actually lost, which is the "
        "entire content of the record. Rewriting them would make the incident report false. Rule 4: "
        "history stays as written.",
    "measurement/profile_train_step.broken_2026-09-29.py":
        "deliberately preserved in a BROKEN state as the record of an approach that corrupted the file; "
        "it does not parse, so it cannot be made portable, and repairing it would destroy its purpose",
}


def main() -> int:
    print("=" * 76)
    print("path check -- every path in the live tree is relative to the project root")
    print("=" * 76)
    print(f"root: {ROOT}")
    print(f"scanning: {', '.join(sorted(SCAN_SUFFIXES))}")

    print(f"\nExemptions in force ({len(EXEMPT)}) -- printed on every run so the list cannot grow quietly:")
    for f, why in EXEMPT.items():
        print(f"  [exempt] {f:36s} {why}")

    violations: dict[str, list[tuple[int, str]]] = {}
    files = 0
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SCAN_SUFFIXES:
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(SKIP_DIRS) or rel in EXEMPT or any(
                rel.startswith(k) for k in EXEMPT if k.endswith("/")):
            continue
        files += 1
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = DRIVE.search(line) or POSIX_ABS.search(line)
            if m:
                violations.setdefault(rel, []).append((i, line.strip()[:100]))

    print(f"\nfiles scanned: {files}")
    if violations:
        total = sum(len(v) for v in violations.values())
        print(f"\nVIOLATIONS ({total} absolute paths across {len(violations)} files):\n")
        for rel, rows in sorted(violations.items(), key=lambda kv: -len(kv[1])):
            print(f"  {rel}   ({len(rows)} lines)")
            for line, txt in rows[:4]:
                print(f"      L{line}: {txt}")
            if len(rows) > 4:
                print(f"      ... and {len(rows)-4} more")
        print("\nRule 1 of docs/PROJECT_RULES.md. Replace with a path derived from __file__, a")
        print("project-root value, or a CLI argument. If a literal is genuinely required (a system")
        print("directory, a mount point), add it to EXEMPT with a written reason.")
        return 1

    print("\nRESULT: PASS -- no machine-specific absolute path in the live tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
