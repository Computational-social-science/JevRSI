"""Git as experiment memory (brief constraint: Git 作为实验记忆).

An accepted candidate is a commit; a rejected candidate is a reset. This is the
brief's own prescription, and it buys three things a score log cannot:

* the *exact bytes* of every surviving prompt are recoverable, not just its score;
* the accept/reject decision is atomic with the state it produced;
* the search lineage is a real DAG, so "which mutation produced the winner" is a
  query rather than a guess.

The repository is dedicated to the run and lives *inside* the run directory. It is
deliberately **not** the project repository: mixing experiment memory into a repo
with unrelated working-tree churn makes `git reset` unsafe, and the brief's
mechanism depends on reset being cheap and safe.

FLAW 8 (`docs/critical_review_2026-09-26.md`): a reject *is* a reset, so the rejected
branch's code is destroyed by design and the search trajectory — the part that
distinguishes searching from guessing — survives only as a results row. Every
`rollback` therefore writes the attempt out first, to
``<run>/trajectory/<attempt>.diff`` + ``<attempt>.json``: the full diff, the stated
reason, a monotonic attempt index and a UTC timestamp. The trajectory directory is a
sibling of the repo (not inside it), so the reset's own `clean -fd` cannot remove the
record it just wrote.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class GitMemoryError(RuntimeError):
    pass


#: Stored in ``reason`` when a caller rolls back without stating why. Kept explicit
#: rather than blank so an audit can tell "no reason given" from "reason lost".
UNSPECIFIED_REASON = "(unspecified)"


@dataclass
class CommitInfo:
    sha: str
    subject: str
    index: int


@dataclass
class TrajectoryRecord:
    """One rejected attempt, persisted before the reset that erased it."""

    attempt: int
    ts: str
    reason: str
    head: str
    diff_path: Path
    record_path: Path
    diff_bytes: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class GitMemory:
    """A tiny commit/reset ledger for one search run."""

    def __init__(
        self,
        root: Path,
        *,
        workdir_name: str = "work",
        trajectory_dir: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.repo = self.root
        self.work = self.root / workdir_name
        # Default: the repo is `<run>/expgit`, so the trajectory lands on
        # `<run>/trajectory` — i.e. `runs/<run_id>/trajectory/` for a real run.
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else self.root.parent / "trajectory"
        self.repo.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        if not (self.repo / ".git").exists():
            self._git("init", "-q")
            self._git("config", "user.email", "oasp@localhost")
            self._git("config", "user.name", "OASP Search")
        # Bytes must survive verbatim: Git for Windows ships a system-level
        # `core.autocrlf=true`, which normalizes CRLF out of the stored blob. That
        # would make the recovered prompt differ from the workspace that was
        # measured, and a persisted diff would no longer replay byte-exactly.
        self._git("config", "core.autocrlf", "false")
        self._ensure_seed()

    # -- shell -------------------------------------------------------------

    def _git(self, *args: str, check: bool = True) -> str:
        p = subprocess.run(
            ["git", *args],
            cwd=str(self.repo),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if check and p.returncode != 0:
            raise GitMemoryError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
        return p.stdout

    def _git_bytes(self, *args: str) -> bytes:
        """Run git and return stdout as raw bytes.

        Diffs must never pass through text mode: universal-newline decoding rewrites
        a CRLF content line to LF, which silently corrupts the artifact (a replayed
        diff then no longer reproduces the working tree it claims to describe).
        """
        p = subprocess.run(["git", *args], cwd=str(self.repo), capture_output=True)
        if p.returncode != 0:
            raise GitMemoryError(
                f"git {' '.join(args)} failed: {p.stderr.decode('utf-8', 'replace').strip()}"
            )
        return p.stdout

    # -- prompt state ------------------------------------------------------

    def prompt_path(self) -> Path:
        return self.work / "prompt.txt"

    def _write_state(self, prompt_text: str, score: float, meta: dict) -> None:
        self.prompt_path().write_text(prompt_text, encoding="utf-8")
        (self.work / "score.json").write_text(
            json.dumps({"score": score, **meta}, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _ensure_seed(self) -> None:
        if not self.prompt_path().exists():
            self._write_state("", float("nan"), {"kind": "seed"})
            self._git("add", "-A")
            self._git("commit", "-q", "-m", "gen -1: seed")

    def read_prompt(self) -> str:
        p = self.prompt_path()
        return p.read_text(encoding="utf-8") if p.exists() else ""

    # -- memory operations -------------------------------------------------

    def write_candidate(self, prompt_text: str, score: float, meta: dict) -> None:
        """Place a candidate in the work tree *without* committing it.

        The reject path calls this before :meth:`rollback` so the persisted diff
        describes the attempt that is about to be discarded. Without it a rejected
        candidate in this loop exists only in memory and its diff is empty.
        """
        self._write_state(prompt_text, score, meta)

    def commit(self, prompt_text: str, score: float, meta: dict) -> CommitInfo:
        """Record a surviving candidate. Returns the new commit."""
        self._write_state(prompt_text, score, meta)
        self._git("add", "-A")
        subj = f"gen {meta.get('generation', '?')}: {meta.get('mechanism', 'mutate')} score={score:.4f}"
        self._git("commit", "-q", "--allow-empty", "-m", subj)
        sha = self._head()
        return CommitInfo(sha=sha, subject=subj, index=self.count() - 1)

    def rollback(
        self,
        *,
        to: str | None = None,
        reason: str = "",
        meta: dict | None = None,
    ) -> TrajectoryRecord:
        """Discard the working-tree candidate. Never removes history.

        The attempt is persisted **before** the reset (FLAW 8): the diff is captured
        while the candidate's code still exists, written next to the run, and only
        then is the tree reset. A reset destroys the rejected branch by design, so
        skipping this step would leave the search's trajectory unrecoverable.
        """
        record = self.persist_trajectory(reason=reason, meta=meta)
        if to:
            self._git("reset", "-q", "--hard", to)
        else:
            # Discard uncommitted changes and restore the last committed prompt.
            self._git("reset", "-q", "--hard", "HEAD")
            self._git("clean", "-q", "-fd")
        return record

    # -- trajectory memory (FLAW 8) ----------------------------------------

    def attempt_diff(self) -> bytes:
        """Full diff of the working tree against HEAD, including added files.

        Staged first so a file the candidate *created* appears in the diff with its
        content; a plain ``git diff`` would omit it and the artifact would be lossy.
        Staging is undone by the reset that follows, so it changes nothing durable.
        """
        self._git("add", "-A")
        return self._git_bytes("diff", "--cached", "--no-color", "HEAD")

    def _next_attempt(self) -> int:
        """Monotonic attempt index, read back from disk so a restart cannot reuse one."""
        seen = 0
        for p in self.trajectory_dir.glob("*.json"):
            if p.stem.isdigit():
                seen = max(seen, int(p.stem))
        return seen + 1

    def persist_trajectory(self, *, reason: str = "", meta: dict | None = None) -> TrajectoryRecord:
        """Write this attempt's diff + stated reason to the trajectory directory."""
        diff = self.attempt_diff()
        head = self._head()
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        attempt = self._next_attempt()
        diff_path = self.trajectory_dir / f"{attempt:04d}.diff"
        record_path = self.trajectory_dir / f"{attempt:04d}.json"

        reason = reason.strip()
        # `diff` is already bytes, straight from git. Written verbatim, first, so
        # the recorded sha256/size describe the file on disk exactly (text mode
        # would rewrite newlines on Windows and break that guarantee) and so a
        # crash never leaves a record pointing at a missing diff.
        diff_path.write_bytes(diff)
        payload = {
            "attempt": attempt,
            "ts": _utc_now(),
            "reason": reason or UNSPECIFIED_REASON,
            "reason_provided": bool(reason),
            "head": head,
            "diff_file": diff_path.name,
            "diff_bytes": len(diff),
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
            "diff_empty": not diff.strip(),
            "meta": meta or {},
        }
        record_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return TrajectoryRecord(
            attempt=attempt,
            ts=payload["ts"],
            reason=payload["reason"],
            head=head,
            diff_path=diff_path,
            record_path=record_path,
            diff_bytes=len(diff),
        )

    def _head(self) -> str:
        return self._git("rev-parse", "HEAD").strip()

    def count(self) -> int:
        out = self._git("rev-list", "--count", "HEAD").strip()
        return int(out) if out else 0

    def history(self) -> list[dict]:
        fmt = "%H%x1f%s%x1f%cI"
        out = self._git("log", f"--format={fmt}")
        rows = []
        for line in out.splitlines():
            if not line.strip():
                continue
            sha, subj, when = line.split("\x1f")
            rows.append({"sha": sha, "subject": subj, "date": when})
        return rows

    def best_commit(self) -> dict | None:
        """Highest score among committed candidates, read back from the objects.

        Read from `score.json` blobs rather than from an in-memory variable, so the
        answer survives a crash and cannot drift from what was actually committed.
        """
        best: dict | None = None
        for row in self.history():
            try:
                blob = self._git("show", f"{row['sha']}:work/score.json")
                data = json.loads(blob)
            except Exception:  # noqa: BLE001
                continue
            score = data.get("score")
            if score is None or score != score:  # NaN guard
                continue
            if best is None or score > best["score"]:
                best = {**row, "score": float(score), "meta": data}
        return best

    def export_prompt(self, dest: Path, *, sha: str | None = None) -> Path:
        sha = sha or self._head()
        text = self._git("show", f"{sha}:work/prompt.txt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return dest
