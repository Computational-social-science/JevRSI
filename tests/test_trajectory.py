"""Regression tests for the FLAW 8 fix: a reject must not destroy the trajectory.

`docs/critical_review_2026-09-26.md` FLAW 8: the object of study is the *search*, and
the protocol's reject path is `git reset` — which deletes the rejected branch's code by
design. Before the fix, a rejected attempt survived only as a `results.tsv` row.

These tests do not merely assert that a file appeared. They assert the artifact is
**faithful**: the persisted diff is replayed onto the restored tree and compared byte
for byte with the pre-reset working tree, and the reset is then shown to have restored
the repository state exactly.

Run directly: ``python tests/test_trajectory.py`` (no pytest, no GPU, no model calls).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oasp.git_memory import GitMemory  # noqa: E402

FAILS: list[str] = []
CHECKS = 0


def check(cond: bool, msg: str) -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  [OK]   {msg}")
    else:
        print(f"  [FAIL] {msg}")
        FAILS.append(msg)


def git(repo: Path, *args: str) -> tuple[int, str, str]:
    """Call git directly, so the test does not verify the fix with the code under test."""
    p = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return p.returncode, p.stdout, p.stderr


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD")[1].strip()


def porcelain(repo: Path) -> str:
    return git(repo, "status", "--porcelain")[1].strip()


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def record(run_dir: Path, attempt: int) -> dict:
    return json.loads((run_dir / "trajectory" / f"{attempt:04d}.json").read_text(encoding="utf-8"))


def main() -> int:
    if shutil.which("git") is None:
        print("[FAIL] trajectory: git not found on PATH")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="oasp_traj_", dir=os.environ.get("TMPDIR") or None))
    run_dir = tmp / "runs" / "test_traj"
    mem = GitMemory(run_dir / "expgit")
    repo = mem.repo
    work = mem.work
    traj = run_dir / "trajectory"

    try:
        print("=== 0. setup: an accepted incumbent is the state a reject starts from ===")
        check(mem.trajectory_dir == traj,
              f"trajectory defaults to runs/<run_id>/trajectory (got {mem.trajectory_dir})")
        mem.commit("BASELINE PROMPT\n", 0.55, {"generation": 0, "mechanism": "seed"})
        head0, count0 = head(repo), int(git(repo, "rev-list", "--count", "HEAD")[1].strip())
        baseline = read_bytes(work / "prompt.txt")
        check(head0 != "" and count0 == 2, f"seed + incumbent committed (count={count0})")

        print("\n=== 1. a rejected attempt persists a non-empty diff BEFORE the reset ===")
        # Written as bytes: the candidate's exact on-disk bytes are the thing the
        # persisted diff must be shown to reproduce, so nothing may rewrite them.
        (work / "prompt.txt").write_bytes(b"CANDIDATE PROMPT v1\nsecond line\n")
        (work / "extra.txt").write_bytes(b"candidate added this file\n")  # untracked new file
        pre_prompt, pre_extra = read_bytes(work / "prompt.txt"), read_bytes(work / "extra.txt")
        reason = "rejected: score 0.41 did not beat best_so_far 0.55 (reason_code=FORMAT_CONFLICT)"
        rec = mem.rollback(
            reason=reason,
            meta={"generation": 1, "reason_code": "FORMAT_CONFLICT", "score": 0.41},
        )
        diff_path = traj / "0001.diff"
        check(rec.attempt == 1, f"attempt index is 1 (got {rec.attempt})")
        check(diff_path.exists(), f"diff persisted at {diff_path.relative_to(tmp)}")
        blob = read_bytes(diff_path) if diff_path.exists() else b""
        check(len(blob) > 0, f"diff file is non-empty ({len(blob)} bytes)")
        text = blob.decode("utf-8", errors="replace")
        check("CANDIDATE PROMPT v1" in text, "diff body contains the candidate's edited prompt")
        check("extra.txt" in text, "diff covers the file the candidate *added* (not just edits)")

        print("\n=== 2. the record stores the stated reason, index and timestamp ===")
        meta = record(run_dir, 1)
        check(meta.get("reason") == reason, "reason string stored verbatim")
        check(meta.get("reason_provided") is True, "reason_provided is True")
        check(meta.get("attempt") == 1, "attempt index stored in the record")
        check(meta.get("diff_file") == "0001.diff", "record points at its diff file")
        check(meta.get("diff_bytes") == len(blob), "diff_bytes matches the file on disk")
        check(meta.get("diff_sha256") == hashlib.sha256(blob).hexdigest(),
              "diff_sha256 matches the bytes on disk")
        check(meta.get("diff_empty") is False, "diff_empty is False for a real attempt")
        check(meta.get("head") == head0, "record carries the HEAD the attempt was built on")
        check(meta.get("meta", {}).get("reason_code") == "FORMAT_CONFLICT",
              "caller meta (blind reason code) is carried through")
        try:
            ts = datetime.fromisoformat(str(meta.get("ts")).replace("Z", "+00:00"))
            check(ts.tzinfo is not None, f"timestamp is timezone-aware UTC ({meta.get('ts')})")
        except ValueError:
            check(False, f"timestamp is not ISO-8601 ({meta.get('ts')!r})")

        print("\n=== 3. the reset still works: git state restored exactly ===")
        check(head(repo) == head0, "HEAD unchanged by the rollback")
        check(int(git(repo, "rev-list", "--count", "HEAD")[1].strip()) == count0,
              "commit count unchanged")
        check(porcelain(repo) == "", f"working tree is clean (got {porcelain(repo)!r})")
        check(read_bytes(work / "prompt.txt") == baseline, "committed prompt restored")
        check(not (work / "extra.txt").exists(), "the candidate's added file was removed")
        check(mem.read_prompt().strip() == "BASELINE PROMPT", "read_prompt returns the restored state")

        print("\n=== 4. the persisted diff reconstructs the pre-reset tree byte for byte ===")
        rc, _, err = git(repo, "apply", "--whitespace=nowarn", str(diff_path))
        check(rc == 0, f"git apply replays the persisted diff (stderr={err.strip()[:80]!r})")
        if rc == 0:
            check(read_bytes(work / "prompt.txt") == pre_prompt,
                  "replayed prompt.txt is byte-identical to the pre-reset working tree")
            check(read_bytes(work / "extra.txt") == pre_extra,
                  "replayed added file is byte-identical to the pre-reset working tree")
        git(repo, "reset", "-q", "--hard", "HEAD")
        git(repo, "clean", "-q", "-fd")
        check(porcelain(repo) == "", "tree clean again after the replay")

        print("\n=== 5. attempt indices are monotonic and earlier records are immutable ===")
        (work / "prompt.txt").write_bytes(b"CANDIDATE PROMPT v2\n")
        reason2 = "rejected: score 0.55 tied best_so_far 0.55 (reason_code=OVER_SPECIFIED)"
        rec2 = mem.rollback(reason=reason2, meta={"generation": 2, "reason_code": "OVER_SPECIFIED"})
        check(rec2.attempt == 2, f"second reject gets attempt 2 (got {rec2.attempt})")
        check((traj / "0002.diff").exists(), "second diff persisted as 0002.diff")
        check(hashlib.sha256(read_bytes(diff_path)).hexdigest() == meta["diff_sha256"],
              "0001.diff is untouched by later attempts")
        check(record(run_dir, 2).get("reason") == reason2, "each attempt carries its own reason")
        check(record(run_dir, 2).get("attempt") == 2, "record 2 self-reports index 2")
        check(porcelain(repo) == "", "tree still clean after the second reject")

        print("\n=== 6. rewind (rollback to a sha) also records, and never drops history ===")
        mem.commit("SECOND ACCEPT\n", 0.60, {"generation": 3, "mechanism": "small"})
        head1 = head(repo)
        (work / "prompt.txt").write_bytes(b"CANDIDATE PROMPT v3\n")
        rec3 = mem.rollback(to=head0, reason="rewind: abandoned lineage back to the incumbent")
        check(rec3.attempt == 3, f"rewind writes attempt 3 (got {rec3.attempt})")
        check(record(run_dir, 3).get("head") == head1, "record captures the pre-rewind HEAD")
        check(head(repo) == head0, "HEAD moved to the requested sha")
        check(int(git(repo, "cat-file", "-e", head1)[0]) == 0,
              "the abandoned commit is still in the object store (history preserved)")
        check(porcelain(repo) == "", "tree clean after the rewind")

        print("\n=== 7. a reject with no stated reason is marked, never invented ===")
        rec4 = mem.rollback()
        meta4 = record(run_dir, 4)
        check(rec4.attempt == 4, f"fourth attempt indexed (got {rec4.attempt})")
        check(meta4.get("reason_provided") is False, "reason_provided is False")
        check(meta4.get("reason") == "(unspecified)", "reason is the explicit placeholder")
        check(meta4.get("diff_empty") is True, "empty attempt is flagged diff_empty, not faked")
        check(hashlib.sha256(read_bytes(traj / "0004.diff")).hexdigest() == meta4.get("diff_sha256"),
              "even the empty attempt records the real (0-byte) diff hash")

        print("\n=== 8. the search loop's reject path: stage the candidate, then roll back ===")
        # A candidate in the search loop exists only in memory until it is written,
        # so the reject path stages it *before* rolling back; otherwise its diff is
        # empty and the artifact says nothing about what was tried.
        reject_meta = {"generation": 9, "mechanism": "medium", "reason_code": "OVER_SPECIFIED"}
        mem.write_candidate(
            json.dumps({"format": "bare", "order": "random"}, sort_keys=True), 0.41, reject_meta
        )
        rec5 = mem.rollback(reason="rejected: score 0.4100 did not beat 0.6000", meta=reject_meta)
        blob5 = read_bytes(traj / "0005.diff")
        check(rec5.attempt == 5, f"staged reject indexed (got {rec5.attempt})")
        check(len(blob5) > 0, f"diff of the staged candidate is non-empty ({len(blob5)} bytes)")
        check('"format": "bare"' in blob5.decode("utf-8", "replace")
              and '"order": "random"' in blob5.decode("utf-8", "replace"),
              "diff body shows the candidate state that was discarded")
        check(record(run_dir, 5).get("meta", {}).get("mechanism") == "medium",
              "staged candidate's meta reaches the record")
        check(read_bytes(work / "prompt.txt") == baseline, "tree restored to the incumbent afterwards")
        check(porcelain(repo) == "", "tree clean after the staged reject")

    finally:
        if FAILS:
            print(f"\n[WARN] artifacts kept for inspection: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILS:
        print(f"[FAIL] trajectory: {len(FAILS)} of {CHECKS} checks failed")
        for f in FAILS:
            print(f"   - {f}")
        return 1
    print(f"[OK] trajectory: {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
