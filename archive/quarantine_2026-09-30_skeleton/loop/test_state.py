"""test_state.py -- prove the JSONL state machine survives the crashes the proposal promises.

The proposal (section 3.4) claims: "All experimental state is persisted to autoresearch.jsonl, which
survives process kills and host reboots ... Control state is recomputed from the JSONL file on every
restart; the cycle resumes from the last interrupted point with no reliance on in-memory context."

A claim like that is worth nothing untested, and it is especially easy to get subtly wrong in a way
that only appears after a week of unattended running. These tests therefore do the following, in order:

  T1  a normal sequence of cycles rebuilds to the same state a second time (determinism)
  T2  a process KILLED mid-cycle leaves a RUNNING record, and the rebuild reports it as INCOMPLETE
      rather than as a failure -- conflating the two would teach the loop that an unjudged candidate
      was a bad one
  T3  a TRUNCATED final line (a crash during the append) is skipped, and the log still loads
  T4  the attempt counter is monotonic ACROSS a restart, so two cycles can never share a number
  T5  the MAP-Elites archive is order-independent: replaying the same set of keeps in any order gives
      the same cell contents
  T6  cool-downs count consecutive failures per module family and reset on a keep
  T7  an empty log and a missing log both rebuild to a usable empty state

Run: python loop/test_state.py
Exit 0 = all pass.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Register the module in sys.modules BEFORE exec_module. @dataclass resolves field types through
# sys.modules[cls.__module__], so a module that is executed without being registered fails at class
# definition time with an opaque "'NoneType' object has no attribute '__dict__'". The standard
# importlib recipe is sys.modules[name] = module, then exec_module.
_spec = importlib.util.spec_from_file_location("loop_state", HERE / "state.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["loop_state"] = S
_spec.loader.exec_module(S)
Record, Log, Action = S.Record, S.Log, S.Action

FAILS: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


def rec(i, action, **kw):
    base = dict(iter=i, ts=S.now_iso(), action=action, modules={"family": kw.pop("family", "cal")},
                mutation=kw.pop("mutation", f"m{i}"), medium=kw.pop("medium", None),
                delta=kw.pop("delta", None), epsilon=kw.pop("epsilon", 0.05),
                iteration=kw.pop("iteration", None), wall_s=kw.pop("wall_s", 10.0),
                peak_vram_gib=kw.pop("peak_vram_gib", 5.0), diagnosis=kw.pop("diagnosis", ""))
    return Record(**base, **kw)


tmp = Path(tempfile.mkdtemp(prefix="jevrsi_state_"))

# ---- T1: determinism of the rebuild ------------------------------------------------------------
print("== T1  a normal sequence rebuilds deterministically ==")
L = Log(tmp / "t1.jsonl")
for i, a, med in [(1, "keep", 0.50), (2, "discard", 0.51), (3, "keep", 0.53), (4, "discard", 0.52)]:
    L.append(rec(i, a, medium=med, iteration=med and int(med * 5)))
r1 = L.rebuild()
r2 = L.rebuild()
check(r1 == r2, "two rebuilds of an unchanged log are identical")
check(r1["n_kept"] == 2 and r1["n_discarded"] == 2, f"kept/discarded counted ({r1['n_kept']}/{r1['n_discarded']})")
check(r1["best"]["medium"] == 0.53, f"incumbent is the best keep ({r1['best']['medium']})")
check(r1["next_iter"] == 5, f"next attempt is 5 ({r1['next_iter']})")

# ---- T2: a killed process leaves RUNNING, rebuilt as INCOMPLETE ---------------------------------
print("\n== T2  a cycle killed mid-flight is INCOMPLETE, not a FAILURE ==")
L2 = Log(tmp / "t2.jsonl")
L2.append(rec(1, "keep", medium=0.50))
L2.append(rec(2, "running", mutation="qLoRA r32"))     # never got a verdict
s2 = L2.rebuild()
check(s2["n_incomplete"] == 1, f"one incomplete cycle detected ({s2['n_incomplete']})")
check(s2["n_discarded"] == 0, "an unjudged cycle is NOT counted as a discard")
check("qLoRA r32" not in s2["failures_by_family"].get("cal", []),
      "an unjudged cycle does not enter the failure log that FAILURES.md is built from")
check(s2["next_iter"] == 3, f"the next attempt does not reuse number 2 ({s2['next_iter']})")

# ---- T3: a truncated final line -----------------------------------------------------------------
print("\n== T3  a crash DURING the append leaves a partial line, and the log still loads ==")
L3 = Log(tmp / "t3.jsonl")
L3.append(rec(1, "keep", medium=0.50))
L3.append(rec(2, "keep", medium=0.52))
with L3.path.open("a", encoding="utf-8") as fh:
    fh.write('{"iter": 3, "ts": "2026-09-29T00:00:00Z", "action": "kee')   # cut mid-write
s3 = L3.rebuild()
check(s3["n_kept"] == 2, f"intact records survive a truncated tail ({s3['n_kept']})")
check(s3["next_iter"] == 3, f"the truncated line does not consume an attempt number ({s3['next_iter']})")

# ---- T4: monotonic attempt numbers across a REAL process restart --------------------------------
print("\n== T4  the attempt counter is monotonic across a real process restart ==")
L4 = Log(tmp / "t4.jsonl")
for i in (1, 2, 3):
    L4.append(rec(i, "keep" if i % 2 else "discard", medium=0.50 + i / 100))
# a separate interpreter: nothing in memory carries over, exactly as after a reboot
child = subprocess.run(
    [sys.executable, "-c",
     f"import importlib.util,sys;"
     f"s=importlib.util.spec_from_file_location('ls', r'{HERE / 'state.py'}');"
     f"m=importlib.util.module_from_spec(s);sys.modules['ls']=m;s.loader.exec_module(m);"
     f"print(m.Log(r'{L4.path}').next_iter())"],
    capture_output=True, text=True)
check(child.stdout.strip() == "4", f"a fresh process reads next_iter=4 (got {child.stdout.strip()!r})")

# ---- T5: MAP-Elites is order-independent --------------------------------------------------------
print("\n== T5  the archive is order-independent ==")
base = [rec(1, "keep", medium=0.50, iteration=2), rec(2, "keep", medium=0.60, iteration=2),
        rec(3, "keep", medium=0.55, iteration=4)]
La, Lb = Log(tmp / "t5a.jsonl"), Log(tmp / "t5b.jsonl")
for r in base:
    La.append(r)
for r in reversed(base):
    Lb.append(r)
sa, sb = La.rebuild(), Lb.rebuild()
check(sa["archive"] == sb["archive"], "reversed order yields the same archive cells")
check(sa["archive"]["(2,)"]["medium"] == 0.60, "the best occupant holds the cell, not the last writer")

# ---- T6: cool-downs ----------------------------------------------------------------------------
print("\n== T6  cool-downs count consecutive failures per family and reset on a keep ==")
L6 = Log(tmp / "t6.jsonl")
L6.append(rec(1, "discard", family="cal")); L6.append(rec(2, "discard", family="cal"))
L6.append(rec(3, "keep", family="cal", medium=0.5))       # a keep RESETS the run
L6.append(rec(4, "discard", family="cal")); L6.append(rec(5, "discard", family="cal"))
L6.append(rec(6, "discard", family="lora")); L6.append(rec(7, "discard", family="lora"))
L6.append(rec(8, "discard", family="lora"))               # lora now has 3 CONSECUTIVE
s6 = L6.rebuild()
check(s6["cooldowns"]["cal"] == 2, f"cal is at 2 consecutive failures ({s6['cooldowns']['cal']})")
check(s6["cooldowns"]["lora"] == 3, f"lora is tracked separately ({s6['cooldowns']['lora']})")
check("cal" not in s6["families_exhausted"],
      "a family whose run was reset by a keep is NOT exhausted")
check("lora" in s6["families_exhausted"],
      "a family at 3 consecutive failures is flagged exhausted")

# ---- T7: empty and missing logs ------------------------------------------------------------------
print("\n== T7  an empty or absent log still yields a usable state ==")
s7 = Log(tmp / "t7_empty.jsonl").rebuild()
check(s7["next_iter"] == 1 and s7["best"] is None, "an empty log starts at attempt 1 with no incumbent")
s8 = Log(tmp / "does_not_exist.jsonl").rebuild()
check(s8["next_iter"] == 1, "a missing log is an empty log, not a crash")

# ---- T8: durability under an actual kill ----------------------------------------------------------
print("\n== T8  records survive a SIGKILL mid-run (fsync is doing its job) ==")
L9 = Log(tmp / "t9.jsonl")
writer = f"""
import importlib.util, os, signal, sys
s = importlib.util.spec_from_file_location('ls', r'{HERE / 'state.py'}')
m = importlib.util.module_from_spec(s); sys.modules['ls'] = m; s.loader.exec_module(m)
log = m.Log(r'{L9.path}')
for i in range(1, 40):
    log.append(m.Record(iter=i, ts=m.now_iso(), action='keep', medium=0.50 + i/1000,
                        modules={{'family': 'cal'}}, mutation=f'm{{i}}'))
    if i == 25:
        os.kill(os.getpid(), signal.SIGKILL)     # no cleanup, no flush, no warning
"""
killed = subprocess.run([sys.executable, "-c", writer], capture_output=True, text=True)
n_after_kill = len(list(L9.read()))
check(killed.returncode != 0, f"the child really was killed (rc={killed.returncode})")
check(n_after_kill >= 20, f"records written before the kill are still on disk ({n_after_kill} records)")
s9 = L9.rebuild()
check(s9["next_iter"] > 20, f"the rebuilt state is usable after the kill (next_iter={s9['next_iter']})")

print()
print("=" * 70)
if FAILS:
    print(f"[FAIL] {len(FAILS)} assertion(s):")
    for f in FAILS:
        print(f"    - {f}")
    raise SystemExit(1)
print("[OK] the JSONL state machine survives every crash the proposal promises to survive")
raise SystemExit(0)
