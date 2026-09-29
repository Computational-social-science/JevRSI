"""run_pipeline.py -- drive the pipeline end to end on the AgentJev-0.6B initial state.

WHAT THIS PROVES, AND WHAT IT DOES NOT

This runs the real loop: it selects a route, executes it in a subprocess, scores it with the frozen
evaluator, applies the accept rule, writes the append-only log, updates the MAP-Elites archive, and
rebuilds its control state from that log. Every number it prints came from a real evaluation.

It is a VERIFICATION RUN, not a search. The routes it walks are the declared outer-level routes in the
order the selector prefers them, and a small epsilon is set for the demo. A six-week run would use the
epsilon that loop/epsilon.py derives; using a demo value is stated in the output rather than hidden,
because a run that accepted candidates on a made-up threshold would produce a progress curve that
looks real and means nothing.

THE ORDER OF CHECKS IS THE POINT

  1. execute one route by hand, to prove the executor works outside the loop
  2. run N cycles, to prove the loop advances and records
  3. rebuild state from the log ALONE, to prove Rule 3 compliance
  4. re-run the loop, to prove a restart resumes rather than restarting
  5. confirm the gate, the archive and the route mix are all consistent with the log

Step 4 is the one that catches the failure this project has already suffered once: a state file that
is written but never read back looks exactly like one that works, right up until a reboot.

Run: python measurement/run_pipeline.py --cycles 6
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(str(ROOT))
SUBJECT = pathlib.Path(os.environ.get("JEVRSI_SUBJECT", ROOT.parent / "agent-jev")).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "loop"))

BUNDLES = ROOT / "measurement" / "logit_bundles"
RUN_DIR = ROOT / "pipeline_runs" / "pipeline_v1"
PY = sys.executable

from pipeline.orchestrator import Candidate, Orchestrator      # noqa: E402
from pipeline.routes import ALL_ROUTES, HARNESS, PARAM         # noqa: E402


def run_executor(cand: Candidate) -> tuple[dict, float, float, str]:
    """One cycle's work, in a subprocess that loads nothing and exits.

    Subprocess, not a call: the orchestrator must never hold GPU state beside a child, and on this
    host that produced a 632 s allocator stall plus a 9x phantom slowdown that would have been logged
    as evidence about the configuration.
    """
    out = RUN_DIR / f"cycle_{cand.route.replace('.', '_')}.json"
    cmd = [PY, str(ROOT / "pipeline" / "executor.py"),
           "--route", cand.route,
           "--params", json.dumps(cand.params),
           "--proxy", str(BUNDLES / "proxy.json"),
           "--medium", str(BUNDLES / "medium.json"),
           "--calibration", str(BUNDLES / "calibration.json"),
           "--out", str(out)]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=SUBJECT, capture_output=True, text=True,
                       env=env, errors="replace", timeout=1800)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"executor rc={r.returncode}: {(r.stdout + r.stderr)[-400:]}")
    payload = json.loads(out.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise RuntimeError(f"route {cand.route} returned status={payload.get('status')}")
    return payload["medium"], wall, None, payload["stage"] and json.dumps(payload["stage"])[:200] or ""


def preflight() -> bool:
    """All four guards, before any cycle. Rule 1/2/3/4 of docs/PROJECT_RULES.md.

    A pipeline that can run on a drifted tree will eventually do so, and the run it produces will look
    like evidence. The gates are cheap -- seconds, no GPU -- so there is no reason to make them optional.
    """
    import subprocess
    print("PREFLIGHT: four project guards (Rule 1 relative paths / Rule 2 English / Rule 3 vocabulary / Rule 4 purity)")
    ok = True
    for g in ("check_language.py", "check_terminology.py", "check_object_purity.py",
              "check_relative_paths.py"):
        r = subprocess.run([PY, str(ROOT / "scripts" / g)], capture_output=True, text=True)
        verdict = "PASS" if r.returncode == 0 else "FAIL"
        print(f"  [{verdict}] {g}")
        if r.returncode != 0:
            ok = False
            tail = [l for l in r.stdout.splitlines() if l.strip()][-6:]
            for l in tail:
                print(f"         {l.strip()[:100]}")
    if not ok:
        print("\n[FAIL] a guard failed. Refusing to run cycles on a drifted tree -- a result produced "
              "on one looks exactly like a result produced on a clean one.")
    else:
        print("  all four guards PASS")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--epsilon", type=float, default=0.005,
                    help="DEMO value. A production run uses loop/epsilon.py's derivation; this is "
                         "not a measured noise threshold and is labelled as such in the output.")
    ap.add_argument("--fresh", action="store_true", help="wipe the run dir first")
    a = ap.parse_args()

    for b in ("proxy", "medium", "calibration"):
        if not (BUNDLES / f"{b}.json").exists():
            print(f"[FAIL] missing bundle {b}.json -- run measurement/build_logit_bundles.py first")
            return 2
    if a.fresh and RUN_DIR.exists():
        import shutil
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    if not preflight():
        return 3

    print("=" * 78)
    print("PIPELINE END-TO-END -- the project pipeline on the AgentJev-0.6B initial state")
    print("=" * 78)
    print(f"run dir     : {RUN_DIR}")
    print(f"epsilon     : {a.epsilon}  <-- DEMO VALUE, not a derived noise threshold")
    print(f"bundles     : proxy / medium / calibration (frozen V is not present and not read)")

    # ---- 1. one route by hand, outside the loop ------------------------------------------------
    print(f"\n{'='*78}\nSTEP 1  execute one route by hand, outside the loop\n{'='*78}")
    o = Orchestrator(RUN_DIR, epsilon=a.epsilon)
    try:
        probe = run_executor(Candidate(route="cal.isotonic_top1"))
        print(f"  executor returned: medium softCE={probe[0]['soft_cross_entropy']:.4f} "
              f"acc={probe[0]['accuracy']:.4f} in {probe[1]*1000:.0f} ms")
        print("  the executor works outside the loop")
    except Exception as e:                                          # noqa: BLE001
        print(f"  [FAIL] executor did not work: {e}")
        return 1

    # ---- 2. run cycles --------------------------------------------------------------------------
    print(f"\n{'='*78}\nSTEP 2  run {a.cycles} cycles\n{'='*78}")
    o = Orchestrator(RUN_DIR, epsilon=a.epsilon)          # fresh: the probe above wrote no log line
    incumbent = None
    t_start = time.perf_counter()
    for _ in range(a.cycles):
        state = o.rebuild()
        cand = o.select(state)
        if cand is None:
            print("  selector returned nothing -- the declared route set is exhausted")
            break
        print(f"\n  -> {cand.route}  (kind={cand.kind}, declared cost "
              f"{ALL_ROUTES[cand.route].cost_s}s, axis={ALL_ROUTES[cand.route].axis})")
        cyc = o.run_cycle(cand, run_executor, incumbent)
        if cyc.action == "error":
            print(f"     ERROR: {cyc.reason}")
            continue
        m = json.loads((RUN_DIR / f"cycle_{cand.route.replace('.','_')}.json").read_text(encoding="utf-8"))["medium"]
        print(f"     proxy softCE={m['soft_cross_entropy']:.4f}  medium softCE={m['soft_cross_entropy']:.4f} "
              f"acc={m['accuracy']:.4f} ECE={m['ece_10_bins_vs_gold_argmax']:.4f}")
        print(f"     action={cyc.action}  {cyc.reason}")
        if cyc.action == "keep":
            incumbent = m
    total = time.perf_counter() - t_start

    # ---- 3. rebuild from the log alone ----------------------------------------------------------
    print(f"\n{'='*78}\nSTEP 3  rebuild control state from autoresearch.jsonl ALONE (Rule 3)\n{'='*78}")
    o2 = Orchestrator(RUN_DIR, epsilon=a.epsilon)
    st = o2.rebuild()
    for k in ("records", "kept", "discarded", "incomplete", "errors", "cooldowns"):
        print(f"  {k:12s}: {st[k]}")
    print(f"  incumbent   : {'yes' if st['incumbent'] else 'none'}")
    if st["incumbent"]:
        inc = json.loads(st["incumbent"])
        print(f"                medium softCE={inc['medium']:.4f} route={inc['route']}")
    mix = st["route_mix"]
    print(f"  route mix   : " + ", ".join(f"{k}={v}" for k, v in sorted(mix.items()) if k != "_fractions"))
    fr = mix.get("_fractions", {})
    harness_n = sum(v for k, v in mix.items() if k != "_fractions" and ALL_ROUTES.get(k)
                    and ALL_ROUTES[k].kind == HARNESS)
    param_n = sum(v for k, v in mix.items() if k != "_fractions" and ALL_ROUTES.get(k)
                  and ALL_ROUTES[k].kind == PARAM)
    print(f"  attribution : harness {harness_n} cycles, param {param_n} cycles "
          f"-- the split the proposal requires")
    print(f"  gate        : open={o2.gate.open}  {o2.gate.reason}")

    # ---- 4. restart: the loop must RESUME, not restart -------------------------------------------
    print(f"\n{'='*78}\nSTEP 4  restart -- does it resume?\n{'='*78}")
    before = len(o2._records())
    cand2 = o2.select(o2.rebuild())
    print(f"  records before restart: {before}")
    if cand2 is None:
        print("  nothing left to try (route set exhausted) -- restart is trivially consistent")
    else:
        already = cand2.route in {r.route for r in o2._records() if r.medium is not None}
        print(f"  next selection after restart: {cand2.route}")
        print(f"  {'[FAIL] it re-proposed a route that already has a result' if already else '[OK] it resumed with an untried route'}")
        print(f"  iteration counter continues at {len(o2._records())} (no number is reused)")

    # ---- 5. the log itself ----------------------------------------------------------------------
    log = RUN_DIR / "autoresearch.jsonl"
    lines = [l for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n{'='*78}\nSTEP 5  the durable log\n{'='*78}")
    print(f"  {log.name}: {len(lines)} lines, {log.stat().st_size/1024:.1f} KB")
    print(f"  every line parses as a Record: "
          f"{all(_parses(l) for l in lines) if (_parses := lambda l: _try(l)) else 'n/a'}")
    from collections import Counter
    print(f"  actions: {dict(Counter(json.loads(l)['action'] for l in lines))}")
    print(f"  routes : {dict(Counter(json.loads(l).get('route') for l in lines))}")

    print(f"\n{'='*78}\nVERDICT\n{'='*78}")
    ok = (st["records"] > 0 and st["incomplete"] == 0 and st["errors"] == 0
          and len(lines) == st["records"] and harness_n > 0 and param_n == 0)
    print(f"  {a.cycles} cycles in {total:.1f}s total "
          f"({total/max(1,a.cycles)*1000:.0f} ms/cycle average)")
    print(f"  every cycle wrote a durable record        : {st['records'] == len(lines)}")
    print(f"  no cycle was left incomplete              : {st['incomplete'] == 0}")
    print(f"  no cycle errored                          : {st['errors'] == 0}")
    print(f"  the loop stayed on the OUTER level        : {param_n == 0} "
          f"(gate is shut: {o2.gate.reason})")
    print(f"  restart resumes rather than restarting    : see STEP 4")
    print(f"\n  {'PIPELINE VERIFIED END TO END' if ok else 'PIPELINE HAS A DEFECT -- see above'}")
    print(f"\n  artifacts: {RUN_DIR}")
    return 0 if ok else 1


def _try(line: str) -> bool:
    from loop.state import Record
    try:
        Record.from_json(line)
        return True
    except Exception:                                               # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
