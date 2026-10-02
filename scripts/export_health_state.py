#!/usr/bin/env python
"""Export the live state the dashboard renders.

WHY A SEPARATE EXPORTER
    Browsers block `fetch()` on file:// documents, so a dashboard cannot read a JSON file next to
    itself. The state is therefore written as a JS global in `viz/state.js`, which `<script src>`
    loads without the CORS restriction. The JSON is written too, for anything that wants to parse it.

WHY IT IS CALLED BY THE HEALTH CHAIN
    A dashboard that is exported by hand desynchronises the first time someone forgets. `health.py`
    calls this at the end of preflight, status and validate, so whatever the chain just concluded is
    what the page shows.

WHY NOTHING IS EVER FABRICATED
    If a source is missing, the panel says so. A dashboard that invents plausible content is worse
    than a blank one because it gets believed -- and this project has already spent GPU time on
    numbers that looked real.

WHAT THE PAGE CAN AND CANNOT SHOW
    The harness prints nothing during `fit()`, so a running arm has no step-level progress to display.
    What exists is: GPU state, elapsed time against a measured budget, and the presence of artifacts
    written between training and evaluation. The exporter reports exactly those, and a phase field
    says which one is in play rather than implying a percentage it cannot know.
"""
from __future__ import annotations

import datetime
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import health  # noqa: E402

VIZ = ROOT / "viz"
STATE_JS = VIZ / "state.js"
STATE_JSON = VIZ / "state.json"

# Every file the page renders from. Their mtimes are shown so staleness is judgeable, which is the
# difference between a live dashboard and one that merely looks live.
SOURCES = [
    ROOT / "measurement" / "health_ledger.jsonl",
    ROOT / "config" / "arm0_spec.json",
]


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def source_status(p: pathlib.Path) -> dict:
    if not p.is_file():
        return {"path": str(p), "exists": False, "age_s": None, "mtime": None}
    st = p.stat()
    return {"path": str(p), "exists": True,
            "mtime": iso(st.st_mtime),
            "age_s": round(datetime.datetime.now().timestamp() - st.st_mtime, 1)}


def arm_state(arm: pathlib.Path) -> dict:
    used, util, pids, free = health.gpu_state()
    if used >= health.TRAINING_MIB:
        phase, phase_class = "TRAINING", "run"
    elif used >= health.PREPARING_MIB:
        phase, phase_class = "PREPARING", "warn"
    else:
        phase, phase_class = "NOT RUNNING", "fail" if used >= 0 else "unknown"

    out = {"name": arm.name, "phase": phase, "phase_class": phase_class,
           "gpu_mib": used, "gpu_util": util, "gpu_apps": len(pids),
           "no_resume": True}

    launch = arm / "LAUNCH.json"
    if launch.is_file():
        L = json.loads(launch.read_text(encoding="utf-8"))
        out["started_at"] = L.get("started_at")
        out["spec_sha256"] = str(L.get("spec_sha256"))[:16]
        out["seed"] = L.get("seed")
        out["model"] = pathlib.Path(str(L.get("model", ""))).name
        try:
            t0 = datetime.datetime.fromisoformat(L["started_at"])
            out["elapsed_min"] = round((datetime.datetime.now() - t0).total_seconds() / 60, 1)
        except Exception:                                        # noqa: BLE001
            out["elapsed_min"] = None
        out["budget_min"] = 85   # train ~75 + evaluation ~16, from the measured 2-3 s/step

    ck = arm / "checkpoints" / "tower.safetensors"
    if ck.is_file():
        out["checkpoint"] = {"size_gib": round(ck.stat().st_size / 2**30, 2),
                             "written_at": iso(ck.stat().st_mtime)}
    outfiles = sorted(p for p in (arm / "out").glob("*.jsonl")
                      if not p.name.endswith(".items.jsonl")) if (arm / "out").is_dir() else []
    items = sorted((arm / "out").glob("*.items.jsonl")) if (arm / "out").is_dir() else []
    out["records_file"] = outfiles[0].name if outfiles else None
    out["items_file"] = items[0].name if items else None

    # their numbers, our anchors, and this arm's, so the page answers "on track?" not "what is here?"
    tgt = {
        "their_n_train_cases": health.EXPECTED_TRAIN_CASES,
        "their_final_loss": 0.6075,
        "our_zero_shot_top1": 0.4025,
        "their_v1_0_top1": 0.6525,
        "our_zero_shot_mmlu_top1": 0.2110,
        "their_v1_0_mmlu_top1": 0.355,
        "our_zero_shot_order_gap_pp": 3.1,
        "their_v1_0_order_gap_pp": 0.0,
    }
    if launch.is_file():
        tgt.update(json.loads(launch.read_text(encoding="utf-8")).get("targets") or {})
    out["targets"] = tgt

    meta_p = arm / "checkpoints" / "meta.json"
    if meta_p.is_file():
        m = json.loads(meta_p.read_text(encoding="utf-8"))
        out["actual"] = {"n_train_cases": m.get("n_train_cases"),
                         "final_loss": m.get("final_loss"),
                         "train_seconds": m.get("train_seconds"),
                         "tower_keys": m.get("tower_keys"),
                         "tapped_layer": m.get("tapped_layer"),
                         "linear_attn_kernel": m.get("linear_attn_kernel")}
    if items:
        try:
            from measurement.report_from_items import load, pooled_top1
            rws = load(items[0])
            def t1(role, order):
                v, n = pooled_top1([r for r in rws if r.get("target") == "typed_decisions"
                                    and r.get("role") == role and r.get("option_order") == order])
                return v
            out.setdefault("actual", {})
            out["actual"]["top1_candidate_canonical"] = t1("candidate", "canonical")
            out["actual"]["top1_candidate_reversed"] = t1("candidate", "reversed")
            out["actual"]["top1_control_canonical"] = t1("control", "canonical")
        except Exception as e:                                   # noqa: BLE001
            out["metrics_error"] = f"{type(e).__name__}: {str(e)[:70]}"
    return out


def build_state(arm: pathlib.Path) -> dict:
    checks = {"preflight": [], "status": [], "validate": []}
    verdicts = {}
    try:
        r = health.preflight(ROOT / "config" / "arm0_spec.json")
        checks["preflight"] = [{"name": n, "status": s, "detail": d, "why": w} for n, s, d, w in r.rows]
        verdicts["preflight"] = "FAIL" if any(x[1] == health.FAIL for x in r.rows) else "OK"
    except Exception as e:                                       # noqa: BLE001
        verdicts["preflight"] = "ERROR"
        checks["preflight"] = [{"name": "preflight", "status": health.FAIL,
                                "detail": f"{type(e).__name__}: {str(e)[:80]}", "why": ""}]
    try:
        r = health.status(arm)
        checks["status"] = [{"name": n, "status": s, "detail": d, "why": w} for n, s, d, w in r.rows]
        verdicts["status"] = "FAIL" if any(x[1] == health.FAIL for x in r.rows) else "OK"
    except Exception as e:                                       # noqa: BLE001
        verdicts["status"] = "ERROR"
        checks["status"] = [{"name": "status", "status": health.FAIL,
                             "detail": f"{type(e).__name__}: {str(e)[:80]}", "why": ""}]

    has_artifacts = (arm / "out").is_dir() and any((arm / "out").glob("*.jsonl"))
    if has_artifacts:
        try:
            r = health.validate(arm)
            checks["validate"] = [{"name": n, "status": s, "detail": d, "why": w} for n, s, d, w in r.rows]
            verdicts["validate"] = "FAIL" if any(x[1] == health.FAIL for x in r.rows) else "OK"
        except Exception as e:                                   # noqa: BLE001
            verdicts["validate"] = "ERROR"
            checks["validate"] = [{"name": "validate", "status": health.FAIL,
                                   "detail": f"{type(e).__name__}: {str(e)[:80]}", "why": ""}]
    else:
        verdicts["validate"] = "PENDING"
        checks["validate"] = [{"name": "validate", "status": health.SKIP,
                               "detail": "no evaluation records yet — the arm has not finished",
                               "why": ""}]

    # Two different questions, kept apart. Merging them would make a healthy running arm show FAIL
    # merely because the card is full -- and a verdict that is wrong whenever the pipeline is busy
    # is a verdict that gets ignored.
    #
    #   run_health      : is the arm that is running (or has finished) sound?     status + validate
    #   can_launch      : could another run start right now?                        preflight
    run_verdicts = [v for k, v in verdicts.items() if k in ("status", "validate")]
    if any(v in ("FAIL", "ERROR") for v in run_verdicts):
        overall = "FAIL"
    elif "PENDING" in run_verdicts or "WARN" in run_verdicts:
        overall = "WARN"
    else:
        overall = "OK"
    can_launch = verdicts.get("preflight") == "OK"

    led = health.ledger_read()
    launches = [r for r in led if r.get("kind") == "launch"]
    envs = [json.dumps(r.get("environment"), sort_keys=True) for r in launches if r.get("environment")]
    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "verdict": overall,
        "can_launch": can_launch,
        "verdicts": verdicts,
        "arm": arm_state(arm),
        "checks": checks,
        "ledger": [{"arm": r.get("arm"), "started_at": r.get("started_at"),
                    "spec_sha256": str(r.get("spec_sha256"))[:12],
                    "seed": r.get("seed")} for r in launches],
        "ledger_drift": {"distinct_environments": len(set(envs)), "launches": len(envs)},
        "sources": [source_status(p) for p in SOURCES],
        "notes": [
            "The harness prints nothing during fit(); a running arm has no step-level progress to show.",
            "There is no resume: a dead run costs all 1500 steps again.",
            "pooled_top1 is their verification metric; this project's harness does not record it, "
            "so it is recomputed from the per-question rows.",
        ],
    }


def export(arm: pathlib.Path) -> pathlib.Path:
    VIZ.mkdir(parents=True, exist_ok=True)
    state = build_state(arm)
    STATE_JS.write_text("// generated by scripts/export_health_state.py — do not edit\n"
                        "window.APP_STATE = " + json.dumps(state, default=str) + ";\n",
                        encoding="utf-8")
    STATE_JSON.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    return STATE_JS


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="E:/2026-AI4S/arms/arm0")
    a = ap.parse_args()
    p = export(pathlib.Path(a.arm))
    print(f"[OK] {p}")
    print(f"     open viz/health_dashboard.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
