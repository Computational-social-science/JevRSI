# JevRSI

> **The objective is [`RTX4070_SelfEvolving_Jev_Research_Proposal.md`](RTX4070_SelfEvolving_Jev_Research_Proposal.md).**
> Read [`CURRENT_OBJECT.md`](CURRENT_OBJECT.md) first — it is the SSOT and it outranks this file.

## The controller is the ecosystem's, not ours

Experiments run through **`agent-jev/scripts/autoresearch_agent.py`** — the upstream 24/7 loop, which
already has a pre-registered τ, a dev/shadow two-signal scheme, crash classification with a circuit
breaker, and trajectory persistence for rejected attempts.

**We drive it. We do not reimplement it.** A second orchestrator is how two copies drift, and the
drift is invisible until a number disagrees. Our own `pipeline/` and `loop/` were retired on
2026-09-30 for exactly that reason. Our layer is `adaptation/` — gates and adapters, no controller.

## BLOCKED — the accept threshold is revoked

`INSTRUMENT_CALIBRATION.json` carried a run-to-run floor of **0.40 pp measured on a retired 322M
full-parameter arm**, not on this seed (596M, QLoRA; dev baseline 0.7955 against that arm's 0.53).
The pre-registration explicitly forbade inheriting it. It was inherited, and `tau_dev = 4.949 pp`
was armed on top of it. The measurement-sampling floor (**1.0668 pp**) is a property of the
*benchmark* and remains valid.

`adaptation/gate_tau.py` **refuses** while `_REVOKED_2026-09-30` is present. No keep/revert decision
may be made until it is cleared — by re-measuring the floor on this seed, or by re-pre-registering on
the sampling floor alone.

## Layout

```
autoresearch/
├── CURRENT_OBJECT.md                                 SSOT: the object, the retirements, the gates
├── RTX4070_SelfEvolving_Jev_Research_Proposal.md     THE OBJECTIVE
├── adaptation/                                       gates and adapters, no controller
│   ├── _ecosystem.py                                 loads pure helpers from agent-jev by path
│   └── gate_tau.py                                   refuses to arm a revoked threshold
├── config/paths.json                                 every machine-specific location, declared once
├── harness/                                          reference implementation of the calibration stages
├── lean/JevRSI/AcceptRule.lean                       Lean 4 formalisation of the accept rule
├── measurement/                                      floors, tau, feature cache, calibration
├── scripts/                                          the five guards
├── docs/                                             decisions, pre-registration, rules, terminology
└── archive/                                          retired and superseded material, with manifests
```

## Gates — all six must pass before a cycle spends GPU

```bash
python scripts/check_object_purity.py     # 15 retired objects stay absent
python scripts/check_language.py          # English is the language of every result
python scripts/check_terminology.py       # the lineage stays correctly attributed
python scripts/check_relative_paths.py    # no machine paths, AND derived roots land correctly
python scripts/check_split_disjoint.py    # fit / filter / decision sets are three different things
python adaptation/gate_tau.py             # refuses to arm a revoked or unregistered threshold
```

Each prints its exemptions in force on every run, so an exemption list cannot grow unnoticed.

## Rules

The full set is in [`docs/PROJECT_RULES.md`](docs/PROJECT_RULES.md) and is binding. Four worth
restating:

1. **A measurement is not a claim until its noise is measured.** A borrowed floor may plan
   *provisionally* if labelled at the point of use; it may never appear in an accept threshold or a
   comparison table un-measured.
2. **State the df beside every SD.** A pilot once overstated the run-to-run floor by 3.53x because it
   was measured at df = 1.
3. **Destructive operations are not delegated.** Files are moved, never deleted, when reclassifying
   work -- and any agent given a reclassification task gets an explicit target and a verification
   step.
4. **The agent does not retire the owner's objective.** An audit may report that a cited figure has
   no source; whether a stated objective is still pursued is the owner's decision.

## Retired

Fifteen research objects are retired and asserted absent by `check_object_purity.py`. They are
listed with reasons in `CURRENT_OBJECT.md`. Do not work on, extend, or cite any of them as live.
