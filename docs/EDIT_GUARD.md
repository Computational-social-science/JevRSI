# EDITABLE / PROTECTED — the guard rail

> Adopted from [RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev)'s `rsijev/README.md`, which is the
> most valuable idea in that repository: **the loop may rewrite the model, the data and the training
> objective; it may not rewrite what a typed decision is, what is scored, or on which split.**
>
> A change to scoring is a change to the task, and goes through a human.

## Why this exists here

On 2026-09-30 this repository's `agent-jev` clone had 18 uncommitted worktree items that a single
`git reset --hard` inside the ecosystem agent would have destroyed. The backup was taken first,
which is the only reason nothing was lost. But the deeper risk is the one this file addresses:

`model.py` is imported by **both** `train.py` (editable) **and** `evaluate_split.py` (protected).
In RSI-Jev the same separation holds — `arch.py` is editable, `evaluate.py` is protected — but
their boundary is drawn differently, because their `evaluate.py` does not import their `arch.py`.

Ours does. So the protected surface cannot be "the file that computes metrics"; it has to be the
**narrow interface between them**. Measured 2026-09-30:

```python
# scripts/evaluate_split.py -- the ONLY contact with the editable model
model = AgentJevModel(args.model_path, dtype=torch.bfloat16).to('cuda:0')
model.load_state_dict(state_dict, strict=True)
```

No internal method is called. The contract is therefore the **`state_dict` key set and shapes**,
plus the probability vector's layout. Everything behind that interface is editable; everything
that produces the number is protected.

## The classification

Assigned by **what the file decides**, not by its name. The two projects have different file sets,
so a name-for-name copy of their table would protect the wrong things.

| Module | Class | What it decides |
|---|---|---|
| `agentjev/model.py` | **EDITABLE** | the model axis: how state and options are encoded and read out |
| `agentjev/data.py` | **EDITABLE** | the data axis: what gets collected, generated or deleted |
| `agentjev/train.py` | **EDITABLE** | the training axis: objective, optimiser, schedule, calibration |
| `agentjev/losses.py` | **EDITABLE** | the training objective, as distinct from the scoring |
| `agentjev/synth.py` | **EDITABLE** | synthetic-question generation |
| `agentjev/routing.py`, `routing_data.py`, `routing_runtime.py` | **EDITABLE** | the serving and routing axis |
| `agentjev/__init__.py` | **EDITABLE** | package surface |
| `scripts/evaluate_split.py` | **PROTECTED** | what is scored, how, and on which split |
| `scripts/autoresearch_agent.py` | **PROTECTED** | the accept rule, the threshold, the reset semantics |
| `typed_decisions/` (in the benchmark repo) | **PROTECTED** | what a typed decision *is*: one `Case`, one prediction |
| `typed_decisions/experiment.py::metrics` | **PROTECTED** | every metric definition, including ECE and Brier |

### `losses.py` is EDITABLE, and that is not a loophole

Training loss and scoring metric are different objects and are allowed to diverge — that
divergence is how an arm's claim gets falsified. What is protected is that **the reported number**
is computed by the protected `metrics`, from logits the protected evaluator produced.

If an arm could edit the training loss and also the reported metric, every result would be
unfalsifiable. If it can edit only the loss, the reported metric is still the arbiter. That
asymmetry is the entire point.

## Enforcement

`scripts/check_edit_guard.py` reads a single declaration file and is run by
`scripts/check_object_purity.py`'s sibling in CI. It enforces:

1. **Every module is declared.** An undeclared file is a FAIL, not a default — a file nobody
   classified is a file whose protection nobody chose.
2. **A protected file's sha256 is unchanged** from the declaration, unless a human re-declares it
   with a reason. This is a hash, not a review, so it cannot be talked past.
3. **No protected module is imported by an editable one for its decisions.** Informational only:
   it is allowed for `train.py` to import `model.py`, and it is the reason the boundary is a
   hash rather than an import rule.
4. **Every exemption prints on every run**, so an exemption list cannot grow unnoticed.

### Re-declaring a protected file

Never to make an arm pass. To change scoring is to change the task, and it goes through a human,
with the old and new hashes both recorded in the commit that made the change.
