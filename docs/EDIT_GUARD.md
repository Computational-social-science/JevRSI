# EDIT_GUARD — the guard rail

> Adopted from [RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev)'s `rsijev/README.md`: **the loop may
> rewrite the model, the data and the training objective. It may not rewrite what a typed decision
> is, what is scored, or on which split.**
>
> A change to scoring is a change to the task, and goes through a human.

## Why it is not a file boundary here

RSI-Jev draws the same line between `arch.py` (editable) and `evaluate.py` (protected). We cannot
copy that division, because our module graph does not permit it:

```
train.py            (editable)  ──imports──>  model.py
evaluate_split.py  (protected)  ──imports──>  model.py
```

Their evaluator does not import their architecture. Ours does. So the protected surface cannot be
"the file that computes the metric" — it has to be the narrow interface between the editable and the
protected. Measured:

```python
# scripts/evaluate_split.py -- the ONLY contact with the editable model
model = AgentJevModel(args.model_path, dtype=torch.bfloat16).to('cuda:0')
model.load_state_dict(state_dict, strict=True)
```

No internal method is called. **The contract is the `state_dict` key set and shapes**, plus the
probability vector's layout. Everything behind that interface is editable; everything that produces
the number is protected.

## The classification

Assigned by what a file decides, not by its name. RSI-Jev's table cannot be copied across because the
two projects have different file sets.

| Module | Class | Decides |
|---|---|---|
| `agentjev/model.py` | **EDITABLE** | the model axis: how state and options are encoded and read out |
| `agentjev/data.py` | **EDITABLE** | the data axis: what is collected, generated or deleted |
| `agentjev/train.py` | **EDITABLE** | the training axis: objective, optimiser, schedule |
| `agentjev/losses.py` | **EDITABLE** | the training objective, as distinct from the scoring |
| `agentjev/synth.py` | **EDITABLE** | synthetic-question generation |
| `agentjev/routing*.py` | **EDITABLE** | the serving and routing axis |
| `scripts/convert_prepared_to_agentjev.py` | **EDITABLE** | how prepared questions enter the training set — a data decision |
| `scripts/evaluate_split.py` | **PROTECTED** | what is scored, how, and on which split |
| `scripts/autoresearch_agent.py` | **PROTECTED** | the accept rule, the threshold, the reset semantics |
| `typed_decisions/experiment.py` | **PROTECTED** | every metric definition |

### `losses.py` is EDITABLE, and that is not a loophole

The training loss and the reported metric are different objects and are allowed to diverge — that
divergence is how an arm's claim gets falsified. What is protected is that **the reported number**
is computed by the protected metrics, from logits the protected evaluator produced.

If an arm could edit both, every result would be unfalsifiable. If it can edit only the loss, the
metric is still the arbiter. That asymmetry is the entire point.

## Enforcement

`scripts/check_edit_guard.py` reads `config/edit_guard.json` — a declaration, not a docstring. A
module that misdeclares its own protection is exactly the case the guard exists for.

1. **Completeness.** Every module under `agentjev/` and `scripts/` is declared. An undeclared file is
   a FAIL, not a default: a file nobody classified is a file whose protection nobody chose.
2. **Integrity.** Each PROTECTED file's sha256 matches the declaration. A hash, not a review, so it
   cannot be argued with and cannot change as a side effect of a refactor.
3. **Reachability.** Each PROTECTED file exists. A protected path that has moved is a guard that has
   silently stopped guarding.

Run it, and prove it can fail:

```bash
python scripts/check_edit_guard.py              # expect: all pass
python scripts/check_edit_guard.py --self-test  # expect: injected defects all rejected
```

The self-test is not optional bookkeeping. A guard that has never rejected anything is
indistinguishable from a guard that cannot, and the difference only shows up on the one run where it
matters.

## Re-declaring a protected file

Never to make an arm pass. To change scoring is to change the task: it goes through a human, with the
old and new hashes both recorded in the commit that made the change.
