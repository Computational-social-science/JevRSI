# Is the JEV pipeline on track? An audit against the BootLoops protocols

Written 2026-10-03, with arm 0 still training at 9.75 h. The question is the owner's, and the answer
is split, because the pipeline has two halves and only one of them is sound.

**The configuration is on track. The verification design is not.**

Concretely: every question of *what* runs is settled and mechanically checked — the spec is the
published one field for field, the code is the release's own snapshot rather than the checkout,
the corpus is the builder's output, and eight guards fail loudly when any of that moves. But almost
every question of *whether the result is right* rests on machinery that has never been shown capable
of saying no.

---

## Where the plan is sound

| protocol | what we have | evidence it works |
|---|---|---|
| `tool-stewardship` — consult before writing | `TOOLS.md` index + four-question pages | each tool's page names the known-answer test it passed |
| `acceptance-gate` step 0 — declare the gate before the answer | `docs/arm0_acceptance_gate.md`, written while arm 0 ran | thresholds stated before the number exists |
| `acceptance-gate` — the check must be able to fail | negative controls on the spec validator | restoring `lr_head: 1e-3` and `eval_batch_size: 32` each fire |
| `independence-bookkeeping` — one-way contamination | the zero-shot anchor (0.4025) was measured **before** arm 0 and entered no part of its fit | recorded in `LAUNCH.json` at launch |
| `reading-contract` — sources only | the spec is copied from the release's `meta.json`, not reconstructed | `check_arm0_spec.py` diffs against it live |
| `timing-discipline` — no projection without a measurement | the budget is now absent until a completed arm supplies one | dashboard shows `budget_source: no completed arm yet` |

**The deepest fix of this whole engagement is in this half:** the pipeline no longer trusts a
reconstruction of the reference, and it no longer trusts a hand-typed number anywhere the release
itself publishes one.

---

## Where it is not — four findings, in descending severity

### 1. The pipeline has never been run on data whose answer was known first

`planted-truth`: *"A pipeline that has only ever seen real data has never been tested, because nobody
knew what the right answer was. … If the pipeline drops half its input on a parsing error, the output
is smaller and still looks like a finding."*

**We have never fed this harness a corpus whose gold labels we wrote down in advance and demanded
back.** Every number it has produced — including every number in `evidence/` — is real-data output.
If the scoring path silently resolved the wrong option index, mis-handled a type, or counted a
question twice, the output would be a slightly different number that reads exactly like a finding.

This is not hypothetical for this project. A converter of ours once **silently dropped 65% of the
corpus while exiting 0 and printing a summary line**; it was caught by a self-test, not by looking at
the output. And `planted-truth` names the sharper version: *"Self-consistency mistaken for a
control."*

**Not built. This is the single largest gap.**

### 2. No independent route to the score

`independence-bookkeeping`: *"The shared-library twins. Two 'independent' implementations both call
the same underlying expansion; a defect there makes them agree, wrongly, to full precision — the
check measures the library's self-consistency."*

`report_from_items.py` does not re-derive a single prediction. It counts agreements between `pred` and
`gold`, **both produced by the harness**. If the harness's `unpermute_logits` step were wrong, the
score would be wrong and this check would agree with it perfectly. The entire verification of our
headline number shares the entire production path of that number.

**Not built.** The fix is a route that re-predicts a handful of questions from raw model outputs
outside the harness, by hand.

### 3. Determinism was offered as evidence

`planted-truth`: *"a pipeline that is consistently wrong is still consistent. Internal agreement,
stability under iterations, and smooth residuals are properties of the machinery, not of the answer.
Only a check with an independent notion of truth counts."*

Two runs agreeing to full float precision was written up in `evidence/arm0_probe_v1_records.md` as
reason to accept the numbers. **It is not evidence of correctness** — it is evidence that repeating a
run is unnecessary. The claim is corrected in place.

**Fixed in the record; the underlying absence remains finding 2.**

### 4. The claim that started this — "the code is what ran" — was a reading-contract violation

`reading-contract`: *"an instrument that reads documents asserts only what the record in front of it
supports… Where that discipline lapses, reading silently becomes remembering, and remembering becomes
inventing."*

`DEFAULTS` says `lr_head: 1e-3` was asserted as a fact about **what ran**. The record that says what
ran — `meta.json` in the published checkpoint — had not been opened. The assertion came from a file
that says what the *current revision* would do, presented in the register of a fact about the release.
That is *"memory dressed as reading"* with a checkout standing in for memory.

**Fixed:** the spec is copied from that record and diffed against it mechanically.

---

## What the plan should be, restated

The arm sequence was right; the checks around it were thin. Revised:

| step | before | now |
|---|---|---|
| 0 | — | **declared gate** (`arm0_acceptance_gate.md`), written before the result |
| 1 | arm 0 finishes → read the numbers | **arm 0 finishes → Gate A/B (invariants and calibration) → then C** |
| 2 | compare to 0.6075 / 0.4025 | C1 against the **anchor** (can fail; has failed); C2 as **headroom consumed**, not a target |
| 3 | test the order-gap hypothesis | unchanged, but declared falsifiable **before** the number |
| 4 | seeds 29/43 | **gated**: only after a planted-truth battery and an independent prediction route exist |
| 5 | arm iteration | unchanged |

**Step 4 is the substantive change.** Running two more seeds before the pipeline can be shown to
score correctly would multiply a possible error by three and cost three more overnight runs.

---

## The one thing to do next, in order

1. **Wait for arm 0.** No action; it is healthy and unkillable without forfeiting 9.75 h.
2. **Run Gates A and B the moment it lands.** They are invariants and calibration — they either hold
   or they do not, and both can fail.
3. **Only then read the score**, against the declared gate.
4. **Build the planted-truth battery and the independent prediction route while arm 1 is not yet
   launched.** These are cheap (CPU, synthetic corpus, no 1500-step run) and they are what turns
   every later arm's number into something that can be believed.
5. **Add the heartbeat to arm 1** — the instrument exists and is mechanism-verified.
