# Decision Record

Decisions are recorded here with the evidence that was available at the time, not with the evidence
available now. A decision that could not be justified from what was known when it was taken is not a
decision this project makes.

The pre-migration Chinese original is preserved unmodified at
`archive/superseded_2026-09-29_language_migration/DECISIONS.zh-CN.pre_translation.md`
(sha256 `212ee53245426076`). Rule 2 of `docs/PROJECT_RULES.md` requires English in the live tree; Rule 4
requires the record of what was actually decided to survive the migration. Every number below is
carried over unchanged.

---

## D1 (2026-09-29) Keep the official 960/120/120+400 split

```json
{
  "id": "D1",
  "date": "2026-09-29",
  "question": "The proposal's 30/80/60 three-way split (Sec 3.2) versus the official 960/120/120+400",
  "decision": "A -- keep the official split: train 960 / dev 120 / calibration 120 / test 400",
  "decided_by": "project owner",
  "evidence": {
    "independent_unit": "the case (5 questions per case; questions within a case are not independent)",
    "scaling_law_verified": "test 400 cases SE=1.067pp; dev 120 cases SE=2.140pp; the 1/sqrt(cases) prediction of 1.948pp is 1.10x the measured value",
    "proposal_medium_M": {
      "tasks": 80,
      "approx_cases": 16,
      "se_pp": 5.33,
      "bonferroni_H1000_pp": 21.63
    },
    "official_dev": {
      "cases": 120,
      "se_pp": 2.14,
      "bonferroni_H1000_pp": 12.38
    },
    "usable_headroom_pp": 20.0,
    "note": "starting point 0.8150, theoretical ceiling 1.0 -> about 18.5pp of usable room; the upper bound is taken as 20pp"
  },
  "consequence": "The proposal's medium-M decision threshold of 21.63pp exceeds the entire usable headroom, so that split cannot separate a real improvement from noise at H~1000. The official dev threshold of 12.38pp is below the usable headroom and can decide.",
  "what_is_given_up": "The proposal's three-way partition (proxy / medium / frozen) itself. Replaced by dev for selection, calibration for temperatures, and test for the final evaluation, which preserves the intent of the proposal's Rule 1 -- the frozen set is used only at pre-registered checkpoints -- at a different scale."
}
```

---

## D2 (2026-09-29) QLoRA necessity: VRAM is not the binding constraint

```json
{
  "id": "D2",
  "date": "2026-09-29",
  "question": "Proposal Sec 2.1/2.3: is QLoRA (4-bit) a required path under 12 GB?",
  "status": "the paper arithmetic is falsified; the measurement is in progress",
  "claim_1_full_param_infeasible": {
    "verdict": "HOLDS",
    "evidence": "0.6B fp32 AdamW = 8.94 GiB (weights 2.24 + grads 2.24 + m/v 4.47) exceeds the 12.28 GiB card"
  },
  "claim_2_4bit_required": {
    "verdict": "DOES NOT FOLLOW from claim 1",
    "evidence": {
      "base": "Qwen3-0.6B: 28 layers, hidden 1024, GQA 16/8, intermediate 3072",
      "lora_r16_all_linear_trainable": "168.82M = 28.1% of backbone",
      "estimated_footprint": {
        "base_bf16_gib": 1.12,
        "lora_params_grads_gib": 0.62,
        "lora_adamw_fpv_gib": 1.26,
        "activations_ctx_gib": 1.89,
        "total_gib": 4.84
      },
      "qlora_estimate_gib": 4.0,
      "saving_from_4bit_gib": 0.84,
      "card_gib": 12.28
    },
    "reading": "4-bit saves 0.84 GiB on a card with 12.28 GiB. VRAM is not the constraint.",
    "the_real_cost": "4-bit injects a weight rounding error into every forward pass, and that error enters Floor B (the run-to-run SD). Floor B sets epsilon; a larger epsilon means fewer detectable improvements. The proposal's Sec 2.3 lists quantisation as a 'fixed baseline' and so treats it as free, but it is in fact deciding how much noise the accept rule must tolerate.",
    "decision_deferred_to": "a two-arm Floor B measurement (BF16-LoRA versus QLoRA), to be run after the decision rule is pre-registered"
  },
  "consequence": "QLoRA is not adopted for VRAM reasons. Whether to adopt it is a numerical question, settled by measuring Floor B. Until then, epsilon must not cite a Floor B from the QLoRA path."
}
```

---

## D3 (2026-09-29) Manuscript provenance: one real Floor B contamination found and corrected

```json
{
  "id": "D3",
  "date": "2026-09-29",
  "question": "Is every number in the manuscript traceable to an artifact, and free of contamination from another substrate?",
  "status": "one real contamination found and corrected; the provenance chain was upgraded",
  "finding": {
    "what": "The first manuscript draft cited Floor B = 0.40 pp as this objective's Floor B, and on that basis presented epsilon(dev) = 12.38 pp as a settled threshold",
    "truth": "The 0.40 pp was measured on a RETIRED 322M full-parameter arm -- a different model on a different training path, and one the pre-registration explicitly forbids inheriting from ('Do not inherit it silently'). The 12.38 pp did inherit that borrowed Floor B (see DAY1_BASELINE_2026-09-29.md lines 67 and 72-73), and the same document also records 'Floor B on the QLoRA seed = not measured'",
    "direction_of_error": "optimistic. Floor B enters as a squared term, so a borrowed value that is too small makes the true threshold LARGER. 12.38 pp is therefore a lower bound",
    "how_found": "measurement/audit_manuscript_numbers.py raised an error when it compared 12.38 against the calibration file's 4.95, which exposed two epsilons coming from two different objectives"
  },
  "corrections": [
    "Manuscript Sec 2 became a table with a 'transferable?' column: Floor A transfers (a property of the split), Floor B does not (a property of the optimiser), and epsilon(dev) is marked provisional",
    "The manuscript now quotes the calibration file's own warning verbatim, so the caution travels with the number",
    "tau_test = 3.11 pp and the accept-rule audit are marked as another substrate's measurement, and this seed's test threshold may not be stated until it is measured",
    "The Sec 6 status table marks 'Floor B on this seed' as blocking rather than done, consistent with Sec 2's provisional marking"
  ],
  "provenance_upgrade": {
    "was": "the sole source of the baseline numbers was a manual transcription of evaluator stdout (docs/DAY1_PROGRESS.md), which permitted only an existence check",
    "weakness_proved_by": "negative control N4: the value 0.8150 appears 6 times in the manuscript, and after changing one occurrence the audit still PASSED. An existence check cannot catch a wrong value",
    "now": "measurement/recompute_day1_baseline.py recomputes both arms from the per-question logits in day1_raw_dev.json, calling the frozen evaluator's own compute_metrics / probabilities / chance_corrected_skill; all 12 metrics agree with the record, written to day1_baseline_metrics.json. The audit now compares by value",
    "why_not_reimplement": "reimplementing ECE would create a second definition of ECE, drifting at exactly the point the manuscript is most sensitive to. The frozen evaluator's metric functions must be reused",
    "bonus": "both arms come from the same logits, so raw versus L1 is a paired comparison on identical input, which makes the 'calibration is neutral' claim in Sec 3.2 clean"
  },
  "guard_has_teeth": {
    "file": "measurement/test_manuscript_audit.py",
    "cases": 5,
    "result": "5/5 injected faults all caught; byte-identical on restore",
    "test_bug_found": "the first negative-control version replaced only the first occurrence, so N2 and N4 passed falsely. Changed to replace all occurrences and assert a count >= 2, so a stale test case fails loudly instead of silently losing its ability to detect anything"
  },
  "gates": "9/9 PASS"
}
```

---

## D4 (2026-09-29) Technical posture is harness-first; parameter updates are gated

```json
{
  "id": "D4",
  "date": "2026-09-29",
  "question": "Should a 10.5 h LoRA Floor B batch run as the primary experiment?",
  "decision": "No. The batch was stopped at step 342/600 and the schedule inverted. The objective's revised technical posture makes parameter updates a gated secondary lever, invoked only after harness families plateau.",
  "decided_by": "project owner",
  "measured_cost_asymmetry": {
    "harness_post_hoc_stage_seconds": 0.01,
    "harness_feature_cache_seconds": 444,
    "head_refit_on_cached_features_seconds": 16,
    "one_600_step_lora_replicate_seconds": 7596,
    "note": "the proposal's nominal '5 minutes per QLoRA pass' is off by more than an order of magnitude on this host, which is why the gate is expressed in measured cycle counts rather than nominal hours"
  },
  "consequence": "the epsilon problem dissolves on the harness route: a deterministic harness has no training variance, so Floor B is 0 by construction and epsilon reduces to a paired bootstrap on the decision split",
  "resumability": "the batch's 25-step snapshots are intact (checkpoint-300, checkpoint-325), so the inner level can be resumed at 25-step granularity if the gate ever opens. docs/prereg_floor_b_lora_2026-09-29.md stays frozen either way."
}
```

---

## D5 (2026-09-29) "autoresearch" means karpathy/autoresearch, and nothing else

```json
{
  "id": "D5",
  "date": "2026-09-29",
  "question": "May our scheduler be named by the compound formed from the optimisation term 'bilevel' and the harness name 'autoresearch'? (the literal compound is in docs/TERMINOLOGY.md; it is not reproduced here because the terminology guard forbids it everywhere except that ruling)",
  "decision": "No. The compound is banned. Our two-level schedule is called the outer level / the inner level, which is a structural description and implies no lineage from the borrowed harness.",
  "decided_by": "project owner",
  "rationale": "the compound welded together two unrelated things: karpathy/autoresearch, a real repository this project borrows from, and bilevel optimization, a technique this project does not use. It implied a derivation that does not exist and made a foreign term look like a design property of the borrowed framework.",
  "enforcement": "scripts/check_terminology.py, with docs/TERMINOLOGY.md as the canonical ruling",
  "why_mechanical": "the failure is silent. A wrong name reads naturally, nothing errors, and the next session inherits it as vocabulary"
}
```

---

## D6 (2026-09-29) Global project rules: Nature-level standards, English as the language of results

```json
{
  "id": "D6",
  "date": "2026-09-29",
  "question": "What language and what standard govern every artefact in this repository?",
  "decision": "Rule 1 of docs/PROJECT_RULES.md: every document follows Nature-level scientific standards. Rule 2: English is the language of every result, including manuscripts, inline code comments, docstrings, commit messages, and the prose fields of JSON artifacts.",
  "decided_by": "project owner",
  "enforcement": "scripts/check_language.py, which scans the live tree for CJK and fails on any occurrence",
  "permitted_exceptions": "quoted foreign-language terms that are functionally load-bearing. The live case is scripts/check_object_purity.py matching two Chinese terms, because a source document used those words and removing the Chinese form would blind the guard to the exact vocabulary it exists to catch. Every exemption is printed on every run, so the list cannot grow unnoticed.",
  "not_exempt": "archive/, which is history kept as written, and binary sources such as the original brief and the cited papers, which are sources rather than results"
}
```

---

## D7 (2026-09-29) Rule 1 audit of the Day-1 baseline found two real errors

```json
{
  "id": "D7",
  "date": "2026-09-29",
  "question": "Does every number in docs/DAY1_BASELINE_2026-09-29.md survive recomputation from its artifact?",
  "trigger": "the language migration required translating the file, and the translating agent was instructed to translate verbatim and REPORT anything numerically suspect rather than fix it. Two items were reported and both were checked against the artifacts.",
  "finding_1": {
    "what": "the accuracy row read '0.8150 (488/600)'",
    "verified": "recomputing with the frozen evaluator over measurement/day1_raw_dev.json gives 0.8150 = 489/600; 488/600 would be 0.8133",
    "correction": "count changed to 489/600. The metric value 0.8150 was already correct; only the prose count was wrong.",
    "why_it_survived": "nobody multiplied 0.8150 by 600. A number can be individually correct and internally inconsistent, and an existence-check audit cannot see it -- the same weakness negative control N4 exposed in D3."
  },
  "finding_2": {
    "what": "the Sec 3 Bonferroni column reported z x SE, while Sec 2.1's derivation includes the parent's own sampling variance as SE x sqrt(1 + replicas)",
    "verified": "the document's own 3.05 / 2.14 implies 1 + replicas = 2.0313; with the document's z of 4.0556 the three thresholds are 50.35 / 30.81 / 35.61 pp, not the 35.33 / 21.63 / 24.98 pp printed",
    "correction": "the three values are restated on Sec 2.1's footing, with the superseded values shown alongside",
    "effect_on_conclusions": "none, and in the same direction. Every threshold rises, so the proposal's split is even less able to separate signal from noise, and D1 -- keep the official 960/120/120+400 split -- stands."
  },
  "consequence": "this is the first audit run under Rule 1 of docs/PROJECT_RULES.md, and it found two real errors in a document that had already passed a 9/9 gate. A gate that had run on a translated-from-memory version of the numbers would not have found either. The lesson is recorded rather than the fix: the recomputation-from-artifact check is what caught these, not the gate."
}
```
