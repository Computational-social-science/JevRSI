# Item expansion plan — ARC-Challenge, 60 → 300 items

**Date:** 2026-09-26 · **Author:** analysis subagent · **Status:** ready to run, not yet run
**Scope constraint honoured:** no GPU/Ollama probe was executed. Every number below is read from the
already-frozen benchmark files or recomputed from the existing `pilot_v2_challenge` artifacts on CPU.
The runner command is **reported, not run** (§d), because a 45-minute GPU training job currently holds
the device.

**Bottom line:** the expansion is a **pure CLI change — no re-freeze, no re-download, no harness edit**.
It costs ~83 min (like-for-like ~93 min) and the pilot's 60 items remain a **strict subset** of the new
300-item dev set, so `pilot_v2_challenge` is extended rather than contradicted. The new run must use a
**new run-id**; reusing `pilot_v2_challenge` would silently rewrite its manifest (§e).

---

## (a) Usable items per benchmark per partition, read from the already-frozen files

The frozen artifacts are **JSONL**, not parquet: `benchmarks/<key>/items.jsonl`, one row per item, with
the *upstream* parquet's SHA-256 recorded in `benchmarks/<key>/provenance.json`. There is no parquet
copy in the tree (`find . -name '*.parquet'` → empty). Nothing was downloaded for this report.

| benchmark | frozen file | **usable items** | malformed / duplicate ids | option-count distribution | max `--items-dev` (test reserves 60) | source SHA-256 (upstream parquet) |
|---|---|---:|---:|---|---:|---|
| **arc_challenge** | `benchmarks/arc_challenge/items.jsonl` | **1172** | 0 / 0 | 4-way ×1165, 3-way ×4, 5-way ×3 | **1112** | `62f03257e737aed2…c1d1e9` |
| arc_easy | `benchmarks/arc_easy/items.jsonl` | 2376 | 0 / 0 | 4-way ×2365, 3-way ×7, 5-way ×4 | 2316 | `4160597d618ae851…27b47177` |
| mmlu_pro | `benchmarks/mmlu_pro/items.jsonl` | 12032 | 0 / 0 | 3–10 way (10-way ×9981) | 11972 | `0e24a191921c2f45…1ecb9ad8` |
| supergpqa | `benchmarks/supergpqa/items.jsonl` | 26529 | 0 / 0 | 4–10 way (10-way ×23158) | 26469 | `28b998e70205ee95…33f7a910` |

**Partition arithmetic (arc_challenge).** `benchmark.stratified_split` always constructs *both*
partitions and requires `n_dev + n_test ≤ n_total` (`oasp/benchmark.py:214-216`). With the design's
`n_items_test = 60`, the dev partition can be scaled to **1172 − 60 = 1112** items. 300 dev items is
**25.6 %** of the pool and uses 360 of 1172 rows; 812 frozen items remain unused — the reserve needed if
300 proves insufficient (§f).

**Validation performed on the frozen files** (not assumed): no duplicate `item_id`; no empty
question/option text; `labels` and `texts` equal length everywhere; every `answer_key` present in its own
label set. ARC-Challenge keys are letters except 22 items that use numeric labels (`1`–`4`), which are
valid for those items and are handled by the evaluator. The recorded mean chance level is 0.2502
(`runs/pilot_v2_challenge/manifest.json`).

---

## (b) How `--items-dev` actually selects items: **seeded shuffle of the whole pool, then a prefix**

Not first-N, not unseeded random. Mechanism, with the exact code lines:

`oasp/runner.py:346` — the flag exists and defaults to the design constant:
```python
ap.add_argument("--items-dev", type=int, default=config.DESIGN["n_items_dev"])
```

`oasp/runner.py:210` — seed resolution (`None` → `DESIGN["seed"]` = 42):
```python
seed = d["seed"] if seed is None else seed
```

`oasp/runner.py:226-228` — the split is computed over **all** frozen items:
```python
split_map = benchmark.stratified_split(
    items, n_dev=n_items_dev, n_test=n_items_test, seed=seed
)
```

`oasp/benchmark.py:211-218` — **the selection rule itself**:
```python
ordered = sorted(items, key=lambda it: it.item_id)   # 211 — deterministic order
rng = random.Random(seed)                            # 212 — seeded
rng.shuffle(ordered)                                 # 213 — Fisher–Yates over the FULL pool
...
dev = ordered[:n_dev]                                # 217 — contiguous PREFIX, not a fresh draw
test = ordered[n_dev : n_dev + n_test]               # 218
```
Guard: `benchmark.py:214-216` raises `ValueError("asked for N items but benchmark has M")` if
`n_dev + n_test > pool`.

**The consequence that matters:** the permutation is over the *entire* frozen pool in both runs, so the
dev set is a **prefix of one fixed permutation** and therefore monotone in `n_dev`. Replaying these exact
lines on the frozen JSONL with seed 42 (CPU, no probe) gives:

| `--items-dev` | 60 | 150 | 300 | 1112 |
|---|---|---|---|---|
| contains the previous dev set as an exact prefix? | — | ✅ dev(60) | ✅ dev(150) | ✅ dev(300) |
| items added | 60 | +90 | +150 | +812 |

Two further identities, both verified against the pilot artifacts rather than assumed:
1. **Variants unchanged.** `prompt_space.sample_variants(n=32, seed=42)` (`oasp/prompt_space.py:79-97`)
   depends only on `n` and `seed`, not on items. Recomputed output is **byte-identical** to
   `runs/pilot_v2_challenge/variants.json` (same 32 `vid`s, canonical `p0` included).
2. **Cell ids unchanged for the already-measured cells.** `cell_id = "dev|<model>|<vid>|<item_id>"`
   (`runner.py:35-36`). Because both the variants and the 60 items are unchanged, **all 3840 pilot cell_ids
   collide exactly** with the new run's cell set — which is both the reason the pilot data is reusable in
   principle and the reason a run-id collision is dangerous in practice (§e).

---

## (c) Re-freezing is **not** required — purely a CLI change

| question | answer | evidence |
|---|---|---|
| Does the frozen file already contain all 1172 items? | **Yes** | `provenance.n_items = 1172` = line count of `items.jsonl` (1172). The freeze ran with `limit=None`, i.e. the full test split (`benchmark.py:159`, called from `load_items` at `:190-193` with no limit). |
| Will the runner re-download anything? | **No** | `load_items` reads the existing file (`benchmark.py:190-200`); `freeze_benchmark` returns early when `items.jsonl` + `provenance.json` both exist (`benchmark.py:149-150`). |
| Does a larger dev set need new stimulus material? | **No** | Only `ordered[:n_dev]` changes (`benchmark.py:217`). Prompt rendering is untouched (`prompt_space.render`). |
| Is the stimulus provably the same as the pilot's? | **Yes** | New manifest re-records the same provenance block, `source_sha256 = 62f03257…c1d1e9` (`benchmark.py:175`, `runner.py:243-251`). |

⚠️ **Do not run `freeze_benchmark(..., force=True)`.** It would re-download the upstream parquet; a hash
change would break comparability with `pilot_v2_challenge` and any other run. Nothing in this plan needs it.

---

## (d) The command, and the runtime estimate

The flag set below is **exactly** `scripts/nightly_campaign.sh:54-58` (the operational unit used for the
pilot, whose manifest records `num_predict: 192`, `concurrency: 1`, `splits: ["dev"]`), with only
`--run-id` and `--items-dev` changed. Model order matches the pilot's log (`models=['mistral_7b','gemma3_4b']`).

```bash
cd /d/2026-AI4S/autoresearch && python -m oasp.runner \
  --run-id pilot_v3_challenge300 --benchmark arc_challenge \
  --models mistral_7b gemma3_4b \
  --variants 32 --items-dev 300 --items-test 60 --splits dev \
  --num-predict 192 --timeout 600 --concurrency 1
```

Why each non-obvious flag:

* `--items-dev 300` — the expansion. 300 ≤ 1112, so the split guard cannot fire.
* `--items-test 60` — keep the design's reservation so the split is unchanged apart from the prefix length
  (and the test partition remains available for a later held-out run, item-by-item disjoint from dev).
* `--num-predict 192` — **required for fidelity.** The pilot manifest records 192, but `config.DESIGN`
  now carries `num_predict: 16`; omitting the flag would silently change the generation budget relative to
  the pilot (0.26 s/cell was measured at 192).
* `--concurrency 1` — the pilot ran sequentially (its log shows `concurrency=1`; per-model times sum to the
  wall time: 621.9 s + 498.6 s = 1120.5 s). One worker keeps the throughput profile — and therefore the
  runtime basis — comparable.
* `--seed` / `--temperature` are omitted because the defaults (42, 0.0) are exactly the pilot's values.

**Runtime.** Cells = 300 items × 32 variants × 2 models = **19 200** (5 × the pilot's 3840).

| basis | rate | projected time |
|---|---|---|
| task-stated rate (gemma3_4b solo: 498.6 s / 1920 cells) | 0.260 s/cell | 4992 s = **83.2 min** |
| like-for-like wall clock (pilot: 1120.5 s / 3840 cells, sequential) | 0.292 s/cell | 5602 s = **93.4 min** |
| per model (sequential, from the pilot's own rates) | gemma3 0.260 · mistral 0.324 | gemma3 ≈ 42 min + mistral ≈ 52 min |

So: **~83 min best case, ~93 min like-for-like**, plus a few minutes of setup. Artifacts ≈ 11.4 MB
(pilot: 2.28 MB / 3840 rows), well inside the campaign script's 5 GB disk floor.

**Pre-flight (run when the GPU is free — not now):**
```bash
ollama list   # digests must equal the pilot manifest:
              #   gemma3:4b  -> 2026-07-06T16:19:28.0481828+08:00
              #   mistral:7b -> 2026-08-10T11:10:30.8631265+08:00
              # a different digest is a different treatment, not a re-run
```

**Analysis afterwards** (writes `analysis/pilot_v3_challenge300/`, never overwrites figures):
```bash
python -m oasp.analyze --run-id pilot_v3_challenge300 --split dev --noise-run-id noise_v1
```
`noise_v1` is the existing noise-floor run (28 s; 5 replicates × 20 items × 2 models at T = 0.7) that the
pilot was analysed against. `scripts/nightly_campaign.sh` would instead create a fresh
`noise_arc_challenge`; either is fine, but the id passed here must match the one used.

---

## (e) Risk to `pilot_v2_challenge`, and why the new run-id is mandatory

**The hazard is real and silent.** Reusing `--run-id pilot_v2_challenge` with `--items-dev 300` would:

1. **Skip the 3840 existing cells** — `_run_one_model` loads completed `cell_id`s and filters them out
   (`runner.py:128-129`), so only the 15 360 new cells would run;
2. **Append to the same file** — results are append-only (`runner.py:72-77`, `:150-173`), producing a
   19 200-row `results.jsonl`;
3. **Destroy the pilot's manifest before the first new cell runs** — `manifest.json` is written *in place*
   at run start (`runner.py:216-219`, `:289`) and rewritten at the end (`:330`). The pilot's
   `n_items_dev: 60`, `wall_seconds: 1120.5` and `results_sha256_after: e44b8cdb…30ce9` would be gone.

Because the pilot's 60 items are a strict subset of the 300 (§b), the appended rows are *consistent*, so
the result would look valid while no longer being attributable to the published numbers — the corruption
would not announce itself. That is exactly the "silent overwrite" to avoid.

**Mitigation (the recommended path):** use the new run-id `pilot_v3_challenge300` and never invoke the
runner against `pilot_v2_challenge` again. The pilot directory is then untouched by construction.
Verification that it survived:

```bash
sha256sum runs/pilot_v2_challenge/results.jsonl
# must equal e44b8cdb08dc9df11c071e4cb89471d01e78a24535ae2f0fee312a8529f30ce9
# (verified at this value on 2026-09-26, matching the manifest's results_sha256_after)
```

Secondary protections: the new run-id contains `pilot`, so the manifest label stays `exploratory`
(`runner.py:242`) as FLAW 2 requires; `oasp.analyze` writes only under `analysis/<run_id>/` and refuses to
overwrite figures; `analysis/pilot_v2_challenge/` is left as published.

*Optional variant (not recommended):* to save the ~19 min the 3840 shared cells would cost, copy the
pilot's `results.jsonl` into `runs/pilot_v3_challenge300/` before starting — the cell_ids collide exactly
(§b), the runner skips them, and the manifest records a non-empty `results_sha256_before`, so the import is
auditable rather than silent. Downside: the run's cells then originate from two sessions, which muddies a
pilot whose whole value is clean provenance. Prefer the clean 83–93 min full re-run.

---

## (f) Will 300 items actually decide the test? (extrapolation — labelled as such)

The decision quantity is the **upper 95 % bound of the spread** from the item bootstrap in
`stats.metric_m1_spread` (`stats.py:143-180`, resampling items with replacement, `B=2000`), compared with
`DESIGN["equivalence_margin"] = 0.05` by `stats.equivalence_verdict` (`stats.py:542-547`). Recomputed
today on the pilot, it reproduces FLAW 10 exactly: **upper = 0.2167 for both models, verdict
"NOT ESTABLISHED"**.

The measured scaling law (`scripts/power_probe.py`, 200 resamples, seed 12345 — rerun today) is
mean range ≈ c/√n, with c = range×√n:

| n | 15 | 30 | 45 | 60 | drift |
|---|---|---|---|---|---|
| gemma3_4b | 0.7436 | 0.7602 | 0.7104 | **0.6452** | ↓ — asymptote below the measured value |
| mistral_7b | 0.6944 | 0.7696 | 0.8318 | **0.9040** | ↑ — a real floor, *not* pure noise |

Extrapolating that law to larger *fresh* item sets (conservative version: pure 1/√n from the n = 60 point,
plus the measured CI half-width law 0.26/√n), with the caveat that this is a 4-point extrapolation, ±0.01:

| n_items | gemma3_4b point | gemma3_4b upper | mistral_7b point | mistral_7b upper |
|---:|---|---|---|---|
| 60 (measured) | 0.0833 | **0.2167** | 0.1167 | **0.2167** |
| 300 (planned) | ≈ 0.037 | **≈ 0.052** | ≈ 0.052 | ≈ 0.067 |
| 600 | ≈ 0.026 | ≈ 0.037 | ≈ 0.037 | ≈ 0.048 |
| 1112 (pool max) | ≈ 0.019 | ≈ 0.027 | ≈ 0.027 | ≈ 0.035 |

Reading this honestly:

* **gemma3_4b at 300 is borderline**: the point estimate is projected to drop below the margin, but the
  bound that decides is projected to land at ≈ 0.052 — within ~0.005 of the margin. 300 items may still
  return "NOT ESTABLISHED". ~600 items are projected to clear it comfortably, and the pool supports 1112.
* **mistral_7b is projected never to be equivalent**: its c rises with n (0.69 → 0.90), i.e. its spread is
  not noise-limited — the fitted asymptote is ≈ 0.052, at the margin. More items will *decide* it the other
  way ("materially different"), which is a reportable result rather than an underpowered null.
* Therefore the expansion buys **decidability in whichever direction the data falls**. That is the correct
  framing of FLAW 10 and should replace "H1 NOT SUPPORTED".
* **Pre-commit the escalation ladder now** (`pilot_v3_challenge300` → `600` → `1112`), before seeing the
  300-item result, so the step is not data-dependent. Every rung is the same CLI change: only
  `--items-dev` and `--run-id` differ, and no re-freeze is ever required.

The 300-item run's **own** bootstrap CI is the measurement; the table above is only a planning device.

---

## (g) What was executed, and what was deliberately not

**Executed (all CPU / local reads):** parsing and validating the four frozen JSONL files; replaying
`stratified_split` (seed 42) for `n_dev ∈ {60,150,300,1112}`; byte-comparison of
`sample_variants(32, 42)` against the pilot's `variants.json`; cell-id collision count; recount of the
pilot's `results.jsonl` (3840 rows, 3840 unique cell_ids, 60 items × 32 variants × 2 models, 0 failed,
3 non-parse cells); recomputation of `metric_m1_spread` + `equivalence_verdict`; `sha256sum` of the pilot
results; `power_probe.py` rerun plus the extrapolation above; runtime arithmetic.

**Not executed:** any Ollama/GPU call — the runner command in §d is reported, not run. A GPU training job
is in flight; a second GPU consumer would corrupt it.

## Files

* This report: `docs/item_expansion_plan.md` (new).
* No source file was modified. No frozen benchmark file was touched. No run directory was created or altered.
* Next artifacts, when the GPU is free: `runs/pilot_v3_challenge300/{results.jsonl,manifest.json,variants.json}`
  → `analysis/pilot_v3_challenge300/report.md`.
