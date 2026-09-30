"""
floor_a_dev_bootstrap.py -- the measurement-sampling floor for a DECISION made on the dev split.

WHY THIS FILE EXISTS
    The accept rule runs on dev, so its threshold must be built from dev's own sampling noise. The
    repository carried two numbers for this and neither could be used:

      * floor_A_measurement_sampling.se_pp = 1.0668 -- a case-level cluster bootstrap over the 400
        TEST cases. The right measurement, on the wrong split. Quoting it for a dev decision
        understates dev's noise, which dev_split_resolution in INSTRUMENT_CALIBRATION.json says
        explicitly: "Quoting a test-derived threshold for a dev decision would make the loop accept
        noise."
      * dev_split_resolution.acc_se_pp = 2.14 -- the correct quantity, but its source
        (measurement/fitness_resolution.json) and its generator (floor_a_cluster_bootstrap.py) are
        both absent from the live tree, so the number has no derivation anyone can check.

    So the number is recomputed here, from the dev bundle, in this file, and the intermediate
    quantities are written out. A threshold whose derivation cannot be re-run is a constant.

THE MEASUREMENT, precisely
    The unit of resampling is the CASE, not the question. The dev split is 88 cases of 5 questions
    each, drawn as states with several questions over shared context; the five questions of a case
    are not independent, so a binomial SE over 440 questions understates the true sampling variance
    by the design effect. The bootstrap resamples whole cases with replacement, which is the only
    resampling scheme that respects the dependence.

    Two quantities come out, and the accept threshold needs both:

      * the CLUSTER SE of absolute accuracy on dev -- how far the number moves when the cases move
      * the PAIRED SE of a DIFFERENCE between two configurations on the same cases -- strictly
        smaller, because the shared case-to-case difficulty cancels in the difference

    A paired difference is what the accept rule actually tests, so the paired SE is the one that
    sets the threshold. Reporting only the absolute SE would be the conservative error; reporting
    only the paired one without the absolute would hide how small dev is.

WHAT THIS DOES NOT DO
    It does not touch the frozen test split, and it does not produce an accept threshold by itself.
    It produces the noise floor that a threshold is built from, and it is deliberately silent about
    the correction for the number of decisions -- that belongs to whoever writes the pre-registration,
    because the horizon is a property of the run's plan and not of the measurement.

Usage:
    python measurement/floor_a_dev_bootstrap.py [--replicates 10000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "measurement" / "logit_bundles"
OUT = ROOT / "measurement" / "floor_a_dev.json"


def load_split(name: str) -> dict:
    p = BUNDLES / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"[fatal] {p} not found. Run measurement/build_logit_bundles.py first.")
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = d["rows"]
    if not rows or "case" not in rows[0]:
        raise SystemExit(
            f"[fatal] {p} carries no case id. A cluster bootstrap needs the unit of dependence, and "
            f"the bundle was written before qid/case were recorded. Rebuild it with "
            f"measurement/build_logit_bundles.py.")
    return {
        "target": [np.asarray(r["target"], dtype=np.float64) for r in rows],
        "type": [r["type"] for r in rows],
        "case": [r["case"] for r in rows],
        "n_cand": [len(r["target"]) for r in rows],
    }


def correctness(d: dict, probs: list[np.ndarray]) -> np.ndarray:
    """Per-question hit, as a float vector. Questions with fewer than two candidates are dropped:
    their argmax is trivially right and they carry no information about a decision."""
    out = []
    for t, p, k in zip(d["target"], probs, d["n_cand"]):
        if k < 2:
            continue
        out.append(float(int(np.argmax(p)) == int(np.argmax(t))))
    return np.asarray(out, dtype=np.float64)


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def case_index(d: dict) -> tuple[list[np.ndarray], list[str]]:
    """Group question positions by case, preserving a deterministic case order."""
    order: dict[str, list[int]] = {}
    for i, c in enumerate(d["case"]):
        order.setdefault(c, []).append(i)
    keys = sorted(order)
    return [np.asarray(order[k], dtype=np.int64) for k in keys], keys


def cluster_bootstrap(stat, groups: list[np.ndarray], replicates: int, seed: int) -> np.ndarray:
    """Resample WHOLE CASES with replacement. The statistic receives the concatenated positions of
    the drawn cases, so it sees the same clustered structure the real sample has."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    out = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        pick = rng.integers(0, n, size=n)
        idx = np.concatenate([groups[i] for i in pick])
        out[r] = stat(idx)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Case-level cluster bootstrap on the dev split.")
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    d = load_split("medium")
    # The identity stage: raw probabilities from the cached logits, no calibration. It is the
    # reference configuration a candidate is compared against, and it is deterministic, so any
    # spread measured here is the benchmark's, not the harness's.
    bundle = json.loads((BUNDLES / "medium.json").read_text(encoding="utf-8"))
    probs = [softmax(np.asarray(z, dtype=np.float64)) for z in bundle["logits"]]

    hit = correctness(d, probs)
    groups, _case_names = case_index(d)
    # correctness() may have dropped short questions; realign the case groups to the kept positions
    kept = [i for i, k in enumerate(d["n_cand"]) if k >= 2]
    remap = {old: new for new, old in enumerate(kept)}
    groups = [np.asarray([remap[i] for i in g if i in remap], dtype=np.int64) for g in groups]
    groups = [g for g in groups if len(g)]

    print("=" * 78)
    print("Floor A on the decision split -- case-level cluster bootstrap")
    print("=" * 78)
    print(f"  split        : medium (the keep/revert decision set)")
    print(f"  cases        : {len(groups)}")
    print(f"  questions    : {len(hit)}  (questions with <2 candidates excluded: "
          f"{len(d['n_cand']) - len(kept)})")
    print(f"  replicates   : {a.replicates}   seed {a.seed}")
    print(f"  observed acc : {hit.mean():.4f}")
    print()

    t0 = time.perf_counter()
    boots = cluster_bootstrap(lambda idx: hit[idx].mean(), groups, a.replicates, a.seed)
    se_abs = float(boots.std(ddof=1))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # Binomial SE for comparison, to show the design effect rather than assert it.
    p = float(hit.mean())
    se_binom = float(np.sqrt(p * (1 - p) / len(hit)))
    design_effect = (se_abs / se_binom) ** 2 if se_binom else float("nan")

    print(f"  cluster SE (absolute accuracy) : {se_abs:.6f}  ({se_abs*100:.4f} pp)")
    print(f"  binomial SE (WRONG here)       : {se_binom:.6f}  ({se_binom*100:.4f} pp)")
    print(f"  design effect                  : {design_effect:.4f}")
    print(f"  95% percentile interval        : [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  ({time.perf_counter()-t0:.1f}s)")

    # A PAIRED difference needs two configurations, and the comparison arm has to be one that can
    # actually MOVE the argmax. The first version used a single scalar temperature, which is a
    # monotone map and therefore leaves argmax invariant -- so it reported a paired SE of exactly
    # 0.0000, which is a true statement about that arm and a useless measurement of the paired
    # noise, because a real candidate is not monotone.
    #
    # The arm used here is a per-dimension VECTOR temperature: an independent positive scale on each
    # logit dimension. It is not monotone, it moves argmax, and it is still a deterministic function
    # of the logits with no fitted parameter -- so the spread measured is the benchmark's, and this
    # is a measurement instrument, NOT a claim that vector temperature is a good idea.
    #
    # Scaled PER QUESTION, because the number of candidates varies (2 for boolean, 4-5 for choice and
    # score) and a single fixed-length vector cannot multiply a 2-vector. The scale is drawn from one
    # seeded generator in question order, so the whole construction is reproducible.
    rng = np.random.default_rng(12345)
    p2 = []
    for z in bundle["logits"]:
        zz = np.asarray(z, dtype=np.float64)
        p2.append(softmax(zz * rng.uniform(0.6, 1.8, size=len(zz))))
    hit2 = correctness(d, p2)
    diff = hit2 - hit
    boots_d = cluster_bootstrap(lambda idx: diff[idx].mean(), groups, a.replicates, a.seed + 1)
    se_paired = float(boots_d.std(ddof=1))

    print()
    n_moved = int((diff != 0).sum())
    print(f"  paired SE (difference vs identity, random VECTOR temperature) : "
          f"{se_paired:.6f}  ({se_paired*100:.4f} pp)")
    print(f"  mean difference observed        : {diff.mean():+.6f}  ({diff.mean()*100:+.4f} pp)")
    print(f"  questions whose hit changed     : {n_moved} of {len(diff)} "
          f"({100*n_moved/len(diff):.1f}%)")
    if n_moved == 0:
        print("  [WARN] the comparison arm moved nothing, so the paired SE is not a usable floor")
    else:
        print("  the arm moves the argmax, so this paired SE is a real measurement of the noise in a")
        print("  difference -- which is the quantity the accept rule actually tests")

    payload = {
        "_what_this_is": "Case-level cluster bootstrap of measurement noise on the DECISION split "
                         "(dev/medium). Recomputed here because the two numbers previously carried "
                         "for this were unusable: 1.0668 pp is the TEST split's, and 2.14 pp had no "
                         "derivation in the live tree.",
        "split": "medium",
        "n_cases": len(groups),
        "n_questions": len(hit),
        "replicates": a.replicates,
        "seed": a.seed,
        "observed_accuracy": float(hit.mean()),
        "cluster_se_pp": se_abs * 100,
        "binomial_se_pp_wrong": se_binom * 100,
        "design_effect": design_effect,
        "accuracy_ci95": ci,
        "paired_se_pp": se_paired * 100,
        "paired_comparison_arm": "a fixed random per-dimension vector temperature (seed 12345). "
                                 "Chosen because it is deterministic, parameter-free, and NOT "
                                 "monotone, so it moves argmax. A scalar temperature is monotone and "
                                 "reports a paired SE of exactly 0, which is true and useless.",
        "paired_mean_difference_pp": float(diff.mean() * 100),
        "what_it_is": "the spread of the score when the CASES change, not when the model changes. "
                      "Independent of which model or harness produced the predictions, which is why "
                      "it transfers to a configuration that has never been run before.",
        "which_one_sets_the_threshold": "paired_se_pp. The accept rule tests a DIFFERENCE between two "
                                        "configurations on the same cases, and the shared "
                                        "case-to-case difficulty cancels in that difference, so the "
                                        "paired SE is strictly the right scale.",
        "does_not_include": "the run-to-run floor. A deterministic harness has none by construction; "
                            "a stochastic one (QLoRA training) has one that must be measured on THIS "
                            "seed before any threshold covering it may be armed.",
        "method": "resample whole cases with replacement, 10000 replicates, numpy default_rng",
    }
    pathlib.Path(a.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
