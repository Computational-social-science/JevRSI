"""M1–M8 statistics for the OASP program (RESEARCH_PROTOCOL.md §4).

Design note that drives every choice here
----------------------------------------
The matrix is **paired**: all prompt variants see the *same* items, decoded
greedily (temperature 0) with a fixed seed. If the determinism check passes, any
difference in accuracy between two variants is a *causal effect of the wording*,
not sampling noise. That is what makes "a static prompt is one sample from prompt
space" a measurable claim rather than a metaphor.

Consequently there are two distinct reference points, and they answer different
questions:

* **Permutation null (within-item variant-label shuffle).** Preserves each item's
  difficulty and the overall accuracy, destroys the association between a variant
  and *which items* it gets right. Tests whether the observed spread is
  attributable to variant identity at all. This is the confirmatory test for H1.
* **Noise floor (M6).** A re-run of one fixed prompt at temperature > 0 over
  several seeds. This is *not* a null test; it is a practical-significance
  yardstick, used because the protocol's decision rule compares the spread to
  ``2 x noise floor``.

Both are reported. A large spread that fails its permutation test is noise; a
spread that passes the permutation test but is smaller than the noise floor is
real but operationally irrelevant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config, evaluator


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_results(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    path = run_dir / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no results.jsonl in {run_dir}")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # Drop duplicates (a crash mid-write can leave a repeated cell) keeping the last.
    df = df.drop_duplicates(subset=["cell_id"], keep="last").reset_index(drop=True)
    df["correct"] = df["correct"].astype(bool)
    df["parse_ok"] = df["parse_ok"].astype(bool)
    return df


def load_manifest(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Core aggregations
# ---------------------------------------------------------------------------


def accuracy_matrix(df: pd.DataFrame, *, model: str | None = None, split: str | None = None) -> pd.DataFrame:
    """``variants x models`` accuracy matrix (mean over items)."""
    d = df
    if model is not None:
        d = d[d["model"] == model]
    if split is not None:
        d = d[d["split"] == split]
    return (
        d.groupby(["variant", "model"], observed=True)["correct"]
        .mean()
        .unstack("model")
    )


def item_outcome_matrix(
    df: pd.DataFrame, *, model: str, split: str | None = None, complete: bool = True
) -> tuple[pd.DataFrame, dict]:
    """``items x variants`` boolean matrix — the paired structure, made explicit.

    With ``complete=True`` (the default) the matrix is restricted to the largest
    rectangle with no missing cells: variants that ran on every retained item, and
    items seen by every retained variant.

    This matters more than it looks. A partially-collected matrix has holes, and a
    hole is not neutral here:

    * ``nan`` propagates through the accuracy mean, so M1 and M3 become ``nan``;
    * ``nan != nan`` is **True**, so a hole is silently counted as an IPS *flip*,
      inflating M2.

    Silently analysing an incomplete grid is exactly the failure class the
    protocol warns about (a measurement describing the plumbing, not the object),
    so holes are removed here and their count is reported rather than ignored.
    """
    d = df[df["model"] == model]
    if split is not None:
        d = d[d["split"] == split]
    M = d.pivot_table(index="item_id", columns="variant", values="correct", aggfunc="last")
    cov = {
        "items_present": int(M.shape[0]),
        "variants_present": int(M.shape[1]),
        "items_used": int(M.shape[0]),
        "variants_used": int(M.shape[1]),
        "items_dropped_incomplete": 0,
        "variants_dropped_incomplete": 0,
        "complete": True,
    }
    if not complete or M.empty:
        return M, cov

    before_v, before_i = M.shape[1], M.shape[0]
    M = M.dropna(axis=1, how="any")   # variants missing any item
    M = M.dropna(axis=0, how="any")   # items missing any retained variant
    cov.update(
        {
            "items_used": int(M.shape[0]),
            "variants_used": int(M.shape[1]),
            "items_dropped_incomplete": int(before_i - M.shape[0]),
            "variants_dropped_incomplete": int(before_v - M.shape[1]),
            "complete": bool(M.shape[0] > 0 and M.shape[1] > 1),
        }
    )
    return M, cov


# ---------------------------------------------------------------------------
# M1 — prompt spread, with item-level bootstrap CI
# ---------------------------------------------------------------------------


def metric_m1_spread(
    df: pd.DataFrame, *, split: str | None = None, B: int = 2000, seed: int = 42
) -> dict:
    out: dict[str, dict] = {}
    rng = np.random.default_rng(seed)
    for model in sorted(df["model"].unique()):
        M, cov = item_outcome_matrix(df, model=model, split=split)
        if M.empty or M.shape[1] < 2:
            continue
        vals = M.to_numpy(dtype=float)          # items x variants
        K, N = vals.shape
        acc = vals.mean(axis=0)                 # per-variant accuracy
        spread_obs = float(acc.max() - acc.min())
        cv_obs = float(acc.std(ddof=1) / acc.mean()) if acc.mean() > 0 else float("nan")

        # Item-level bootstrap: which items were drawn is the dominant uncertainty.
        idx = rng.integers(0, K, size=(B, K))
        boot_spread = np.empty(B)
        for b in range(B):
            a = vals[idx[b]].mean(axis=0)
            boot_spread[b] = a.max() - a.min()

        out[model] = {
            "n_items": int(K),
            "n_variants": int(N),
            "coverage": cov,
            "accuracy_mean": round(float(acc.mean()), 4),
            "accuracy_min": round(float(acc.min()), 4),
            "accuracy_max": round(float(acc.max()), 4),
            "range": round(spread_obs, 4),
            "range_ci95": [round(float(np.percentile(boot_spread, 2.5)), 4),
                           round(float(np.percentile(boot_spread, 97.5)), 4)],
            "sd_across_variants": round(float(acc.std(ddof=1)), 4),
            "cv": round(cv_obs, 4),
            "best_variant": str(M.columns[int(np.argmax(acc))]),
            "worst_variant": str(M.columns[int(np.argmin(acc))]),
        }
    return out


# ---------------------------------------------------------------------------
# Permutation null — within-item variant-label shuffle
# ---------------------------------------------------------------------------


def _spread_stat(vals: np.ndarray) -> float:
    a = vals.mean(axis=0)
    return float(a.max() - a.min())


def _eta2_prompt(vals: np.ndarray) -> float:
    """eta^2 of the variant factor on a single-model items x variants matrix."""
    N = vals.shape[1]
    if N < 2:
        return 0.0
    grand = vals.mean()
    ss_total = float(((vals - grand) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ss_variant = float(((vals.mean(axis=0) - grand) ** 2).sum() * vals.shape[0])
    return ss_variant / ss_total


def permutation_null(
    df: pd.DataFrame,
    *,
    model: str,
    split: str | None = None,
    B: int = 2000,
    seed: int = 42,
) -> dict:
    """Null for H1: shuffle variant labels *within each item*.

    Preserves item difficulty and total accuracy; destroys variant identity.
    Returns the null distributions and one-sided p-values for the spread and for
    eta^2_prompt.
    """
    M, cov = item_outcome_matrix(df, model=model, split=split)
    vals = M.to_numpy(dtype=float)
    K, N = vals.shape
    if K == 0 or N < 2:
        return {"model": model, "n_items": int(K), "n_variants": int(N),
                "note": "incomplete grid — no testable rectangle", "coverage": cov}

    obs_spread = _spread_stat(vals)
    obs_eta2 = _eta2_prompt(vals)

    rng = np.random.default_rng(seed)
    null_spread = np.empty(B)
    null_eta2 = np.empty(B)
    for b in range(B):
        perm = np.empty_like(vals)
        for i in range(K):
            perm[i] = rng.permutation(vals[i])
        null_spread[b] = _spread_stat(perm)
        null_eta2[b] = _eta2_prompt(perm)

    # +1 smoothing: a p-value of exactly 0 is not reportable with finite B.
    p_spread = float((np.sum(null_spread >= obs_spread) + 1) / (B + 1))
    p_eta2 = float((np.sum(null_eta2 >= obs_eta2) + 1) / (B + 1))

    return {
        "model": model,
        "n_items": int(K),
        "n_variants": int(N),
        "observed_spread": round(obs_spread, 4),
        "null_spread_mean": round(float(null_spread.mean()), 4),
        "null_spread_p95": round(float(np.percentile(null_spread, 95)), 4),
        "p_spread": p_spread,
        "observed_eta2_prompt": round(obs_eta2, 4),
        "null_eta2_mean": round(float(null_eta2.mean()), 4),
        "null_eta2_p95": round(float(np.percentile(null_eta2, 95)), 4),
        "p_eta2": p_eta2,
        "excess_spread_ratio": round(obs_spread / float(null_spread.mean()), 3)
        if null_spread.mean() > 0
        else None,
    }


# ---------------------------------------------------------------------------
# M2 — Item-level Prompt Sensitivity
# ---------------------------------------------------------------------------


def metric_m2_ips(
    df: pd.DataFrame, *, split: str | None = None, B: int = 2000, seed: int = 42
) -> dict:
    """Fraction of items whose correctness is not constant across variants.

    Under exact prompt-invariance this statistic is **identically 0**, so it needs
    no null distribution — only a CI.

    Note on the reference value: a within-item permutation of variant labels leaves
    ``min != max`` **invariant** (it permutes the same multiset), so permutation can
    never provide a null for this statistic — an earlier version of this module
    reported one and it was misleading by construction. The reference reported here
    is instead the *independent-outcome* baseline: if an item's N outcomes were
    independent Bernoulli draws at that item's own marginal accuracy ``p_i``, the
    expected flip probability would be ``1 - p_i^N - (1-p_i)^N``.

    Read the comparison as a correlation diagnostic:

    * ``observed > reference`` → outcomes are anti-correlated within items; variants
      systematically disagree, which is a prompt effect rather than item noise;
    * ``observed < reference`` → outcomes are positively correlated; items are
      consistently right or wrong regardless of wording.
    """
    out: dict[str, dict] = {}
    rng = np.random.default_rng(seed)
    for model in sorted(df["model"].unique()):
        M, cov = item_outcome_matrix(df, model=model, split=split)
        if M.empty or M.shape[1] < 2:
            continue
        vals = M.to_numpy(dtype=float)
        K, N = vals.shape
        flips = vals.min(axis=1) != vals.max(axis=1)
        ips_obs = float(flips.mean())

        idx = rng.integers(0, K, size=(B, K))
        boot = flips[idx].mean(axis=1)

        p_item = vals.mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ref = 1.0 - np.power(p_item, N) - np.power(1.0 - p_item, N)
        reference = float(np.mean(ref))

        out[model] = {
            "n_items": int(K),
            "n_variants": int(N),
            "coverage": cov,
            "ips": round(ips_obs, 4),
            "ips_ci95": [round(float(np.percentile(boot, 2.5)), 4),
                         round(float(np.percentile(boot, 97.5)), 4)],
            "n_agree": int((~flips).sum()),
            "n_flip": int(flips.sum()),
            "invariance_null": 0.0,
            "independent_reference": round(reference, 4),
            "ips_minus_reference": round(ips_obs - reference, 4),
        }
    return out


# ---------------------------------------------------------------------------
# M4 / M5 — ranking stability
# ---------------------------------------------------------------------------


def _kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.kendalltau(a, b).statistic)


def metric_m4_m5_rank_stability(df: pd.DataFrame, *, split: str | None = None) -> dict:
    A = accuracy_matrix(df, split=split)
    # Restrict to variants complete across ALL models. A NaN (from a partially
    # collected model) makes `np.sign(nan - x)` NaN, which then compares `!=` the
    # reference sign and is counted as an order FLIP — the same contamination class
    # that inflated M2's flip count. Here it produced a flip rate of 1.0 on data
    # where one model was ahead on every single variant.
    n_variants_before = int(A.shape[0])
    A = A.dropna(axis=0, how="any")
    models = list(A.columns)
    if A.shape[0] < 2:
        return {"models": models, "n_variants": int(A.shape[0]),
                "n_variants_dropped_incomplete": n_variants_before - int(A.shape[0]),
                "note": "no variant is complete across all models"}
    if len(models) < 2:
        return {"models": models, "n_variants": int(A.shape[0]),
                "note": "needs >=2 models with complete data"}

    taus: list[float] = []
    flips = 0
    ties = 0
    decided = 0
    ref_tied_pairs = 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            vi = A[models[i]].to_numpy(dtype=float)
            vj = A[models[j]].to_numpy(dtype=float)
            taus.append(_kendall_tau_b(vi, vj))
            # Reference order = which model wins on average across variants.
            ref = np.sign(np.mean(vi) - np.mean(vj))
            d = np.sign(vi - vj)
            ties += int(np.sum(d == 0))
            if ref == 0:
                # The pair ties on average, so there is no reference order for a
                # "flip" to depart from. Emitting 0/mask = 0.0 here would present an
                # undefined quantity as a reassuring one — the same failure mode as
                # the NaN propagation this function was fixed for.
                ref_tied_pairs += 1
                continue
            # A tie gives sign 0, and `0 != sign(+/-1)` is True, so counting every
            # comparison as informative made ties read as ORDER FLIPS and inflated
            # this rate. Only decided comparisons belong in the denominator.
            mask = d != 0
            decided += int(mask.sum())
            flips += int(np.sum(d[mask] != ref))

    # Ranking induced by each variant. With only 2 models a per-variant ranking is a
    # 2-element vector, so a single tie makes it constant and kendalltau returns NaN,
    # which then propagated through np.mean and made this whole metric read `nan`.
    # The leg is only defined once there are >=3 models to order.
    rank_taus: list[float] = []
    if len(models) >= 3:
        for vi in range(A.shape[0]):
            for vj in range(vi + 1, A.shape[0]):
                r1 = A.iloc[vi].rank().to_numpy()
                r2 = A.iloc[vj].rank().to_numpy()
                rank_taus.append(_kendall_tau_b(r1, r2))

    def _safe_mean(xs: list[float]) -> tuple[float | None, int]:
        """Mean over the non-NaN entries, plus how many were usable.

        Reported instead of a bare mean so a metric computed from 2 of 32 pairs can
        never masquerade as one computed from all 32.
        """
        if not xs:
            return None, 0
        arr = np.asarray(xs, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return None, 0
        return round(float(arr.mean()), 4), int(arr.size)

    mean_tau, n_tau = _safe_mean(taus)
    mean_rank_tau, n_rank_tau = _safe_mean(rank_taus)
    min_tau = float(np.nanmin(taus)) if (taus and not np.all(np.isnan(taus))) else None
    min_rank_tau = (
        float(np.nanmin(rank_taus)) if (rank_taus and not np.all(np.isnan(rank_taus))) else None
    )
    frac_nonident = None
    if rank_taus:
        arr = np.asarray(rank_taus, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size:
            frac_nonident = round(float(np.mean(arr < 1.0)), 4)

    return {
        "models": models,
        "n_variants": int(A.shape[0]),
        "mean_pairwise_tau_accuracy_profile": mean_tau,
        "n_valid_tau_pairs": n_tau,
        "min_pairwise_tau": None if min_tau is None else round(min_tau, 4),
        "mean_tau_between_variant_rankings": mean_rank_tau,
        "n_valid_rank_tau_pairs": n_rank_tau,
        "rank_tau_undefined_reason": (
            "needs >=3 models: a per-variant ranking of 2 models is constant when they tie, "
            "and kendalltau of a constant vector is NaN"
        ) if len(models) < 3 else None,
        "min_tau_between_variant_rankings": None if min_rank_tau is None else round(min_rank_tau, 4),
        "frac_variant_pairs_with_nonidentical_ranking": frac_nonident,
        "model_pair_order_flip_rate": round(flips / decided, 4) if decided else None,
        "flip_rate_undefined_reason": (
            "every model pair ties on average across variants, so there is no reference "
            "ranking to flip away from"
        ) if (not decided and ref_tied_pairs) else None,
        "n_ref_tied_model_pairs": ref_tied_pairs,
        "n_decided_comparisons": decided,
        "n_tied_comparisons": ties,
        "ranking_by_variant": {
            str(v): list(A.loc[v].sort_values(ascending=False).index) for v in A.index
        },
    }


# ---------------------------------------------------------------------------
# Capability-floor screen (§3)
# ---------------------------------------------------------------------------


def capability_floor(
    df: pd.DataFrame, *, chance: float, split: str | None = None, alpha: float = 0.05,
    band: tuple[float, float] | None = None,
) -> dict:
    """Per-model capability screen, including the interior-band selection rule.

    A model outside ``band`` is reported but **excluded from H1**: a ceiling-bound
    model has no headroom for wording to move, and a floor-bound model has no signal
    to move. Reporting H1 for either would be uninformative rather than negative.
    """
    band = band or config.DESIGN["interior_band"]
    out: dict[str, dict] = {}
    d = df if split is None else df[df["split"] == split]
    for model, grp in d.groupby("model", observed=True):
        acc = float(grp["correct"].mean())
        n = int(len(grp))
        # Wilson lower bound — the honest lower CI for a proportion near 0/1.
        z = stats.norm.ppf(1 - alpha / 2)
        phat = acc
        denom = 1 + z**2 / n
        centre = phat + z**2 / (2 * n)
        half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
        lo, hi = (centre - half) / denom, (centre + half) / denom
        clears = bool(lo > chance)
        if not clears:
            verdict = "FLOOR_BOUND"
        elif acc > band[1]:
            verdict = "CEILING_BOUND"
        elif acc < band[0]:
            verdict = "BELOW_BAND"
        else:
            verdict = "INTERIOR"
        out[str(model)] = {
            "n": n,
            "accuracy": round(acc, 4),
            "wilson_ci95": [round(lo, 4), round(hi, 4)],
            "chance": round(chance, 4),
            "clears_floor": clears,
            "interior_band": list(band),
            "verdict": verdict,
            "eligible_for_h1": verdict == "INTERIOR",
        }
    return out


# ---------------------------------------------------------------------------
# M6 — noise floor
# ---------------------------------------------------------------------------


def metric_m6_noise_floor(noise_dir: str | Path) -> dict:
    """SD of accuracy across seeds for a *fixed* prompt at temperature > 0."""
    df = load_results(noise_dir)
    rep = (
        df.groupby(["model", "replicate"], observed=True)["correct"].mean().reset_index()
    )
    out = {}
    for model, grp in rep.groupby("model", observed=True):
        accs = grp["correct"].to_numpy(dtype=float)
        out[str(model)] = {
            "n_replicates": int(len(accs)),
            "mean_accuracy": round(float(accs.mean()), 4),
            "sd": round(float(accs.std(ddof=1)), 4) if len(accs) > 1 else 0.0,
            "range": round(float(accs.max() - accs.min()), 4),
        }
    return {"per_model": out, "max_sd": max([v["sd"] for v in out.values()], default=0.0)}


# ---------------------------------------------------------------------------
# H1 decision rule
# ---------------------------------------------------------------------------


def equivalence_verdict(m1: dict, *, margin: float, alpha: float = 0.05) -> dict:
    """TOST-style verdict for the claim that the variants are interchangeable.

    The difference test in :func:`permutation_null` asks "is the spread LARGER than
    chance relabeling?" and can therefore only ever *fail to reject* when the truth is
    equivalence — it cannot establish equivalence. This is the complementary direction:
    the variants are declared behaviourally equivalent when the **upper** confidence
    bound of the spread falls below ``margin``.

    This matters because the data can point this way. gemma3_4b on ARC-Challenge has an
    observed spread of 0.0833 against a shuffle-null mean of 0.1144 (p = 0.9915): the
    variants agree *more tightly* than independence predicts, which is evidence FOR
    equivalence. Reporting that as an inconclusive null inverts its meaning.
    """
    rows = []
    n_equiv = 0
    n_eligible = 0
    for model, row in m1.items():
        ci = row.get("range_ci95") or [None, None]
        upper = ci[1] if len(ci) > 1 else None
        equivalent = bool(upper is not None and upper < margin)
        n_eligible += 1
        n_equiv += int(equivalent)
        rows.append(
            {
                "model": model,
                "range": row.get("range"),
                "range_ci95_upper": upper,
                "margin": margin,
                "equivalent": equivalent,
                "verdict": (
                    "EQUIVALENT (variants interchangeable within margin)"
                    if equivalent
                    else "NOT ESTABLISHED (upper CI exceeds margin)"
                ),
            }
        )
    return {
        "margin": margin,
        "models_equivalent": n_equiv,
        "models_tested": n_eligible,
        "all_equivalent": bool(n_eligible and n_equiv == n_eligible),
        "rows": rows,
        "note": "Complementary to the difference test, not a substitute: the difference "
        "test speaks to HPFE, this one to prompt-insensitivity. Report both.",
    }


def h1_decision(
    m1: dict, perm: dict, m6: dict | None, design: dict | None = None, floor: dict | None = None
) -> dict:
    design = design or config.DESIGN
    floor_sd = (m6 or {}).get("max_sd", 0.0)
    floor = floor or {}
    verdict_rows = []
    n_pass = 0
    n_eligible = 0
    for model, row in m1.items():
        needs = design["h1_range_over_floor"] * floor_sd
        rng_val = row.get("range")
        spread_ok = (rng_val > needs) if (floor_sd > 0 and rng_val is not None) else None
        p = perm.get(model, {}).get("p_spread")
        perm_ok = bool(p is not None and p < design["alpha"])
        eligible = floor.get(model, {}).get("eligible_for_h1")
        if eligible is None:
            # No capability screen supplied: fall back to testing every model but
            # flag that the interior-band rule was not applied.
            eligible = None
        if eligible:
            n_eligible += 1
        ok_and = bool(perm_ok and spread_ok is True and eligible is not False)
        if ok_and:
            n_pass += 1
        verdict_rows.append(
            {
                "model": model,
                "verdict": floor.get(model, {}).get("verdict"),
                "eligible_for_h1": eligible,
                "range": rng_val,
                "needed_range": round(needs, 4),
                "spread_exceeds_floor": spread_ok,
                "perm_p_spread": p,
                "perm_significant": perm_ok,
                "counts_toward_h1": ok_and,
            }
        )
    return {
        "noise_floor_sd": floor_sd,
        "min_models_required": design["h1_min_models"],
        "models_eligible": n_eligible,
        "models_passing": n_pass,
        "h1_supported": n_pass >= design["h1_min_models"],
        "rows": verdict_rows,
        "note": "ranking-stability leg (M4 tau) is evaluated separately in m4_m5. "
        "Models outside the interior band are reported but cannot count toward H1.",
    }


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


def compute_all(run_dir: str | Path, *, noise_dir: str | Path | None = None,
                split: str | None = None, B: int | None = None) -> dict:
    df = load_results(run_dir)
    man = load_manifest(run_dir)
    B = B or man.get("design", {}).get("bootstrap_B", 2000)
    chance = man.get("benchmark", {}).get("chance_level", 0.25)

    m1 = metric_m1_spread(df, split=split, B=B)
    m2 = metric_m2_ips(df, split=split, B=B)
    m4 = metric_m4_m5_rank_stability(df, split=split)
    floor = capability_floor(df, chance=chance, split=split)
    perm = {m: permutation_null(df, model=m, split=split, B=B) for m in sorted(df["model"].unique())}
    m6 = metric_m6_noise_floor(noise_dir) if noise_dir else None

    return {
        "run_dir": str(run_dir),
        "run_manifest": {k: man.get(k) for k in ("run_id", "label", "benchmark", "design", "models")},
        "split": split or "all",
        "n_rows": int(len(df)),
        "coverage": {
            "models_present": sorted(df["model"].unique().tolist()),
            "rows_per_model": {str(k): int(v) for k, v in df["model"].value_counts().items()},
            "partial_run": bool(
                len(df) < int(man.get("n_cells_planned") or len(df))
            ),
            "n_cells_planned": man.get("n_cells_planned"),
            "n_cells_present": int(len(df)),
            "note": "M1/M2/M3 use the largest complete items x variants rectangle "
                    "per model; dropped-cell counts are reported per metric.",
        },
        "parse_rates": evaluator.summarize_parse_rates(df),
        "m1_spread": m1,
        "m2_ips": m2,
        "m3_permutation": perm,
        "m4_m5_rank_stability": m4,
        "capability_floor": floor,
        "m6_noise_floor": m6,
        "h1_decision": h1_decision(m1, perm, m6, floor=floor),
        "equivalence": equivalence_verdict(m1, margin=config.DESIGN["equivalence_margin"]),
    }


def to_markdown(rep: dict) -> str:
    L: list[str] = []
    man = rep["run_manifest"]
    L.append(f"# OASP analysis — `{man.get('run_id')}`\n")
    L.append(f"- split: **{rep['split']}** · rows: {rep['n_rows']:,} · label: **{man.get('label')}**")
    b = man.get("benchmark") or {}
    L.append(f"- benchmark: `{b.get('key')}` · chance = {b.get('chance_level')} · "
             f"items dev/test = {b.get('n_dev')}/{b.get('n_test')}")
    d = man.get("design") or {}
    L.append(f"- design: N_variants={d.get('n_variants')} · T={d.get('temperature')} · seed={d.get('seed')}\n")

    L.append("## M1 — prompt spread\n")
    L.append("| model | acc mean | min | max | **range** | range CI95 | SD | CV | best | worst |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for m, r in rep["m1_spread"].items():
        L.append(
            f"| {m} | {r['accuracy_mean']} | {r['accuracy_min']} | {r['accuracy_max']} | "
            f"**{r['range']}** | [{r['range_ci95'][0]}, {r['range_ci95'][1]}] | "
            f"{r['sd_across_variants']} | {r['cv']} | {r['best_variant']} | {r['worst_variant']} |"
        )

    L.append("\n## M2 — item-level prompt sensitivity (IPS)\n")
    L.append("Invariance null = **0** exactly; the reference column is the independent-outcome baseline.")
    L.append("")
    L.append("| model | items | variants | IPS | CI95 | agree | flip | reference | IPS−ref |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m, r in rep["m2_ips"].items():
        L.append(
            f"| {m} | {r['n_items']} | {r['n_variants']} | **{r['ips']}** | "
            f"[{r['ips_ci95'][0]}, {r['ips_ci95'][1]}] | "
            f"{r['n_agree']} | {r['n_flip']} | {r.get('independent_reference')} | "
            f"{r.get('ips_minus_reference')} |"
        )

    L.append("\n## M3 — permutation null (within-item variant shuffle)\n")
    L.append("| model | obs spread | null spread mean | p | obs \u03b7\u00b2_prompt | null \u03b7\u00b2 mean | p | excess ratio |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m, r in rep["m3_permutation"].items():
        if "observed_spread" not in r:
            L.append(f"| {m} | — | — | — | — | — | — | {r.get('note', 'no testable grid')} |")
            continue
        L.append(
            f"| {m} | {r['observed_spread']} | {r['null_spread_mean']} | {r['p_spread']:.4f} | "
            f"{r['observed_eta2_prompt']} | {r['null_eta2_mean']} | {r['p_eta2']:.4f} | "
            f"{r['excess_spread_ratio']} |"
        )

    L.append("\n## M4/M5 — ranking stability\n")
    r4 = rep["m4_m5_rank_stability"]
    L.append(
        f"- mean pairwise Kendall τ (accuracy profile across variants): "
        f"**{r4.get('mean_pairwise_tau_accuracy_profile')}** "
        f"over {r4.get('n_valid_tau_pairs')} model pair(s)"
    )
    if r4.get("rank_tau_undefined_reason"):
        L.append(f"- mean τ between variant rankings: **undefined** — {r4['rank_tau_undefined_reason']}")
    else:
        L.append(
            f"- mean τ between variant rankings: "
            f"**{r4.get('mean_tau_between_variant_rankings')}** "
            f"over {r4.get('n_valid_rank_tau_pairs')} variant pair(s)"
        )
        L.append(
            f"- fraction of variant pairs with non-identical ranking: "
            f"**{r4.get('frac_variant_pairs_with_nonidentical_ranking')}**"
        )
    L.append(
        f"- model-pair order flip rate: **{r4.get('model_pair_order_flip_rate')}** over "
        f"{r4.get('n_decided_comparisons')} decided comparisons "
        f"({r4.get('n_tied_comparisons')} tied, excluded from the denominator)"
    )

    L.append("\n## Capability floor / interior-band screen\n")
    L.append("Only `INTERIOR` models can count toward H1 "
             f"(band {config.DESIGN['interior_band']}): a ceiling-bound model has no headroom, "
             "a floor-bound model has no signal.")
    L.append("")
    L.append("| model | acc | Wilson CI95 | chance | clears | verdict | eligible |")
    L.append("|---|---|---|---|---|---|---|")
    for m, r in rep["capability_floor"].items():
        L.append(f"| {m} | {r['accuracy']} | [{r['wilson_ci95'][0]}, {r['wilson_ci95'][1]}] | "
                 f"{r['chance']} | {'YES' if r['clears_floor'] else 'NO'} | "
                 f"**{r.get('verdict','—')}** | {'yes' if r.get('eligible_for_h1') else 'no'} |")

    L.append("\n## H1 decision\n")
    h = rep["h1_decision"]
    L.append(f"- noise-floor SD (M6): {h['noise_floor_sd']}")
    L.append(f"- models passing: **{h['models_passing']}** / required {h['min_models_required']} "
             f"→ **H1 {'SUPPORTED' if h['h1_supported'] else 'NOT SUPPORTED'}**")
    L.append("")
    L.append("| model | verdict | eligible | range | needed | spread>floor | perm p | counts |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in h["rows"]:
        p = r.get("perm_p_spread")
        p_str = f"{p:.4f}" if isinstance(p, float) else str(p)
        L.append(
            f"| {r['model']} | {r.get('verdict')} | {r.get('eligible_for_h1')} | "
            f"{r['range']} | {r['needed_range']} | {r['spread_exceeds_floor']} | "
            f"{p_str} | {r['counts_toward_h1']} |"
        )

    L.append("\n## Equivalence test (the complementary direction)\n")
    eq = rep["equivalence"]
    L.append(
        "A difference test can only *fail to reject* when the truth is equivalence — it can "
        "never **establish** it. This test does the complementary job: the variants are "
        "declared behaviourally interchangeable when the **upper** confidence bound of the "
        f"spread falls below the margin **{eq['margin']}**."
    )
    L.append("")
    L.append("| model | range | upper CI95 | margin | verdict |")
    L.append("|---|---|---|---|---|")
    for r in eq["rows"]:
        L.append(
            f"| {r['model']} | {r['range']} | {r['range_ci95_upper']} | {r['margin']} | "
            f"**{r['verdict']}** |"
        )
    L.append("")
    L.append(
        f"-> **{eq['models_equivalent']}/{eq['models_tested']}** models are equivalent "
        "within margin. Report alongside the difference test, never instead of it."
    )

    L.append("\n## Parse rates\n")
    L.append("| model | n | parse rate | accuracy | accuracy|parsed |")
    L.append("|---|---|---|---|---|")
    for m, r in rep["parse_rates"].items():
        L.append(f"| {m} | {r['n']} | {r['parse_rate']} | {r['accuracy']} | {r['accuracy_on_parsed']} |")
    return "\n".join(L)
