"""The prompt-space landscape: hill-climb with fallback points.

Brief constraint #5 asks for a LANDSCAPE built from two streams:

* the **hill-climb curve** — the score of surviving solutions per generation;
* the **fallback points** — the discarded candidates, which are the sampling of the
  region the search walked through and rejected.

Constraint #2 asks that fallbacks be *quantified*: every discard carries a
**reason code**, and the codes are assigned from the candidate's *structure* before
its score is known. Assigning them blind is what makes the later M8 test
("do discard classes have distinct score signatures?") evidence rather than a
narrative fitted to the scores. If the classes turn out to be score-identical,
"death valleys" is decorative language and must be dropped.

Constraint #4 asks for three specific defences against local optima, each of which
is logged so its effect is auditable rather than asserted:

1. **periodic stochastic cold-start** — every ``cold_start_every`` generations the
   incumbent is abandoned and the search restarts from a uniformly random point;
2. **population diversity** — parents are drawn by tournament from a diverse
   top-K set rather than from the single incumbent;
3. **forced large mutation on plateau** — after ``plateau_window`` generations
   without improvement, the next candidate is forced to differ in >=3 features.

Design property worth noting: the search walks the **same feature space** that
``prompt_space.sample_variants`` samples. The hill-climb result is therefore
directly comparable to the random sample — "is the static prompt far from optimal"
becomes a within-space comparison instead of a cross-study one.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import benchmark, config, evaluator, llm, prompt_space
from .git_memory import GitMemory

# ---------------------------------------------------------------------------
# Blind reason codes (assigned from structure, before any score exists)
# ---------------------------------------------------------------------------

REASON_CODES = (
    "NO_CHANGE",
    "UNCONSTRAINED_OUTPUT",
    "FORMAT_CONFLICT",
    "LABEL_STYLE_MISMATCH",
    "ORDER_TEMPLATE_CONFLICT",
    "AMBIGUOUS_INSTRUCTION",
    "OVER_SPECIFIED",
    "OFF_TASK",
    "OTHER",
)


def assign_reason_code(parent: dict, child: dict) -> str:
    """Structural reason code for a mutation. MUST NOT look at any score."""
    if child == parent:
        return "NO_CHANGE"

    f, ol, ne, od = child["format"], child["opt_label"], child["no_explanation"], child["order"]

    # Instruction promises a bare letter while the option list is bracketed/labelled
    # differently — the model is told two different output alphabets.
    if f == "bare" and ol == "paren":
        return "LABEL_STYLE_MISMATCH"

    # Instruction demonstrates "(X)" while options are rendered "A." — mismatch
    # between the exemplar and the list the model must read.
    if f == "paren_letter" and ol == "upper":
        return "FORMAT_CONFLICT"

    # "only the letter" without an explicit no-explanation clause: the two clauses
    # pull in opposite directions.
    if f == "bare" and ne is False:
        return "FORMAT_CONFLICT"

    # Strict output template combined with the question placed first, so the
    # instruction arrives after the model may already have begun answering.
    if od == "question_first" and f == "answer_colon":
        return "ORDER_TEMPLATE_CONFLICT"

    # Generic request verb with a strict template — the least specific pairing.
    if child["verb"] == "Respond to" and f != "bare":
        return "AMBIGUOUS_INSTRUCTION"

    # Dropping the no-explanation clause while keeping a structured template
    # removes the only length constraint the prompt had.
    if ne is False and parent.get("no_explanation") is True and f != "bare":
        return "UNCONSTRAINED_OUTPUT"

    return "OTHER"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    features: dict
    generation: int
    parent_key: str | None
    mechanism: str
    reason_code: str
    n_features_changed: int = 0

    @property
    def key(self) -> str:
        return prompt_space.feature_key(self.features)


def mutate(
    parent: dict,
    *,
    n_changes: int,
    rng: random.Random,
) -> dict:
    keys = list(prompt_space.FEATURES)
    chosen = rng.sample(keys, k=min(n_changes, len(keys)))
    child = dict(parent)
    for k in chosen:
        options = [v for v in prompt_space.FEATURES[k] if v != parent[k]]
        child[k] = rng.choice(options)
    return child


def mutate_random(*, rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in prompt_space.FEATURES.items()}


def evaluate(
    features: dict,
    items: list[benchmark.Item],
    model_keys: list[str],
    *,
    temperature: float,
    seed: int,
    num_predict: int,
    timeout: int,
) -> tuple[float, list[dict]]:
    """Accuracy of one prompt candidate over the given items and models."""
    per_model: list[dict] = []
    scores: list[float] = []
    for mk in model_keys:
        spec = config.TARGET_MODELS[mk]
        correct = 0
        parsed = 0
        for it in items:
            comp = llm.chat(
                spec,
                prompt_space.render(features, it),
                temperature=temperature,
                seed=seed,
                num_predict=num_predict,
                timeout=timeout,
            )
            if not comp.ok:
                continue
            sc = evaluator.score(comp.text, it)
            correct += int(sc.correct)
            parsed += int(sc.parse_ok)
        acc = correct / len(items) if items else 0.0
        scores.append(acc)
        per_model.append(
            {"model": mk, "accuracy": round(acc, 4), "parse_rate": round(parsed / len(items), 4)}
        )
    return (sum(scores) / len(scores) if scores else 0.0), per_model


def run_search(
    *,
    run_id: str,
    benchmark_key: str = "arc_easy",
    model_keys: list[str] | None = None,
    generations: int = 40,
    items: int = 30,
    population_size: int = 5,
    cold_start_every: int = 10,
    plateau_window: int = 6,
    seed: int | None = None,
    num_predict: int | None = None,
    timeout: int = 600,
    split: str = "dev",
) -> dict:
    """Run the search loop; write ``landscape.jsonl`` and ``hillclimb.json``."""
    d = config.DESIGN
    seed = d["seed"] if seed is None else seed
    num_predict = d["num_predict"] if num_predict is None else num_predict
    model_keys = model_keys or ["gemma3_4b"]
    config.assert_decoupled(model_keys)

    rng = random.Random(seed)

    items_all = benchmark.load_items(benchmark_key)
    split_map = benchmark.stratified_split(
        items_all, n_dev=max(items, 1), n_test=max(items, 1), seed=seed
    )
    probe = split_map[split]

    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    landscape_path = run_dir / "landscape.jsonl"
    mem = GitMemory(run_dir / "expgit")

    # --- seed: the canonical heuristic point (what published practice reports) --
    seed_features = dict(prompt_space.CANONICAL)
    best_score, _ = evaluate(
        seed_features, probe, model_keys,
        temperature=d["temperature"], seed=seed, num_predict=num_predict, timeout=timeout,
    )
    mem.commit(
        json.dumps(seed_features, sort_keys=True), best_score,
        {"generation": -1, "mechanism": "seed", "reason_code": "SEED"},
    )

    population: list[tuple[float, dict]] = [(best_score, seed_features)]
    history: list[dict] = [
        {
            "generation": -1,
            "candidate_key": prompt_space.feature_key(seed_features),
            "features": seed_features,
            "parent_key": None,
            "mechanism": "seed",
            "reason_code": "SEED",
            "n_features_changed": 0,
            "score": round(best_score, 4),
            "accepted": True,
            "is_cold_start": False,
            "is_forced_large": False,
            "best_so_far": round(best_score, 4),
            "ts": time.time(),
        }
    ]

    best_so_far = best_score
    since_improvement = 0
    accepted_total = 1
    t0 = time.time()

    for gen in range(generations):
        is_cold = (gen + 1) % cold_start_every == 0
        is_forced_large = since_improvement >= plateau_window

        if is_cold:
            child = mutate_random(rng=rng)
            mechanism = "cold_start"
            parent_key = None
            parent = dict(prompt_space.CANONICAL)
        else:
            # tournament over a diverse top-K population
            k = min(population_size, len(population))
            contenders = rng.sample(population, k) if k else [(best_score, seed_features)]
            parent = max(contenders, key=lambda t: t[0])[1]
            parent_key = prompt_space.feature_key(parent)
            if is_forced_large:
                n_changes = 3
                mechanism = "plateau_forced_large"
            else:
                n_changes = rng.choice([1, 1, 1, 2])
                mechanism = "small" if n_changes == 1 else "medium"
            child = mutate(parent, n_changes=n_changes, rng=rng)

        reason = assign_reason_code(parent, child)

        # A structurally identical prompt is a no-op; record it as a discard
        # WITHOUT spending a model call. This is the only pre-evaluation discard.
        if reason == "NO_CHANGE" or any(
            h["candidate_key"] == prompt_space.feature_key(child) and h["score"] is not None
            for h in history
        ):
            row = {
                "generation": gen,
                "candidate_key": prompt_space.feature_key(child),
                "features": child,
                "parent_key": parent_key,
                "mechanism": mechanism,
                "reason_code": "NO_CHANGE",
                "n_features_changed": 0,
                "score": None,
                "accepted": False,
                "is_cold_start": is_cold,
                "is_forced_large": is_forced_large,
                "best_so_far": round(best_so_far, 4),
                "evaluated": False,
                "ts": time.time(),
            }
            history.append(row)
            continue

        score, per_model = evaluate(
            child, probe, model_keys,
            temperature=d["temperature"], seed=seed, num_predict=num_predict, timeout=timeout,
        )
        accepted = score > best_so_far

        n_changed = sum(1 for k in prompt_space.FEATURES if child[k] != parent[k])
        row = {
            "generation": gen,
            "candidate_key": prompt_space.feature_key(child),
            "features": child,
            "parent_key": parent_key,
            "mechanism": mechanism,
            "reason_code": reason,
            "n_features_changed": n_changed,
            "score": round(score, 4),
            "per_model": per_model,
            "accepted": bool(accepted),
            "is_cold_start": is_cold,
            "is_forced_large": is_forced_large,
            "best_so_far": round(max(best_so_far, score), 4),
            "evaluated": True,
            "ts": time.time(),
        }

        if accepted:
            best_so_far = score
            since_improvement = 0
            accepted_total += 1
            mem.commit(
                json.dumps(child, sort_keys=True), score,
                {
                    "generation": gen,
                    "mechanism": mechanism,
                    "reason_code": reason,
                    "parent_key": parent_key,
                },
            )
            population.append((score, child))
            population.sort(key=lambda t: t[0], reverse=True)
            population = population[:population_size]
        else:
            # The brief's fallback mechanism: a reject is a reset, and the reset is
            # recorded as a *sampling point* of the rejected region. FLAW 8: the
            # attempt's diff + stated reason are persisted to
            # runs/<run_id>/trajectory/ *before* the reset erases its code, so the
            # candidate is staged into the tree first (it exists only in memory
            # until then, and an unstaged candidate would diff as empty).
            reject_meta = {
                "generation": gen,
                "mechanism": mechanism,
                "reason_code": reason,
                "candidate_key": prompt_space.feature_key(child),
                "parent_key": parent_key,
                "score": round(score, 4),
                "best_so_far": round(best_so_far, 4),
            }
            mem.write_candidate(json.dumps(child, sort_keys=True), score, reject_meta)
            mem.rollback(
                reason=(
                    f"rejected: score {score:.4f} did not beat best_so_far "
                    f"{best_so_far:.4f} (reason_code={reason})"
                ),
                meta=reject_meta,
            )
            since_improvement += 1

        history.append(row)
        print(
            f"  gen {gen:>3} {mechanism:<20} {reason:<24} score={score:.3f} "
            f"best={best_so_far:.3f} {'ACCEPT' if accepted else 'reset'}",
            flush=True,
        )

    # --- persist both streams -------------------------------------------------
    with landscape_path.open("w", encoding="utf-8") as fh:
        for row in history:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    evaluated = [h for h in history if h.get("evaluated")]
    fallbacks = [h for h in evaluated if not h["accepted"]]

    hillclimb = {
        "run_id": run_id,
        "generations": generations,
        "items": len(probe),
        "split": split,
        "search_models": model_keys,
        "seed_score": round(best_score, 4),
        "best_score": round(best_so_far, 4),
        "gain_over_seed": round(best_so_far - best_score, 4),
        "accepted_total": accepted_total,
        "evaluated_total": len(evaluated),
        "fallback_total": len(fallbacks),
        "curve": [
            {"generation": h["generation"], "best_so_far": h["best_so_far"]} for h in history
        ],
        "accepted": [
            {
                "generation": h["generation"],
                "score": h["score"],
                "mechanism": h["mechanism"],
                "features": h["features"],
            }
            for h in evaluated
            if h["accepted"]
        ],
        "wall_seconds": round(time.time() - t0, 1),
        "mechanism_counts": {
            m: sum(1 for h in evaluated if h["mechanism"] == m)
            for m in {h["mechanism"] for h in evaluated}
        },
        "reason_code_counts": {
            r: sum(1 for h in fallbacks if h["reason_code"] == r) for r in REASON_CODES
        },
        # Anti-spin (iterative-pipeline-anti-spin): a loop that accepted nothing is
        # loud, not silently "successful".
        "stalled": accepted_total <= 1,
    }
    (run_dir / "hillclimb.json").write_text(
        json.dumps(hillclimb, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    best = mem.best_commit()
    if best:
        mem.export_prompt(run_dir / "best_prompt.txt", sha=best["sha"])
        hillclimb["best_commit"] = best

    if hillclimb["stalled"]:
        print("\nIDLE SEARCH: no candidate improved on the seed prompt.")
        print("  remediation: raise generations, widen the mutation set, or add a")
        print("  benchmark/model on which the seed prompt is not already at ceiling.")

    return hillclimb
