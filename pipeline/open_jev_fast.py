"""open_jev_fast.py -- the prior-art tier: what to BORROW, from AnyJev-class open resources.

WHAT THIS TIER IS FOR

The other two tiers of the pipeline are an execution skeleton (the loop) and its first
generation (AgentJev-0.6B). Neither of them can tell the loop what it has NOT yet tried. This tier
answers a different question, and it is the question that decides where the six weeks go:

    an open resource (AnyJev and its class) contains some idea, some calibration routine, some
    scoring trick. Is it real, is it applicable to a NON-GENERATIVE scoring model, and is it worth
    the cost of a cycle here?

The failure mode this exists to prevent is specific and expensive. An autonomous loop with a mutation
generator will happily invent forty variants of the same idea, and will also happily adopt a clever
trick from a neighbouring field that cannot work on this architecture and burn a day proving it. A
prior-art tier turns both into a filter that runs BEFORE a cycle is spent.

THE THREE VERDICTS, AND WHY THEY ARE SEPARATE

    ADOPT     applicable, cheap, and untried here -> becomes a concrete proposal
    NOTE      applicable but already covered, or too expensive to be worth a cycle now
    REJECT    not applicable to a non-generative scoring model -> never retried; goes to FAILURES.md

REJECT is the verdict that matters most, because it is the one that must be durable. A rejected idea
that reappears as a "new" mutation in hour 40 has cost the run a day and taught it nothing, so
rejections are written to the log with their REASON and are consulted on every later selection.

THE APPLICABILITY TEST IS THE INTERESTING PART

Most of what circulates in the open resources for this task family assumes a model that GENERATES
tokens: prompt templates, few-shot exemplars, chain-of-thought scaffolding, output parsing. AgentJev
has no LM head. It emits one scalar per candidate and a distribution over a supplied option set. So
the first filter is not "is this a good idea" but "does this idea have anything to act on" -- and the
overwhelming majority of prompt-space ideas fail it, immediately and cheaply, before any GPU time.

That is not a dismissal of the prior art. It is a statement about which part of it transfers: the
CALIBRATION and SCORING-PIPELINE half of this literature does apply in full, and that half is exactly
the outer level the objective ranks as P0. This tier exists to make that split explicit and reusable
rather than rediscovered per candidate.

THE INTERFACE IS A CARD, NOT A SCRAPER

Cards are plain JSON. Nothing here fetches from the network, by Rule 2 -- zero external API, and a
tier whose job is to read other people's work is exactly where that rule would otherwise be quietly
broken. Cards are written by hand or by an offline batch process, and every card must carry its
source, because Rule 4's comparison table requires that a borrowed method be attributable.

Run: python pipeline/open_jev_fast.py --list        # show the current ledger
    python pipeline/open_jev_fast.py --card X.json  # adjudicate one card
"""
from __future__ import annotations

# TERMINOLOGY: the term "prompt-space" below names a retired research object of this project.
# Do not work on it. It is referenced in exactly one place -- the NON_TRANSFERABLE table --
# purely to mark that family of methods as inapplicable to a non-generative scoring model,
# which emits one scalar per candidate and has no LM head. It is not an approach this project
# uses, is evaluating, or plans to evaluate.




import json
import pathlib
import time
from dataclasses import dataclass, field, asdict

LEDGER = pathlib.Path(__file__).with_name("idea_ledger.jsonl")

ADOPT, NOTE, REJECT = "adopt", "note", "reject"

# The transfer test, in one place. A method needs at least one of these surfaces to be adoptable.
TRANSFERABLE_SURFACES = {
    "calibration": "acts on the probability distribution; applicable in full",
    "scoring_pipeline": "acts on how candidates are encoded/scored; applicable in full",
    "inference_backend": "batching, graphs, precision; applicable in full",
    "decision_rule": "argmax/threshold logic over the option set; applicable",
    "probe": "a small learned map on the candidate features; applicable, cheap",
    "adapter": "a low-rank update to the backbone; applicable but expensive (inner level)",
}
NON_TRANSFERABLE = {
    "prompt_template": "no LM head: there is no prompt to template",
    "few_shot": "no token generation: exemplars cannot be placed in a context",
    "chain_of_thought": "no token generation: there is no reasoning trace to elicit",
    "output_parser": "output is already a fixed-size scalar per candidate",
    "reward_model": "the model IS the scorer; there is no separate reward to model",
    "self_refine": "requires generating a revision; the model cannot emit one",
    "tool_use": "requires the model to emit a tool call; it emits a scalar",
    "multi_agent_debate": "requires multiple generative passes with distinct roles",
}


@dataclass
class IdeaCard:
    """One borrowed idea, with everything needed to judge it and to attribute it later."""

    id: str
    title: str
    source: str                       # repo, paper, commit -- required by Rule 4
    surfaces: list[str]                # which transfer surfaces it touches
    claimed_effect: str = ""           # what the source claims, in ITS terms
    estimated_cost_s: float = 0.0      # our cost to try here, not theirs
    axis: str = "calibration"
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


@dataclass
class Verdict:
    idea_id: str
    verdict: str
    reason: str
    route: str = ""                    # the pipeline route this would run as, when ADOPT
    surfaces_transferable: list[str] = field(default_factory=list)
    surfaces_blocked: list[str] = field(default_factory=list)
    ts: str = ""


def adjudicate(card: IdeaCard) -> Verdict:
    """Decide ADOPT / NOTE / REJECT. Pure function of the card, so it is testable and replayable.

    The order of the tests matters. Transferability is checked FIRST because it is free and
    disqualifying: a prompt-space idea on a model with no LM head is not a bad idea, it is an
    inapplicable one, and spending a cycle discovering that is pure waste. Only among transferable
    ideas does cost decide, and only among affordable ones does novelty matter.
    """
    transfer = [s for s in card.surfaces if s in TRANSFERABLE_SURFACES]
    blocked = [s for s in card.surfaces if s in NON_TRANSFERABLE]
    unknown = [s for s in card.surfaces
               if s not in TRANSFERABLE_SURFACES and s not in NON_TRANSFERABLE]

    if not transfer and blocked:
        return Verdict(card.id, REJECT,
                       f"touches only non-transferable surfaces ({', '.join(blocked)}). AgentJev emits "
                       f"one scalar per candidate and has no LM head, so there is nothing here for the "
                       f"method to act on. Rejected permanently: do not re-propose.",
                       surfaces_blocked=blocked, ts=time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    if blocked and transfer:
        # Mixed. Take the transferable half and name the half being dropped, because the dropped half
        # usually contains the source's headline claim -- so the ledger has to say which part of the
        # borrowed result we are NOT reproducing.
        return Verdict(card.id, ADOPT,
                       f"transferable via {', '.join(transfer)}; the {', '.join(blocked)} half does not "
                       f"apply here and is not being reproduced, so the source's headline number is "
                       f"not the number to expect.",
                       route=_route_for(card, transfer),
                       surfaces_transferable=transfer, surfaces_blocked=blocked,
                       ts=time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    if unknown:
        return Verdict(card.id, NOTE,
                       f"surfaces not in the transfer table: {', '.join(unknown)}. Classify before "
                       f"spending a cycle -- an unknown surface is not a rejected one.",
                       surfaces_transferable=transfer, ts=time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    if card.estimated_cost_s > 60.0 and card.axis == "calibration":
        return Verdict(card.id, NOTE,
                       f"applicable via {', '.join(transfer)} but costs {card.estimated_cost_s:.0f}s. "
                       f"A calibration idea that costs a minute is usually a parameter mutation in "
                       f"disguise, and belongs on the inner level behind the gate.",
                       route=_route_for(card, transfer), surfaces_transferable=transfer,
                       ts=time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    return Verdict(card.id, ADOPT,
                   f"applicable via {', '.join(transfer)} at ~{card.estimated_cost_s:.2f}s. Eligible for "
                   f"the outer level.",
                   route=_route_for(card, transfer), surfaces_transferable=transfer,
                   ts=time.strftime("%Y-%m-%dT%H:%M:%SZ"))


def _route_for(card: IdeaCard, transfer: list[str]) -> str:
    """Map an adopted idea onto an existing pipeline route. Never invents one.

    An adopted idea that cannot be expressed as a route is a NOTE, not an ADOPT, because the loop can
    only execute routes it knows how to evaluate. Returning "" here makes the caller demote it.
    """
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from pipeline.routes import ALL_ROUTES
    if card.axis == "calibration" and "calibration" in transfer:
        for name in ("cal.isotonic_top1", "cal.platt_per_type", "cal.vector_temperature",
                     "cal.scalar_temperature"):
            if name in ALL_ROUTES:
                return name
    if card.axis == "intelligence" and ("probe" in transfer or "adapter" in transfer):
        return "head.probe_refit"
    if card.axis == "speed":
        for name in ("infer.batch_policy", "infer.cuda_graphs", "infer.precision_policy"):
            if name in ALL_ROUTES:
                return name
    return ""


def append_verdict(v: Verdict, path: pathlib.Path = LEDGER) -> None:
    """Append-only, because a rejection that gets overwritten is a rejection that will be re-proposed."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(v), ensure_ascii=False, sort_keys=True) + "\n")


def ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def rejected_ids() -> set[str]:
    """Every idea already refused. Consulted on EVERY selection, so a refusal is durable."""
    return {r["idea_id"] for r in ledger() if r["verdict"] == REJECT}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the ledger")
    ap.add_argument("--card", help="adjudicate one card JSON file")
    ap.add_argument("--seed-ledger", action="store_true",
                    help="write the example cards below, to show the shape of a well-formed card")
    a = ap.parse_args()

    if a.seed_ledger:
        examples = [
            IdeaCard("anyjev-vec-temp", "Vector temperature scaling",
                     "AnyJev (open-source calibration suite)",
                     ["calibration"], "vector-scaled temperature, reported ECE gains",
                     0.02, "calibration",
                     "The transferable half of a method whose headline framing is generative."),
            IdeaCard("anyjev-cot-verify", "Chain-of-thought self-verification",
                     "AnyJev (open-source)", ["chain_of_thought", "output_parser"],
                     "self-verification via generated reasoning traces", 60.0, "intelligence",
                     "Included to show what a REJECT looks like and why it is permanent."),
            IdeaCard("anyjev-platt", "Platt scaling on the top-1 margin",
                     "AnyJev (open-source)", ["calibration", "decision_rule"],
                     "per-type logistic recalibration", 0.01, "calibration"),
        ]
        for c in examples:
            v = adjudicate(c)
            append_verdict(v)
            print(f"  {c.id:22s} -> {v.verdict.upper():6s} {v.reason[:96]}")
        print(f"\nledger: {LEDGER}")
        return 0

    if a.card:
        card = IdeaCard(**json.loads(pathlib.Path(a.card).read_text(encoding="utf-8")))
        v = adjudicate(card)
        append_verdict(v)
        print(json.dumps(asdict(v), indent=2, ensure_ascii=False))
        return 0

    rows = ledger()
    if not rows:
        print("ledger is empty. Run with --seed-ledger for worked examples.")
        print("\nTransferable surfaces (act on this architecture):")
        for k, v in TRANSFERABLE_SURFACES.items():
            print(f"  [ok]      {k:20s} {v}")
        print("\nNon-transferable surfaces (require a generative model):")
        for k, v in NON_TRANSFERABLE.items():
            print(f"  [blocked] {k:20s} {v}")
        return 0

    print(f"ledger: {len(rows)} verdicts, {len({r['idea_id'] for r in rows})} distinct ideas")
    for r in rows:
        print(f"  {r['idea_id']:22s} {r['verdict']:6s} {r.get('route') or '-':24s} {r['reason'][:70]}")
    print(f"\npermanently rejected (never re-propose): {sorted(rejected_ids()) or 'none yet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
