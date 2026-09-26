"""Slot definitions for the program.md variant space (docs/nanochat_only_design.md §6).

WHY THIS FILE EXISTS
--------------------
The manipulated variable must be exactly the *strategy* content of program.md, not its
structure. So rather than hand-writing 13 whole files (which would silently differ in
layout, formatting and the order of bullet points), each variant is defined as a set of
exact string substitutions applied to the stock program.md.

Consequences, all deliberate:
  * v00_stock applies ZERO substitutions and is therefore byte-identical to the shipped
    program.md. The builder asserts this, so "the untranslated baseline" is provable
    rather than claimed.
  * the diff between any two variants is exactly the strategy passages that differ,
    which makes the manipulation auditable.
  * the harness contract (prepare.py read-only, results.tsv schema, the trajectory
    protocol, the fixed 300s budget, NEVER STOP) is untouched by every variant, because
    none of it appears in a slot.

Every alternative is meant to be a choice a competent researcher could defensibly make
*either way*. If an alternative were obviously smarter than the stock text, the spread
across variants would measure prompt quality rather than prompt arbitrariness, which
would change what HPFE means.

The stock strings below are copied verbatim from program.md. The builder FAILS LOUDLY if
any of them does not occur exactly once, so a divergence cannot pass silently.
"""

# --- Slot: how much code complexity is tolerated -----------------------------
SIMPLICITY_STOCK = (
    "**Simplicity criterion**: All else being equal, simpler is better. A small "
    "improvement that adds ugly complexity is not worth it. Conversely, removing "
    "something and getting equal or better results is a great outcome \u2014 that's a "
    "simplification win. When evaluating whether to keep a change, weigh the complexity "
    "cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 "
    "lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from "
    "deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep."
)

SIMPLICITY = {
    "stock": SIMPLICITY_STOCK,
    "strict": (
        "**Simplicity criterion**: Prefer the smallest change that could work. Never keep "
        "an edit that adds more than 10 lines unless it improves val_bpb by at least "
        "0.005. Deleting code for equal or better val_bpb is always a win. If two changes "
        "give the same val_bpb, keep the one with fewer lines; if they give nearly the "
        "same val_bpb, keep the simpler one."
    ),
    "none": (
        "**Simplicity criterion**: Judge changes on val_bpb alone. Code complexity is not "
        "a reason to reject a change \u2014 a lower val_bpb is kept regardless of how much "
        "code it requires. Do not spend effort simplifying the code; spend that effort on "
        "finding a lower val_bpb."
    ),
}

# --- Slot: VRAM posture ------------------------------------------------------
VRAM_STOCK = (
    "**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb "
    "gains, but it should not blow up dramatically."
)

VRAM = {
    "stock": VRAM_STOCK,
    "conservative": (
        "**VRAM** is a hard budget: keep peak VRAM under 6 GB. Prefer changes that reduce "
        "or hold memory flat, and reject any change that raises peak usage above that "
        "ceiling even when val_bpb improves."
    ),
    "aggressive": (
        "**VRAM** is not a real constraint: this GPU has 12 GB and nothing else is using "
        "it. Use as much of it as the machine allows \u2014 a change that buys a lower "
        "val_bpb by expanding memory use is a good trade."
    ),
}

# --- Slot: what to do when progress stalls (rollback policy) -----------------
REWIND_STOCK = (
    "If you feel like you're getting stuck in some way, you can rewind but you should "
    "probably do this very very sparingly (if ever)."
)

REWIND = {
    "stock": REWIND_STOCK,
    "after_three": (
        "If three consecutive experiments fail to lower val_bpb, rewind the branch to the "
        "best-known commit and start again from there, recording why the streak failed. "
        "Do not rewind more than once in every five experiments."
    ),
    "never": (
        "Never rewind. The branch only moves forward: each experiment either advances it "
        "or is discarded, and you continue from wherever you are. Treat a losing streak as "
        "information about which direction is exhausted, not as a reason to go back."
    ),
}

# --- Slot: where new ideas come from when stuck ------------------------------
IDEAS_STOCK = (
    "If you run out of ideas, think harder \u2014 read papers referenced in the code, "
    "re-read the in-scope files for new angles, try combining previous near-misses, try "
    "more radical architectural changes."
)

IDEAS = {
    "stock": IDEAS_STOCK,
    "architecture_first": (
        "If you run out of ideas, work through the architecture systematically: model "
        "depth, then width, then attention pattern, then activation, then normalization. "
        "Sweep one axis at a time with every other setting held fixed, and move to the "
        "next axis only when the current one stops yielding gains."
    ),
    "optimizer_first": (
        "If you run out of ideas, work through the optimization stack systematically: "
        "learning rate, then schedule and warmup, then optimizer hyperparameters, then "
        "weight decay, then batch size and gradient accumulation. Sweep one axis at a "
        "time and move on only when an axis is exhausted."
    ),
    "empirical_only": (
        "If you run out of ideas, ignore the literature and rely only on what you can "
        "measure here: re-read your own results.tsv and trajectory records, and try "
        "variations of the changes that came closest to winning. Trust the log, not the "
        "papers."
    ),
    "hypothesis_first": (
        "If you run out of ideas, stop and write a one-line prediction before every "
        "experiment from now on: what you will change, the direction you expect val_bpb "
        "to move, and roughly how much. Run it, then record whether the prediction held, "
        "and use the misses to choose the next change."
    ),
}

SLOTS = {"SIMPLICITY": SIMPLICITY, "VRAM": VRAM, "REWIND": REWIND, "IDEAS": IDEAS}

# --- The variant set ---------------------------------------------------------
# v01..v10 each change EXACTLY ONE slot, so any difference they show is attributable to
# that dimension alone. v11 and v12 change several slots coherently, to probe interaction.
# "stock" means "keep the shipped text for this slot".
VARIANTS = [
    {"id": "v00", "name": "stock",              "slots": {}},
    {"id": "v01", "name": "strict_simplicity",  "slots": {"SIMPLICITY": "strict"}},
    {"id": "v02", "name": "no_simplicity_penalty", "slots": {"SIMPLICITY": "none"}},
    {"id": "v03", "name": "conservative_vram",  "slots": {"VRAM": "conservative"}},
    {"id": "v04", "name": "aggressive_vram",    "slots": {"VRAM": "aggressive"}},
    {"id": "v05", "name": "rewind_after_three", "slots": {"REWIND": "after_three"}},
    {"id": "v06", "name": "never_rewind",       "slots": {"REWIND": "never"}},
    {"id": "v07", "name": "architecture_first", "slots": {"IDEAS": "architecture_first"}},
    {"id": "v08", "name": "optimizer_first",    "slots": {"IDEAS": "optimizer_first"}},
    {"id": "v09", "name": "empirical_only",     "slots": {"IDEAS": "empirical_only"}},
    {"id": "v10", "name": "hypothesis_first",   "slots": {"IDEAS": "hypothesis_first"}},
    {"id": "v11", "name": "disciplined_bundle", "slots": {
        "SIMPLICITY": "strict", "VRAM": "conservative",
        "REWIND": "after_three", "IDEAS": "architecture_first"}},
    {"id": "v12", "name": "aggressive_bundle",  "slots": {
        "SIMPLICITY": "none", "VRAM": "aggressive", "IDEAS": "optimizer_first"}},
]

# Which slot each single-dimension variant probes, for automatic reporting.
DIMENSION_OF = {
    "v01": "SIMPLICITY", "v02": "SIMPLICITY",
    "v03": "VRAM", "v04": "VRAM",
    "v05": "REWIND", "v06": "REWIND",
    "v07": "IDEAS", "v08": "IDEAS", "v09": "IDEAS", "v10": "IDEAS",
}
