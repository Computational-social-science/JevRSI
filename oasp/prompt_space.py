"""The prompt space and its sampling.

Brief constraint #1 requires that "random solution" be *operationalized*: not one
static prompt, but N semantically-equivalent prompts whose score distribution is
measured. Constraint #6 (our own addition, §7.6) requires that "semantically
equivalent" be **auditable** rather than asserted.

This module therefore enumerates an explicit **surface-feature factorial**. Every
variant is a point in that space, so a reader can see exactly which surface
features were varied and which were held constant. The full product is 5*3*2*3*2
= 180 prompts; a fixed seed draws N=32 of them (always including the canonical
baseline ``p0``), so the sample is reproducible byte-for-byte.

The generator is deliberately **not** an LLM in the primary condition: an LLM
generator would inject its own prior over wordings, confounding "prompt space" with
"this optimizer's preferred region of it" (§7.5). An LLM condition exists as a
secondary check; divergence between the two is reported, never averaged away.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field

from .benchmark import Item

# ---------------------------------------------------------------------------
# The surface-feature space
# ---------------------------------------------------------------------------

FEATURES: dict[str, list] = {
    "verb": ["Answer", "Choose", "Select", "Respond to", "Pick"],
    "format": ["bare", "answer_colon", "paren_letter"],
    "no_explanation": [True, False],
    "opt_label": ["upper", "lower", "paren"],
    "order": ["instr_first", "question_first"],
}

#: The heuristic point that published practice actually reports.
CANONICAL: dict = {
    "verb": "Answer",
    "format": "answer_colon",
    "no_explanation": True,
    "opt_label": "upper",
    "order": "instr_first",
}


@dataclass(frozen=True)
class Variant:
    features: dict
    ordinal: int

    @property
    def vid(self) -> str:
        return feature_key(self.features)

    @property
    def label(self) -> str:
        """Human-readable id: ordinal + a hash of the features."""
        return f"v{self.ordinal:02d}_{feature_key(self.features)[:8]}"

    def as_row(self) -> dict:
        return {"vid": self.vid, "ordinal": self.ordinal, **self.features}


def feature_key(features: dict) -> str:
    parts = [f"{k}={features[k]}" for k in sorted(FEATURES)]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def enumerate_space() -> list[dict]:
    keys = list(FEATURES)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(FEATURES[k] for k in keys))]


def sample_variants(n: int = 32, seed: int = 42) -> list[Variant]:
    """Canonical baseline first, then a deterministic draw from the full space.

    Guarantees the returned set contains ``CANONICAL`` exactly once and is stable
    across runs and machines (``random.Random(seed)`` on a sorted space).
    """
    space = enumerate_space()
    space.sort(key=feature_key)

    keep = [v for v in space if v == CANONICAL]
    if not keep:
        raise AssertionError("canonical baseline is not a point in FEATURES space")
    pool = [v for v in space if v != CANONICAL]

    rng = random.Random(seed)
    rng.shuffle(pool)

    chosen = [CANONICAL] + pool[: max(0, n - 1)]
    return [Variant(features=dict(f), ordinal=i) for i, f in enumerate(chosen)]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_FORMAT_CLAUSE = {
    "bare": "Respond with only the letter of the correct option.",
    "answer_colon": 'Respond in the form "Answer: X", where X is the letter of the correct option.',
    "paren_letter": "Put the letter of the correct option in parentheses, for example (X).",
}

# Each verb value carries its *own* complete clause, so that the verb manipulation
# varies wording while holding the request constant. Writing "Select the question
# below" (the naive composition) would not be a surface variant at all — it would
# change what is asked. Equivalence is therefore constructed, not assumed (§7.6).
_VERB_CLAUSE = {
    "Answer": "Answer the question below.",
    "Choose": "Choose the correct option for the question below.",
    "Select": "Select the correct option for the question below.",
    "Respond to": "Respond to the question below.",
    "Pick": "Pick the correct option for the question below.",
}

_OPT_LABEL_FMT = {
    "upper": "{L}. {T}",
    "lower": "{l}. {T}",
    "paren": "({L}) {T}",
}


def _options_block(item: Item, opt_label: str) -> str:
    lines = []
    for lbl, txt in zip(item.labels, item.texts):
        lines.append(_OPT_LABEL_FMT[opt_label].format(L=lbl.upper(), l=lbl.lower(), T=txt))
    return "\n".join(lines)


def _instruction(v: dict) -> str:
    head = _VERB_CLAUSE[v["verb"]]
    if v["no_explanation"]:
        # Keep the clause order fixed so this is a pure add/remove manipulation.
        return f"{head} {_FORMAT_CLAUSE[v['format']]} Do not explain."
    return f"{head} {_FORMAT_CLAUSE[v['format']]}"


def render(features: dict, item: Item) -> str:
    """Render one variant of the stimulus for one item.

    Only surface features vary. The question text and the option texts are
    byte-identical across all variants for a given item, which is what makes
    "semantically equivalent" auditable.
    """
    instr = _instruction(features)
    qblock = f"Question: {item.question}\n{_options_block(item, features['opt_label'])}"
    if features["order"] == "instr_first":
        return f"{instr}\n\n{qblock}"
    return f"{qblock}\n\n{instr}"


# ---------------------------------------------------------------------------
# LLM-generated condition (secondary, §7.5)
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM_PROMPT = (
    "You write evaluation prompt templates. You never answer the question. "
    "Output only the requested prompt text."
)

GENERATOR_TEMPLATE = """Write a prompt template for a multiple-choice benchmark.

REQUIREMENTS
- It must ask the solver to answer a multiple-choice question by choosing one option letter.
- It must be semantically identical in what it asks for; vary only wording, clause order, and phrasing.
- It must use the literal placeholders {{QUESTION}} and {{OPTIONS}} where the question and the
  option list go. Do not invent example questions or options.
- It must not add reasoning instructions, chain-of-thought requests, or extra constraints.
- Output the template only, with no commentary, no quotes, no markdown fences.

VARIATION FOCUS for this sample: {focus}

Template:"""

FOCUSES = [
    "terse imperative phrasing",
    "polite request phrasing",
    "explicit output-format specification",
    "question-first ordering with the instruction last",
    "formal academic register",
]


def llm_generated_templates(optimizer_spec, *, n: int = 5, temperature: float = 0.9, seed: int = 42):
    """Generate additional templates with an LLM optimizer (secondary condition).

    Returns a list of ``(template_str, reason_or_error)``. Never raises: a failed
    generation is recorded as an error string so the run continues.
    """
    from . import llm

    out: list[tuple[str, str]] = []
    for i in range(n):
        focus = FOCUSES[i % len(FOCUSES)]
        prompt = GENERATOR_TEMPLATE.format(focus=focus)
        comp = llm.chat(
            optimizer_spec,
            prompt,
            temperature=temperature,
            seed=seed + i,
            num_predict=256,
            system=GENERATOR_SYSTEM_PROMPT,
        )
        if not comp.ok:
            out.append(("", f"ERROR: {comp.error}"))
            continue
        tmpl = comp.text.strip().strip("`").strip()
        if tmpl.startswith('"') and tmpl.endswith('"'):
            tmpl = tmpl[1:-1]
        if "{QUESTION}" not in tmpl or "{OPTIONS}" not in tmpl:
            out.append(("", "REJECTED: missing {QUESTION}/{OPTIONS} placeholder"))
            continue
        out.append((tmpl, f"ok:focus={focus}"))
    return out


def render_from_template(template: str, item: Item) -> str:
    qblock = f"Question: {item.question}\n{_options_block(item, 'upper')}"
    return template.replace("{QUESTION}", item.question).replace("{OPTIONS}", qblock)
