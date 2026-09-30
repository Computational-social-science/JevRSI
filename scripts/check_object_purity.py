#!/usr/bin/env python3
"""check_object_purity.py -- assert the ABSENCE of retired research objects from the live tree.

Why this exists
---------------
This repository has hosted four research objects in sequence:

  1. the NHB LLM-mistranslation workspace
  2. prompt-multiplier / HPFE / OASP / kappa      (optimise the *prompt*)
  3. Gate B / val_bpb / recursive collapse        (nanochat, data policy)
  4. (wrong turn) Zarankiewicz / cap-set selection

The current object is JevRSI (project code; formerly *Jevolution*, a name now retired): a 24/7 autoresearch loop over the *training code* of
`AgentJev-0.6B`, a non-generative typed-decision model. See CURRENT_OBJECT.md.

Each earlier object left artifacts behind, and reading them silently re-targets the work. A validator
that only asserts the *presence* of correct values cannot catch this -- a silent revert passes cleanly.
This script therefore asserts **absence**:

  CHECK A (fail) -- no retired artifact may exist in the live tree.
  CHECK B (fail) -- no live file may carry an *actionable* retired instruction
                    (a command, module path, or identifier that would run the old object).
  CHECK C (fail) -- a live document may name a retired *concept* only if that same file also
                    declares the retirement; naming it as if live is a fail.
  CHECK F (fail) -- a document retired by the 2026-09-29 object cleanup must not reappear in the
                    live tree. This is the guard for the cleanup itself: retiring a document by
                    MOVING it only works if something asserts it stayed moved, because a later
                    session that finds the file useful will happily copy it back.

Exit code 0 = pure, 1 = interference detected.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------------------------
# CHECK A -- retired artifacts that must NOT be present in the live tree.
# Paths are relative to the repo root. Globs allowed.
# ---------------------------------------------------------------------------------------------
FORBIDDEN_PATHS = [
    # object 2: prompt-multiplier / HPFE / OASP
    "oasp", "RESEARCH_PROTOCOL.md", "docs/research_program.md", "configs",
    "benchmarks", "prompts", "runs", "analysis", "tests", "logs", "formal", "outputs",
    "scripts/power_probe.py", "scripts/run_landscape.py", "scripts/run_noise_floor.py",
    "scripts/screen_models.py", "scripts/nightly_campaign.sh",
    # object 3: Gate B / nanochat
    "scripts/session_runner.py", "scripts/fs_harness.py", "scripts/fs_policy.py",
    "scripts/fs_build_pools.py", "scripts/run_arm.sh", "scripts/build_program_variants.py",
    "scripts/figures", "configs/program_variants", "tests/test_session_runner.py",
    # wrong turn: Zarankiewicz
    "scripts/zarankiewicz.py", "scripts/bench_certificate_evaluators.py",
    # stale duplicates of canonical docs
    "docs/research_program_v4_prompt_multiplier.md",
]

# ---------------------------------------------------------------------------------------------
# CHECK B -- actionable retired instructions: a COMMAND or IMPORT that would actually run the
# retired object. Bare path mentions are NOT violations -- naming the trap is the point, and the
# directory's absence is already covered by CHECK A. Patterns therefore require an execution or
# import context.
# ---------------------------------------------------------------------------------------------
FORBIDDEN_INSTRUCTIONS = [
    r"python[23]?\s+(?:-m\s+)?oasp\b",              # python -m oasp ... / python oasp ...
    r"from\s+oasp[\s.]",                            # importing the old package
    r"^\s*import\s+oasp\b",
    r"oasp\.(runner|analyze|stats|landscape|evaluator|config|figures)",
    r"python[23]?\s+\S*\b(screen_models|run_noise_floor|run_landscape|session_runner|"
    r"fs_harness|fs_policy|fs_build_pools|zarankiewicz|bench_certificate_evaluators)\.py",
    r"(?:bash|sh|\./)\s*\S*run_arm\.sh",
    r"--items-dev\b|--num-predict\b|--generations\b|--items-dev\b",
]

# Files exempt from CHECK B (they define the patterns, or are historical records that must quote
# the old commands verbatim to be intelligible).
INSTRUCTION_SCAN_SKIP = {
    "scripts/check_object_purity.py",
    "docs/incident_2026-09-25_tree_loss.md",
}

# ---------------------------------------------------------------------------------------------
# CHECK C -- retired concepts. Allowed in a file ONLY if that file declares the retirement.
# ---------------------------------------------------------------------------------------------
RETIRED_CONCEPTS = {
    "HPFE": r"HPFE",
    "OASP": r"\bOASP\b",
    "kappa": r"\bκ\b|\bkappa\b",
    "prompt-multiplier": r"prompt[- _]multiplier|PromptMultiplier",
    "prompt-space": r"prompt[- _]space",
    "landing-form": r"landing form|落地形态",
    "control-variable": r"control variable|控制变量",
    "static-template": r"static template|static prompt",
    "gate-B": r"Gate[- ]?B|self-training collapse",
    "MentalBench": r"MentalBench|MentalHealthBench",
    # ---- object: the laya-multilingual 322M substrate --------------------------------------------
    # The largest surviving contamination, found 2026-09-30. It was never in this list, which is why
    # 21 live files still referenced it after the earlier purge. The damage was not cosmetic: its
    # Floor B (0.40 pp, 5 seeds, full-parameter) had been carried into INSTRUMENT_CALIBRATION.json
    # and thence into tau_dev, so the pre-registered accept threshold of the LIVE seed was a function
    # of a measurement on a DIFFERENT MODEL -- 322M vs 596M, full-parameter vs QLoRA, replicate
    # accuracies ~0.53 vs a dev baseline of 0.7955. docs/prereg_floor_b_lora_2026-09-29.md had
    # already forbidden exactly this ("Do not inherit it silently"), and the threshold was armed
    # anyway. That file is now revoked and adaptation/gate_tau.py refuses to arm it.
    #
    # Enforcement is on the SUBSTRATE NAME, not on the word "Floor B". Floor A is a property of the
    # benchmark and stays inheritable; Floor B is a property of a training path, and every training
    # path in this project is the live seed's.
    "laya-substrate": r"laya[-_ ]?multilingual|laya[-_ ]?typed[-_ ]?decisions"
                       r"|\blaya\b|convaiinnovolutions/laya|evaluate_laya|train_laya_arm",
    # The project's FORMER name. Renamed to JevRSI on 2026-09-27 (the brief's term; the user retired
    # it). Treating it as live is drift, so naming it now requires a retirement marker in the file.
    "Jevolution": r"\bJevolution\b|\bJEVO\b|\bJevo\b",
    # ---- object 1: the NHB LLM-mistranslation workspace -----------------------------------------
    # The docstring lists this as retired object #1, but it was never enforced: none of its
    # vocabulary appeared in RETIRED_CONCEPTS, so a stray FDLH / acronym-expansion / self-referential
    # mistranslation reference would silently re-target the work at a programme that was falsified
    # (its MLE objective was audited as tautological) -- with the guard reporting PASS. This is the
    # single largest blind spot in the guard, and it is the one the operator flagged.
    "NHB-mistranslation": (
        r"\bNHB\b|mistranslat|Master of Laws|\bFDLH\b|Frequency-Dependent"
        r"|self-referential|\u81ea\u6211\u8bd1|\u7f29\u5199\u8bcd|acronym expansion"
    ),
    # the wrong turn: Zarankiewicz / cap-set *selection* was previously enforced only as the absence
    # of a script path, so the vocabulary could drift back in freely. Enforced as vocabulary now.
    # A prior-art annex must be able to cite the published literature by name, so docs/evidence/ is
    # exempt from this one rule -- see PRIOR_ART_PREFIXES.
    "zarankiewicz": r"[Zz]arankiewicz",
    "cap-set": r"cap[- ]set",
}

# Prior-art annexes survey and cite the EXTERNAL literature, so they must be able to name a
# publication's topic (e.g. "the largest cap set in F_3^8" when citing FunSearch). They remain
# subject to CHECK A/B and to every object-name rule; only CHECK C's concept-naming rule is relaxed
# for them. Nothing in this directory is an instruction to work on the cited topic.
PRIOR_ART_PREFIXES = ("docs/evidence/",)

# ---- CHECK E: the subject repository is where experiments actually RUN ----------------------------
# This guard covers only the JevRSI repository, but a full replicate executes in
# ../agent-jev (weights, configs, CI workflow, results ledger). A retired objective
# surviving THERE can re-target the work while this guard still reports PASS -- exactly the
# contamination path the operator asked to close. Override the location with JEVRSI_SUBJECT; if the
# path is absent the check skips with a note (the subject repo is a clone and may live elsewhere).
SUBJECT_REPO_ENV = "JEVRSI_SUBJECT"
SUBJECT_REPO_DEFAULT = Path(r"../agent-jev")
SUBJECT_MAX_BYTES = 400_000  # skip anything larger: weights/indices are not prose

# Harness concepts: retired as research OBJECTS (the Gate B / recursive-collapse line of work) but
# LIVE as the sanctioned instrument for JevRSI -- `autoresearch-win-rtx` is the substrate the 24/7
# loop runs on and `val_bpb` is the metric it optimises. Naming them is legitimate only in a file
# that also declares the harness by name; a file that treats them as the object of study is still a
# violation. CHECK D asserts the SSOT declares the harness, so this allowance cannot outlive it.
HARNESS_CONCEPTS = {
    "val_bpb": r"\bval_bpb\b|TIME_BUDGET",
    "nanochat": r"\bnanochat\b|TinyStories",
}
# Precise repo identifiers, not the bare word "harness": the marker must be unambiguous.
HARNESS_DECLARATIONS = ["autoresearch-win-rtx", "jsegov/autoresearch", "nanochat-autoresearch"]
SSOT = "CURRENT_OBJECT.md"

# A file "declares the retirement" if it carries one of these markers.
RETIREMENT_MARKERS = [
    "retired", "previous object", "superseded", "archive/", "do not work on",
    "do NOT work on", "no longer the object", "wrong turn", "⛔",
]

LIVE_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".json", ".txt", ".sh", ".toml"}
SKIP_DIRS = {".git", "archive", "__pycache__", ".cache", ".pytest_cache", "literature"}
# Evidence that legitimately quotes retired material from *other authors*, not our objects.
SKIP_FILES = {"docs/incident_2026-09-25_tree_loss.md"}
# The subject repository is a clone of upstream; skip the heavy artifact directories too.
SUBJECT_SKIP_DIRS = SKIP_DIRS | {"checkpoints", "data", "wandb", "runs", "outputs", "logs"}

# The subject repository is a CLONE of another project's history, and that history is not this
# project's object. agent-jev's author ran their own experiments on laya-multilingual before settling
# on the Qwen3-0.6B + LoRA seed; those files are part of commit a965ca8 (2026-09-23) and are the
# upstream record, not contamination introduced here.
#
# The exemption is deliberately narrow. It lists files that are BOTH tracked in the subject repo AND
# present in upstream's history, so a laya file added by THIS project -- untracked, or new -- is
# still caught. An earlier untracked typed_decisions/evaluate_laya_variant.py was exactly that case
# and was removed; had the exemption been a blanket skip of the directory, it would have survived.
#
# protocol.json is the single source for the split definition and was verified case-by-case on day 1.
# Exempting it is not a convenience: deleting it would destroy the provenance of every split.
SUBJECT_UPSTREAM_EXEMPT = {
    "typed_decisions/protocol.json":
        "upstream a965ca8; the split definition's single source, verified case-by-case on day 1",
    "typed_decisions/evaluate_laya.py": "upstream a965ca8; the upstream author's own substrate study",
    "typed_decisions/smoke_laya.py": "upstream a965ca8; the upstream author's own substrate study",
    "typed_decisions/summarize.py": "upstream a965ca8; summariser for the upstream author's study",
    "typed_decisions/comparison.json": "upstream a965ca8; the upstream author's comparison table",
    "typed_decisions/laya_model_metadata.json": "upstream a965ca8; upstream model card",
    "typed_decisions/laya_test_report.json": "upstream a965ca8; upstream test report",
    "typed_decisions/download_manifest.json": "upstream a965ca8; upstream download manifest",
}

# Resolved once. `git ls-files` asks the clone itself whether a path is part of its history, which
# is the definition of "upstream" that cannot be faked by adding a path to the dict above: a file
# created here is untracked, and an untracked file stays untracked until someone commits it.
_SUBJECT_TRACKED: set[str] | None = None


def subject_tracked_files() -> set[str]:
    """Paths the subject repo's git history knows about, or an empty set if git is unavailable.

    Empty on failure is the safe direction: no tracked files means no exemptions, so an unreadable
    subject repo produces violations rather than silent passes.
    """
    global _SUBJECT_TRACKED
    if _SUBJECT_TRACKED is not None:
        return _SUBJECT_TRACKED
    _SUBJECT_TRACKED = set()
    try:
        repo = Path(os.environ.get(SUBJECT_REPO_ENV, str(SUBJECT_REPO_DEFAULT)))
        if not repo.is_dir():
            return _SUBJECT_TRACKED
        res = subprocess.run(["git", "-C", str(repo), "ls-files"],
                             capture_output=True, text=True, timeout=60, errors="replace")
        if res.returncode == 0:
            _SUBJECT_TRACKED = {ln.strip().replace("\\", "/")
                                for ln in res.stdout.splitlines() if ln.strip()}
    except Exception:                                              # noqa: BLE001
        pass
    return _SUBJECT_TRACKED


def subject_is_upstream(rel: str) -> bool:
    """True only if `rel` is a path the agent-jev clone's own git history contains."""
    return rel in subject_tracked_files()


def live_files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        if rel in SKIP_FILES:
            continue
        if p.suffix.lower() in LIVE_SUFFIXES:
            yield rel, p


# ---------------------------------------------------------------------------------------------
# CHECK F -- documents retired by the 2026-09-29 object cleanup.
#
# WHY A SEPARATE CHECK. CHECK A lists retired *artifacts* of the prompt-multiplier era. The 2026-09-29
# cleanup retired a different class: documents that were live, internally consistent, and still
# readable, but which defined the RETIRED OBJECT -- they made AgentJev-0.6B the retraining substrate
# and treated 0.704/0.735 as attainable-accuracy ceilings. Neither check could catch their return,
# because they are not artifacts of an old object; they are current-looking documents about a dead one.
# The general failure this prevents: a session reads a plausible program document, finds it internally
# coherent, and re-targets the work to an objective that was retired two days earlier.
# ---------------------------------------------------------------------------------------------
RETIRED_DOCS_2026_09_29 = [
    # The four documents that defined the SUPERSEDED objective (prompt-multiplier / OASP / kappa axis,
    # and the AgentJev-0.6B-as-retraining-substrate framing).
    #
    # NOT in this list, and this is a correction: `RTX4070_SelfEvolving_Jev_Research_Proposal.md/.pdf`,
    # `docs/audit_RTX4070_proposal_2026-09-29.md`, `CURRENT_OBJECT.md` and `docs/DOCUMENT_MAP.md`. An
    # earlier revision of this guard archived all four on the reasoning that the proposal's three cited
    # figures had no source and that its scale conflicts with the measured tau. That reasoning was a
    # correct AUDIT and an illegitimate DECISION: whether an objective is pursued is the owner's call.
    # The proposal is the LIVE research objective, and the audit document is the evidence for it.
    "docs/JevRSI_program.md",
    "docs/JEVO_program.md",
    "docs/JevRSI_retraining_readiness.md",
    "docs/EVIDENCE_prior_art_and_feasibility.md",
    # Documents written by the agent on the mistaken premise that the proposal was superseded. They are
    # archived, and their reappearance would mean the mistake is being repeated.
    "docs/RESEARCH_GOAL_v2_2026-09-29.md",
    "CURRENT_OBJECT.v2_draft.md",
]

# No content markers are asserted here. An earlier revision flagged the phrases
# "a 24/7 autoresearch retraining loop for AgentJev-0.6B" and "Baseline to beat | 79.25%" inside the
# SSOT as a revert to a retired framing. They are not: the AgentJev-0.6B retraining loop at a 79.25%
# baseline is the LIVE objective, and CURRENT_OBJECT.md says so. Asserting their absence would have
# made the guard forbid the correct work -- which is exactly what it did before this correction.
# The check that matters is structural, not lexical: the retired DOCUMENTS must be absent from the live
# tree, and the SSOT must name the live objective. Both are asserted below.

# The live objective must be named by the SSOT. If this assertion is what stops a future session from
# re-reading a retired objective, then the retirement record and the live objective are anchored to each
# other: the SSOT cannot be rewritten into a dead object without this failing.
# The live objective. Corrected 2026-09-29: this pointed at RESEARCH_GOAL_v2, which the agent had
# derived on the assumption that the RTX4070 proposal was superseded. The owner clarified that the
# proposal IS the objective. The guard now requires the SSOT to name the proposal, so a future session
# cannot quietly redirect the work back to the agent's own reading.
LIVE_OBJECT = "RTX4070_SelfEvolving_Jev_Research_Proposal.md"

# Two files were REPLACED rather than moved during that mistaken cleanup, then restored. Listed so the
# archive-presence rule does not demand an archive copy for a file that is live again.
# Documents the operator ordered DESTROYED on 2026-09-30 rather than archived, on the grounds that a
# polluting document must not exist at all. sha256 copies are held OUTSIDE the repository, at
# a sibling directory outside the repository, deliberately: keeping them in-tree would defeat the order,
# and the point of a deletion here is that a later session cannot read them by accident.
#
# This is the third terminal state for a retired document -- live, archived, deleted. CHECK F
# previously accepted only the first two, so complying with the order would have made the guard fail
# permanently. A guard that cannot be satisfied is a guard that gets disabled.
DELETED_DOCS_2026_09_30 = {
    "docs/JevRSI_program.md",
    "docs/JEVO_program.md",
    "docs/JevRSI_retraining_readiness.md",
    "docs/EVIDENCE_prior_art_and_feasibility.md",
    "docs/RESEARCH_GOAL_v2_2026-09-29.md",
    "CURRENT_OBJECT.v2_draft.md",
    "manuscript.html",
    "docs/audit_RTX4070_proposal_2026-09-29.md",
    "docs/incident_2026-09-25_tree_loss.md",
}

REPLACED_IN_PLACE = {
    "CURRENT_OBJECT.md",
    "docs/DOCUMENT_MAP.md",
}

# The archive that holds them, and the manifest that explains the move. CHECKED PRESENT: a retirement
# that lost its own record is indistinguishable from a deletion.
CLEANUP_ARCHIVE = "archive/superseded_2026-09-29_object_cleanup"
CLEANUP_MANIFEST = f"{CLEANUP_ARCHIVE}/MANIFEST.json"


def check_f() -> list[str]:
    """Assert the 2026-09-29 cleanup held: retired docs absent, archive intact, SSOT live."""
    out: list[str] = []
    for rel in RETIRED_DOCS_2026_09_29:
        p = ROOT / rel
        if p.exists():
            out.append(f"  [F] retired document back in the live tree: {rel}\n"
                       f"      -> it carries the retired objective; the live objective is {LIVE_OBJECT}")
        # It must be in the archive OR recorded as deliberately destroyed. The original rule was
        # "neither live nor archived is indistinguishable from a deletion" -- true, and the operator
        # then ordered that polluting documents must NOT EXIST. Those two requirements are only
        # compatible if DELETED is a third, recorded terminal state, so it is one now: a document
        # listed in DELETED_DOCS_2026_09_30 is expected to be absent, and its absence is the correct
        # outcome rather than a missing record. Anything neither live, nor archived, nor listed here
        # is still a genuine gap.
        archived = (ROOT / CLEANUP_ARCHIVE / Path(rel).name).exists() or \
                   (ROOT / CLEANUP_ARCHIVE / "SSOT" / Path(rel).name).exists()
        if rel in DELETED_DOCS_2026_09_30:
            # Recorded as destroyed. Re-appearance is the failure this must still catch.
            if archived:
                out.append(f"  [F] {rel} was recorded as DELETED on 2026-09-30 but an archived copy "
                           f"exists -- the retirement is inconsistent, pick one")
            continue
        if not archived and rel not in REPLACED_IN_PLACE:
            out.append(f"  [F] {rel} is neither live, archived, nor recorded as deleted -- a "
                       f"retirement that lost its record is indistinguishable from an accident")
    if not (ROOT / CLEANUP_MANIFEST).exists():
        out.append(f"  [F] cleanup manifest missing: {CLEANUP_MANIFEST}")

    # The SSOT must still exist AND must name the live objective. A missing SSOT is the worst state:
    # a session with no authority invents one, and invention is how dead objectives come back.
    ssot = ROOT / SSOT
    if not ssot.exists():
        out.append(f"  [F] no SSOT at {SSOT} -- a session with no authority will invent an objective")
    else:
        text = ssot.read_text(encoding="utf-8", errors="ignore")
        if LIVE_OBJECT not in text:
            out.append(f"  [F] {SSOT} does not name the live objective ({LIVE_OBJECT}) -- a reader "
                       f"cannot tell what the project is working on")
    if not (ROOT / LIVE_OBJECT).exists():
        out.append(f"  [F] live objective missing: {LIVE_OBJECT}")
    return out


def main() -> int:
    failures: list[str] = []
    notes: list[str] = []

    # ---- CHECK A -----------------------------------------------------------------------------
    for rel in FORBIDDEN_PATHS:
        target = ROOT / rel
        if target.exists():
            failures.append(f"[A] retired artifact present in the live tree: {rel}")

    # ---- CHECK B + C -------------------------------------------------------------------------
    allowed_mentions: list[tuple[str, str, int]] = []
    for rel, path in live_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:                                    # pragma: no cover
            notes.append(f"[!] unreadable: {rel} ({exc})")
            continue

        for rx in FORBIDDEN_INSTRUCTIONS:
            if rel in INSTRUCTION_SCAN_SKIP:
                break
            for m in re.finditer(rx, text):
                line = text[: m.start()].count("\n") + 1
                failures.append(f"[B] actionable retired instruction in {rel}:{line} -> {m.group(0)!r}")

        hits = {name: len(re.findall(rx, text)) for name, rx in RETIRED_CONCEPTS.items()}
        hits = {k: v for k, v in hits.items() if v}
        # Two narrow exemptions, both of them upstream material rather than this project's objects:
        #   PRIOR_ART_PREFIXES          -- docs/evidence/, which cites other authors by name
        #   SUBJECT_UPSTREAM_EXEMPT     -- named files inside the agent-jev CLONE, which are that
        #                                    project's own history (commit a965ca8, 2026-09-23)
        # The second is keyed by exact path, not by directory, precisely so that a laya file ADDED
        # to the subject repo by this project is still caught. A blanket directory skip would have
        # let typed_decisions/evaluate_laya_variant.py -- untracked, introduced here -- through, and
        # it was only removed because the exemption was this narrow.
        upstream_ok = (rel in SUBJECT_UPSTREAM_EXEMPT
                       and subject_is_upstream(rel))
        if hits and not rel.startswith(PRIOR_ART_PREFIXES) and not upstream_ok:
            declares = any(mk in text for mk in RETIREMENT_MARKERS)
            total = sum(hits.values())
            if declares:
                allowed_mentions.append((rel, ", ".join(f"{k}x{v}" for k, v in sorted(hits.items())), total))
            else:
                failures.append(
                    f"[C] {rel} names retired concepts as if live ({', '.join(sorted(hits))}) "
                    f"but declares no retirement"
                )

        # ---- CHECK C2: harness concepts (live as substrate, retired as research object) -------
        hhits = {name: len(re.findall(rx, text)) for name, rx in HARNESS_CONCEPTS.items()}
        hhits = {k: v for k, v in hhits.items() if v}
        if hhits:
            declares_harness = any(mk in text for mk in HARNESS_DECLARATIONS)
            total = sum(hhits.values())
            if declares_harness:
                allowed_mentions.append(
                    (rel, ", ".join(f"{k}x{v}" for k, v in sorted(hhits.items())) + " [harness]", total))
            else:
                failures.append(
                    f"[C] {rel} names harness concepts ({', '.join(sorted(hhits))}) without "
                    f"declaring the harness ({HARNESS_DECLARATIONS[0]}): the instrument is live, "
                    f"the retired research object is not"
                )

    # ---- CHECK D: the harness allowance must be anchored in the SSOT --------------------------
    ssot_path = ROOT / SSOT
    if not ssot_path.exists():
        failures.append(f"[D] SSOT {SSOT} missing -- cannot anchor the harness allowance")
    else:
        ssot_text = ssot_path.read_text(encoding="utf-8", errors="ignore")
        if not any(mk in ssot_text for mk in HARNESS_DECLARATIONS):
            failures.append(
                f"[D] {SSOT} does not declare the harness substrate ({HARNESS_DECLARATIONS[0]}), "
                f"so the HARNESS_CONCEPTS allowance is stale"
            )

    # ---- CHECK E: the subject repository carries the same vocabulary ban ----------------------
    subject = Path(os.environ.get(SUBJECT_REPO_ENV, str(SUBJECT_REPO_DEFAULT)))
    subject_scanned = 0
    if not subject.exists():
        notes.append(f"[!] subject repo not found ({subject}); CHECK E skipped -- set "
                     f"{SUBJECT_REPO_ENV} if the clone lives elsewhere")
    else:
        for p in sorted(subject.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in LIVE_SUFFIXES:
                continue
            try:
                rel_parts = p.relative_to(subject).parts
            except ValueError:                                    # pragma: no cover
                continue
            if any(part in SUBJECT_SKIP_DIRS for part in rel_parts):
                continue
            try:
                if p.stat().st_size > SUBJECT_MAX_BYTES:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:                                # pragma: no cover
                notes.append(f"[!] unreadable (subject): {p} ({exc})")
                continue
            subject_scanned += 1
            rel = p.relative_to(subject).as_posix()

            hits = {name: len(re.findall(rx, text)) for name, rx in RETIRED_CONCEPTS.items()}
            hits = {k: v for k, v in hits.items() if v}
            declares = any(mk in text for mk in RETIREMENT_MARKERS)
            # CHECK E is a SEPARATE scan from CHECK B/C over the same vocabulary, so it needs its own
            # copy of the upstream exemption. Adding it in only one place produced the worst kind of
            # result: a guard that reported the subject repo as impure while an identical file in the
            # live tree passed, which reads as two different truths about the same content.
            upstream_ok = subject_is_upstream(rel)
            if hits and not (declares or rel.startswith(PRIOR_ART_PREFIXES) or upstream_ok):
                failures.append(
                    f"[E] subject repo {rel} names retired concepts "
                    f"({', '.join(sorted(hits))}) but declares no retirement"
                )
            if upstream_ok and hits:
                notes.append(f"[ok] subject/{rel} {sum(hits.values())}x "
                             f"({', '.join(sorted(hits))}) -- upstream history, exempt")
            hhits = {name: len(re.findall(rx, text)) for name, rx in HARNESS_CONCEPTS.items()}
            hhits = {k: v for k, v in hhits.items() if v}
            if hhits and not any(mk in text for mk in HARNESS_DECLARATIONS):
                failures.append(
                    f"[E] subject repo {rel} names harness concepts "
                    f"({', '.join(sorted(hhits))}) without declaring the harness"
                )

    # ---- CHECK F: the 2026-09-29 document cleanup held ------------------------------------------
    failures.extend(check_f())

    # ---- report ------------------------------------------------------------------------------
    print("=" * 78)
    print("object-purity check -- asserting ABSENCE of retired research objects")
    print("=" * 78)
    n_live = sum(1 for _ in live_files())
    print(f"live files scanned : {n_live}")
    print(f"forbidden paths    : {len(FORBIDDEN_PATHS)} checked")
    print(f"subject repo       : {subject}  ({subject_scanned} file(s) scanned)")
    print(f"retired concepts   : {len(RETIRED_CONCEPTS)} enforced")
    print(f"cleanup-retired    : {len(RETIRED_DOCS_2026_09_29)} documents asserted absent from the live tree")
    print()

    if allowed_mentions:
        print("Allowed mentions -- file declares the retirement, so naming a retired concept is OK:")
        for rel, what, total in sorted(allowed_mentions, key=lambda r: -r[2]):
            print(f"  [ok] {rel:44s} {total:>3}x  ({what})")
        print()

    if failures:
        print("VIOLATIONS")
        for f in failures:
            print(f"  [FAIL] {f}")
        print(f"\nRESULT: {len(failures)} violation(s) -- the live tree is NOT pure.")
        return 1

    print("RESULT: PASS -- no retired artifact, no actionable retired instruction,")
    print("        and no live document naming a retired concept without declaring it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
