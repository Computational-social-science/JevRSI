# BootLoops as the project's foundation and global toolbox

Adopted 2026-10-02 at the owner's instruction. This file records **what is adopted, what is not, and
why** — because the honest answer is that BootLoops is two things and only one of them applies here,
and a project that claims to have adopted both would be lying about half of it.

Source: `https://github.com/BootLoops-ai/bootloops` (MIT, © 2026 Anthropic PBC, by Matthew D.
Schwartz; code written by Claude under his direction). The protocols live in the sibling repository
`BootLoops-ai/skills`. Both are cloned beside each other, as their README requires, at:

```
E:/2026-AI4S/bootloops/bootloops/   ->  the toolkit (49 packages) + house engines
E:/2026-AI4S/bootloops/skills/      ->  the protocols (12 Agent Skills)
```

---

## What BootLoops is

A harness for LLMs doing precision quantitative science. Two halves:

1. **A toolkit** — 49 Python (some Julia) packages of *mathematical physics instruments*: Feynman
   integrals in the polylogarithmic/elliptic/K3/Calabi–Yau classes, Picard–Fuchs operators, Landau
   alphabets, Bayesian evidence integrals, plus applied packages in phylogenetics, population
   genetics, ecology and seismology. Their own engines ship in full source under `upgrades/`.

2. **A protocol set** — the working protocols, as plain-markdown Agent Skills, "the other half",
   which encode what *done* means.

---

## What we adopt, and what we do not

### Adopted in full: the 12 protocols

Installed as global Hermes skills under `bootloops/`, read from the clone rather than from memory:

| protocol | the rule it carries |
|---|---|
| `acceptance-gate` | **A result is done when it has survived a check that could have failed, run by a route that could not have known the answer.** Never merely because the computation finished. |
| `independence-bookkeeping` | Independence is an **auditable property of records**, never an assumption. An oracle that fed a fit may never certify the result — contamination is one-way and permanent. |
| `timing-discipline` | An estimate not derived from a measurement of *this* calculation, in *this* configuration, on *this* machine, **is a guess**. Three legitimate sources; there is no fourth. |
| `planted-truth` | Before real data: prove the pipeline recovers a known planted answer **and** catches a deliberately corrupted input. |
| `reading-contract` | Sources-only discipline for anything that reads documents and asserts facts from them. |
| `constant-recognition` | When no relation is found, **refusal over invention**. |
| `tool-stewardship` | Consult the index before writing; patch in place, never fork; the four-question page written the day of the build. |
| `lit-review` | The papers must be actually read; thoroughness made falsifiable by ledgers and search receipts. |
| `ref-check` | Every author, year, DOI verified against an authoritative source before entering a `.bib`. |
| `referee-sim` | Simulate the strongest objection from every audience the document claims. |
| `prose-lint` | Hype vocabulary, agentless prose, number discipline, honesty failures. |
| `prove-protocol` | Multi-agent protocol for proving or refuting a mathematical statement. |

**These are domain-general.** Nothing in them mentions physics. Every one of them describes a failure
this project has already made or already guards against — see the mapping below.

### Adopted as conventions: the structure, not the content

| BootLoops convention | what we take from it |
|---|---|
| `tools/<pkg>/GUIDE.md` — the **four-question page** | every tool here gets one: what it does, when to reach for it, what its output means, what test its answer must pass. "A tool missing any of the four is not finished, however well it runs." |
| `tools/README.md` — a flat index table | ours is [`../TOOLS.md`](../TOOLS.md). "The unregistered capability": built, tested, documented in its own directory, but no index row — so searching agents conclude it does not exist and rebuild it. |
| **Verification classes** — selftest / partial / smoke / data-gated | adopted verbatim as the battery column in our index. |
| **banked value / value of record** | maps onto our `evidence/` and, more importantly, onto **the release's own `meta.json`** — the single authoritative object when several candidates exist. |
| `tools/` vs `ops/` separation | instruments vs infrastructure, with nothing in one importing from the other. Our `measurement/` is instruments; `scripts/health.py` is `ops`. |

### Not adopted: the 49 toolkit packages

**They do not apply to this project, and pretending otherwise would be the kind of claim this
project has spent the session retracting.** BootLoops' instruments compute Feynman integrals,
algebraic periods of Calabi–Yau varieties, phylogenetic likelihoods and population-genetic
demographics. Our work is:

- **FDLH** — corpus frequency analysis over C4, cross-lingual probing of LLM self-referential
  mistranslation, temporal n-gram gradients.
- **JevRSI** — reproducing a recursive self-improvement loop over a small language model: training
  runs, evaluation harnesses, arms and a curve.

There is no object in our work that a Landau alphabet, a Picard–Fuchs operator or an AMFlow
integration has anything to say about. **The right posture is: a resource we know where to find if a
problem ever does land in their domain, not a dependency of every session.** Concretely — if this
project ever needs a certified high-precision constant or an integer-relation search that cannot be
concluded by ordinary means, `tools/lockpick` and `tools/baller` are there, and `constant-recognition`
says how to use their output honestly.

### Partially adopted: `ops/turnstile`

`turnstile` is admission control for long jobs on a **shared Linux machine** — a priority token plus
RAM and CPU-width ledgers. We built an equivalent this session (`scripts/health.py preflight`) for a
single Windows workstation, independently and before reading theirs. Where theirs is better:

- it **leases** a priority token rather than merely checking, so two jobs cannot both pass a check
  and then collide;
- it keeps **RAM and CPU-width ledgers**, not just a yes/no.

Ours is better adapted to this host (WDDM's 26-process GPU reporting, the measured 11,874 MiB peak).
**Recorded as a known gap, not claimed as covered**: if this project ever runs two arms
concurrently, our preflight's capacity check is not sufficient and turnstile's lease model is the
design to copy.

---

## Why this adoption is not cosmetic: it names failures this project already made

Read against our own record, the protocols are a diagnosis. Each row is a thing that actually
happened here, and the protocol that already had a name for it:

| what happened here | the protocol that names it |
|---|---|
| `23 s/step` written down as a measurement with **no run behind it**, sizing a job at 9.6 h | `timing-discipline`: *"The intuition ETA… The tell is that no log exists from which the number could have been derived. The question that kills it is three words: measured from what?"* |
| the replacement, `2 s/step` from a **1-step run**, low by 2×+ | `timing-discipline`: *"The startup-transient fit"* — and *"The unfaithful pilot… The pilot is the run, scaled down, with nothing else changed"* |
| reading `DEFAULTS` out of a **later revision** and calling it "what ran" | `independence-bookkeeping`: a reference's **birth record** — generator + version — is not decoration |
| the audit's `read_text` comparison reporting **"byte-identical"** after normalising away the difference it measured | `acceptance-gate`: *"The gate that cannot fail"* — the comparator must be shown to fail on a planted mismatch |
| a watcher keyed on one memory threshold reporting **NOT RUNNING** during healthy preparation | `timing-discipline`: *"Log lines mistaken for progress… Liveness is measured on the output class the run exists to produce"* |
| a `freeze_base: true` control used to retract an OOM claim, where the control **removed half the cause** | `acceptance-gate`: *"The gate that cannot fail… every clause of it might be broken and you would see only passes"* |
| keeping the OOM from a contended run while discarding the same run's duration | `independence-bookkeeping`: influence is one-way and permanent; the record must say what touched what |
| `linear_attn_kernel` recorded as a real incomparability when our backbone **has no such layer** | `reading-contract` / `acceptance-gate`: audit the claim, not the resemblance |

**Eight for eight. That is the argument for the adoption, and it is stronger than any claim that the
toolkit is useful here.**

---

## The immediate consequence: arm 0 cannot be monitored, and that is a defect

`timing-discipline` step 6: *"Define real progress before launch. Name the output unit the run exists
to produce and the file or table where it accumulates. Monitoring watches that count. Log volume, CPU
load, and process liveness are not progress; they are signs of activity, and activity is what a stuck
job emits too."*

We did not do this. Applying it now, honestly:

| | |
|---|---|
| output unit the run exists to produce | `checkpoints/meta.json` + `out/arm0.jsonl` |
| where it accumulates | both written **once**, after training and after evaluation respectively |
| intermediate count to watch | **there is none** |

So there is **no progress signal for arm 0 at all** — only completion. Everything the dashboard shows
(GPU band, elapsed time) is activity, which is exactly what a wedged job emits too. The protocol
names the underlying fault precisely: *"If the job cannot be scaled down, that is itself a design
defect worth fixing before launch, because it also means the job cannot be checkpointed or partially
salvaged."*

That is our harness: **no intermediate output and no resume.** Both are properties of the reference
code, which we run unmodified by design. Recorded here as a defect with its consequence stated, and
the fix — for the arms we control, not for the reference's code — is a wrapper that writes a
heartbeat row per N steps, so the next arm has a progress unit and this analysis does not have to be
repeated.

---

## How to keep this current

- The clone is the source. Read the protocols from `E:/2026-AI4S/bootloops/skills/skills/`, never
  from memory.
- Re-installing after an upstream update: copy `skills/*/SKILL.md` to
  `hermes/skills/bootloops/<name>/SKILL.md`. Provenance is in `bootloops/_SOURCE.json`.
- A protocol that turns out to need a local amendment gets one **in the copy**, with the amendment
  marked, and the reason recorded here — not a private rewrite that stops matching upstream.
