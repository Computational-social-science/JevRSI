# Theory: HPFE (effect) and OASP (insight)

Two names, two jobs, deliberately **disjoint roots** — the brief warns that mixing
them turns the paper into a word game (¶7: 两者不要混用同一个词根).

| | **HPFE** — Heuristic Prompt Fixation Effect | **OASP** — Optimization-as-Search Principle |
|---|---|---|
| Kind | **Effect**: observable, quantifiable, describes the phenomenon | **Insight**: falsifiable, mechanism-explaining, method-guiding |
| Claim | strategies collapse to one hand-authored static template, evaluated once | LLM optimization should be modelled as a search loop, not a single-point score |
| Unit of analysis | the paper, and the prompt it reports | the method of evaluation |
| Primary metric | M1 spread, M2 IPS | M2 IPS, M7 held-out gain |
| Falsifier | all variants inside the noise floor **and** rankings stable (τ ≈ 1) | search yields no held-out gain over the sampled upper tail |

---

## Why HPFE rather than PBCE

The brief offered **PBCE — Prompt–Benchmark Coupling Effect** first. It is retained
as an alternative label, but HPFE is preferred for the paper's headline because:

1. **PBCE names the wrong mechanism.** "Coupling" says score = f(prompt, benchmark,
   model) — true, but already accepted. The contested claim is that strategies
   **fixate**, i.e. `S ↦ p₀` is a degeneration, not a search. HPFE says that; PBCE
   does not.
2. **PBCE's term is already taken.** "Coupling effect" carries specific meanings in
   physics and in statistics; importing it invites a reviewer objection that has
   nothing to do with the finding.
3. **"Heuristic" is the accurate modifier.** A static prompt is not random. It is a
   *human heuristic choice* — which is exactly why it can be arbitrarily far from
   optimal while looking principled. Calling it a "random solution" (as the brief
   initially does) would be a strawman; calling it a heuristic sample is the
   defensible version, and it is the version our measurement supports.

PBCE remains useful as the *interaction* description when reporting variance
decomposition (M3): η²_interaction *is* a coupling quantity. Use PBCE for the
statistic, HPFE for the phenomenon.

---

## Why OASP must be falsifiable or it is vacuous

The brief itself flags the weakness: 太宽泛, 可能被批评为"所有优化都是搜索"的同义反复.

"Optimization is search" is analytically true — every optimizer searches something.
So OASP cannot be defended as a truth claim. It is defended only as a **claim about
measurable benefit**:

> OASP is supported only if searching prompt space measurably beats the sampled
> upper tail on data the search never saw.

That makes H2 the load-bearing hypothesis, and it makes H2's failure a *result*, not
an inconvenience. If H2 fails while H1 holds, the honest paper is: *static prompts
are arbitrary samples whose spread is large (HPFE supported), but prompt-space
search over surface features does not convert that spread into held-out gains* — a
narrower and more interesting claim than either the brief's original framing or its
negation.

---

## The three-cell consequence of the brief's own framing

The brief fixes the object as the *landing form*, which forces a distinction that
must be reported separately:

| Cell | Situation | What the paper may claim |
|---|---|---|
| **A** | strategy has no static prompt at all | outside scope — not a fixation |
| **B** | strategy has a static prompt, and variants are equivalent | HPFE falsified **for this strategy**; report the null |
| **C** | strategy has a static prompt, and variants differ beyond the floor | HPFE supported; report spread, IPS, and rank flips |

A paper that reports only cell C is selecting on the outcome. The protocol requires
the count of each cell across the surveyed strategies, so the base rate of fixation
is visible.

---

## What would falsify the whole program

- If a large survey finds that reported strategies overwhelmingly *do* search prompt
  space (cell A dominates), HPFE has no population to describe.
- If prompt-induced spread is within the noise floor across models and benchmarks,
  then fixation is benign and the critique is immaterial.
- If search gains vanish under held-out evaluation at scale, OASP reduces to a
  restatement and should be dropped from the title.

Each of these is reportable with the current harness; none of them is currently
excluded by the design.
