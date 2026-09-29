/-
  JevRSI.AcceptRule -- the statistical core of the accept rule, stated so the kernel can check it.

  The loop accepts a generation iff the measured improvement exceeds tau, where tau is built from two
  independently measured noise sources and Bonferroni-corrected over the loop's horizon. The results
  below are the properties that make that rule valid rather than decorative.

  A NOTE ON SCOPE, because it is load-bearing. What follows is deliberately elementary. A small claim
  that the kernel actually checks is worth more than an elaborate claim that is not checked, and the
  earlier state of this repository contained Lean files with no theorem in them at all -- only
  `#check` probes. Every theorem in this file is closed by a proof that compiles; claimed-but-unproved
  statements are not permitted to appear here, not even with a comment explaining why they are hard.

  A NOTE ON IMPORTS, because it is a measurement rather than a style choice. The first version of this
  file said `import Mathlib`, which forces the full olean closure of Mathlib -- roughly 5 GB of download
  and hours of build -- to check four statements that touch nothing but ordered-field arithmetic on
  `Rat`. A probe of what Lean 4.34.1's core can do on its own showed the core is NOT sufficient:
  `omega` settles linear goals but cannot establish the squaring step that `combinedVar_mono` needs, and
  `ring_nf` does not exist outside Mathlib. So a dependency is genuinely required. What is not required
  is all of Mathlib, and the imports below name only what is actually used: the order theory and field
  interface for `Rat`, and the `linarith`/`nlinarith` pair. The saving is not cosmetic; it is the
  difference between a check that completes and one that does not.

  `Rat` itself is a kernel type, so the arithmetic these theorems reason about is checked by the
  kernel rather than by a library axiom -- which is the property that makes stating them over `Rat`
  rather than over the reals the honest choice. -/
import Mathlib.Data.Rat.Order
import Mathlib.Data.Rat.Lemmas
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

namespace JevRSI

/-- The Bonferroni-allocated per-decision level for a horizon `H` at family level `alpha`. -/
def perDecisionLevel (alpha : Rat) (H : Nat) : Rat := alpha / (2 * (H : Rat))

/-- **A longer planned horizon can only lower the per-decision level.**

    This is the formal content of a defect the driver actually had: it hard-coded the factor for a
    single decision, so the accept rule behaved as if the horizon were 1 whatever the horizon was.
    Understating the horizon is therefore a validity failure, while overstating it is merely
    conservative -- the two errors are not symmetric, and this theorem is why.

    Stated for `0 < H1` because Lean's field division sends `alpha / 0` to `0`, not to infinity: with
    `H1 = 0` the claimed inequality is genuinely false, so the hypothesis is not cosmetic. -/
theorem perDecisionLevel_antitone (alpha : Rat) (hα : 0 < alpha) :
    ∀ H1 H2 : Nat, 0 < H1 → H1 ≤ H2 →
      perDecisionLevel alpha H2 ≤ perDecisionLevel alpha H1 := by
  intro H1 H2 h1 hle
  unfold perDecisionLevel
  have h1pos : (0 : Rat) < (H1 : Rat) := by exact_mod_cast h1
  have hleR : (H1 : Rat) ≤ (H2 : Rat) := by exact_mod_cast hle
  have hden : 2 * (H1 : Rat) ≤ 2 * (H2 : Rat) := by linarith
  exact div_le_div_of_nonneg_left (le_of_lt hα) (by linarith) hden

/-- **The correction is strict, not merely weak.** A horizon that is genuinely longer allocates a
    strictly smaller per-decision level, so the threshold strictly rises. This is the formal reason
    the horizon must be declared rather than assumed away. -/
theorem perDecisionLevel_strict (alpha : Rat) (hα : 0 < alpha) :
    ∀ H1 H2 : Nat, 0 < H1 → H1 < H2 →
      perDecisionLevel alpha H2 < perDecisionLevel alpha H1 := by
  intro H1 H2 h1 hlt
  unfold perDecisionLevel
  have h1pos : (0 : Rat) < (H1 : Rat) := by exact_mod_cast h1
  have hltR : (H1 : Rat) < (H2 : Rat) := by exact_mod_cast hlt
  have hden : 2 * (H1 : Rat) < 2 * (H2 : Rat) := by linarith
  exact div_lt_div_of_pos_left hα (by linarith) hden

/-- The two-source threshold, in the form the driver uses: the combined standard error is the root of
    the sum of squares of the measurement floor and twice the training floor, and the threshold is its
    standard-error multiple. Stated over `Rat` via the squared identity so no real analysis is needed.

    Consequence proved here: the threshold is non-decreasing in each floor. An underestimated floor
    therefore yields an anti-conservative rule -- the mechanism by which the earlier driver accepted
    more than its stated level. -/
def combinedVar (seMeas seTrain : Rat) : Rat := seMeas ^ 2 + 2 * seTrain ^ 2

theorem combinedVar_mono (seM seT seM' seT' : Rat)
    (hM : seM ≤ seM') (hT : seT ≤ seT') (hM0 : 0 ≤ seM) (hT0 : 0 ≤ seT) :
    combinedVar seM seT ≤ combinedVar seM' seT' := by
  unfold combinedVar
  have hM2 : seM ^ 2 ≤ seM' ^ 2 := by nlinarith
  have hT2 : seT ^ 2 ≤ seT' ^ 2 := by nlinarith
  linarith

/-- A nonnegative multiplier preserves the ordering of the combined variance, so the threshold built
    from it is monotone in the floors as well. This is the property that makes "report both floors"
    a requirement rather than a courtesy: the rule's stated level depends on them. -/
theorem threshold_mono (z seM seT seM' seT' : Rat)
    (hz : 0 ≤ z) (hM : seM ≤ seM') (hT : seT ≤ seT') (hM0 : 0 ≤ seM) (hT0 : 0 ≤ seT) :
    z * combinedVar seM seT ≤ z * combinedVar seM' seT' := by
  have h := combinedVar_mono seM seT seM' seT' hM hT hM0 hT0
  exact mul_le_mul_of_nonneg_left h hz

end JevRSI
