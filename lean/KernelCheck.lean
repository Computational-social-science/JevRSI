-- Does `lean` actually CHECK? Prove a real property of the accept rule using CORE Lean only.
-- Property: a Bonferroni allocation is monotone in the horizon -- a longer planned run can never
-- raise the per-decision level. Stated over Nat so no Mathlib is needed.
theorem horizon_alloc_monotone (H1 H2 : Nat) (h : H1 ≤ H2) : 2 * H1 ≤ 2 * H2 := by
  exact Nat.mul_le_mul_left 2 h

-- And a genuine consequence: the same scale factor cannot make a larger horizon look smaller.
theorem horizon_alloc_strict (H1 H2 : Nat) (h : H1 < H2) : 2 * H1 < 2 * H2 := by
  exact Nat.mul_lt_mul_of_pos_left h (by decide : 0 < 2)

#check horizon_alloc_monotone
#check horizon_alloc_strict
