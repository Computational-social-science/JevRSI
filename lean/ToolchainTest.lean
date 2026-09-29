-- Minimal kernel check: does `lean` actually verify, not merely parse?
theorem tau_monotone_in_se {se1 se2 z : Float} (hz : 0 < z) (h : se1 ≤ se2) :
    z * se1 ≤ z * se2 := by
  exact mul_le_mul_of_nonneg_left h (le_of_lt hz)
