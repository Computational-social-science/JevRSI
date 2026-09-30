"""
adaptation/__init__.py -- the project's adaptations of the ecosystem loop, and why each exists.

WHAT THIS PACKAGE IS
    The ecosystem already ships a working 24/7 loop: agent-jev/scripts/autoresearch_agent.py. It has
    the pieces this project would otherwise have to build, and they are better than a fresh version
    would be -- a pre-registered tau with a Bonferroni correction, a dev/shadow two-signal scheme,
    crash classification with a consecutive-crash circuit breaker, and trajectory persistence for
    rejected attempts.

    This package does not reimplement any of that. It adapts the loop to this project's proposal,
    at the seams where the two genuinely disagree. Each adaptation is one file, and each states the
    proposal clause that requires it.

THE SEAM, PRECISELY
    The loop is not wrong; it is parameterised for a different objective. Six differences matter, and
    they are listed in ADAPTATIONS.md with the clause each one answers. The rule followed here is
    that a difference which can be fixed by supplying a value (a threshold, a split name) is fixed by
    supplying a value. A difference which requires a different control flow (a search space with no
    harness routes, a three-stage fidelity ladder, a multi-objective archive) is NOT patched around
    here -- those are recorded as open, because patching them inside the loop would fork the thing
    this project decided to reuse.

WHY NOT FORK IT
    Forking a 479-line battle-tested loop to add features is how the two copies drift, and the drift
    is invisible until a number disagrees. Keeping the loop intact and putting the project's
    requirements in front of it as explicit checks means a requirement that is not yet met shows up
    as a FAILING CHECK rather than as a subtly different metric.
"""
