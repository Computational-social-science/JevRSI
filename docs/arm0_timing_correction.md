# arm 0 timing: a correct number was discarded and a wrong one adopted

Written 2026-10-03 06:58, nine hours and forty-five minutes into arm 0.

## What is measured

| | |
|---|---|
| process | pid 42308 (the shell wrapper pid the terminal tool returned, 4048, is not it) |
| CPU time | **25,963 s of a 35,100 s wall** = 74% — sampled twice, 19.8 s of CPU per 20 s of wall |
| GPU | **100% utilization, 11.86 GB, stable** across three samples |
| elapsed | 9.75 h |
| artifacts | **none** — `out/` carries an mtime of 2026-09-30 23:30 from the *previous* aborted launch |
| progress unit | **does not exist** |

**The process is working, not wedged.** A wedged job does not hold 100% GPU and 74% CPU for nine
hours with stable memory.

## What is not measured, and that is the finding

**There is no way to know how many of the 1,500 steps are done.** The harness writes nothing during
training, its stdout is empty by design (`fit()` prints nothing; the driver prints only after the
loop), there is no checkpoint until the end, and there is no resume. Every observable — GPU band,
CPU time, memory — is *activity*, which is precisely what the protocol says is not progress:

> "Log volume, CPU load, and process liveness are not progress; they are signs of activity, and
> activity is what a stuck job emits too." — `timing-discipline`

So the honest projection is the third legitimate category and nothing else: **unknown — measuring
now.** And there is no instrument that can measure it. That is a defect of the harness we chose to
run unmodified, and `timing-discipline` already explains why it matters beyond scheduling:

> "If the job cannot be scaled down, that is itself a design defect worth fixing before launch,
> because it also means the job cannot be checkpointed or partially salvaged."

## The error, stated exactly

Two step-time figures were produced for this project. **The one that was discarded was right. The one
that replaced it was wrong.**

| figure | provenance | withdrawn? | verdict |
|---|---|---|---|
| **23 s/step** | no traceable run | **withdrawn** | **closest to the truth: 585 min / 1500 = 23.4 s/step is exactly what has elapsed** |
| 2-3 s/step | a real 1-step run's `meta.json` | adopted | **wrong by ~8×** |

The reasoning that discarded the first was sound as far as it went — a number with no run behind it
is not a measurement, and that rule caught a genuine problem. **But it was applied as if provenance
were sufficient.** It is not. The replacement had provenance and was still wrong, because its
provenance was a *different configuration*: a 1-step run measures kernel autotuning, allocator
warm-up and gradient-checkpoint hook installation, not steady state. `timing-discipline` names this
in advance:

> "**The unfaithful pilot.** The pilot ran with different flags, lower precision, warm caches, or the
> easy slice of the input, so the projection describes a different computation."

And the rule that covers both errors at once:

> "An estimate not derived from a measurement of **this calculation**, **in this configuration**, on
> **this machine**, is a guess."

**The new failure mode, which neither skill names explicitly:** *discarding a correct number for lack
of provenance, and adopting an incorrect one because it had provenance.* Provenance is necessary and
not sufficient; the provenance must match the configuration the projection is about. Removing a
number requires replacing it with a measurement under the same conditions, not with any measurement.

## What is being done about it

1. **arm 0 is left running.** It is healthy, there is no restructure available — the 1,500-step
   schedule is the reference's, and shortening it destroys the only comparison this project exists to
   make — and killing forfeits nine hours for nothing, since there is no resume. This is explicitly
   *not* sunk-cost reasoning: the live comparison is "unknown remaining time of a healthy run"
   against "a restructured route", and no restructured route exists.

2. **A heartbeat is the missing instrument.** The next arm gets a wrapper that appends one row per N
   steps to `<arm>/progress.jsonl` with a timestamp, so a rate can be fit from the job's own log —
   category (a) of the three legitimate sources. Without it, every future arm repeats this analysis
   and every projection remains a guess.

3. **The budget stays absent until it is measured.** The dashboard shows no bar and
   `budget_source: no completed arm yet`. That is correct and stays correct until arm 0 writes its
   `train_seconds`. **No third estimate is being produced.**
