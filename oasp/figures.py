"""Publication figures for the OASP program.

House conventions applied here (they are requirements, not preferences):

* Arial first in the font stack; never a mono or serif fallback;
* top and right spines removed, no top/right ticks;
* in-axes titles carry extra vertical padding so they sit clear of the axes;
* x axes are labelled at the first and last tick only;
* golden-ratio panel proportions (0.618 : 0.382);
* figures are written as **new files** — an existing figure is never overwritten,
  so a regenerated figure can always be compared against what was published.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOUBLE_COL_IN = 7.16  # Nature double-column width
GOLD = 0.618
DPI = 400


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _title(ax, text: str) -> None:
    """In-axes title with padding, kept clear of the axes box."""
    ax.set_title(text, pad=10, loc="left")


def _end_labels_only(ax, values) -> None:
    ax.set_xticks([values[0], values[-1]])
    ax.set_xticklabels([str(values[0]), str(values[-1])])


def _new_path(out_dir: Path, stem: str, ext: str = "png") -> Path:
    """Never overwrite: v1, v2, ... so a regenerated figure is comparable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{stem}.{ext}"
    if not p.exists():
        return p
    i = 2
    while (out_dir / f"{stem}_v{i}.{ext}").exists():
        i += 1
    return out_dir / f"{stem}_v{i}.{ext}"


# ---------------------------------------------------------------------------
# Fig 1 — the prompt sample distribution (the headline figure)
# ---------------------------------------------------------------------------


def fig_prompt_sample_distribution(m1: dict, *, out_dir: Path, chance: float, noise_sd: float = 0.0) -> Path:
    models = sorted(m1)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN * GOLD, DOUBLE_COL_IN * GOLD * GOLD))
    rng = np.random.default_rng(0)

    for i, m in enumerate(models):
        r = m1[m]
        lo, hi, mean = r["accuracy_min"], r["accuracy_max"], r["accuracy_mean"]
        y = i
        ax.plot([lo, hi], [y, y], color="#4d4d4d", lw=0.9, zorder=2)
        ax.plot([lo, lo], [y - 0.12, y + 0.12], color="#4d4d4d", lw=0.9)
        ax.plot([hi, hi], [y - 0.12, y + 0.12], color="#4d4d4d", lw=0.9)
        # jittered variant positions are not available here (only summary), so mark
        # the three order statistics the protocol reports.
        ax.scatter([lo, mean, hi], [y, y, y], s=11, zorder=3,
                   color=["#c0392b", "#2c3e50", "#27ae60"], edgecolor="white", linewidth=0.4)
        ax.annotate(f"{r['range']:.2f}", (hi, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color="#c0392b")

    if chance > 0:
        ax.axvline(chance, color="#7f8c8d", lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.annotate("chance", (chance, len(models) - 0.45), xytext=(3, 0),
                    textcoords="offset points", fontsize=6, color="#7f8c8d")
    if noise_sd > 0:
        ax.axvspan(m1[models[0]]["accuracy_mean"] - noise_sd,
                   m1[models[0]]["accuracy_mean"] + noise_sd,
                   color="#f1c40f", alpha=0.18, zorder=0)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlabel("accuracy over the frozen item set")
    _title(ax, "Fig 1  Static-prompt sample\nrange across 32 semantically equivalent prompts")
    return _save(fig, out_dir, "fig1_prompt_sample_distribution")


# ---------------------------------------------------------------------------
# Fig 2 — hill-climb curve + fallback sampling points
# ---------------------------------------------------------------------------


def fig_landscape(hill: dict, landscape_rows: list[dict], *, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN * GOLD, DOUBLE_COL_IN * GOLD * GOLD))
    gens = [h["generation"] for h in landscape_rows]
    best = [h["best_so_far"] for h in landscape_rows]
    ax.step(gens, best, where="post", color="#2c3e50", lw=1.0, label="best so far (survivors)")

    fb_x = [h["generation"] for h in landscape_rows if h.get("evaluated") and not h["accepted"]]
    fb_y = [h["score"] for h in landscape_rows if h.get("evaluated") and not h["accepted"]]
    ax.scatter(fb_x, fb_y, s=8, color="#c0392b", alpha=0.55, linewidths=0,
               label="fallback (discarded)")

    cs = [h["generation"] for h in landscape_rows if h.get("is_cold_start")]
    for g in cs:
        ax.axvline(g, color="#8e44ad", lw=0.5, ls=(0, (2, 3)), alpha=0.7)
    if cs:
        ax.annotate("cold start", (cs[0], ax.get_ylim()[0]), xytext=(2, 6),
                    textcoords="offset points", fontsize=6, color="#8e44ad")

    seed = hill.get("seed_score")
    if seed is not None:
        ax.axhline(seed, color="#7f8c8d", lw=0.7, ls=":")
        ax.annotate("seed prompt (published practice)", (gens[-1], seed), xytext=(-4, 4),
                    textcoords="offset points", ha="right", fontsize=6, color="#7f8c8d")

    _end_labels_only(ax, gens if gens else [0, 1])
    ax.set_xlabel("generation")
    ax.set_ylabel("accuracy (search split)")
    ax.legend(loc="lower right")
    _title(ax, "Fig 2  Prompt-space landscape\nsurvivor curve and fallback sampling points")
    return _save(fig, out_dir, "fig2_landscape")


# ---------------------------------------------------------------------------
# Fig 3 — item-level prompt sensitivity
# ---------------------------------------------------------------------------


def fig_ips(m2: dict, *, out_dir: Path, zero_null: bool = True) -> Path:
    models = sorted(m2)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN * GOLD, DOUBLE_COL_IN * GOLD * GOLD))
    x = np.arange(len(models))
    vals = [m2[m]["ips"] for m in models]
    lo = [m2[m]["ips"] - m2[m]["ips_ci95"][0] for m in models]
    hi = [m2[m]["ips_ci95"][1] - m2[m]["ips"] for m in models]
    ax.bar(x, vals, width=0.55, color="#2c3e50", yerr=[lo, hi], capsize=2.5,
           error_kw={"lw": 0.7, "ecolor": "#4d4d4d"})
    ax.axhline(0, color="#c0392b", lw=0.8)
    if zero_null:
        ax.annotate("exact-invariance null = 0", (0, 0), xytext=(2, 6),
                    textcoords="offset points", fontsize=6, color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("fraction of items whose\ncorrectness flips across prompts")
    _title(ax, "Fig 3  Item-level prompt sensitivity (IPS)")
    return _save(fig, out_dir, "fig3_ips")


# ---------------------------------------------------------------------------
# Fig 4 — discard classes vs score (death-valley test)
# ---------------------------------------------------------------------------


def fig_reason_codes(hill: dict, landscape_rows: list[dict], *, out_dir: Path) -> Path:
    fb = [h for h in landscape_rows if h.get("evaluated") and not h["accepted"]]
    classes: dict[str, list[float]] = {}
    for h in fb:
        classes.setdefault(h["reason_code"], []).append(h["score"])
    if not classes:
        classes = {"(none)": [0.0]}

    order = sorted(classes, key=lambda k: -len(classes[k]))
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN * GOLD, DOUBLE_COL_IN * GOLD * GOLD))
    rng = np.random.default_rng(1)
    for i, c in enumerate(order):
        ys = classes[c]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(ys)), ys, s=9, alpha=0.7,
                   color="#c0392b", linewidths=0)
        ax.plot([i - 0.22, i + 0.22], [float(np.mean(ys))] * 2, color="#2c3e50", lw=1.0)
    seed = hill.get("seed_score")
    if seed is not None:
        ax.axhline(seed, color="#7f8c8d", lw=0.7, ls=":")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([c.replace("_", "\n") for c in order], fontsize=6)
    ax.set_ylabel("accuracy of discarded candidate")
    _title(ax, "Fig 4  Discard classes (death valleys)\nreason codes assigned blind to score")
    return _save(fig, out_dir, "fig4_reason_codes")


def _save(fig, out_dir: Path, stem: str) -> Path:
    p = _new_path(Path(out_dir), stem, "png")
    fig.savefig(p)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def load_landscape(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "landscape.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
