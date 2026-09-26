"""OASP configuration — the single source of truth for paths, models, benchmarks,
and the pre-registered design constants.

Nothing in this module is allowed to be duplicated elsewhere: any script that needs
a model id, a benchmark path, a threshold, or a directory resolves it here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCH_DIR = PROJECT_ROOT / "benchmarks"
PROMPT_DIR = PROJECT_ROOT / "prompts"
RUNS_DIR = PROJECT_ROOT / "runs"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
CONFIG_DIR = PROJECT_ROOT / "configs"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
DOCS_DIR = PROJECT_ROOT / "docs"

for _d in (BENCH_DIR, PROMPT_DIR, RUNS_DIR, ANALYSIS_DIR, CONFIG_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

OLLAMA_BASE = os.environ.get("OASP_OLLAMA_BASE", "http://127.0.0.1:11434")
OPENROUTER_BASE = os.environ.get("OASP_OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------
# Frozen, provenance-pinned. `hf_path` is resolved against HF_ENDPOINT as
#   {HF_ENDPOINT}/datasets/{hf_repo}/resolve/main/{hf_path}


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    hf_repo: str
    hf_path: str
    notes: str = ""
    fmt: str = "arc"  # row parser: "arc" | "mmlu_pro"
    release: str = ""  # publication date — for contamination accounting
    # Nominal chance. ARC is a fixed 4-way (0.25) and MMLU-Pro is nominally 10-way,
    # but MMLU-Pro's option arrays are VARIABLE length (observed 4-10), so the exact
    # per-item chance is 1/len(options) and is averaged at analysis time. Reporting a
    # single 0.10 for MMLU-Pro would understate chance for short-option items.
    chance: float = 0.25

    def url(self) -> str:
        return f"{HF_ENDPOINT}/datasets/{self.hf_repo}/resolve/main/{self.hf_path}"


BENCHMARKS: dict[str, BenchmarkSpec] = {
    # --- legacy / pre-cutoff: retained as a *contamination contrast*, not primary ---
    "arc_easy": BenchmarkSpec(
        key="arc_easy",
        hf_repo="allenai/ai2_arc",
        hf_path="ARC-Easy/test-00000-of-00001.parquet",
        notes="2018 benchmark. Ceiling-bound for 4B+ models (gemma3:4b scored 0.906), "
        "so prompt-variance tests here are uninformative. Retained as a pre-cutoff contrast.",
        release="2018-09",
    ),
    "arc_challenge": BenchmarkSpec(
        key="arc_challenge",
        hf_repo="allenai/ai2_arc",
        hf_path="ARC-Challenge/test-00000-of-00001.parquet",
        notes="2018 benchmark, harder sibling. Still pre-cutoff for every 2024 model; "
        "serves as the contamination contrast condition.",
        release="2018-09",
    ),
    # --- post-cutoff / contamination-controlled ---
    "mmlu_pro": BenchmarkSpec(
        key="mmlu_pro",
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_path="data/test-00000-of-00001.parquet",
        notes="2024-06 (ICML'24). 10-way, harder than MMLU, so 7-9B targets land in the "
        "interior band instead of at ceiling. CAVEAT: built from MMLU (2020) items, so "
        "the *content* may still be in pretraining even though format is new.",
        fmt="mmlu_pro",
        release="2024-06",
        chance=0.10,
    ),
    "supergpqa": BenchmarkSpec(
        key="supergpqa",
        hf_repo="m-a-p/SuperGPQA",
        hf_path="SuperGPQA-all.jsonl",
        notes="2025-02. Expert-written, post-cutoff for every target below — the genuine "
        "contamination control. Expect a lower band; band placement is measured, not assumed.",
        fmt="supergpqa",
        release="2025-02",
        chance=0.25,
    ),
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    key: str
    backend: str  # "ollama" | "openrouter"
    model_id: str
    family: str = ""
    role: str = "target"  # "target" | "optimizer"
    think: bool | None = None  # Ollama thinking control; None = omit the field
    # Checkpoint release (YYYY-MM). Recorded so that a benchmark's release date can be
    # compared against the model's cutoff: a benchmark published *after* the checkpoint
    # is a post-cutoff probe, which is what makes the contamination contrast auditable
    # rather than asserted.
    release: str = ""

    def digest_hint(self) -> str:
        return f"{self.backend}:{self.model_id}"


# Target models. Chosen for: (a) locally servable at pilot scale, (b) span of
# families/sizes, (c) likelihood of clearing the ARC-Easy capability floor.
TARGET_MODELS: dict[str, ModelSpec] = {
    # --- canonical academic baselines (Problem-2 rev: 学术名星模型 for credibility) ---
    # All are the checkpoints that appear as baselines throughout the 2024-2026 open-model
    # literature, all instruction-tuned, all with cutoffs <= 2024-06 so that a 2024-06+
    # benchmark is a genuine post-cutoff probe for them.
    "llama31_8b": ModelSpec(
        "llama31_8b", "ollama", "llama3.1:8b-instruct-q4_K_M", "meta", "target", release="2024-07"
    ),
    "qwen25_7b": ModelSpec(
        "qwen25_7b", "ollama", "qwen2.5:7b-instruct-q4_K_M", "qwen", "target", release="2024-09"
    ),
    "gemma2_9b": ModelSpec(
        "gemma2_9b", "ollama", "gemma2:9b-instruct-q4_K_M", "google", "target", release="2024-06"
    ),
    "mistral_v03_7b": ModelSpec(
        "mistral_v03_7b", "ollama", "mistral:7b-instruct-v0.3-q4_K_M", "mistral", "target",
        release="2024-05",
    ),
    "olmo2_7b": ModelSpec(
        "olmo2_7b", "ollama", "olmo2:7b", "allenai", "target", release="2024-11"
    ),
    # --- previously screened legacy targets ---
    "gemma3_4b": ModelSpec("gemma3_4b", "ollama", "gemma3:4b", "gemma", "target", release="2025-03"),
    "qwen3_4b": ModelSpec(
        "qwen3_4b", "ollama", "qwen3:4b", "qwen", "target", think=False, release="2025-04"
    ),
    "mistral_7b": ModelSpec("mistral_7b", "ollama", "mistral:7b", "mistral", "target", release="2023-09"),
    "qwen2.5_7b": ModelSpec("qwen2.5_7b", "ollama", "qwen2.5:7b", "qwen", "target", release="2024-09"),
    "gemma3_12b": ModelSpec("gemma3_12b", "ollama", "gemma3:12b", "gemma", "target", release="2025-03"),
    "deepseek_r1_7b": ModelSpec(
        "deepseek_r1_7b", "ollama", "deepseek-r1-qwen-7b:Q4_K_M", "deepseek", "target", think=False
    ),
}

# Optimizer models. MUST be disjoint from target model ids (see assert_decoupled).
#
# Local-only by requirement (Problem-2.docx ¶14: 全程在本机 24/7 部署，并全程免费).
# The optimizer writes prompts and must never be scored: it is chosen to be *larger*
# than every target so it is not merely the target's own idiosyncrasy, and it is
# checked disjoint at load time by assert_decoupled().
OPTIMIZER_MODELS: dict[str, ModelSpec] = {
    "local_qwen3_27b": ModelSpec(
        "local_qwen3_27b", "ollama", "qwen3.8:27b", "qwen", "optimizer"
    ),
}

# ---------------------------------------------------------------------------
# Pre-registered design constants (mirrored into configs/oasp.yaml at run start)
# ---------------------------------------------------------------------------

DESIGN = {
    "temperature": 0.0,
    "seed": 42,
    "num_predict": 16,
    "n_variants": 32,
    "n_items_dev": 60,
    "n_items_test": 60,
    "n_items_test_confirmatory": 150,
    "noise_floor_seeds": 5,
    "noise_floor_temp": 0.7,
    "noise_floor_items": 20,
    "determinism_check_items": 20,
    "bootstrap_B": 2000,
    "permutation_B": 2000,
    # Decision rules (§5 of RESEARCH_PROTOCOL.md)
    "h1_min_models": 2,
    "h1_range_over_floor": 2.0,
    "h1_max_tau": 0.95,
    "h3_kruskal_p": 0.05,
    "h3_min_epsilon_sq": 0.06,
    "bh_q": 0.05,
    "alpha": 0.05,
    # Practical-equivalence margin for the "the variants are interchangeable" claim.
    # A difference test can only ever FAIL TO REJECT when the truth is equivalence, so
    # establishing prompt-insensitivity requires the complementary test: declare the
    # variants interchangeable when the spread's UPPER confidence bound falls below this
    # margin. 0.05 = five accuracy points, a spread too small to change any ranking
    # conclusion on a 4-way benchmark. See docs/critical_review_2026-09-26.md FLAW 1 —
    # this margin must be fixed before the confirmatory run, not chosen after seeing it.
    "equivalence_margin": 0.05,
    # Target-selection rule (implementation_plan.md §0). Selected on *baseline*
    # accuracy against the canonical prompt — never on prompt-variance, which would
    # make H1 tautological. Outside this band a model cannot exhibit the effect:
    # ceiling-bound models have no headroom, floor-bound models have no signal.
    "interior_band": (0.35, 0.85),
}

DTYPES = {
    "split": "category",
    "model": "category",
    "variant": "category",
    "item_id": "string",
}


# ---------------------------------------------------------------------------
# Decoupling guard (brief constraint #3)
# ---------------------------------------------------------------------------


class DecouplingViolation(RuntimeError):
    pass


def assert_decoupled(
    target_keys: list[str] | None = None,
    optimizer_keys: list[str] | None = None,
) -> None:
    """Hard-fail if any optimizer is also a target.

    The brief requires that the agent writing prompts must not be the model being
    scored (优化器模型与目标模型解耦), otherwise evolved prompts converge on the
    optimizer's own idiosyncrasies. A guideline would be violated silently; this
    is a load-time assertion on purpose.
    """
    target_keys = target_keys if target_keys is not None else list(TARGET_MODELS)
    optimizer_keys = optimizer_keys if optimizer_keys is not None else list(OPTIMIZER_MODELS)

    target_ids = {TARGET_MODELS[k].model_id for k in target_keys if k in TARGET_MODELS}
    optimizer_ids = {
        OPTIMIZER_MODELS[k].model_id for k in optimizer_keys if k in OPTIMIZER_MODELS
    }

    overlap = target_ids & optimizer_ids
    if overlap:
        raise DecouplingViolation(
            f"optimizer/target decoupling violated: {sorted(overlap)} appear in both roles. "
            "RESEARCH_PROTOCOL.md §2 constraint 3 forbids this."
        )


def resolve_targets(keys: list[str] | None = None) -> list[ModelSpec]:
    keys = keys or list(TARGET_MODELS)
    return [TARGET_MODELS[k] for k in keys]
