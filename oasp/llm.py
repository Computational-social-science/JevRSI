"""Unified chat client for Ollama and OpenRouter.

Deliberately dependency-light: ``requests`` only, so the harness runs on the
system interpreter with no install step.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import requests

from . import config


@dataclass
class Completion:
    text: str
    ok: bool
    error: str = ""
    latency_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class OptionScores:
    """Per-option log-likelihoods for one item — the second point on the scoring axis."""

    logprobs: dict[str, float]
    predicted: str | None
    ok: bool
    error: str = ""
    latency_ms: int = 0
    raw_tokens: int = 0

    def margin(self) -> float | None:
        """Gap between the top two options. Small margins flag fragile items."""
        if len(self.logprobs) < 2:
            return None
        vals = sorted(self.logprobs.values(), reverse=True)
        return vals[0] - vals[1]


class BackendError(RuntimeError):
    pass


def _retry_post(url: str, *, payload: dict, headers: dict, timeout: int, attempts: int = 3) -> dict:
    last: Exception | None = None
    for i in range(attempts):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = BackendError(f"HTTP {r.status_code}: {r.text[:300]}")
        except Exception as e:  # noqa: BLE001 - network layer, retried below
            last = e
        time.sleep(min(2 ** i, 8))
    raise BackendError(str(last))


def chat_ollama(
    model_id: str,
    user_content: str,
    *,
    temperature: float,
    seed: int,
    num_predict: int,
    timeout: int = 120,
    system: str | None = None,
    think: bool | None = None,
) -> Completion:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
            "num_ctx": 2048,
            "top_p": 1.0,
        },
    }
    if think is not None:
        # Ollama's thinking control. Sent only for models known to expose it, so a
        # non-reasoning model never sees an option it might reject.
        payload["think"] = think
    t0 = time.time()
    try:
        data = _retry_post(
            f"{config.OLLAMA_BASE}/api/chat",
            payload=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    except BackendError as e:
        return Completion(text="", ok=False, error=str(e), latency_ms=int((time.time() - t0) * 1000))

    text = (data.get("message") or {}).get("content", "") or ""
    return Completion(
        text=text,
        ok=True,
        latency_ms=int((time.time() - t0) * 1000),
        prompt_tokens=data.get("prompt_eval_count"),
        completion_tokens=data.get("eval_count"),
    )


def score_options_ollama(
    model_id: str,
    stem: str,
    labels: tuple[str, ...] | list[str],
    *,
    timeout: int = 120,
    top_logprobs: int = 20,
    think: bool | None = None,
) -> OptionScores:
    """Score an item by reading the model's per-option token log-probabilities.

    This is the second point on the **scoring axis**. Paper A (arXiv 2608.21382,
    "There Is No Neutral Harness") finds that "the scoring choice, not the option
    order that protocols usually fix, is the load-bearing axis" of harness
    sensitivity. A pipeline that only parses generated text samples a single point
    of that axis, so an FSI measured from text parsing alone cannot be attributed to
    the wording rather than to the read-out.

    ``stem`` must already enumerate the options and end just before the answer token.
    """
    prompt = f"{stem}\nAnswer:"
    payload = {
        "model": model_id,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "keep_alive": "10m",
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "options": {"temperature": 0.0, "num_predict": 1, "num_ctx": 2048, "top_p": 1.0},
    }
    if think is not None:
        payload["think"] = think
    t0 = time.time()
    try:
        data = _retry_post(
            f"{config.OLLAMA_BASE}/api/generate",
            payload=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    except BackendError as e:
        return OptionScores(logprobs={}, predicted=None, ok=False, error=str(e),
                            latency_ms=int((time.time() - t0) * 1000))

    lp = data.get("logprobs") or []
    if not lp:
        return OptionScores(logprobs={}, predicted=None, ok=False,
                            error="backend returned no logprobs",
                            latency_ms=int((time.time() - t0) * 1000))

    # A label can surface under several tokenisations (' B' with a leading space,
    # 'B' bare). Collapse by the stripped form and keep the highest logprob seen,
    # which is the probability the model assigns to that answer surface form.
    want = {str(l).strip().upper() for l in labels}
    best: dict[str, float] = {}
    n_raw = 0
    for entry in lp:
        for alt in (entry.get("top_logprobs") or []):
            n_raw += 1
            tok = str(alt.get("token", "")).strip().upper()
            if tok in want:
                v = float(alt.get("logprob", float("-inf")))
                if tok not in best or v > best[tok]:
                    best[tok] = v

    if not best:
        return OptionScores(logprobs={}, predicted=None, ok=False,
                            error=f"no option token among top_logprobs ({n_raw} seen)",
                            latency_ms=int((time.time() - t0) * 1000), raw_tokens=n_raw)

    predicted = max(best, key=lambda k: best[k])
    return OptionScores(
        logprobs=best,
        predicted=predicted,
        ok=True,
        latency_ms=int((time.time() - t0) * 1000),
        raw_tokens=n_raw,
    )


def chat_openrouter(
    model_id: str,
    user_content: str,
    *,
    temperature: float,
    seed: int,
    max_tokens: int,
    timeout: int = 120,
    system: str | None = None,
) -> Completion:
    key = os.environ.get(config.OPENROUTER_KEY_ENV, "")
    if not key:
        return Completion(text="", ok=False, error=f"missing ${config.OPENROUTER_KEY_ENV}")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    t0 = time.time()
    try:
        data = _retry_post(
            f"{config.OPENROUTER_BASE}/chat/completions",
            payload=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
    except BackendError as e:
        return Completion(text="", ok=False, error=str(e), latency_ms=int((time.time() - t0) * 1000))

    choices = data.get("choices") or []
    text = (choices[0].get("message") or {}).get("content", "") if choices else ""
    usage = data.get("usage") or {}
    return Completion(
        text=text or "",
        ok=True,
        latency_ms=int((time.time() - t0) * 1000),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def chat(
    spec,
    user_content: str,
    *,
    temperature: float,
    seed: int,
    num_predict: int,
    timeout: int = 120,
    system: str | None = None,
) -> Completion:
    """Dispatch on ``ModelSpec.backend``."""
    if spec.backend == "ollama":
        return chat_ollama(
            spec.model_id,
            user_content,
            temperature=temperature,
            seed=seed,
            num_predict=num_predict,
            timeout=timeout,
            system=system,
            think=spec.think,
        )
    if spec.backend == "openrouter":
        return chat_openrouter(
            spec.model_id,
            user_content,
            temperature=temperature,
            seed=seed,
            max_tokens=num_predict,
            timeout=timeout,
            system=system,
        )
    raise ValueError(f"unknown backend {spec.backend!r}")


def server_models() -> list[str]:
    """Names currently served by the local Ollama daemon."""
    try:
        r = requests.get(f"{config.OLLAMA_BASE}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as e:  # noqa: BLE001
        raise BackendError(f"cannot reach Ollama at {config.OLLAMA_BASE}: {e}") from e


def model_digest(model_id: str) -> str:
    """Ollama digest for provenance manifests."""
    try:
        r = requests.post(f"{config.OLLAMA_BASE}/api/show", json={"model": model_id}, timeout=15)
        if r.status_code == 200:
            return str(r.json().get("digest") or r.json().get("modified_at") or "")
    except Exception:  # noqa: BLE001
        pass
    return ""
