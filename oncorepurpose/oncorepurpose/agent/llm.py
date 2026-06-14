"""Minimal, provider-agnostic hosted-LLM client (OpenAI-compatible chat API).

Configure via environment variables:
  ONCO_LLM_API_KEY    (required to enable the LLM; if unset, calls are skipped)
  ONCO_LLM_BASE_URL   (default: https://api.openai.com/v1)
  ONCO_LLM_MODEL      (default: gpt-4o-mini)

Responses are cached on disk (keyed by a hash of the request) so re-runs and the
LLM-as-judge step do not re-bill or re-call the API.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests

from oncorepurpose.config import MODELS_DIR

CACHE_DIR = MODELS_DIR / "llm_cache"


def llm_available() -> bool:
    return bool(os.environ.get("ONCO_LLM_API_KEY"))


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 900,
    json_mode: bool = False,
    use_cache: bool = True,
) -> Optional[str]:
    """Call an OpenAI-compatible chat endpoint. Returns text, or None if disabled/failed."""
    api_key = os.environ.get("ONCO_LLM_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("ONCO_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("ONCO_LLM_MODEL", "gpt-4o-mini")

    payload = {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    key = hashlib.md5(
        json.dumps({"u": base_url, "m": model, "p": payload}, sort_keys=True).encode()
    ).hexdigest()
    cp = _cache_path(key)
    if use_cache and cp.exists():
        return json.loads(cp.read_text())["content"]

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=90,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"  [llm] call failed: {exc}")
        return None

    cp.write_text(json.dumps({"content": content}))
    return content
