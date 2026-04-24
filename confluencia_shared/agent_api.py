from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class AgentAPIResult:
    ok: bool
    content: Optional[str]
    raw: Optional[Any]
    error: Optional[str]


def _extract_text_from_response(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj.strip() or None
    if isinstance(obj, dict):
        for key in ["content", "text", "output", "answer", "response", "message"]:
            v = obj.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # OpenAI-style
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    c = msg.get("content")
                    if isinstance(c, str) and c.strip():
                        return c.strip()
                c = first.get("text")
                if isinstance(c, str) and c.strip():
                    return c.strip()
    return None


def _prepare_headers(api_key: Optional[str], extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        for k, v in extra_headers.items():
            if v is None:
                continue
            headers[str(k)] = str(v)
    return headers


def call_openai_chat(
    *,
    endpoint: str,
    api_key: Optional[str],
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = 512,
    timeout: float = 60.0,
    extra_headers: Optional[Dict[str, str]] = None,
) -> AgentAPIResult:
    payload: Dict[str, Any] = {
        "model": str(model),
        "messages": [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": str(user_prompt)},
        ],
        "temperature": float(temperature),
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    headers = _prepare_headers(api_key, extra_headers=extra_headers)

    try:
        resp = requests.post(str(endpoint), json=payload, headers=headers, timeout=float(timeout))
    except Exception as e:
        return AgentAPIResult(ok=False, content=None, raw=None, error=f"请求失败: {e}")

    if resp.status_code != 200:
        return AgentAPIResult(ok=False, content=None, raw=resp.text, error=f"HTTP {resp.status_code}")

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    content = _extract_text_from_response(data)
    if not content:
        return AgentAPIResult(ok=False, content=None, raw=data, error="未在响应中解析到文本内容")

    return AgentAPIResult(ok=True, content=content, raw=data, error=None)


def call_raw_json(
    *,
    endpoint: str,
    api_key: Optional[str],
    payload: Dict[str, Any],
    timeout: float = 60.0,
    headers: Optional[Dict[str, str]] = None,
) -> AgentAPIResult:
    hdr = _prepare_headers(api_key, extra_headers=headers)

    try:
        resp = requests.post(str(endpoint), json=payload, headers=hdr, timeout=float(timeout))
    except Exception as e:
        return AgentAPIResult(ok=False, content=None, raw=None, error=f"请求失败: {e}")

    if resp.status_code != 200:
        return AgentAPIResult(ok=False, content=None, raw=resp.text, error=f"HTTP {resp.status_code}")

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    content = _extract_text_from_response(data)
    return AgentAPIResult(ok=True, content=content, raw=data, error=None)


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None
