"""Provider-independent language-model access.

One small interface (`LLM.json` / `LLM.text`) with an adapter per provider: Anthropic (the
official SDK), OpenAI, OpenRouter, Google Gemini and a local Ollama server (plain HTTPS).
API keys are read server-side (settings table or environment) and never leave the backend.

A model here only ever produces *text or a JSON object*. It cannot execute code, query the
database or produce a financial result; everything it returns is validated by the caller.
"""
from __future__ import annotations

import json
import re

import httpx

from strata.api.settings import get_secret, get_setting

PROVIDERS = {
    "anthropic": {"label": "Anthropic Claude", "secret": "ANTHROPIC_API_KEY", "default_model": "claude-opus-5-5"},
    "openai": {"label": "OpenAI", "secret": "OPENAI_API_KEY", "default_model": "gpt-4.1-mini"},
    "gemini": {"label": "Google Gemini", "secret": "GEMINI_API_KEY", "default_model": "gemini-2.5-flash"},
    "openrouter": {"label": "OpenRouter", "secret": "OPENROUTER_API_KEY", "default_model": "openai/gpt-4.1-mini"},
    "ollama": {"label": "Ollama (local)", "secret": None, "default_model": ""},
}
AUTO_ORDER = ["anthropic", "openai", "gemini", "openrouter"]


class LLMError(RuntimeError):
    """Any provider failure, with a message that is safe to show the user (never contains a key)."""


def extract_json(text: str) -> dict:
    """First JSON object in a model reply (tolerates code fences and surrounding prose)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        v = json.loads(text)
        if isinstance(v, dict):
            return v
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(text[start:i + 1])
                        if isinstance(v, dict):
                            return v
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    raise LLMError("the model did not return a JSON object")


class LLM:
    provider = "base"
    model = ""

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def text(self, system: str, messages: list[dict], max_tokens: int = 1500) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def json(self, system: str, messages: list[dict], schema: dict | None = None, max_tokens: int = 4000) -> dict:
        return extract_json(self.text(system + "\n\nReply with a single JSON object and nothing else.", messages, max_tokens))


def _http(method: str, url: str, *, headers: dict, body: dict, timeout: float, provider: str) -> dict:
    try:
        r = httpx.request(method, url, headers=headers, json=body, timeout=timeout)
    except httpx.TimeoutException:
        raise LLMError(f"{provider} did not answer within {timeout:.0f}s") from None
    except httpx.HTTPError as e:
        raise LLMError(f"could not reach {provider}: {type(e).__name__}") from None
    if r.status_code >= 400:
        try:
            j = r.json()
            detail = (j.get("error") or {}).get("message") if isinstance(j.get("error"), dict) else j.get("error") or j.get("message")
        except Exception:  # noqa: BLE001
            detail = r.text[:200]
        hint = {401: "the API key was rejected", 403: "the API key lacks permission", 404: "model or endpoint not found", 429: "rate limited or out of quota"}.get(r.status_code, f"HTTP {r.status_code}")
        raise LLMError(f"{provider}: {hint}" + (f" ({str(detail)[:240]})" if detail else ""))
    try:
        return r.json()
    except ValueError:
        raise LLMError(f"{provider} returned a non-JSON response") from None


class OpenAICompat(LLM):
    """OpenAI chat-completions wire format (OpenAI and OpenRouter)."""

    def __init__(self, provider: str, base_url: str, key: str, model: str, timeout: float = 90):
        self.provider, self.base_url, self.key, self.model, self.timeout = provider, base_url.rstrip("/"), key, model, timeout

    def _call(self, system, messages, max_tokens, json_mode):
        body = {"model": self.model, "messages": [{"role": "system", "content": system}] + messages, "max_completion_tokens" if self.provider == "openai" else "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        j = _http("POST", f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.key}"}, body=body, timeout=self.timeout, provider=PROVIDERS[self.provider]["label"])
        try:
            return j["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"{self.provider} returned an unexpected response shape") from None

    def text(self, system, messages, max_tokens=1500):
        return self._call(system, messages, max_tokens, False)

    def json(self, system, messages, schema=None, max_tokens=4000):
        return extract_json(self._call(system + "\n\nReply with a single JSON object and nothing else.", messages, max_tokens, True))


class Ollama(LLM):
    provider = "ollama"

    def __init__(self, url: str, model: str, timeout: float = 180):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def _call(self, system, messages, max_tokens, fmt):
        body = {"model": self.model, "stream": False, "messages": [{"role": "system", "content": system}] + messages, "options": {"temperature": 0, "num_predict": max_tokens}}
        if fmt is not None:
            body["format"] = fmt
        j = _http("POST", f"{self.url}/api/chat", headers={}, body=body, timeout=self.timeout, provider="Ollama")
        return (j.get("message") or {}).get("content") or ""

    def text(self, system, messages, max_tokens=1500):
        return self._call(system, messages, max_tokens, None)

    def json(self, system, messages, schema=None, max_tokens=4000):
        return extract_json(self._call(system + "\n\nReply with a single JSON object and nothing else.", messages, max_tokens, "json"))


class Gemini(LLM):
    provider = "gemini"

    def __init__(self, key: str, model: str, timeout: float = 90):
        self.key, self.model, self.timeout = key, model, timeout

    def _call(self, system, messages, max_tokens, json_mode):
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages]
        cfg = {"maxOutputTokens": max_tokens, "temperature": 0}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        j = _http("POST", f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent", headers={"x-goog-api-key": self.key},
                  body={"systemInstruction": {"parts": [{"text": system}]}, "contents": contents, "generationConfig": cfg}, timeout=self.timeout, provider="Google Gemini")
        try:
            return "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError):
            reason = ((j.get("candidates") or [{}])[0] or {}).get("finishReason") or (j.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(f"Gemini returned no text{' (' + str(reason) + ')' if reason else ''}") from None

    def text(self, system, messages, max_tokens=1500):
        return self._call(system, messages, max_tokens, False)

    def json(self, system, messages, schema=None, max_tokens=4000):
        return extract_json(self._call(system + "\n\nReply with a single JSON object and nothing else.", messages, max_tokens, True))


class Anthropic(LLM):
    """Claude through the official SDK.

    Thinking is adaptive (the parameter is omitted) and depth is set with `output_config.effort`.
    Requests opt in to server-side refusal fallbacks (`fallbacks="default"`), so a declined
    request is retried on another Claude model inside the same call; a final refusal is
    reported as an error rather than read as an answer.
    """
    provider = "anthropic"
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, key: str, model: str, timeout: float = 120, effort: str = "medium"):
        import anthropic
        self._sdk = anthropic
        self.client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=2)
        self.model, self.effort = model, effort

    def _create(self, system, messages, max_tokens, schema):
        a = self._sdk
        out_cfg: dict = {"effort": self.effort}
        if schema is not None:
            out_cfg["format"] = {"type": "json_schema", "schema": schema}
        try:
            try:
                return self.client.beta.messages.create(model=self.model, max_tokens=max_tokens, system=system, messages=messages, output_config=out_cfg,
                                                        betas=[self.FALLBACK_BETA], fallbacks="default")
            except a.BadRequestError:
                # e.g. a schema the structured-output grammar cannot express, or a platform without
                # server-side fallbacks: retry as a plain request and validate the JSON ourselves
                return self.client.messages.create(model=self.model, max_tokens=max_tokens, system=system, messages=messages, output_config={"effort": self.effort})
        except a.AuthenticationError:
            raise LLMError("Anthropic: the API key was rejected") from None
        except a.PermissionDeniedError:
            raise LLMError("Anthropic: the API key lacks permission for this model") from None
        except a.NotFoundError:
            raise LLMError(f"Anthropic: model {self.model!r} was not found") from None
        except a.RateLimitError:
            raise LLMError("Anthropic: rate limited or out of credit") from None
        except a.APIStatusError as e:
            raise LLMError(f"Anthropic: HTTP {e.status_code} ({str(getattr(e, 'message', ''))[:240]})") from None
        except a.APIConnectionError:
            raise LLMError("could not reach Anthropic") from None

    def _text_of(self, resp) -> str:
        if getattr(resp, "stop_reason", None) == "refusal":
            cat = getattr(getattr(resp, "stop_details", None), "category", None)
            raise LLMError("Claude declined this request" + (f" ({cat})" if cat else ""))
        txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if getattr(resp, "stop_reason", None) == "max_tokens" and not txt.strip():
            raise LLMError("Claude ran out of output tokens before answering")
        return txt

    def text(self, system, messages, max_tokens=16000):
        return self._text_of(self._create(system, messages, max(max_tokens, 4000), None))

    def json(self, system, messages, schema=None, max_tokens=16000):
        return extract_json(self._text_of(self._create(system + "\n\nReply with a single JSON object and nothing else.", messages, max(max_tokens, 8000), schema)))


# ----------------------------------------------------------------------------- selection
def _model_for(provider: str) -> str:
    return str(get_setting(f"llm.model.{provider}") or PROVIDERS[provider]["default_model"])


def ollama_models(url: str | None = None, timeout: float = 2.5) -> list[str]:
    url = (url or str(get_setting("llm.ollama_url") or "http://127.0.0.1:11434")).rstrip("/")
    try:
        r = httpx.get(f"{url}/api/tags", timeout=timeout)
        return [m["name"] for m in r.json().get("models", [])] if r.status_code == 200 else []
    except Exception:  # noqa: BLE001
        return []


def build(provider: str) -> LLM:
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r}")
    if provider == "ollama":
        url = str(get_setting("llm.ollama_url") or "http://127.0.0.1:11434")
        model = _model_for("ollama")
        if not model:
            found = ollama_models(url)
            if not found:
                raise LLMError(f"no Ollama server with a model was found at {url}")
            model = found[0]
        return Ollama(url, model)
    key = get_secret(PROVIDERS[provider]["secret"])
    if not key:
        raise LLMError(f"no API key is configured for {PROVIDERS[provider]['label']} (add one under Settings)")
    model = _model_for(provider)
    if provider == "anthropic":
        return Anthropic(key, model)
    if provider == "gemini":
        return Gemini(key, model)
    if provider == "openai":
        return OpenAICompat("openai", "https://api.openai.com/v1", key, model)
    return OpenAICompat("openrouter", "https://openrouter.ai/api/v1", key, model)


def active() -> LLM | None:
    """The model the AI Lab should use, or None for the offline interpreter.

    'auto' picks the first cloud provider with a key. A local Ollama model is only used when
    selected explicitly, since small local models interpret less reliably than the offline rules.
    """
    choice = str(get_setting("llm.provider") or "auto")
    if choice == "offline":
        return None
    if choice == "auto":
        for p in AUTO_ORDER:
            if get_secret(PROVIDERS[p]["secret"]):
                return build(p)
        return None
    return build(choice)


def status() -> dict:
    """What the Settings page shows: which providers are usable, and which one is active. No secrets."""
    choice = str(get_setting("llm.provider") or "auto")
    out = {"choice": choice, "providers": [], "active": None, "error": None}
    for p, info in PROVIDERS.items():
        ready = bool(get_secret(info["secret"])) if info["secret"] else None
        out["providers"].append({"id": p, "label": info["label"], "model": _model_for(p) or None, "default_model": info["default_model"], "configured": ready, "needs_key": bool(info["secret"])})
    try:
        m = active()
        out["active"] = m.name if m else None
    except LLMError as e:
        out["error"] = str(e)
    return out
