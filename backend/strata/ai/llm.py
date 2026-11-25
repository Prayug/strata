"""Provider-independent language-model access.

One small interface (`LLM.json` / `LLM.text`) with an adapter per provider: Anthropic (the
official SDK), OpenAI, OpenRouter, Google Gemini and a local Ollama server (plain HTTPS).
API keys are read server-side (OS keychain, protected file or environment) and never leave
the backend: they are not logged, not returned by any endpoint, not placed in prompts, and
any provider error message is scrubbed of them before it is shown.

A model here only ever produces *text or a JSON object*. It cannot execute code, query the
database or produce a financial result; everything it returns is validated by the caller.

Reliability
  * structured output: each adapter uses the provider's own JSON / JSON-schema output mode
    where it has one (Anthropic output_config.format, OpenAI and OpenRouter response_format,
    Gemini responseMimeType, Ollama format) and the reply is still validated by the caller
  * transient failures (timeouts, 429, 5xx) are retried with backoff; the timeout and the
    number of retries are settings
  * a call can be cancelled between attempts
  * token usage of every call is recorded (provider, model, purpose)
"""
from __future__ import annotations

import json
import re
import time

import httpx

from strata.api.settings import get_secret, get_setting, redact

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

    def __init__(self, msg: str = ""):
        super().__init__(redact(str(msg)))


RETRYABLE = (408, 409, 425, 429, 500, 502, 503, 504, 529)


def _timeout(default: float) -> float:
    try:
        return float(get_setting("llm.timeout_seconds") or default)
    except (TypeError, ValueError):
        return default


def _retries() -> int:
    try:
        return max(0, min(5, int(get_setting("llm.max_retries") if get_setting("llm.max_retries") is not None else 2)))
    except (TypeError, ValueError):
        return 2


def record_usage(llm: "LLM", input_tokens, output_tokens) -> None:
    """Token usage of one call. Failing to record must never fail the call."""
    llm.last_usage = {"input_tokens": int(input_tokens or 0), "output_tokens": int(output_tokens or 0)}
    try:
        from strata.api import db
        db.add_usage(llm.provider, llm.model, getattr(llm, "purpose", "") or "", llm.last_usage["input_tokens"], llm.last_usage["output_tokens"])
    except Exception:  # noqa: BLE001
        pass


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
    purpose = "interpret"  # what the call is for (recorded with its token usage)
    cancel = None  # optional callable: return True to abandon the call before the next attempt
    last_usage: dict | None = None

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def text(self, system: str, messages: list[dict], max_tokens: int = 1500) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def json(self, system: str, messages: list[dict], schema: dict | None = None, max_tokens: int = 4000) -> dict:
        return extract_json(self.text(system + "\n\nReply with a single JSON object and nothing else.", messages, max_tokens))


def _http(method: str, url: str, *, headers: dict, body: dict | None, timeout: float, provider: str, retries: int | None = None, cancel=None) -> dict:
    """One provider request with bounded retries. Timeouts, connection failures and
    transient statuses (429, 5xx) are retried with exponential backoff; anything else is
    reported at once. The error text never contains a credential."""
    attempts = (retries if retries is not None else _retries()) + 1
    last: LLMError | None = None
    for k in range(attempts):
        if cancel is not None and cancel():
            raise LLMError("cancelled")
        if k:
            time.sleep(min(8.0, 0.8 * (2 ** (k - 1))))
        try:
            r = httpx.request(method, url, headers=headers, timeout=timeout, **({"json": body} if body is not None else {}))
        except httpx.TimeoutException:
            last = LLMError(f"{provider} did not answer within {timeout:.0f}s")
            continue
        except httpx.HTTPError as e:
            last = LLMError(f"could not reach {provider}: {type(e).__name__}")
            continue
        if r.status_code >= 400:
            try:
                j = r.json()
                detail = (j.get("error") or {}).get("message") if isinstance(j.get("error"), dict) else j.get("error") or j.get("message")
            except Exception:  # noqa: BLE001
                detail = r.text[:200]
            hint = {401: "the API key was rejected", 403: "the API key lacks permission", 404: "model or endpoint not found", 429: "rate limited or out of quota"}.get(r.status_code, f"HTTP {r.status_code}")
            err = LLMError(f"{provider}: {hint}" + (f" ({str(detail)[:240]})" if detail else ""))
            err.status = r.status_code
            if r.status_code in RETRYABLE and k < attempts - 1:
                last = err
                continue
            raise err
        try:
            return r.json()
        except ValueError:
            raise LLMError(f"{provider} returned a non-JSON response") from None
    raise last or LLMError(f"{provider} request failed")


class OpenAICompat(LLM):
    """OpenAI chat-completions wire format (OpenAI and OpenRouter)."""

    def __init__(self, provider: str, base_url: str, key: str, model: str, timeout: float | None = None):
        self.provider, self.base_url, self.key, self.model, self.timeout = provider, base_url.rstrip("/"), key, model, timeout or _timeout(90)

    def _call(self, system, messages, max_tokens, fmt):
        body = {"model": self.model, "messages": [{"role": "system", "content": system}] + messages, "max_completion_tokens" if self.provider == "openai" else "max_tokens": max_tokens}
        if fmt is not None:
            body["response_format"] = fmt
        j = _http("POST", f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.key}"}, body=body, timeout=self.timeout, provider=PROVIDERS[self.provider]["label"], cancel=self.cancel)
        u = j.get("usage") or {}
        record_usage(self, u.get("prompt_tokens"), u.get("completion_tokens"))
        try:
            return j["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"{self.provider} returned an unexpected response shape") from None

    def text(self, system, messages, max_tokens=1500):
        return self._call(system, messages, max_tokens, None)

    def json(self, system, messages, schema=None, max_tokens=4000):
        sys_ = system + "\n\nReply with a single JSON object and nothing else."
        if schema is not None:
            # the provider's JSON-schema output mode; models or routes without it fall back to plain JSON mode
            try:
                return extract_json(self._call(sys_, messages, max_tokens, {"type": "json_schema", "json_schema": {"name": "action", "schema": schema, "strict": False}}))
            except LLMError as e:
                if getattr(e, "status", None) not in (400, 404, 422):
                    raise
        return extract_json(self._call(sys_, messages, max_tokens, {"type": "json_object"}))


class Ollama(LLM):
    provider = "ollama"

    def __init__(self, url: str, model: str, timeout: float | None = None):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout or _timeout(180)

    def _call(self, system, messages, max_tokens, fmt):
        body = {"model": self.model, "stream": False, "messages": [{"role": "system", "content": system}] + messages, "options": {"temperature": 0, "num_predict": max_tokens}}
        if fmt is not None:
            body["format"] = fmt
        j = _http("POST", f"{self.url}/api/chat", headers={}, body=body, timeout=self.timeout, provider="Ollama", retries=0, cancel=self.cancel)
        record_usage(self, j.get("prompt_eval_count"), j.get("eval_count"))
        return (j.get("message") or {}).get("content") or ""

    def text(self, system, messages, max_tokens=1500):
        return self._call(system, messages, max_tokens, None)

    def json(self, system, messages, schema=None, max_tokens=4000):
        sys_ = system + "\n\nReply with a single JSON object and nothing else."
        if schema is not None:
            try:  # Ollama accepts a JSON schema as `format`; older servers only know "json"
                return extract_json(self._call(sys_, messages, max_tokens, schema))
            except LLMError as e:
                if getattr(e, "status", None) not in (400, 404, 422):
                    raise
        return extract_json(self._call(sys_, messages, max_tokens, "json"))


class Gemini(LLM):
    provider = "gemini"

    def __init__(self, key: str, model: str, timeout: float | None = None):
        self.key, self.model, self.timeout = key, model, timeout or _timeout(90)

    def _call(self, system, messages, max_tokens, json_mode):
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages]
        cfg = {"maxOutputTokens": max_tokens, "temperature": 0}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        j = _http("POST", f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent", headers={"x-goog-api-key": self.key},
                  body={"systemInstruction": {"parts": [{"text": system}]}, "contents": contents, "generationConfig": cfg}, timeout=self.timeout, provider="Google Gemini", cancel=self.cancel)
        u = j.get("usageMetadata") or {}
        record_usage(self, u.get("promptTokenCount"), u.get("candidatesTokenCount"))
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

    def __init__(self, key: str, model: str, timeout: float | None = None, effort: str = "medium"):
        import anthropic
        self._sdk = anthropic
        self.client = anthropic.Anthropic(api_key=key, timeout=timeout or _timeout(120), max_retries=_retries())
        self.model, self.effort = model, effort

    def _create(self, system, messages, max_tokens, schema):
        a = self._sdk
        if self.cancel is not None and self.cancel():
            raise LLMError("cancelled")
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
        u = getattr(resp, "usage", None)
        if u is not None:
            record_usage(self, getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))
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


def list_models(provider: str) -> list[str]:
    """Models the configured account can use, asked from the provider itself."""
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r}")
    if provider == "ollama":
        return ollama_models()
    key = get_secret(PROVIDERS[provider]["secret"])
    label = PROVIDERS[provider]["label"]
    if provider == "openrouter":  # the catalogue is public
        j = _http("GET", "https://openrouter.ai/api/v1/models", headers={}, body=None, timeout=20, provider=label, retries=1)
        return sorted(m["id"] for m in j.get("data", []) if m.get("id"))
    if not key:
        raise LLMError(f"no API key is configured for {label}")
    if provider == "openai":
        j = _http("GET", "https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"}, body=None, timeout=20, provider=label, retries=1)
        ids = sorted(m["id"] for m in j.get("data", []) if m.get("id"))
        return [i for i in ids if re.match(r"(gpt-|o\d|chatgpt)", i) and not re.search(r"audio|realtime|tts|transcribe|image|embedding|moderation|search", i)] or ids
    if provider == "gemini":
        j = _http("GET", "https://generativelanguage.googleapis.com/v1beta/models", headers={"x-goog-api-key": key}, body=None, timeout=20, provider=label, retries=1)
        return sorted(m["name"].split("/", 1)[-1] for m in j.get("models", []) if "generateContent" in (m.get("supportedGenerationMethods") or []))
    import anthropic
    try:
        return [m.id for m in anthropic.Anthropic(api_key=key, timeout=20, max_retries=1).models.list(limit=100)]
    except anthropic.APIError as e:
        raise LLMError(f"Anthropic: could not list models ({type(e).__name__})") from None


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
