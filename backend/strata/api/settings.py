"""Server-side settings. Secrets (API keys) are stored in the local SQLite database or read
from environment variables; they are never returned to the browser - only a masked hint."""
from __future__ import annotations

import json
import os

from strata.api import db

SECRETS = {
    "OPENAI_API_KEY": "OpenAI", "ANTHROPIC_API_KEY": "Anthropic", "GEMINI_API_KEY": "Google Gemini", "OPENROUTER_API_KEY": "OpenRouter",
    "ALPACA_API_KEY": "Alpaca key id", "ALPACA_API_SECRET": "Alpaca secret", "POLYGON_API_KEY": "Polygon / Massive", "DATABENTO_API_KEY": "Databento", "ORATS_API_KEY": "ORATS",
}
ENV_ALIASES = {"ALPACA_API_KEY": ["APCA_API_KEY_ID"], "ALPACA_API_SECRET": ["APCA_API_SECRET_KEY"], "GEMINI_API_KEY": ["GOOGLE_API_KEY"]}
DEFAULTS = {
    "llm.provider": "auto",  # auto | offline | ollama | openai | anthropic | gemini | openrouter
    "llm.model.openai": "gpt-4.1-mini", "llm.model.anthropic": "claude-opus-5-5", "llm.model.gemini": "gemini-2.5-flash",
    "llm.model.openrouter": "openai/gpt-4.1-mini", "llm.model.ollama": "", "llm.ollama_url": "http://127.0.0.1:11434",
    "ui.timezone": "America/Los_Angeles", "data.default_symbol": "SPY",
}


def _row(key: str):
    with db.connect() as con:
        return con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()


def get_setting(key: str, default=None):
    try:
        r = _row(key)
    except Exception:  # table not created yet (CLI before init)
        r = None
    if r is None:
        return DEFAULTS.get(key, default)
    try:
        return json.loads(r["value"])
    except (TypeError, json.JSONDecodeError):
        return r["value"]


def set_setting(key: str, value) -> None:
    db.init_db()
    with db.connect() as con:
        if value is None or value == "":
            con.execute("DELETE FROM settings WHERE key=?", (key,))
        else:
            con.execute("INSERT INTO settings (key, value, updated_at) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                        (key, json.dumps(value), db.now()))


def get_secret(name: str) -> str | None:
    v = get_setting(f"secret.{name}")
    if v:
        return str(v)
    for env in [name] + ENV_ALIASES.get(name, []):
        if os.environ.get(env):
            return os.environ[env]
    return None


def public_settings() -> dict:
    out = {k: get_setting(k) for k in DEFAULTS}
    secrets = {}
    for name, label in SECRETS.items():
        v = get_secret(name)
        src = "settings" if get_setting(f"secret.{name}") else ("environment" if v else None)
        secrets[name] = {"label": label, "set": bool(v), "source": src, "hint": (f"…{v[-4:]}" if v and len(v) > 8 else None)}
    return {"settings": out, "secrets": secrets}
