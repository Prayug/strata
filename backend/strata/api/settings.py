"""Server-side settings and credentials.

Settings live in the SQLite database. Credentials (API keys) do NOT: they are kept in the
operating system's keychain where one is available (macOS Keychain, or the Secret Service
through `secret-tool` on Linux) and otherwise in a file readable only by the owner
(`<data>/secrets.json`, mode 0600). They can also be supplied as environment variables.

A credential is never returned by an endpoint (only whether it is set, where it is stored
and its last four characters), never written to a log, never placed in a prompt, and is
scrubbed from any provider error before that error is shown or stored.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys

from strata.api import db
from strata.config import get_settings

SERVICE = "strata-research"

SECRETS = {
    "OPENAI_API_KEY": "OpenAI", "ANTHROPIC_API_KEY": "Anthropic", "GEMINI_API_KEY": "Google Gemini", "OPENROUTER_API_KEY": "OpenRouter",
    "ALPACA_API_KEY": "Alpaca key id", "ALPACA_API_SECRET": "Alpaca secret", "POLYGON_API_KEY": "Polygon / Massive", "DATABENTO_API_KEY": "Databento", "ORATS_API_KEY": "ORATS",
}
ENV_ALIASES = {"ALPACA_API_KEY": ["APCA_API_KEY_ID"], "ALPACA_API_SECRET": ["APCA_API_SECRET_KEY"], "GEMINI_API_KEY": ["GOOGLE_API_KEY"]}
DEFAULTS = {
    "llm.provider": "auto",  # auto | offline | ollama | openai | anthropic | gemini | openrouter
    "llm.model.openai": "gpt-4.1-mini", "llm.model.anthropic": "claude-opus-5-5", "llm.model.gemini": "gemini-2.5-flash",
    "llm.model.openrouter": "openai/gpt-4.1-mini", "llm.model.ollama": "", "llm.ollama_url": "http://127.0.0.1:11434",
    "llm.fallback": "rules",  # rules = use the built-in interpreter when the model fails (and say so) | none = stop and report the failure
    "llm.timeout_seconds": 90, "llm.max_retries": 2,
    "ai.exec_mode": "quick",  # quick | review (default for new research sessions)
    "ui.timezone": "America/Los_Angeles", "data.default_symbol": "SPY",
}


# ----------------------------------------------------------------------------- credential storage
def secret_backend() -> str:
    """keychain | secret-tool | file. STRATA_SECRET_STORE can force one (tests use "file")."""
    forced = os.environ.get("STRATA_SECRET_STORE", "").lower()
    if forced in ("keychain", "secret-tool", "file"):
        return forced
    if sys.platform == "darwin" and shutil.which("security"):
        return "keychain"
    if sys.platform.startswith("linux") and shutil.which("secret-tool") and os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        return "secret-tool"
    return "file"


def _secrets_file():
    return get_settings().data_dir / "secrets.json"


def _file_read() -> dict:
    p = _secrets_file()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _file_write(d: dict) -> None:
    p = _secrets_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(d, f)
    os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)


def _run(cmd: list[str], data: str | None = None) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, input=data, capture_output=True, text=True, timeout=10)
        return r.returncode, r.stdout
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def _store_get(name: str) -> str | None:
    b = secret_backend()
    if b == "keychain":
        code, out = _run(["security", "find-generic-password", "-s", SERVICE, "-a", name, "-w"])
        return out.strip() or None if code == 0 else None
    if b == "secret-tool":
        code, out = _run(["secret-tool", "lookup", "service", SERVICE, "account", name])
        return out.strip() or None if code == 0 else None
    return _file_read().get(name) or None


def _store_set(name: str, value: str | None) -> None:
    b = secret_backend()
    if b == "keychain":
        _run(["security", "delete-generic-password", "-s", SERVICE, "-a", name])
        if value:
            if re.search(r"[\s\"'\\]", value):
                raise RuntimeError("the credential contains characters that cannot be passed to the keychain safely")
            # the command is fed to `security -i` on stdin, so the value never appears in the process list
            _run(["security", "-i"], data=f'add-generic-password -s {SERVICE} -a {name} -l "Strata {name}" -U -w {value}\n')
            code, out = _run(["security", "find-generic-password", "-s", SERVICE, "-a", name, "-w"])
            if code != 0 or out.strip() != value:
                raise RuntimeError("the macOS keychain did not store the credential")
        return
    if b == "secret-tool":
        if value:
            code, _ = _run(["secret-tool", "store", "--label", f"Strata {name}", "service", SERVICE, "account", name], data=value)
            if code != 0:
                raise RuntimeError("the system keyring refused to store the credential")
        else:
            _run(["secret-tool", "clear", "service", SERVICE, "account", name])
        return
    d = _file_read()
    if value:
        d[name] = value
    else:
        d.pop(name, None)
    _file_write(d)


_cache: dict[str, str | None] = {}


def set_secret(name: str, value: str | None) -> None:
    if name not in SECRETS:
        raise ValueError(f"unknown credential {name!r}")
    _store_set(name, (value or "").strip() or None)
    _cache.pop(name, None)
    _legacy_remove(name)


def _legacy_remove(name: str) -> None:
    try:
        with db.connect() as con:
            con.execute("DELETE FROM settings WHERE key=?", (f"secret.{name}",))
    except Exception:  # noqa: BLE001
        pass


def migrate_legacy_secrets() -> int:
    """Strata V2 kept credentials in the settings table. Move any that are still there into
    the credential store and remove them from the database."""
    moved = 0
    try:
        with db.connect() as con:
            rows = con.execute("SELECT key, value FROM settings WHERE key LIKE 'secret.%'").fetchall()
    except Exception:  # noqa: BLE001
        return 0
    for r in rows:
        name = r["key"].split(".", 1)[1]
        try:
            val = json.loads(r["value"])
        except (TypeError, json.JSONDecodeError):
            val = r["value"]
        if name in SECRETS and val:
            try:
                _store_set(name, str(val))
            except RuntimeError:
                continue  # leave it where it is rather than lose it
            moved += 1
        _legacy_remove(name)
    _cache.clear()
    return moved


def known_secret_values() -> list[str]:
    out = []
    for name in SECRETS:
        v = get_secret(name)
        if v and len(v) >= 6:
            out.append(v)
    return out


_KEY_SHAPES = re.compile(r"\b(sk-ant-[A-Za-z0-9_\-]{8,}|sk-or-[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{12,}|AIza[0-9A-Za-z_\-]{20,}|PK[A-Z0-9]{16,}|AK[A-Z0-9]{16,})")


def redact(text: str) -> str:
    """Remove credentials from text that will be shown, stored or logged: every configured
    value, and anything shaped like a provider key."""
    if not text:
        return text
    try:
        for v in known_secret_values():
            if v in text:
                text = text.replace(v, "[redacted]")
    except Exception:  # noqa: BLE001
        pass
    return _KEY_SHAPES.sub("[redacted]", text)


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
    if name not in _cache:
        v = None
        try:
            v = _store_get(name)
        except Exception:  # noqa: BLE001
            v = None
        if not v:
            legacy = get_setting(f"secret.{name}")  # a V2 database that has not been migrated yet
            v = str(legacy) if legacy else None
        _cache[name] = v
    if _cache[name]:
        return _cache[name]
    for env in [name] + ENV_ALIASES.get(name, []):
        if os.environ.get(env):
            return os.environ[env]
    return None


def secret_source(name: str) -> str | None:
    if _cache.get(name) or (name not in _cache and get_secret(name) and _cache.get(name)):
        return {"keychain": "macOS Keychain", "secret-tool": "system keyring", "file": "protected file (owner-only)"}[secret_backend()]
    return "environment variable" if get_secret(name) else None


def public_settings() -> dict:
    out = {k: get_setting(k) for k in DEFAULTS}
    secrets = {}
    for name, label in SECRETS.items():
        v = get_secret(name)
        secrets[name] = {"label": label, "set": bool(v), "source": secret_source(name), "hint": (f"…{v[-4:]}" if v and len(v) > 8 else None)}
    return {"settings": out, "secrets": secrets, "secret_store": {"backend": secret_backend(),
                                                                  "label": {"keychain": "macOS Keychain", "secret-tool": "system keyring (Secret Service)", "file": "protected file readable only by you"}[secret_backend()]}}
