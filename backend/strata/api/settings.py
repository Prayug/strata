"""Server-side settings and credentials.

Settings live in the SQLite database. Credentials (API keys) do NOT: they are kept in the
operating system's keychain where one is available (macOS Keychain, or the Secret Service
through `secret-tool` on Linux) and otherwise in a file readable only by the owner
(`<data>/secrets.json`, mode 0600). They can also be supplied as environment variables.

A data directory can be used by the app running directly on the computer and, at another
time, by a container: the two do not see the same credential store. A key saved inside a
container lands in the protected file, which the native app reads as well (after its own
keychain). A key saved in the macOS Keychain cannot be read from a container; the settings
table records WHERE each key was stored (never the key), so the other side can say so
instead of reporting that no key exists.

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
import time

from strata.api import db
from strata.config import get_settings

SERVICE = "strata-research"

SECRETS = {
    "OPENAI_API_KEY": "OpenAI", "ANTHROPIC_API_KEY": "Anthropic", "GEMINI_API_KEY": "Google Gemini", "OPENROUTER_API_KEY": "OpenRouter",
    "ALPACA_API_KEY": "Alpaca key id", "ALPACA_API_SECRET": "Alpaca secret", "POLYGON_API_KEY": "Polygon / Massive", "DATABENTO_API_KEY": "Databento", "ORATS_API_KEY": "ORATS", "ALPHAVANTAGE_API_KEY": "Alpha Vantage",
}
ENV_ALIASES = {"ALPACA_API_KEY": ["APCA_API_KEY_ID"], "ALPACA_API_SECRET": ["APCA_API_SECRET_KEY"], "GEMINI_API_KEY": ["GOOGLE_API_KEY"], "POLYGON_API_KEY": ["MASSIVE_API_KEY"]}
DEFAULTS = {
    "llm.provider": "auto",  # auto | offline | ollama | openai | anthropic | gemini | openrouter
    "llm.model.openai": "gpt-4.1-mini", "llm.model.anthropic": "claude-opus-5-5", "llm.model.gemini": "gemini-2.5-flash",
    "llm.model.openrouter": "openai/gpt-4.1-mini", "llm.model.ollama": "", "llm.ollama_url": "http://127.0.0.1:11434",
    "llm.fallback": "rules",  # rules = use the built-in interpreter when the model fails (and say so) | none = stop and report the failure
    # who reads a request when a model is configured: model = the model reads everything | assist = the built-in readers first,
    # the model only for what they could not place (its reading is used only if it compiles and leaves nothing unplaced) |
    # check = both read every strategy and comparison, and where they differ the difference is asked instead of run |
    # auto = check for a local model, model for a hosted one
    "llm.mode": "auto",
    # how a model is asked: tools = the full planner (every tool, free expressions) | form = a short form of typed fields that a
    # small model can fill, every value checked against the sentence | auto = form for a local model, tools for a hosted one
    "llm.planner": "auto",
    "llm.ollama.context_tokens": 12288,  # context asked of a local model; left to the server it is the model's maximum
    "llm.timeout_seconds": 90, "llm.max_retries": 2,
    "ai.exec_mode": "quick",  # quick | review (default for new research sessions)
    "ui.timezone": "America/Los_Angeles", "data.default_symbol": "SPY",
    "ui.mode": "simple",  # simple | advanced: how much of each screen is shown. One calculation engine either way.
    "ui.onboarded": "",
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


def _file_drop(name: str) -> None:
    """Remove one key from the protected file (a copy a container left there), if the file holds it."""
    d = _file_read()
    if name in d:
        d.pop(name)
        if d:
            _file_write(d)
        else:
            _secrets_file().unlink(missing_ok=True)


def _run(cmd: list[str], data: str | None = None) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, input=data, capture_output=True, text=True, timeout=10)
        return r.returncode, r.stdout
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def _store_get(name: str) -> str | None:
    return _store_find(name)[0]


def _store_find(name: str) -> tuple[str | None, str | None]:
    """(value, where it was found). The operating system's store is asked first; the protected file in the data directory
    second, because a copy of Strata running in a container on this data directory can only have saved a key there."""
    b = secret_backend()
    if b == "keychain":
        code, out = _run(["security", "find-generic-password", "-s", SERVICE, "-a", name, "-w"])
        if code == 0 and out.strip():
            return out.strip(), "keychain"
    elif b == "secret-tool":
        code, out = _run(["secret-tool", "lookup", "service", SERVICE, "account", name])
        if code == 0 and out.strip():
            return out.strip(), "secret-tool"
    v = _file_read().get(name) or None
    return (v, "file") if v else (None, None)


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
        _file_drop(name)
        return
    if b == "secret-tool":
        if value:
            code, _ = _run(["secret-tool", "store", "--label", f"Strata {name}", "service", SERVICE, "account", name], data=value)
            if code != 0:
                raise RuntimeError("the system keyring refused to store the credential")
        else:
            _run(["secret-tool", "clear", "service", SERVICE, "account", name])
        _file_drop(name)
        return
    d = _file_read()
    if value:
        d[name] = value
    else:
        d.pop(name, None)
    _file_write(d)


_cache: dict[str, tuple[str | None, str | None, float]] = {}  # name -> (value, where, good until)
FOUND_TTL, MISSING_TTL = 60.0, 3.0  # an absence is only remembered briefly: another process (or a container) may have just saved the key
WHERE_KEY = "secrets.where"  # settings row: which store each key was saved to, and from where. Never the key.
STORE_LABEL = {"keychain": "macOS Keychain", "secret-tool": "system keyring", "file": "protected file (owner-only)"}


def _where_map() -> dict:
    try:
        m = get_setting(WHERE_KEY)
        return m if isinstance(m, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _record_where(name: str, store: str | None) -> None:
    from strata.core import instance
    try:
        m = _where_map()
        if store:
            m[name] = {"store": store, "label": STORE_LABEL.get(store, store), "from": instance.where(), "container": instance.in_container(), "at": db.now()}
        else:
            m.pop(name, None)
        set_setting(WHERE_KEY, m or None)
    except Exception:  # noqa: BLE001
        pass  # the note is a convenience for the other side; storing the key does not depend on it


def set_secret(name: str, value: str | None) -> None:
    if name not in SECRETS:
        raise ValueError(f"unknown credential {name!r}")
    value = (value or "").strip() or None
    _store_set(name, value)
    _cache.pop(name, None)
    _legacy_remove(name)
    _record_where(name, secret_backend() if value else None)


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
            _record_where(name, secret_backend())
            moved += 1
        _legacy_remove(name)
    _cache.clear()
    return moved


def backfill_where() -> int:
    """Note where each key that is readable here is stored, for keys saved before that note existed. Returns how many were added."""
    m, n = _where_map(), 0
    for name in SECRETS:
        v, where = _lookup(name)
        if v and where in STORE_LABEL and name not in m:
            _record_where(name, where)
            n += 1
    return n


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
    from strata.core.instance import Busy
    try:
        r = _row(key)
    except Busy:
        raise  # the database may not be opened here: that is not "no such setting"
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


def _lookup(name: str) -> tuple[str | None, str | None]:
    hit = _cache.get(name)
    if hit is not None and hit[2] > time.monotonic():
        return hit[0], hit[1]
    v, where = None, None
    try:
        v, where = _store_find(name)
    except Exception:  # noqa: BLE001
        v, where = None, None
    if not v:
        from strata.core.instance import Busy
        try:
            legacy = get_setting(f"secret.{name}")  # a V2 database that has not been migrated yet
        except Busy:
            legacy = None  # the database is held by another copy: the credential stores above are still readable
        v, where = (str(legacy), "legacy") if legacy else (None, None)
    _cache[name] = (v, where, time.monotonic() + (FOUND_TTL if v else MISSING_TTL))
    return v, where


def get_secret(name: str) -> str | None:
    v, _ = _lookup(name)
    if v:
        return v
    for env in [name] + ENV_ALIASES.get(name, []):
        if os.environ.get(env):
            return os.environ[env]
    return None


def secret_source(name: str) -> str | None:
    v, where = _lookup(name)
    if v:
        return STORE_LABEL.get(where or "", "the database of an older version")
    return "environment variable" if get_secret(name) else None


def secret_elsewhere(name: str) -> dict | None:
    """A key that is not readable here but was saved from the other side of a container boundary: where it is, never what it is."""
    if get_secret(name):
        return None
    w = _where_map().get(name)
    if not isinstance(w, dict) or w.get("store") == "file":
        return None  # a key saved to the protected file would have been found
    from strata.core import instance
    if bool(w.get("container")) == instance.in_container() and w.get("store") == secret_backend():
        return None  # same side, same store: it was removed there, not hidden from here
    return {"label": w.get("label") or w.get("store"), "from": w.get("from"), "at": w.get("at"),
            "note": f"A key was saved in the {w.get('label') or w.get('store')} by {w.get('from') or 'another copy of Strata'}. This copy cannot read that store: enter the key here as well to use it from here."}


def public_settings() -> dict:
    out = {k: get_setting(k) for k in DEFAULTS}
    secrets = {}
    for name, label in SECRETS.items():
        v = get_secret(name)
        secrets[name] = {"label": label, "set": bool(v), "source": secret_source(name), "hint": (f"…{v[-4:]}" if v and len(v) > 8 else None)}
        other = None if v else secret_elsewhere(name)
        if other:
            secrets[name]["elsewhere"] = other
    return {"settings": out, "secrets": secrets, "secret_store": {"backend": secret_backend(),
                                                                  "label": {"keychain": "macOS Keychain", "secret-tool": "system keyring (Secret Service)", "file": "protected file readable only by you"}[secret_backend()]}}
