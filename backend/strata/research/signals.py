"""Named signal lists: sets of session dates produced outside the rule language.

The ML workspace writes one when a model's WALK-FORWARD predictions are turned into a
signal: each date's prediction came from a model fitted only on sessions before it. A rule
reads it with `signal("name")`, so a model-generated signal goes through the same
deterministic study and backtest engines as any hand-written rule, and nothing about the
model has to be trusted by them: to the engines it is a list of dates with a stated moment
at which each date's value became known ("open" or "close").
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from strata.config import get_settings

_CACHE: dict[tuple, dict] = {}


def _dir() -> Path:
    return Path(get_settings().data_dir) / "signals"


def valid_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,60}", name or ""))


def save(name: str, symbol: str, dates: list[str], known_at: str, source: dict) -> dict:
    if not valid_name(name):
        raise ValueError("a signal name may contain letters, digits, underscore, dot and dash (at most 60 characters)")
    if known_at not in ("open", "close"):
        raise ValueError("known_at must be open or close")
    from strata.api import db
    d = _dir()
    d.mkdir(parents=True, exist_ok=True)
    doc = {"name": name, "symbol": symbol.upper(), "known_at": known_at, "dates": sorted({str(x)[:10] for x in dates}), "source": source, "created_at": db.now()}
    (d / f"{name}.json").write_text(json.dumps(doc))
    return {k: v for k, v in doc.items() if k != "dates"} | {"n": len(doc["dates"]), "first": doc["dates"][0] if doc["dates"] else None, "last": doc["dates"][-1] if doc["dates"] else None}


def load(name: str) -> dict | None:
    p = _dir() / f"{name}.json"
    if not valid_name(name) or not p.exists():
        return None
    key = (str(p), p.stat().st_mtime_ns)
    if key not in _CACHE:
        for k in [k for k in _CACHE if k[0] == str(p)]:
            del _CACHE[k]
        doc = json.loads(p.read_text())
        doc["set"] = set(doc["dates"])
        _CACHE[key] = doc
    return _CACHE[key]


def listing() -> list[dict]:
    out = []
    for p in sorted(_dir().glob("*.json")) if _dir().exists() else []:
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out.append({"name": d["name"], "symbol": d["symbol"], "known_at": d["known_at"], "n": len(d["dates"]), "first": d["dates"][0] if d["dates"] else None, "last": d["dates"][-1] if d["dates"] else None,
                    "source": d.get("source"), "created_at": d.get("created_at")})
    return out


def delete(name: str) -> None:
    p = _dir() / f"{name}.json"
    if valid_name(name) and p.exists():
        p.unlink()
