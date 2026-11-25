"""In-app notifications: things that happened while nobody was looking (a collection cycle
finished or failed, a session could not be recovered, a forward experiment was evaluated).

They are rows in the application database, shown in the interface. Strata sends nothing
to any outside service."""
from __future__ import annotations

from strata.api import db

LEVELS = ("info", "warn", "error")


def add(kind: str, title: str, body: str = "", level: str = "info", link: str = "") -> int:
    with db.connect() as con:
        return con.execute("INSERT INTO notifications (kind, level, title, body, link, read, created_at) VALUES (?,?,?,?,?,0,?)",
                           (kind, level if level in LEVELS else "info", title[:200], body[:2000], link[:200], db.now())).lastrowid


def listing(limit: int = 50, unread_only: bool = False) -> dict:
    with db.connect() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM notifications" + (" WHERE read=0" if unread_only else "") + " ORDER BY id DESC LIMIT ?", (int(limit),))]
        unread = int(con.execute("SELECT COUNT(*) FROM notifications WHERE read=0").fetchone()[0])
    return {"notifications": rows, "unread": unread}


def mark_read(ids: list[int] | None = None) -> int:
    with db.connect() as con:
        if ids:
            cur = con.execute(f"UPDATE notifications SET read=1 WHERE id IN ({','.join('?' * len(ids))})", [int(i) for i in ids])
        else:
            cur = con.execute("UPDATE notifications SET read=1 WHERE read=0")
        return cur.rowcount
