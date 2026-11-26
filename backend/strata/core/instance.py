"""One machine at a time may open a data directory's database.

SQLite coordinates its write-ahead log through shared memory and file locks. Neither crosses
the boundary between a computer and a container's virtual machine: Docker Desktop shares the
folder, not the kernel. Two copies of Strata on opposite sides of that boundary each believe
they are alone, and a transaction one of them committed is dropped when the other checkpoints
the log. (Observed on 2026-11-25: a credential saved through a container never reached the
database file, because a process on the host truncated the log seconds later.)

The guard is a lease file in the data directory, `instance.json`. It names the kernel that
holds the directory and carries a counter that the holder advances every few seconds.

  - Processes on the SAME kernel share the lease: SQLite coordinates them correctly, so the
    web application, the collector and a command may run side by side.
  - A process on ANOTHER kernel that finds a live lease does not open the database. It says
    who holds the directory and how to free it, and takes over by itself once the counter
    has stopped.

Whether the holder is alive is judged by the counter CHANGING, timed on the observer's own
monotonic clock. Wall clocks are only compared to decide quickly in the obvious cases: a
virtual machine's clock can be minutes off after the host has slept.

Taking a lease is a claim, not a fact. The taker writes its record and then waits for longer
than one heartbeat: a holder that is alive and has been beating normally writes its own record
back (it never looked silent, so the claim is wrong: a lease file that could not be read for a
moment across the folder share, for instance), and the taker, finding the holder's record
again, stands down. Only a holder that really was silent for the whole waiting time gives way.
(The first version treated a momentarily unreadable file as "nobody holds it" and a running
copy lost its lease to a container that was just starting: caught by running both for real.)
"""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HEARTBEAT = 3.0  # seconds between two advances of the counter
TTL = 15.0  # a counter that has not moved for this long (observer's monotonic clock) belongs to a holder that is gone
RELEASED_TTL = 6.0  # the same, when the holder said goodbye on its way out
ANCIENT = 600.0  # a lease this old by the wall clock is taken at once
RECHECK = 1.0  # a refused process looks at the lease file again at most this often
CLAIM_MARGIN = 1.5  # a claim stands if nobody has objected one heartbeat plus this many seconds later
READ_TRIES = 6  # a lease file that cannot be read is read again before it counts as absent (a replace in progress on a shared folder)
FILE = "instance.json"
DEFAULT_WAIT = 0.0  # a command sets this: it has nothing else to do, so it waits out a lease whose holder may be gone
ROLES = {"server": 3, "collector": 2, "command": 1}


class Busy(RuntimeError):
    """Another machine holds the data directory. `holder` describes it."""

    def __init__(self, holder: dict, checking: bool = False):
        self.holder, self.checking = holder or {}, checking
        super().__init__(message(self.holder, checking))


def message(holder: dict, checking: bool = False) -> str:
    who = holder.get("where") or "another machine"
    extra = ", ".join(x for x in (f"engine {holder['engine_version']}" if holder.get("engine_version") else "", f"started {holder['started'][:16].replace('T', ' ')} UTC" if holder.get("started") else "") if x)
    url = f" Use that copy at {holder['url']}," if holder.get("url") else " Use that copy,"
    return (f"This data directory is in use by another copy of Strata: {who}" + (f" ({extra})" if extra else "") + ". "
            "Two copies on opposite sides of a container boundary lose each other's changes, so this one has not opened the database."
            + url + " or stop it: this copy continues by itself within about 20 seconds."
            + (" (Checking whether that copy is still running.)" if checking else ""))


_kernel: str | None = None


def kernel_id() -> str:
    """Identity of the running kernel: the same for every process and container that shares it, different across a virtual-machine boundary."""
    global _kernel
    forced = os.environ.get("STRATA_INSTANCE_KERNEL")
    if forced:
        return forced
    if _kernel is None:
        k = ""
        try:
            if sys.platform.startswith("linux"):
                k = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            elif sys.platform == "darwin":
                k = subprocess.run(["sysctl", "-n", "kern.bootsessionuuid"], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            k = ""
        _kernel = f"{sys.platform}:{k or platform.node()}"
    return _kernel


def in_container() -> bool:
    return Path("/.dockerenv").exists() or bool(os.environ.get("STRATA_IN_CONTAINER"))


def where() -> str:
    if in_container():
        return "a Docker container"
    return {"darwin": "the app running directly on this Mac", "win32": "the app running directly on this PC"}.get(sys.platform, "the app running directly on this computer")


def enabled() -> bool:
    return os.environ.get("STRATA_INSTANCE_GUARD", "1").lower() not in ("0", "off", "false", "no")


class _State:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()
        self.owned = False
        self.holder: dict = {}
        self.checked = 0.0  # monotonic time of the last look at the file
        self.seen_seq: int | None = None  # the foreign counter last observed
        self.seen_at = 0.0  # monotonic time at which that value was first observed
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.role = "command"
        self.url: str | None = None
        self.started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.seq = 0
        self.checking = False
        self.last_beat = 0.0  # monotonic time of this process's last successful write of the lease


_states: dict[str, _State] = {}
_glock = threading.Lock()


def _state(root: Path) -> _State:
    key = str(root)
    with _glock:
        st = _states.get(key)
        if st is None:
            st = _states[key] = _State(root)
        return st


def _read(root: Path) -> dict | None:
    """The lease record, or None when there is none. A file that is missing or half-written for an instant (the holder is
    replacing it, seen through a folder share) is not "no lease": it is read again a few times first."""
    for i in range(READ_TRIES):
        try:
            d = json.loads((root / FILE).read_text())
            if isinstance(d, dict) and d.get("kernel"):
                return d
        except (OSError, ValueError):
            pass
        if i + 1 < READ_TRIES:
            time.sleep(0.05)
    return None


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _write(st: _State, cur: dict | None, released: bool = False) -> None:
    """Advance the counter. The description of the holder is kept when a longer-lived process of this kernel already wrote it."""
    from strata.config import ENGINE_VERSION
    me = {"where": where(), "pid": os.getpid(), "role": st.role, "url": st.url, "engine_version": ENGINE_VERSION, "started": st.started}
    holder = me
    if cur and cur.get("kernel") == kernel_id() and isinstance(cur.get("holder"), dict):
        h = cur["holder"]
        if h.get("pid") != os.getpid() and _alive(h.get("pid")) and ROLES.get(h.get("role"), 0) >= ROLES.get(st.role, 0):
            holder = h
    st.seq = max(st.seq, int((cur or {}).get("seq") or 0) if (cur or {}).get("kernel") == kernel_id() else 0) + 1
    rec = {"kernel": kernel_id(), "seq": st.seq, "beat": time.time(), "holder": holder, "released": bool(released and holder is me)}
    st.root.mkdir(parents=True, exist_ok=True)
    tmp = st.root / f".{FILE}.{os.getpid()}.{threading.get_ident()}.tmp"
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, st.root / FILE)
    st.last_beat = time.monotonic()


def _beat(st: _State) -> None:
    while not st.stop.wait(HEARTBEAT):
        if not st.root.exists():
            return  # the directory is gone (a temporary one): nothing left to guard
        try:
            with st.lock:
                if not st.owned:
                    continue
                cur = _read(st.root)
                if cur and cur.get("kernel") != kernel_id():
                    if time.monotonic() - st.last_beat < TTL:
                        # A copy on another kernel wrote a claim although this process never fell silent: the claim is wrong.
                        # Write our record back; the claimant is waiting to see exactly this and stands down.
                        _write(st, None)
                    else:
                        # this process was frozen (or could not write) for longer than a lease lasts: the other copy took the
                        # directory rightfully. Yield rather than write over each other.
                        st.owned, st.holder, st.seen_seq, st.seen_at = False, cur.get("holder") or {}, int(cur.get("seq") or 0), time.monotonic()
                    continue
                _write(st, cur)
        except OSError:
            continue


def _take(st: _State, cur: dict | None) -> bool:
    """Claim the directory. A record of this kernel is simply joined. Anything else is a claim that must go unanswered for
    longer than one heartbeat: a holder that is alive writes its record back in that time, and then this process stands down."""
    mine = bool(cur and cur.get("kernel") == kernel_id())
    fresh_directory = cur is None and not (st.root / "app.db").exists()  # nothing was ever opened here: nobody to object
    _write(st, cur if mine else None)
    if not mine and not fresh_directory:
        time.sleep(HEARTBEAT + CLAIM_MARGIN + random.random() * 0.5)
        again = _read(st.root)
        if again is None or again.get("kernel") != kernel_id():
            st.owned, st.holder = False, (again or {}).get("holder") or {}
            st.seen_seq, st.seen_at = int((again or {}).get("seq") or 0), time.monotonic()
            return False
    st.owned, st.holder, st.seen_seq = True, {}, None
    if st.thread is None or not st.thread.is_alive():
        st.stop.clear()
        st.thread = threading.Thread(target=_beat, args=(st,), name="strata-instance-lease", daemon=True)
        st.thread.start()
    return True


def _decide(st: _State) -> tuple[bool, bool]:
    """(owned, still checking). Called with the state lock held."""
    now = time.monotonic()
    cur = _read(st.root)
    if cur is None or cur.get("kernel") == kernel_id():
        return _take(st, cur), False
    seq = int(cur.get("seq") or 0)
    st.holder = cur.get("holder") or {}
    if st.seen_seq != seq:
        first_look = st.seen_seq is None
        st.seen_seq, st.seen_at = seq, now
        age = time.time() - float(cur.get("beat") or 0)
        if first_look and age > ANCIENT:
            return _take(st, cur), False
        # the counter moved since the last look (alive), or this is the first look: fresh by the wall clock means alive
        return False, bool(first_look and not (-TTL <= age <= TTL))
    limit = RELEASED_TTL if cur.get("released") else TTL
    if now - st.seen_at >= limit:
        return _take(st, cur), False
    return False, False


def ensure(root: Path | None = None, wait: float | None = None) -> None:
    """Hold the lease on a data directory, or raise Busy. Cheap when the lease is already held.

    wait: how long a caller that cannot serve anything else (a command) is prepared to wait for a lease whose holder may be gone."""
    if not enabled():
        return
    if root is None:
        from strata.config import get_settings
        root = get_settings().data_dir
    st = _state(Path(root))
    if st.owned:
        return
    wait = DEFAULT_WAIT if wait is None else wait
    deadline = time.monotonic() + max(0.0, wait)
    while True:
        with st.lock:
            if st.owned:
                return
            now = time.monotonic()
            if now - st.checked >= RECHECK or st.checked == 0.0 or wait:
                st.checked = now
                owned, checking = _decide(st)
                st.checking = checking
            else:
                owned, checking = st.owned, st.checking
            if owned:
                return
            holder = dict(st.holder)
        if time.monotonic() >= deadline:
            raise Busy(holder, checking)
        time.sleep(1.0)


def configure(role: str | None = None, url: str | None = None, root: Path | None = None) -> None:
    """Describe this process for whoever finds the lease: what it is and where it can be opened."""
    if root is None:
        from strata.config import get_settings
        root = get_settings().data_dir
    st = _state(Path(root))
    if role:
        st.role = role
    if url:
        st.url = url


def status(root: Path | None = None) -> dict:
    """{"standby": False} when this process holds the lease (or the guard is off), else who holds it."""
    if not enabled():
        return {"standby": False}
    try:
        ensure(root)
        return {"standby": False}
    except Busy as e:
        return {"standby": True, "holder": e.holder, "checking": e.checking, "message": str(e)}


def release(root: Path | None = None) -> None:
    """On the way out: stop the heartbeat and say goodbye, so that a waiting copy can take over sooner."""
    if root is None:
        from strata.config import get_settings
        root = get_settings().data_dir
    st = _states.get(str(Path(root)))
    if st is None:
        return
    st.stop.set()
    with st.lock:
        if st.owned:
            try:
                cur = _read(st.root)
                if cur is None or cur.get("kernel") == kernel_id():
                    _write(st, cur, released=True)
            except OSError:
                pass
            st.owned = False


def forget() -> None:
    """Drop every lease this process holds (tests)."""
    with _glock:
        for st in _states.values():
            st.stop.set()
        _states.clear()
