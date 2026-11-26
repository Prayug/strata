"""One machine at a time opens a data directory's database (strata.core.instance), and a credential saved on one side of a
container boundary is found, or at least accounted for, on the other.

A second kernel is simulated two ways: a lease file written by hand with another kernel's identity, and a real second
process started with STRATA_INSTANCE_KERNEL set."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture()
def guard(tmp_path, monkeypatch):
    """A temporary data directory with the lease switched on and short timings."""
    from strata.config import get_settings, override_settings
    from strata.core import instance
    old = get_settings().data_dir
    override_settings(data_dir=tmp_path)
    monkeypatch.setenv("STRATA_INSTANCE_GUARD", "1")
    monkeypatch.setattr(instance, "HEARTBEAT", 0.15)
    monkeypatch.setattr(instance, "TTL", 1.5)  # ten times the heartbeat: a loaded machine must not make a live holder look silent
    monkeypatch.setattr(instance, "RELEASED_TTL", 0.4)
    monkeypatch.setattr(instance, "DEFAULT_WAIT", 0.0)
    monkeypatch.setattr(instance, "RECHECK", 0.05)
    monkeypatch.setattr(instance, "CLAIM_MARGIN", 0.15)
    instance.forget()
    yield tmp_path
    instance.forget()
    override_settings(data_dir=old)


class Foreign:
    """A copy of Strata on another kernel: it writes the lease file and keeps its counter moving until stopped."""

    def __init__(self, root: Path, beat_offset: float = 0.0, url: str | None = "http://localhost:8000", every: float = 0.1):
        self.root, self.off, self.url, self.every = root, beat_offset, url, every
        self.seq, self.stop, self.thread = 0, threading.Event(), None

    def write(self, released: bool = False, beat: float | None = None):
        self.seq += 1
        rec = {"kernel": "linux:another-virtual-machine", "seq": self.seq, "beat": (time.time() + self.off) if beat is None else beat, "released": released,
               "holder": {"where": "a Docker container", "pid": 1, "role": "server", "url": self.url, "engine_version": "6.0.0", "started": "2026-11-25T21:00:00+00:00"}}
        tmp = self.root / ".foreign.tmp"
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, self.root / "instance.json")

    def start(self):
        self.write()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop.wait(self.every):
            self.write()

    def halt(self, goodbye: bool = False):
        self.stop.set()
        if self.thread:
            self.thread.join()
        if goodbye:
            self.write(released=True)


def _until(fn, seconds: float):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if fn():
            return True
        time.sleep(0.05)
    return fn()


def _held(root) -> bool:
    from strata.core import instance
    try:
        instance.ensure(root)
        return True
    except instance.Busy:
        return False


def test_an_unclaimed_directory_is_taken_and_processes_of_one_kernel_share_it(guard):
    from strata.core import instance
    instance.ensure(guard)
    rec = json.loads((guard / "instance.json").read_text())
    assert rec["kernel"] == instance.kernel_id() and rec["holder"]["pid"] == os.getpid() and rec["released"] is False
    assert "host" not in rec["holder"], "the machine's name is not written into the data directory"
    first = rec["seq"]
    assert _until(lambda: json.loads((guard / "instance.json").read_text())["seq"] > first, 2.0), "the holder keeps its counter moving"
    # another process of the SAME kernel (SQLite coordinates those correctly): it joins instead of being refused
    out = subprocess.run([sys.executable, "-c", "from strata.core import instance; import sys; instance.ensure(); print('joined')"], cwd=BACKEND, capture_output=True, text=True, timeout=60,
                         env={**os.environ, "STRATA_DATA_DIR": str(guard), "STRATA_INSTANCE_GUARD": "1", "STRATA_INSTANCE_KERNEL": instance.kernel_id()})
    assert out.stdout.strip() == "joined", out.stderr[-400:]


def test_a_live_copy_on_another_kernel_keeps_the_database_closed(guard):
    from strata.api import db
    from strata.core import instance
    other = Foreign(guard).start()
    try:
        with pytest.raises(instance.Busy) as e:
            db.connect()
        msg = str(e.value)
        assert "a Docker container" in msg and "http://localhost:8000" in msg and "lose each other's changes" in msg
        assert not (guard / "app.db").exists(), "the database file was not even created"
        # it stays closed for as long as the other copy's counter moves, well past the point where a silent holder would be replaced
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            assert not _held(guard)
            time.sleep(0.1)
        assert json.loads((guard / "instance.json").read_text())["kernel"] == "linux:another-virtual-machine", "the lease was never overwritten"
    finally:
        other.halt()
    # the other copy is gone: this one takes over by itself, only after the counter has been still for the whole waiting time
    t0 = time.monotonic()
    assert _until(lambda: _held(guard), 8.0)
    assert time.monotonic() - t0 >= 1.3
    db.init_db()
    assert (guard / "app.db").exists()
    assert json.loads((guard / "instance.json").read_text())["kernel"] == instance.kernel_id()


def test_liveness_is_the_counter_moving_not_the_clock(guard):
    from strata.core import instance
    # the other machine's clock is five minutes behind: its lease looks stale by the wall clock, and it is very much alive
    slow = Foreign(guard, beat_offset=-300).start()
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            assert not _held(guard), "a moving counter is a live holder, whatever its clock says"
            time.sleep(0.1)
    finally:
        slow.halt()
    assert _until(lambda: _held(guard), 8.0)
    instance.forget()
    # the reverse: a holder that died with its clock five minutes ahead looks fresh for five minutes by the wall clock
    dead = Foreign(guard, beat_offset=+300)
    dead.write()
    assert not _held(guard)
    assert _until(lambda: _held(guard), 8.0), "a counter that stands still is a holder that is gone, whatever its clock said"


def test_an_ancient_lease_is_taken_at_once_and_a_goodbye_shortens_the_wait(guard, monkeypatch):
    from strata.core import instance
    monkeypatch.setattr(instance, "TTL", 4.0)  # far apart from the short wait, so the two cannot be mistaken for each other
    old = Foreign(guard)
    old.write(beat=time.time() - 3 * 3600)
    assert _held(guard), "a lease untouched for hours needs no waiting"
    instance.forget()
    polite = Foreign(guard).start()
    assert not _held(guard)
    polite.halt(goodbye=True)
    t0 = time.monotonic()
    assert _until(lambda: _held(guard), 3.0), "after a goodbye the wait is the short one, not the four seconds of a silent holder"
    assert time.monotonic() - t0 >= 0.4


def test_a_holder_that_was_frozen_yields_instead_of_writing_over_the_new_one(guard):
    from strata.api import db
    from strata.core import instance
    instance.ensure(guard)
    db.init_db()
    # this process is frozen for longer than the lease lasts; a copy on another kernel takes the directory meanwhile
    st = instance._state(guard)
    with st.lock:
        st.last_beat = time.monotonic() - 100.0
        thief = Foreign(guard).start()
    try:
        assert _until(lambda: not st.owned, 3.0), "the heartbeat notices the new holder"
        with pytest.raises(instance.Busy):
            db.connect()
        assert json.loads((guard / "instance.json").read_text())["kernel"] == "linux:another-virtual-machine"
    finally:
        thief.halt()


def test_a_live_holder_refuses_a_wrong_claim(guard):
    """Found by running a container beside the app: the container could not read the lease file for an instant (the holder
    was replacing it, seen through the folder share), took that for "nobody holds it" and claimed the directory from a
    running copy. A claim now has to go unanswered for longer than a heartbeat, and a healthy holder answers it."""
    from strata.api import db
    from strata.core import instance
    instance.ensure(guard)
    db.init_db()
    st = instance._state(guard)
    # 1. a claim written by a copy on another kernel, while this one is beating normally: our record is back within a heartbeat
    wrong = Foreign(guard)
    wrong.write()
    assert _until(lambda: json.loads((guard / "instance.json").read_text())["kernel"] == instance.kernel_id(), 2.0), "the holder writes its record back"
    assert st.owned and db.connect() is not None
    # 2. the claimant itself: a real second process whose first look at the lease fails. It claims, waits, finds our record and stands down
    code = ("from strata.core import instance\ninstance.HEARTBEAT, instance.CLAIM_MARGIN, instance.RECHECK = 0.15, 0.6, 0.05\nreal, n = instance._read, {'n': 0}\n"
            "def flaky(root):\n    n['n'] += 1\n    return None if n['n'] == 1 else real(root)\ninstance._read = flaky\n"
            "try:\n    instance.ensure()\n    print('TOOK')\nexcept instance.Busy as e:\n    print('STOOD DOWN', e.holder.get('pid'))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True, timeout=60,
                         env={**os.environ, "STRATA_DATA_DIR": str(guard), "STRATA_INSTANCE_GUARD": "1", "STRATA_INSTANCE_KERNEL": "linux:a-container-that-misread"})
    assert out.stdout.strip() == f"STOOD DOWN {os.getpid()}", (out.stdout, out.stderr[-300:])
    assert st.owned and json.loads((guard / "instance.json").read_text())["kernel"] == instance.kernel_id(), "the running copy kept the directory"
    # 3. a lease file that is unreadable for a moment is read again, not taken for an empty one
    instance.forget()
    live = Foreign(guard).start()
    try:
        real, n = instance._read.__globals__["json"].loads, {"n": 0}

        def flaky(text):
            n["n"] += 1
            if n["n"] <= 2:
                raise ValueError("half-written")
            return real(text)
        import unittest.mock as um
        with um.patch.object(instance.json, "loads", flaky):
            assert not _held(guard), "two failed reads of a live lease do not make the directory free"
        assert n["n"] >= 3
    finally:
        live.halt()


def test_two_real_processes_on_different_kernels(guard):
    """The same thing with a real second process, its kernel identity forced through the environment."""
    from strata.core import instance
    code = ("import sys, time\nfrom strata.core import instance\ninstance.HEARTBEAT = 0.15\ninstance.configure(role='server', url='http://localhost:8000')\n"
            "instance.ensure()\nprint('holding', flush=True)\ntime.sleep(60)\n")
    env = {**os.environ, "STRATA_DATA_DIR": str(guard), "STRATA_INSTANCE_GUARD": "1", "STRATA_INSTANCE_KERNEL": "linux:a-container"}
    p = subprocess.Popen([sys.executable, "-c", code], cwd=BACKEND, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        assert p.stdout.readline().strip() == "holding"
        with pytest.raises(instance.Busy) as e:
            instance.ensure(guard)
        assert e.value.holder.get("url") == "http://localhost:8000" and e.value.holder.get("role") == "server"
        # a command from this side is refused with the explanation and its own exit code, and changes nothing
        r = subprocess.run([sys.executable, "-m", "strata.cli", "usecase"], cwd=BACKEND, capture_output=True, text=True, timeout=120,
                           env={**os.environ, "STRATA_DATA_DIR": str(guard), "STRATA_INSTANCE_GUARD": "1", "STRATA_INSTANCE_KERNEL": "darwin:this-mac", "STRATA_OFFLINE": "1", "STRATA_INSTANCE_WAIT": "1.5"})
        assert r.returncode == 3 and "NOT RUN" in r.stderr and "another copy of Strata" in r.stderr, (r.returncode, r.stderr[-400:])
        assert not (guard / "app.db").exists()
    finally:
        p.kill()
        p.wait()
    assert _until(lambda: _held(guard), 8.0), "killed without a goodbye: taken over after the waiting time"


def test_the_web_application_goes_on_standby_and_continues_by_itself(guard):
    from fastapi.testclient import TestClient

    from strata.api import app as A
    other = Foreign(guard).start()
    try:
        with TestClient(A.app) as c:
            h = c.get("/api/health").json()
            assert h["ok"] and h["standby"] and h["standby"]["holder"]["where"] == "a Docker container" and h["standby"]["holder"]["url"] == "http://localhost:8000"
            for path in ("/api/settings", "/api/backtests", "/api/data/readiness", "/api/forward"):
                r = c.get(path)
                assert r.status_code == 503 and "another copy of Strata" in r.json()["detail"], (path, r.status_code)
            r = c.put("/api/settings", json={"settings": {}, "secrets": {"ALPHAVANTAGE_API_KEY": "written-on-standby"}})
            assert r.status_code == 503, "nothing can be saved into a database this copy has not opened"
            assert not (guard / "app.db").exists() and not (guard / "secrets.json").exists()
            other.halt(goodbye=True)
            assert _until(lambda: c.get("/api/health").json()["standby"] is None, 8.0), "it starts by itself once the other copy has gone"
            assert c.get("/api/settings").status_code == 200 and (guard / "app.db").exists()
    finally:
        other.halt()
        A.standby = False


# ------------------------------------------------------------------ credentials on both sides of the boundary
@pytest.fixture()
def settings_dir(tmp_path):
    from strata.api import db, settings
    from strata.config import get_settings, override_settings
    old = get_settings().data_dir
    override_settings(data_dir=tmp_path)
    db.init_db()
    settings._cache.clear()
    yield tmp_path
    settings._cache.clear()
    override_settings(data_dir=old)


def test_a_key_saved_by_another_process_is_seen_within_seconds(settings_dir, monkeypatch):
    """The fault behind 'I saved it and it says not set': an absence was remembered for the life of the process."""
    from strata.api import settings as S
    monkeypatch.setattr(S, "MISSING_TTL", 0.2)
    assert S.get_secret("ALPACA_API_KEY") is None and S.public_settings()["secrets"]["ALPACA_API_KEY"]["set"] is False
    # another process (the command line, or a container on this data directory) writes the protected file
    (settings_dir / "secrets.json").write_text(json.dumps({"ALPACA_API_KEY": "PKTESTONLY0000000001"}))
    time.sleep(0.3)
    assert S.get_secret("ALPACA_API_KEY") == "PKTESTONLY0000000001"
    pub = S.public_settings()["secrets"]["ALPACA_API_KEY"]
    assert pub["set"] is True and pub["source"] == "protected file (owner-only)" and "PKTESTONLY" not in json.dumps(pub)


def test_the_native_app_reads_a_key_a_container_saved(settings_dir, monkeypatch):
    """On a Mac the keychain is asked first; a key that only exists in the data directory's protected file is still found."""
    from strata.api import settings as S
    monkeypatch.setenv("STRATA_SECRET_STORE", "keychain")
    calls = []

    def fake_run(cmd, data=None):
        calls.append(cmd[:2])
        return 44, ""  # the keychain holds nothing under this name

    monkeypatch.setattr(S, "_run", fake_run)
    (settings_dir / "secrets.json").write_text(json.dumps({"ALPACA_API_SECRET": "containersavedsecret0001"}))
    assert S.get_secret("ALPACA_API_SECRET") == "containersavedsecret0001"
    assert S.secret_source("ALPACA_API_SECRET") == "protected file (owner-only)"
    assert calls and calls[0] == ["security", "find-generic-password"], "the keychain was asked first"


def test_a_key_in_the_keychain_is_accounted_for_inside_a_container(settings_dir, monkeypatch):
    """A container cannot read the macOS Keychain. It says where the key is instead of reporting that none exists; the key itself is never recorded."""
    from strata.api import db, settings as S
    from strata.core import instance
    # the native app saves the key: keychain backend, simulated
    kc: dict = {}

    def fake_run(cmd, data=None):
        if cmd[:2] == ["security", "find-generic-password"]:
            name = cmd[cmd.index("-a") + 1]
            return (0, kc[name] + "\n") if name in kc else (44, "")
        if cmd[:2] == ["security", "delete-generic-password"]:
            kc.pop(cmd[cmd.index("-a") + 1], None)
            return 0, ""
        if cmd == ["security", "-i"]:
            parts = data.split()
            kc[parts[parts.index("-a") + 1]] = parts[parts.index("-w") + 1]
            return 0, ""
        return 1, ""

    monkeypatch.setenv("STRATA_SECRET_STORE", "keychain")
    monkeypatch.setattr(S, "_run", fake_run)
    monkeypatch.setattr(instance, "in_container", lambda: False)
    S.set_secret("ALPACA_API_KEY", "PKNATIVEONLY00000001")
    assert kc == {"ALPACA_API_KEY": "PKNATIVEONLY00000001"} and not (settings_dir / "secrets.json").exists()
    with db.connect() as con:
        dump = " ".join(str(r[0]) + str(r[1]) for r in con.execute("SELECT key, value FROM settings"))
    assert "PKNATIVEONLY" not in dump and "macOS Keychain" in dump, "the database holds where the key is, never the key"
    assert S.public_settings()["secrets"]["ALPACA_API_KEY"].get("elsewhere") is None, "where the key is readable nothing is said"
    # the same data directory, opened inside a container: file backend, no keychain
    S._cache.clear()
    monkeypatch.setenv("STRATA_SECRET_STORE", "file")
    monkeypatch.setattr(instance, "in_container", lambda: True)
    pub = S.public_settings()["secrets"]["ALPACA_API_KEY"]
    assert pub["set"] is False and pub["elsewhere"]["label"] == "macOS Keychain" and "enter the key here as well" in pub["elsewhere"]["note"]
    assert "PKNATIVEONLY" not in json.dumps(S.public_settings())
    # entered in the container too: stored in the protected file, the note disappears
    S.set_secret("ALPACA_API_KEY", "PKNATIVEONLY00000001")
    pub = S.public_settings()["secrets"]["ALPACA_API_KEY"]
    assert pub["set"] is True and "elsewhere" not in pub and (settings_dir / "secrets.json").stat().st_mode & 0o777 == 0o600
    # removed where it was readable: no stale note is left behind
    S.set_secret("ALPACA_API_KEY", None)
    assert "elsewhere" not in S.public_settings()["secrets"]["ALPACA_API_KEY"]
