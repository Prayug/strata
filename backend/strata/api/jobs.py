"""Background job execution outside the HTTP request lifecycle.

Each job runs in its own spawned process (so a crash or a CPU-bound optimisation never
blocks the API). Workers report progress through a multiprocessing queue; a reader
thread fans updates out to Server-Sent-Event subscribers. Cancellation sets a shared
Event that the engine polls between sessions; unresponsive workers are terminated.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import threading
import time
import traceback
from collections import deque

from strata.api import db
from strata.config import get_settings


def _worker(run_id: str, kind: str, payload: dict, data_dir: str, q, cancel_ev):
    from strata.config import override_settings
    override_settings(data_dir=data_dir)
    from strata.api import db as _db
    from strata.api.tasks import execute

    last = [0.0]

    def progress(frac: float, msg: str = ""):
        t = time.time()
        if t - last[0] > 0.25 or frac >= 1:
            last[0] = t
            q.put({"id": run_id, "progress": max(0.0, min(1.0, float(frac))), "message": msg})

    _db.update_run(run_id, status="running", started_at=_db.now())
    q.put({"id": run_id, "status": "running", "progress": 0.0, "message": "started"})
    try:
        summary = execute(run_id, kind, payload, progress, cancel_ev.is_set)
        _db.update_run(run_id, status="done", progress=1.0, summary=summary, finished_at=_db.now(), message="complete")
        q.put({"id": run_id, "status": "done", "progress": 1.0, "message": "complete"})
    except Exception as e:  # noqa: BLE001
        from strata.engine.engine import BacktestCancelled
        if isinstance(e, BacktestCancelled) or cancel_ev.is_set():
            _db.update_run(run_id, status="cancelled", finished_at=_db.now(), message="cancelled")
            q.put({"id": run_id, "status": "cancelled", "message": "cancelled"})
        else:
            err = f"{type(e).__name__}: {e}"
            _db.update_run(run_id, status="error", error=err + "\n" + traceback.format_exc()[-4000:], finished_at=_db.now(), message=err[:300])
            q.put({"id": run_id, "status": "error", "message": err[:300]})


class JobManager:
    def __init__(self, max_concurrent: int = 2):
        self.ctx = mp.get_context("spawn")
        self.q = self.ctx.Queue()
        self.max_concurrent = max_concurrent
        self.pending: deque = deque()
        self.running: dict[str, tuple] = {}
        self.state: dict[str, dict] = {}
        self.subs: dict[str, set] = {}
        self.lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self._stop = False
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._scheduler, daemon=True).start()

    def submit(self, run_id: str, kind: str, payload: dict) -> None:
        with self.lock:
            self.state[run_id] = {"id": run_id, "kind": kind, "status": "queued", "progress": 0.0, "message": "queued"}
            self.pending.append((run_id, kind, payload))
        self._publish(run_id)

    def cancel(self, run_id: str) -> bool:
        with self.lock:
            for i, (rid, *_rest) in enumerate(self.pending):
                if rid == run_id:
                    del self.pending[i]
                    db.update_run(run_id, status="cancelled", finished_at=db.now(), message="cancelled before start")
                    self.state[run_id] = {**self.state.get(run_id, {}), "status": "cancelled", "message": "cancelled"}
                    self._publish(run_id)
                    return True
            item = self.running.get(run_id)
        if not item:
            return False
        proc, ev, started = item
        ev.set()

        def killer():
            proc.join(8)
            if proc.is_alive():
                proc.terminate()
                db.update_run(run_id, status="cancelled", finished_at=db.now(), message="terminated")
                self._on_msg({"id": run_id, "status": "cancelled", "message": "terminated"})
        threading.Thread(target=killer, daemon=True).start()
        return True

    def active(self) -> list[dict]:
        with self.lock:
            return [s for s in self.state.values() if s["status"] in ("queued", "running")]

    def snapshot(self, run_id: str) -> dict | None:
        return self.state.get(run_id)

    # ---------------- internals
    def _scheduler(self):
        while not self._stop:
            with self.lock:
                for rid, (proc, ev, started) in list(self.running.items()):
                    if not proc.is_alive():
                        proc.join(0.1)
                        del self.running[rid]
                        st = self.state.get(rid, {})
                        if st.get("status") in ("queued", "running"):  # died without reporting
                            run = db.get_run(rid) or {}
                            self.state[rid] = {**st, "status": run.get("status", "error"), "message": run.get("message") or "worker exited"}
                            if run.get("status") in (None, "running", "queued"):
                                db.update_run(rid, status="error", error=f"worker exited with code {proc.exitcode}", finished_at=db.now())
                                self.state[rid]["status"] = "error"
                            self._publish(rid)
                while self.pending and len(self.running) < self.max_concurrent:
                    rid, kind, payload = self.pending.popleft()
                    ev = self.ctx.Event()
                    proc = self.ctx.Process(target=_worker, args=(rid, kind, payload, str(get_settings().data_dir), self.q, ev), daemon=False)
                    proc.start()
                    self.running[rid] = (proc, ev, time.time())
                    self.state[rid] = {**self.state.get(rid, {}), "status": "running", "message": "starting"}
            time.sleep(0.2)

    def _reader(self):
        while not self._stop:
            try:
                msg = self.q.get(timeout=0.5)
            except Exception:  # noqa: BLE001
                continue
            self._on_msg(msg)

    def _on_msg(self, msg: dict):
        rid = msg["id"]
        with self.lock:
            st = self.state.setdefault(rid, {"id": rid, "status": "running", "progress": 0.0, "message": ""})
            st.update({k: v for k, v in msg.items() if k != "id"})
        if "progress" in msg and st.get("status") == "running":
            db.update_run(rid, progress=st["progress"], message=st.get("message", ""))
        self._publish(rid)

    def _publish(self, rid: str):
        if self.loop is None:
            return
        st = dict(self.state.get(rid, {}))
        for q in list(self.subs.get(rid, set())) + list(self.subs.get("*", set())):
            self.loop.call_soon_threadsafe(q.put_nowait, st)

    def subscribe(self, rid: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.subs.setdefault(rid, set()).add(q)
        return q

    def unsubscribe(self, rid: str, q) -> None:
        self.subs.get(rid, set()).discard(q)
