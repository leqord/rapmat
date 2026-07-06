"""Run spglib symmetry calls in an isolated, supervised subprocess.
This exists just to fix spglib' crashes.
"""

from __future__ import annotations

import atexit
import os
import pickle
import queue
import struct
import subprocess
import sys
import threading

# ------------------------------------------------------------------ #
#  Messaging
# ------------------------------------------------------------------ #


def _read_msg(f):
    hdr = f.read(4)
    if not hdr or len(hdr) < 4:
        return None
    (n,) = struct.unpack(">I", hdr)
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(bytes(buf))


def _write_msg(f, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    f.write(struct.pack(">I", len(data)))
    f.write(data)
    f.flush()


# ------------------------------------------------------------------ #
#  Worker
# ------------------------------------------------------------------ #


def _serve() -> None:
    import numpy as np
    import spglib

    inp = sys.stdin.buffer
    outp = sys.stdout.buffer
    while True:
        req = _read_msg(inp)
        if req is None:
            return
        op, payload = req
        try:
            if op == "spg":
                cell, scaled, numbers, symprec = payload
                ds = spglib.get_symmetry_dataset(
                    (np.asarray(cell, float), np.asarray(scaled, float), list(numbers)),
                    symprec=symprec,
                )
                resp = None if ds is None else (str(ds.international), int(ds.number))
            elif op == "std":
                cell, scaled, numbers, symprec, to_primitive, no_idealize = payload
                r = spglib.standardize_cell(
                    (np.asarray(cell, float), np.asarray(scaled, float), list(numbers)),
                    to_primitive=to_primitive, no_idealize=no_idealize, symprec=symprec,
                )
                resp = None if r is None else (
                    np.asarray(r[0]).tolist(), np.asarray(r[1]).tolist(), list(r[2])
                )
            else:
                resp = ("__err__", f"unknown op {op!r}")
        except BaseException as exc:  # error -> report, keep working
            resp = ("__err__", f"{type(exc).__name__}: {exc}")
        try:
            _write_msg(outp, resp)
        except (BrokenPipeError, OSError):
            return


# ------------------------------------------------------------------ #
#  Parent
# ------------------------------------------------------------------ #

_SENTINEL = object()


class _Worker:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue | None = None
        self._lock = threading.Lock()

    def _spawn(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "rapmat.utils.spg_isolated", "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0,
        )
        self._q = queue.Queue()
        threading.Thread(
            target=self._read_loop, args=(self._proc.stdout, self._q), daemon=True
        ).start()

    @staticmethod
    def _read_loop(stdout, q: queue.Queue) -> None:
        while True:
            msg = _read_msg(stdout)
            q.put(_SENTINEL if msg is None else ("msg", msg))
            if msg is None:
                return

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None
        self._q = None

    def call(self, op: str, payload, timeout: float = 30.0):
        """Return the worker's response, or None on failure."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._kill()
                try:
                    self._spawn()
                except Exception:
                    self._kill()
                    return None
            try:
                _write_msg(self._proc.stdin, (op, payload))
            except Exception:
                self._kill()
                return None
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                self._kill()
                return None
            if item is _SENTINEL:
                self._kill()
                return None
            resp = item[1]
            if isinstance(resp, tuple) and len(resp) == 2 and resp[0] == "__err__":
                return None
            return resp


_worker: _Worker | None = None
_worker_lock = threading.Lock()


def _get_worker() -> _Worker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _Worker()
            atexit.register(_worker._kill)
        return _worker


def isolation_enabled() -> bool:
    return os.environ.get("RAPMAT_SPGLIB_ISOLATE", "1") != "0"


def spacegroup(cell, scaled, numbers, symprec):
    """(symbol, number) | None"""
    return _get_worker().call("spg", (cell, scaled, numbers, symprec))


def standardize(cell, scaled, numbers, symprec, to_primitive, no_idealize):
    """(lattice, scaled_positions, numbers) | None."""
    return _get_worker().call(
        "std", (cell, scaled, numbers, symprec, to_primitive, no_idealize)
    )


if __name__ == "__main__":
    if "--serve" in sys.argv:
        _serve()
