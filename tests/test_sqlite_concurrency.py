"""Concurrency and locking for the SQLite backend.
"""

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from conftest import _BACKENDS

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(params=list(_BACKENDS))
def store(request, tmp_path):
    s = _BACKENDS[request.param](tmp_path / "conc_db")
    yield s
    s.close()


def _setup_run(store, name="run"):
    store.create_study(name, "Si", "bulk", "MATTERSIM", config={})
    store.create_run(name=name, study_id=name)


def test_concurrent_claim_single_winner(store):
    """N threads race to claim the same run, exactly one wins."""
    _setup_run(store, "race")
    n = 16
    barrier = threading.Barrier(n)
    results: list[bool] = []
    lock = threading.Lock()

    def worker(wid: int) -> None:
        barrier.wait()
        won = store.claim_run("race", f"w{wid}")
        with lock:
            results.append(won)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1, results
    assert results.count(False) == n - 1

    meta = store.get_run_metadata("race")
    assert meta.run_status == "processing"
    assert meta.worker_id is not None


def test_concurrent_heartbeats_only_owner(store):
    """Concurrent heartbeats from non-owners are ignored, the owner's sticky."""
    _setup_run(store, "hb")
    assert store.claim_run("hb", "owner")

    def beat(wid: int) -> None:
        store.update_heartbeat("hb", wid)

    threads = [threading.Thread(target=beat, args=(f"intruder{i}",)) for i in range(8)]
    threads.append(threading.Thread(target=beat, args=("owner",)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    meta = store.get_run_metadata("hb")
    assert meta.worker_id == "owner"
    assert meta.run_status == "processing"


_CHILD = (
    "import sys; from pathlib import Path; from rapmat.storage import SQLiteStore;\n"
    "try:\n"
    "    s = SQLiteStore.from_path(Path(sys.argv[1])); s.close(); print('ACQUIRED')\n"
    "except Exception as e:\n"
    "    print('BLOCKED:' + type(e).__name__); sys.exit(3)\n"
)


def _try_open_in_subprocess(db_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _CHILD, str(db_dir)],
        cwd=str(REPO), capture_output=True, text=True, timeout=60,
    )


def test_lockfile_rejects_second_opener(tmp_path):
    """While one process holds the DB, a second opener is rejected. Once released,
    it succeeds."""
    from rapmat.storage import SQLiteStore

    db_dir = tmp_path / "locked_db"
    holder = SQLiteStore.from_path(db_dir)
    try:
        blocked = _try_open_in_subprocess(db_dir)
        assert blocked.returncode == 3, blocked.stdout + blocked.stderr
        assert "BLOCKED" in blocked.stdout
    finally:
        holder.close()

    freed = _try_open_in_subprocess(db_dir)
    assert freed.returncode == 0, freed.stdout + freed.stderr
    assert "ACQUIRED" in freed.stdout
