"""Tests for DB config resolution, store factory, and run-level locking."""

from datetime import datetime, timedelta

import pytest

from conftest import _BACKENDS, force_heartbeat, read_run_field
from rapmat.db_config import (
    clear_db_config,
    load_db_config,
    resolve_store,
    save_db_config,
)
from rapmat.storage import SQLiteStore

# ------------------------------------------------------------------ #
#  load_db_config / clear_db_config
# ------------------------------------------------------------------ #


class TestDbConfig:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "db.toml"
        monkeypatch.setattr("rapmat.db_config._DB_CONFIG_FILE", cfg_file)
        monkeypatch.setattr("rapmat.db_config.APP_CONFIG_DIR", tmp_path)

        save_db_config(general={"mode": "sqlite", "db_path": str(tmp_path / "data")})
        full = load_db_config()
        assert full is not None
        assert full["general"]["mode"] == "sqlite"
        assert full["general"]["db_path"] == str(tmp_path / "data")

    def test_legacy_server_section_ignored(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "db.toml"
        cfg_file.write_text(
            '[general]\nmode = "sqlite"\ndb_path = ""\n\n'
            '[server]\nurl = "ws://localhost:8000/rpc"\nusername = "root"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr("rapmat.db_config._DB_CONFIG_FILE", cfg_file)
        monkeypatch.setattr("rapmat.db_config.APP_CONFIG_DIR", tmp_path)

        full = load_db_config()
        assert full["general"]["mode"] == "sqlite"
        assert "server" not in full

        save_db_config(general={"mode": "sqlite"})
        assert "[server]" not in cfg_file.read_text(encoding="utf-8")

    @pytest.mark.parametrize("legacy_mode", ["local", "remote"])
    def test_legacy_surreal_mode_resolves_to_sqlite(
        self, tmp_path, monkeypatch, legacy_mode
    ):
        cfg_file = tmp_path / "db.toml"
        cfg_file.write_text(
            f'[general]\nmode = "{legacy_mode}"\ndb_path = "{(tmp_path / "surreal").as_posix()}"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr("rapmat.db_config._DB_CONFIG_FILE", cfg_file)
        monkeypatch.setattr(
            "rapmat.db_config._DEFAULT_SQLITE_PATH", str(tmp_path / "sqlite_default")
        )

        store = resolve_store()
        try:
            assert isinstance(store, SQLiteStore)
            assert store.get_url() == str(
                tmp_path / "sqlite_default" / "rapmat.sqlite"
            )
        finally:
            store.close()

    def test_clear_removes_file(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "db.toml"
        cfg_file.write_text("[server]\n")
        monkeypatch.setattr("rapmat.db_config._DB_CONFIG_FILE", cfg_file)
        assert clear_db_config() is True
        assert not cfg_file.exists()
        assert clear_db_config() is False


# ------------------------------------------------------------------ #
#  Run-level locking
# ------------------------------------------------------------------ #


class TestRunLocking:
    @pytest.fixture(params=list(_BACKENDS))
    def store(self, request, tmp_path):
        s = _BACKENDS[request.param](tmp_path / "lock_db")
        yield s
        s.close()

    def test_claim_and_release(self, store):
        store.create_study(
            study_id="lock-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="lock-run", worker_id="w1", study_id="lock-run")
        assert store.claim_run("lock-run", "w1")

        meta = store.get_run_metadata("lock-run")
        assert meta.run_status == "processing"
        assert meta.worker_id == "w1"

        store.release_run("lock-run", "completed")
        meta = store.get_run_metadata("lock-run")
        assert meta.run_status == "completed"
        assert meta.worker_id is None

    def test_double_claim_fails(self, store):
        store.create_study(
            study_id="dc-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="dc-run", worker_id="w1", study_id="dc-run")
        assert store.claim_run("dc-run", "w1")

        # Second claim should fail (status is now "processing")
        assert store.claim_run("dc-run", "w2") is False

    def test_claim_after_release(self, store):
        store.create_study(
            study_id="re-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="re-run", worker_id="w1", study_id="re-run")
        store.claim_run("re-run", "w1")
        store.release_run("re-run", "pending")

        assert store.claim_run("re-run", "w2")
        meta = store.get_run_metadata("re-run")
        assert meta.worker_id == "w2"

    def test_heartbeat_update(self, store):
        store.create_study(
            study_id="hb-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="hb-run", worker_id="w1", study_id="hb-run")
        store.claim_run("hb-run", "w1")
        store.update_heartbeat("hb-run", "w1")

        meta = store.get_run_metadata("hb-run")
        assert meta.run_status == "processing"

    def test_reclaim_stale_runs(self, store):
        store.create_study(
            study_id="stale-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="stale-run", worker_id="old-w", study_id="stale-run")
        store.claim_run("stale-run", "old-w")

        # Manually set heartbeat to the past
        past_ts = (datetime.now() - timedelta(minutes=20)).isoformat()
        force_heartbeat(store, "stale-run", past_ts)

        reclaimed = store.reclaim_stale_runs(timeout_minutes=10)
        assert "stale-run" in reclaimed

        meta = store.get_run_metadata("stale-run")
        assert meta.run_status == "pending"
        assert meta.worker_id is None

    def test_heartbeat_wrong_worker_ignored(self, store):
        store.create_study(
            study_id="hb-wrong",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="hb-wrong", worker_id="w1", study_id="hb-wrong")
        store.claim_run("hb-wrong", "w1")

        past_ts = (datetime.now() - timedelta(minutes=20)).isoformat()
        force_heartbeat(store, "hb-wrong", past_ts)

        store.update_heartbeat("hb-wrong", "intruder")

        row = read_run_field(store, "hb-wrong", "heartbeat", "worker_id")
        assert row["worker_id"] == "w1"
        assert row["heartbeat"] == past_ts

    def test_reclaim_ignores_active_runs(self, store):
        store.create_study(
            study_id="active-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="active-run", worker_id="w1", study_id="active-run")
        store.claim_run("active-run", "w1")
        store.update_heartbeat("active-run", "w1")

        reclaimed = store.reclaim_stale_runs(timeout_minutes=10)
        assert "active-run" not in reclaimed

    def test_create_run_sets_initial_status(self, store):
        store.create_study(
            study_id="init-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="init-run", worker_id="w1", study_id="init-run")
        meta = store.get_run_metadata("init-run")
        assert meta.run_status == "generating"
        assert meta.worker_id == "w1"

    def test_create_run_without_worker(self, store):
        store.create_study(
            study_id="no-w-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="no-w-run", study_id="no-w-run")
        meta = store.get_run_metadata("no-w-run")
        assert meta.run_status == "generating"
        assert meta.worker_id is None
