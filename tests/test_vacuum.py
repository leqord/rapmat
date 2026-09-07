import os

import pytest

from rapmat.storage.engine import (
    full_vacuum,
    incremental_vacuum,
    make_engine,
    page_stats,
    run_migrations,
)
from rapmat.storage.sqlite_store import SQLiteStore
from rapmat.utils.common import format_bytes


def _bloat(engine, rows: int = 400, blob: int = 8000) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS bloat (x BLOB)")
        for _ in range(rows):
            conn.exec_driver_sql("INSERT INTO bloat VALUES (?)", (os.urandom(blob),))
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM bloat")


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(str(tmp_path / "t.sqlite").replace(os.sep, "/"))
    run_migrations(eng)
    yield eng
    eng.dispose()


class TestPageStats:
    def test_new_database_is_incremental(self, engine):
        assert page_stats(engine).auto_vacuum == 2

    def test_reports_reclaimable_space(self, engine):
        _bloat(engine)
        stats = page_stats(engine)

        assert stats.free_pages > 0
        assert stats.reclaimable_bytes == stats.free_pages * stats.page_size
        assert stats.total_bytes == stats.page_count * stats.page_size
        assert 0.0 < stats.reclaimable_fraction <= 1.0

    def test_vacuumed_database_has_no_free_pages(self, engine):
        full_vacuum(engine)

        assert page_stats(engine).free_pages == 0
        assert page_stats(engine).reclaimable_fraction == 0.0


class TestIncrementalVacuum:
    def test_drains_the_whole_freelist(self, engine):
        _bloat(engine)
        before = page_stats(engine)

        freed = incremental_vacuum(engine)

        assert freed == before.free_pages
        assert page_stats(engine).free_pages == 0

    def test_respects_the_page_budget(self, engine):
        _bloat(engine)
        before = page_stats(engine).free_pages
        assert before > 10

        freed = incremental_vacuum(engine, 10)

        assert freed == 10
        assert page_stats(engine).free_pages == before - 10

    def test_budget_larger_than_freelist_is_clamped(self, engine):
        _bloat(engine)
        before = page_stats(engine).free_pages

        assert incremental_vacuum(engine, before * 100) == before

    def test_no_op_once_the_freelist_is_empty(self, engine):
        _bloat(engine)
        incremental_vacuum(engine)

        assert incremental_vacuum(engine) == 0

    def test_shrinks_the_file(self, engine, tmp_path):
        path = tmp_path / "t.sqlite"
        _bloat(engine)
        before = path.stat().st_size

        incremental_vacuum(engine)

        assert path.stat().st_size < before


class TestFullVacuum:
    def test_reclaims_everything(self, engine, tmp_path):
        path = tmp_path / "t.sqlite"
        _bloat(engine)
        before = path.stat().st_size

        full_vacuum(engine)

        assert page_stats(engine).free_pages == 0
        assert path.stat().st_size < before

    def test_upgrades_a_legacy_database(self, tmp_path):
        import sqlite3

        path = tmp_path / "legacy.sqlite"
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE t (x INTEGER)")
        con.commit()
        con.close()
        assert sqlite3.connect(path).execute("PRAGMA auto_vacuum").fetchone()[0] == 0

        eng = make_engine(str(path).replace(os.sep, "/"))
        try:
            full_vacuum(eng)
            assert page_stats(eng).auto_vacuum == 2
        finally:
            eng.dispose()


class TestStoreIntegration:
    def test_vacuum_returns_fresh_stats(self, tmp_path):
        store = SQLiteStore.from_path(tmp_path / "db")
        try:
            _bloat(store._engine)
            assert store.storage_stats().free_pages > 0

            after = store.vacuum()

            assert after.free_pages == 0
            assert store.storage_stats().free_pages == 0
        finally:
            store.close()

    def test_close_reclaims_free_pages(self, tmp_path):
        store = SQLiteStore.from_path(tmp_path / "db")
        _bloat(store._engine)
        path = tmp_path / "db" / "rapmat.sqlite"
        before = path.stat().st_size
        assert store.storage_stats().free_pages > 0

        store.close()

        assert path.stat().st_size < before

    def test_close_bounds_the_work_it_does(self, tmp_path, monkeypatch):
        monkeypatch.setattr("rapmat.storage.sqlite_store._EXIT_VACUUM_MAX_PAGES", 5)
        store = SQLiteStore.from_path(tmp_path / "db")
        _bloat(store._engine)
        before = store.storage_stats().free_pages
        assert before > 5

        store.close()

        eng = make_engine(str(tmp_path / "db" / "rapmat.sqlite").replace(os.sep, "/"))
        try:
            assert page_stats(eng).free_pages == before - 5
        finally:
            eng.dispose()

    def test_close_survives_a_broken_engine(self, tmp_path):
        store = SQLiteStore.from_path(tmp_path / "db")
        store._engine.dispose()
        store._engine = None

        store.close()


class TestFormatBytes:
    @pytest.mark.parametrize(
        "size,expected",
        [
            (0, "0 B"),
            (512, "512 B"),
            (4096, "4.0 KB"),
            (1024 * 1024 * 3, "3.0 MB"),
            (1024**3 * 2, "2.0 GB"),
            (1024**4, "1024.0 GB"),
        ],
    )
    def test_scales_units(self, size, expected):
        assert format_bytes(size) == expected
