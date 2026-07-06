"""Alembic migration tests."""

import sqlite3

import pytest
from ase.build import bulk

from conftest import add_relaxed_structure
from rapmat.storage import SQLiteStore

_LEGACY_DDL_PRE_EXCLUDED = """
CREATE TABLE study (
    id TEXT PRIMARY KEY, system TEXT, domain TEXT, calculator TEXT,
    config_json TEXT, timestamp TEXT
);
CREATE TABLE run (
    name TEXT PRIMARY KEY, batch_config_json TEXT, timestamp TEXT,
    study TEXT, run_status TEXT, worker_id TEXT, heartbeat TEXT
);
CREATE TABLE structure (
    id TEXT PRIMARY KEY, run TEXT, status TEXT, gen_spg INTEGER,
    gen_fu INTEGER, energy_per_atom REAL, fmax REAL, converged INTEGER,
    duplicate INTEGER, initial_atoms_json TEXT, final_atoms_json TEXT
);
CREATE TABLE evaluation (
    id TEXT PRIMARY KEY, structure TEXT, run TEXT, calculator TEXT,
    config_json TEXT, energy_per_atom REAL, energy_total REAL,
    min_phonon_freq REAL
);
CREATE TABLE phonon (
    structure TEXT PRIMARY KEY, run TEXT, min_phonon_freq REAL,
    supercell TEXT, mesh TEXT, displacement REAL, symprec REAL,
    calculator TEXT
);
CREATE TABLE phonon_params (
    structure TEXT PRIMARY KEY, run TEXT, params_gz TEXT
);
CREATE INDEX idx_struct_run        ON structure(run);
CREATE INDEX idx_struct_status     ON structure(status);
CREATE INDEX idx_struct_run_status ON structure(run, status);
CREATE INDEX idx_eval_run          ON evaluation(run);
CREATE INDEX idx_eval_struct       ON evaluation(structure);
CREATE INDEX idx_phonon_run        ON phonon(run);
CREATE INDEX idx_phonon_params_run ON phonon_params(run);
"""

_LEGACY_EXCLUDED_COLUMN = "ALTER TABLE structure ADD COLUMN excluded INTEGER;"

_TABLES = {"study", "run", "structure", "evaluation", "phonon", "phonon_params"}


def _make_legacy_db(db_dir, *, with_excluded: bool) -> str:
    """Create a database exactly as the legacy raw-sqlite3 store would."""
    db_dir.mkdir(parents=True, exist_ok=True)
    db_file = db_dir / "rapmat.sqlite"
    conn = sqlite3.connect(db_file)
    conn.executescript(_LEGACY_DDL_PRE_EXCLUDED)
    if with_excluded:
        conn.executescript(_LEGACY_EXCLUDED_COLUMN)
    conn.commit()
    conn.close()
    return str(db_file)


def _sqlite(db_file: str) -> sqlite3.Connection:
    return sqlite3.connect(db_file)


def _table_names(db_file: str) -> set[str]:
    with _sqlite(db_file) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    return {r[0] for r in rows}


def _fk_list(db_file: str, table: str) -> list[tuple]:
    with _sqlite(db_file) as conn:
        return conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()


def _version(db_file: str) -> str:
    with _sqlite(db_file) as conn:
        return conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]


def _columns(db_file: str, table: str) -> set[str]:
    with _sqlite(db_file) as conn:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


# ------------------------------------------------------------------ #
#  Fresh databases
# ------------------------------------------------------------------ #


def test_fresh_db_created_at_head(tmp_path):
    store = SQLiteStore.from_path(tmp_path / "fresh")
    db_file = store.get_url()
    try:
        assert _TABLES <= _table_names(db_file)
        assert _version(db_file) == "0004"
        assert "excluded" in _columns(db_file, "structure")
        for table in ("run", "structure", "evaluation", "phonon", "phonon_params"):
            assert _fk_list(db_file, table), f"{table} has no FKs"
    finally:
        store.close()


def test_reopen_is_idempotent_and_preserves_data(tmp_path):
    store = SQLiteStore.from_path(tmp_path / "reopen")
    store.create_study("s", "Si", "bulk", "mattersim")
    store.create_run(name="r", study_id="s")
    store.close()

    store2 = SQLiteStore.from_path(tmp_path / "reopen")
    try:
        assert store2.get_run_metadata("r") is not None
        assert _version(store2.get_url()) == "0004"
    finally:
        store2.close()


# ------------------------------------------------------------------ #
#  Legacy databases
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("with_excluded", [False, True])
def test_legacy_db_stamped_and_upgraded(tmp_path, with_excluded):
    db_dir = tmp_path / "legacy"
    db_file = _make_legacy_db(db_dir, with_excluded=with_excluded)

    with _sqlite(db_file) as conn:
        conn.execute(
            "INSERT INTO study VALUES ('s', 'Si', 'bulk', 'mattersim', '{}', 't0')"
        )
        conn.execute(
            "INSERT INTO run (name, batch_config_json, timestamp, study) "
            "VALUES ('r', '{}', 't0', 's')"
        )
        cols = "id, run, status, energy_per_atom, fmax, converged"
        conn.execute(
            f"INSERT INTO structure ({cols}) VALUES ('r/1', 'r', 'relaxed', -5.0, 0.01, 1)"
        )

        conn.execute(
            f"INSERT INTO structure ({cols.replace(', converged', '')}, converged) "
            "VALUES ('r/2', 'r', 'generated', 0.0, 0.0, NULL)"
        )
        conn.execute(
            f"INSERT INTO structure ({cols}, initial_atoms_json, final_atoms_json) "
            "VALUES ('r/3', 'r', 'generating', 0.0, 0.0, 0, '', '')"
        )
        conn.commit()

    store = SQLiteStore.from_path(db_dir)
    try:
        assert _version(db_file) == "0004"
        assert "excluded" in _columns(db_file, "structure")
        assert _fk_list(db_file, "structure")

        with _sqlite(db_file) as conn:
            empty_atoms = conn.execute(
                "SELECT count(*) FROM structure "
                "WHERE initial_atoms_json = '' OR final_atoms_json = ''"
            ).fetchone()[0]
        assert empty_atoms == 0

        with _sqlite(db_file) as conn:
            rows = dict(
                conn.execute("SELECT id, converged FROM structure").fetchall()
            )
        assert rows == {"r/1": 1, "r/2": 0, "r/3": 0}
        with _sqlite(db_file) as conn:
            excl = conn.execute(
                "SELECT count(*) FROM structure WHERE excluded IS NULL"
            ).fetchone()[0]
        assert excl == 0

        assert store.get_run_metadata("r") is not None
        assert store.count() == 3
    finally:
        store.close()


def test_legacy_orphans_swept(tmp_path):
    db_dir = tmp_path / "orphans"
    db_file = _make_legacy_db(db_dir, with_excluded=True)

    with _sqlite(db_file) as conn:
        conn.execute(
            "INSERT INTO study VALUES ('s', 'Si', 'bulk', 'mattersim', '{}', 't0')"
        )
        conn.execute(
            "INSERT INTO run (name, batch_config_json, timestamp, study) "
            "VALUES ('r', '{}', 't0', 's')"
        )
        conn.execute(
            "INSERT INTO structure (id, run, status) VALUES ('r/1', 'r', 'relaxed')"
        )

        conn.execute(
            "INSERT INTO structure (id, run, status) "
            "VALUES ('gone/1', 'gone-run', 'relaxed')"
        )
        conn.execute(
            "INSERT INTO evaluation (id, structure, run) VALUES ('e1', 'r/1', 'gone-run')"
        )
        conn.execute(
            "INSERT INTO phonon (structure, run) VALUES ('missing-struct', 'r')"
        )
        conn.execute(
            "INSERT INTO phonon_params (structure, run) VALUES ('r/1', 'gone-run')"
        )

        conn.execute(
            "INSERT INTO run (name, batch_config_json, timestamp, study) "
            "VALUES ('free-run', '{}', 't0', NULL)"
        )
        conn.commit()

    store = SQLiteStore.from_path(db_dir)
    try:
        with _sqlite(db_file) as conn:
            structs = {r[0] for r in conn.execute("SELECT id FROM structure")}
            evals = conn.execute("SELECT count(*) FROM evaluation").fetchone()[0]
            phonons = conn.execute("SELECT count(*) FROM phonon").fetchone()[0]
            blobs = conn.execute("SELECT count(*) FROM phonon_params").fetchone()[0]
            runs = {r[0] for r in conn.execute("SELECT name FROM run")}
        assert structs == {"r/1"}
        assert evals == 0
        assert phonons == 0
        assert blobs == 0
        assert runs == {"r", "free-run"}
    finally:
        store.close()


# ------------------------------------------------------------------ #
#  Cascade deletes
# ------------------------------------------------------------------ #


def test_delete_run_cascades(tmp_path):
    store = SQLiteStore.from_path(tmp_path / "cascade_run")
    try:
        store.create_study("s", "Si", "bulk", "mattersim")
        store.create_run(name="r", study_id="s", config={"formula": {"Si": 1}})
        atoms = bulk("Si", "diamond", a=5.43)
        add_relaxed_structure(store, "r", atoms, -5.0, "r/1")
        store.save_phonon_result("r/1", "r", -0.01, params_gz="BLOB")
        store.add_evaluation("r/1", "r", "emt", "{}", -5.0, -10.0)

        store.delete_run("r")

        db_file = store.get_url()
        with _sqlite(db_file) as conn:
            for table in ("run", "structure", "evaluation", "phonon", "phonon_params"):
                assert (
                    conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
                ), table

        assert store.get_study("s") is not None
    finally:
        store.close()


def test_delete_study_cascades_transitively(tmp_path):
    store = SQLiteStore.from_path(tmp_path / "cascade_study")
    try:
        store.create_study("s", "Si", "bulk", "mattersim")
        store.create_run(name="r1", study_id="s", config={"formula": {"Si": 1}})
        store.create_run(name="r2", study_id="s", config={"formula": {"Si": 2}})
        atoms = bulk("Si", "diamond", a=5.43)
        add_relaxed_structure(store, "r1", atoms, -5.0, "r1/1")
        add_relaxed_structure(store, "r2", atoms, -5.1, "r2/1")
        store.save_phonon_result("r1/1", "r1", -0.01, params_gz="BLOB")

        store.delete_study("s")

        db_file = store.get_url()
        with _sqlite(db_file) as conn:
            for table in _TABLES:
                assert (
                    conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
                ), table
    finally:
        store.close()


# ------------------------------------------------------------------ #
#  Autogenerate cleanliness
# ------------------------------------------------------------------ #


def test_migrations_match_models(tmp_path):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from rapmat.storage.models import Base

    store = SQLiteStore.from_path(tmp_path / "autogen")
    try:
        with store._engine.connect() as conn:
            ctx = MigrationContext.configure(
                conn,
                opts={"compare_type": False, "render_as_batch": True},
            )
            diffs = compare_metadata(ctx, Base.metadata)
        assert diffs == [], f"models drifted from migrations: {diffs}"
    finally:
        store.close()
