"""Engine construction and schema migration.
"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.pool import StaticPool

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_BASELINE_PRE_EXCLUDED = "0001"
_BASELINE_WITH_EXCLUDED = "0002"


def _set_sqlite_pragmas(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA busy_timeout=15000")
    # NOTE: NFS-HPC-safe, do not use WAL; maybe guess the mode at runtime later
    cur.execute("PRAGMA journal_mode=TRUNCATE")  
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


def make_engine(db_file: str) -> Engine:
    """One shared connection.
    """
    if db_file == ":memory:":
        eng = create_engine(
            "sqlite://",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    else:
        eng = create_engine(
            f"sqlite:///{db_file}",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False, "timeout": 15.0},
        )
    event.listen(eng, "connect", _set_sqlite_pragmas)
    return eng


def _raw_pragma(engine: Engine, sql: str) -> None:
    """Execute a PRAGMA outside any transaction.
    """
    raw = engine.raw_connection()
    try:
        raw.driver_connection.execute(sql)
    finally:
        raw.close()


def _alembic_config(connection) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.attributes["connection"] = connection
    return cfg


def run_migrations(engine: Engine) -> None:
    _raw_pragma(engine, "PRAGMA auto_vacuum=INCREMENTAL")

    _raw_pragma(engine, "PRAGMA foreign_keys=OFF")
    try:
        with engine.begin() as conn:
            cfg = _alembic_config(conn)
            insp = inspect(conn)
            tables = set(insp.get_table_names())
            if "alembic_version" not in tables and "structure" in tables:
                cols = {c["name"] for c in insp.get_columns("structure")}
                baseline = (
                    _BASELINE_WITH_EXCLUDED
                    if "excluded" in cols
                    else _BASELINE_PRE_EXCLUDED
                )
                command.stamp(cfg, baseline)
            command.upgrade(cfg, "head")
    finally:
        _raw_pragma(engine, "PRAGMA foreign_keys=ON")
