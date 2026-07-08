import tomllib

import tomli_w
from pathlib import Path

from rapmat.config import APP_CONFIG_DIR, APP_DATA_DIR
from rapmat.storage.base import StructureStore
from rapmat.utils.console import get_logger

_DB_CONFIG_FILE = APP_CONFIG_DIR / "db.toml"

_GENERAL_DEFAULTS: dict[str, str] = {
    "mode": "sqlite",
    "db_path": "",
}

_LEGACY_MODES = ("local", "remote")


# ------------------------------------------------------------------ #
#  Load / save
# ------------------------------------------------------------------ #


def load_db_config() -> dict:
    general: dict | None = None # TODO: move to Optional[X]

    if _DB_CONFIG_FILE.is_file():
        with open(_DB_CONFIG_FILE, "rb") as f:
            raw = tomllib.load(f)
        general = raw.get("general", {})

    return {"general": {**_GENERAL_DEFAULTS, **(general or {})}}


def save_db_config(*, general: dict | None = None) -> None:
    existing = load_db_config()

    gen = {**existing["general"], **(general or {})}

    APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    doc = {"general": {k: gen.get(k, "") for k in ("mode", "db_path")}}
    _DB_CONFIG_FILE.write_text(tomli_w.dumps(doc), encoding="utf-8")


def clear_db_config() -> bool:
    if _DB_CONFIG_FILE.is_file():
        _DB_CONFIG_FILE.unlink()
        return True
    return False


# ------------------------------------------------------------------ #
#  Store resolution
# ------------------------------------------------------------------ #

_DEFAULT_SQLITE_PATH = str(APP_DATA_DIR / "sqlite")


def resolve_store() -> StructureStore:
    full = load_db_config()
    general = full.get("general", {})
    mode = general.get("mode", "sqlite")
    custom_path = general.get("db_path", "")

    if mode in _LEGACY_MODES:
        get_logger("rapmat.db_config").warning(
            "db.toml mode=%r refers to the removed backend, "
            "switching to the SQLite backend.",
            mode,
        )
        custom_path = ""

    return _make_sqlite_local(Path(custom_path or _DEFAULT_SQLITE_PATH))


def _make_sqlite_local(db_path: Path) -> "StructureStore":
    from rapmat.storage.sqlite_store import SQLiteStore

    return SQLiteStore.from_path(db_path)
