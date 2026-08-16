import os
import tomllib
from pathlib import Path

import tomli_w

from rapmat.config import APP_CONFIG_DIR
from rapmat.utils.console import get_logger

_SETTINGS_FILE = APP_CONFIG_DIR / "settings.toml"

# NOTE: VASP_SCRIPT is excluded deliberately
_VASP_COMMAND_ENV = ("ASE_VASP_COMMAND", "VASP_COMMAND")


def load_app_settings() -> dict:
    if not _SETTINGS_FILE.is_file():
        return {}
    try:
        with open(_SETTINGS_FILE, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        get_logger("rapmat.app_config").warning(
            "Ignoring unreadable %s: %s", _SETTINGS_FILE, exc
        )
        return {}


def resolve_vasp_command() -> str:
    saved = load_app_settings().get("vasp", {}).get("command", "")
    if saved:
        return str(saved)
    for name in _VASP_COMMAND_ENV:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def persist_vasp_command(command: str) -> bool:
    command = command.strip()
    settings = load_app_settings()
    if not command or settings.get("vasp", {}).get("command", "") == command:
        return False

    settings.setdefault("vasp", {})["command"] = command
    APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _SETTINGS_FILE.write_text(tomli_w.dumps(settings), encoding="utf-8")
    return True


def settings_file_path() -> Path:
    return _SETTINGS_FILE
