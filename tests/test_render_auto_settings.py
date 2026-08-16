"""Explodes?"""

import sys

import pytest

if sys.platform == "win32":
    sys.modules.pop("urwid.display.curses", None)

from rapmat.storage.sqlite_store import SQLiteStore
from rapmat.tui.app import RapmatApp
from rapmat.tui.state import AppState
from rapmat.tui.widgets.calc_fields import SETTINGS_AUTO

_SIZE = (110, 30)


@pytest.fixture
def eval_screen(monkeypatch):
    monkeypatch.setattr("rapmat.app_config.resolve_vasp_command", lambda: "")

    store = SQLiteStore(":memory:")
    store.create_study(
        "s", system="Mo-S", domain="monolayer", calculator="VASP", config={}
    )
    store.create_run(name="r", study_id="s", config={"formula": {"Mo": 1, "S": 2}})

    state = AppState(store=store)
    app = RapmatApp(state)
    state.active_run = "r"

    from rapmat.tui.screens.eval import EvalScreen

    screen = EvalScreen(state, app._router, run_name="r")
    widget = screen.build()
    yield screen, widget
    store.close()


def _render(widget) -> str:
    canvas = widget.render(_SIZE, focus=True)
    return "\n".join(row.decode("utf-8") for row in canvas.text)


def _select(form, key, value):
    widget = form.get_widget(key)
    widget._pick(None, widget.options.index(value))


def test_fields_are_visible(eval_screen):
    _screen, widget = eval_screen
    text = _render(widget)
    assert "Settings:" in text
    assert "VASP command:" in text
    assert "Config TOML Path:" in text


def test_eval_defaults_to_vasp_and_auto(eval_screen):
    screen, widget = eval_screen
    text = _render(widget)
    assert "VASP" in text
    assert SETTINGS_AUTO in text
    assert screen._form.get_values()["calculator"] == "VASP"


def test_auto_locks_the_toml_path(eval_screen):
    screen, _widget = eval_screen
    assert screen._form.is_field_disabled("calculator_config")
    assert not screen._form.is_field_disabled("vasp_command")


def test_switching_to_toml_unlocks_the_path(eval_screen):
    screen, _widget = eval_screen
    _select(screen._form, "calculator_settings", "TOML file")
    assert not screen._form.is_field_disabled("calculator_config")


def test_still_renders_after_switching_modes(eval_screen):
    screen, widget = eval_screen
    _select(screen._form, "calculator_settings", "TOML file")
    _select(screen._form, "calculator", "MATTERSIM")
    _select(screen._form, "calculator_settings", SETTINGS_AUTO)
    for size in (_SIZE, (80, 24)):
        assert widget.render(size, focus=True) is not None


def test_blank_command_blocks_the_run(eval_screen, monkeypatch):
    for name in ("ASE_VASP_COMMAND", "VASP_COMMAND", "VASP_SCRIPT"):
        monkeypatch.delenv(name, raising=False)

    screen, _widget = eval_screen
    screen._form.set_values({"vasp_command": ""})
    screen._on_start()

    assert "VASP command is required" in screen._error_text.get_text()[0]
    assert screen._running is False
