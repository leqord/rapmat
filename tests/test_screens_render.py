"""Render smoke tests for every TUI screen.

Each screen is built using in-memory db, rendered at two terminal sizes. Task-owning screens must
cancel their background tasks in ``on_leave``.
"""

import sys

import pytest
import urwid

if sys.platform == "win32":
    sys.modules.pop("urwid.display.curses", None)

from rapmat.storage.sqlite_store import SQLiteStore
from rapmat.tui.app import RapmatApp
from rapmat.tui.state import AppState
from rapmat.tui.tasks import BackgroundTask

_SIZES = [(121, 13), (80, 24)]


def _make_eval_rows():
    from ase.build import bulk

    from rapmat.core.entities import ResultRow, Structure

    data = [
        ("smoke-run/1", -5.00, -5.10, False),
        ("smoke-run/2", -4.50, -4.40, False),
        ("smoke-run/3", -4.50, -4.45, True),
    ]
    rows = []
    for idx, (sid, mlip, ref, dup) in enumerate(data, 1):
        struct = Structure(
            id=sid,
            status="relaxed",
            energy_per_atom=mlip,
            converged=True,
            duplicate=dup,
            final_atoms=bulk("Cu", cubic=True),
        )
        rows.append(
            ResultRow(
                structure=struct,
                index=idx,
                run_name="smoke-run",
                ref_energy_per_atom=ref,
            )
        )
    return rows


@pytest.fixture
def app_env():
    store = SQLiteStore(":memory:")
    store.create_study(
        "smoke-study",
        system="Al-O",
        domain="bulk",
        calculator="MATTERSIM",
        config={},
    )
    store.create_run(
        name="smoke-run",
        study_id="smoke-study",
        config={"formula": {"Al": 2, "O": 3}},
    )
    state = AppState(store=store)
    app = RapmatApp(state)
    state.active_study = "smoke-study"
    state.active_run = "smoke-run"
    yield state, app
    store.close()


def _make_screen(name: str, state, router):
    if name == "home":
        from rapmat.tui.screens.home import HomeScreen

        return HomeScreen(state, router)
    if name == "status":
        from rapmat.tui.screens.status import StatusScreen

        return StatusScreen(state, router)
    if name == "study_list":
        from rapmat.tui.screens.study_list import StudyListScreen

        return StudyListScreen(state, router)
    if name == "study_create":
        from rapmat.tui.screens.study_create import StudyCreateScreen

        return StudyCreateScreen(state, router)
    if name == "study_detail":
        from rapmat.tui.screens.study_detail import StudyDetailScreen

        return StudyDetailScreen(state, router)
    if name == "db_settings":
        from rapmat.tui.screens.db_settings import DbSettingsScreen

        return DbSettingsScreen(state, router)
    if name == "csp_search":
        from rapmat.tui.screens.csp_search import CSPSearchScreen

        return CSPSearchScreen(state, router)
    if name == "csp_resume":
        from rapmat.tui.screens.csp_resume import CSPResumeScreen

        return CSPResumeScreen(state, router)
    if name == "phonon":
        from rapmat.tui.screens.phonon import PhononDispersionScreen

        return PhononDispersionScreen(state, router)
    if name == "dedup":
        from rapmat.tui.screens.dedup import DedupScreen

        return DedupScreen(state, router)
    if name == "eval":
        from rapmat.tui.screens.eval import EvalScreen

        return EvalScreen(state, router, run_name="smoke-run")
    if name == "eval_results":
        from rapmat.tui.screens.eval import EvalResultsScreen

        return EvalResultsScreen(
            state,
            router,
            eval_rows=_make_eval_rows(),
            phonon_cutoff=-0.15,
            stable_only=False,
            run_name="smoke-run",
        )
    if name == "results":
        from rapmat.tui.screens.results import ResultsScreen

        return ResultsScreen(state, router)
    if name == "hull":
        from rapmat.tui.screens.hull import PhaseAnalysisScreen

        return PhaseAnalysisScreen(state, router)
    raise ValueError(name)


_ALL_SCREENS = [
    "home",
    "status",
    "study_list",
    "study_create",
    "study_detail",
    "db_settings",
    "csp_search",
    "csp_resume",
    "phonon",
    "dedup",
    "eval",
    "eval_results",
    "results",
    "hull",
]


_SINGLE_TASK_SCREENS = ["csp_search", "csp_resume", "phonon", "dedup", "eval"]


_RESULTS_TASK_SCREENS = ["results", "hull"]


@pytest.mark.parametrize("name", _ALL_SCREENS)
def test_screen_builds_renders_and_cycles(app_env, name):
    from rapmat.tui.router import Screen

    state, app = app_env
    screen = _make_screen(name, state, app._router)
    assert isinstance(screen, Screen)

    widget = screen.build()
    for size in _SIZES:
        canvas = widget.render(size, focus=True)
        assert canvas.cols() == size[0]
        assert canvas.rows() == size[1]

    screen.on_resume()
    screen.on_leave()


def _dummy_task(state) -> BackgroundTask:
    return BackgroundTask(fn=lambda prog: None, loop=state.loop)


@pytest.mark.parametrize("name", _SINGLE_TASK_SCREENS)
def test_on_leave_cancels_task(app_env, name):
    state, app = app_env
    screen = _make_screen(name, state, app._router)
    screen.build()

    task = _dummy_task(state)
    screen._task = task
    screen.on_leave()
    assert task._progress.cancelled is True


@pytest.mark.parametrize("name", _RESULTS_TASK_SCREENS)
def test_on_leave_cancels_results_tasks(app_env, name):
    state, app = app_env
    screen = _make_screen(name, state, app._router)
    screen.build()

    loading = _dummy_task(state)
    phonon = _dummy_task(state)
    screen._loading_task = loading
    screen._phonon_task = phonon
    screen.on_leave()
    assert loading._progress.cancelled is True
    assert phonon._progress.cancelled is True


def test_eval_results_filter_recomputes_metrics(app_env):
    """Hiding duplicates on the eval screen recomputes the metric."""
    state, app = app_env
    state.loop = None
    screen = _make_screen("eval_results", state, app._router)
    widget = screen.build()
    widget.render((121, 13), focus=True)

    assert screen._show_duplicate_col is True
    assert screen._ranking["n_structures"] == 3

    screen.keypress((), "d")
    assert screen._hide_duplicates is True
    assert screen._ranking["n_structures"] == 2

    widget.render((121, 13), focus=True)


class TestStudyListSearch:
    """Search keys must flow through the widget tree to the focused edit."""

    _SIZE = (80, 24)

    @pytest.fixture
    def search_env(self, app_env):
        state, app = app_env
        state.store.create_study(
            "zeta-study",
            system="Cu",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        state.invalidate_studies()
        screen = _make_screen("study_list", state, app._router)
        widget = screen.build()
        return screen, widget

    def _canvas_text(self, widget) -> str:
        canvas = widget.render(self._SIZE, focus=True)
        return b"\n".join(canvas.text).decode("utf-8", errors="replace")

    def test_typing_reaches_edit_and_filters(self, search_env):
        screen, widget = search_env
        assert "zeta-study" in self._canvas_text(widget)
        assert "smoke-study" in self._canvas_text(widget)

        screen.keypress((), "/")
        assert screen._searching is True

        widget.keypress(self._SIZE, "z")
        assert screen._search_edit.edit_text == "z"
        text = self._canvas_text(widget)
        assert "zeta-study" in text
        assert "smoke-study" not in text

    def test_cursor_keys_do_not_crash(self, search_env):
        screen, widget = search_env
        screen.keypress((), "/")
        widget.keypress(self._SIZE, "z")
        for key in ("left", "right", "home", "end", "backspace"):
            widget.keypress(self._SIZE, key)

    def test_q_is_consumed_by_edit(self, search_env):
        screen, widget = search_env
        screen.keypress((), "/")

        assert widget.keypress(self._SIZE, "q") is None
        assert screen._search_edit.edit_text == "q"

    def test_esc_exits_and_clears_filter(self, search_env):
        screen, widget = search_env
        screen.keypress((), "/")
        widget.keypress(self._SIZE, "z")
        assert "smoke-study" not in self._canvas_text(widget)

        widget.keypress(self._SIZE, "esc")
        assert screen._searching is False
        text = self._canvas_text(widget)
        assert "smoke-study" in text
        assert "zeta-study" in text

    def test_enter_exits_keeping_filter(self, search_env):
        screen, widget = search_env
        screen.keypress((), "/")
        widget.keypress(self._SIZE, "z")

        widget.keypress(self._SIZE, "enter")
        assert screen._searching is False
        text = self._canvas_text(widget)
        assert "zeta-study" in text
        assert "smoke-study" not in text


class TestResultsDialogs:
    """The results screen modal dialogs open, render and close on Esc."""

    _SIZE = (121, 38)

    @pytest.fixture
    def results_env(self, app_env):
        from ase.build import bulk

        from conftest import add_relaxed_structure

        state, app = app_env
        add_relaxed_structure(
            state.store, "smoke-run", bulk("Al", "fcc", a=4.05), -3.0, "smoke-run/1"
        )
        state.loop = None
        screen = _make_screen("results", state, app._router)
        widget = screen.build()
        widget.render(self._SIZE, focus=True)
        return screen, widget

    def _open_and_close(self, screen, widget, key, dialog_cls):
        base_body = screen._main_frame.body
        screen.keypress((), key)
        assert isinstance(screen._main_frame.body, dialog_cls)
        widget.render(self._SIZE, focus=True)

        widget.keypress(self._SIZE, "esc")
        assert screen._main_frame.body is base_body
        widget.render(self._SIZE, focus=True)

    def test_options_dialog(self, results_env):
        from rapmat.tui.widgets.dialog import FormDialog

        screen, widget = results_env
        self._open_and_close(screen, widget, "o", FormDialog)

    def test_thickness_dialog(self, results_env):
        from rapmat.tui.widgets.dialog import FormDialog

        screen, widget = results_env
        screen._show_thickness = True
        self._open_and_close(screen, widget, "t", FormDialog)

    def test_phonon_dialog(self, results_env):
        from rapmat.tui.widgets.dialog import FormDialog

        screen, widget = results_env
        self._open_and_close(screen, widget, "p", FormDialog)

    def test_save_dialog(self, results_env):
        from rapmat.tui.screens.base_results import _SaveDialog

        screen, widget = results_env
        self._open_and_close(screen, widget, "s", _SaveDialog)
