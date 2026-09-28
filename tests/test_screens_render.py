import sys

import pytest

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
    if name == "calc_settings":
        from rapmat.tui.screens.calc_settings import CalcSettingsScreen

        return CalcSettingsScreen(state, router)
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
    if name == "structure_view":
        from rapmat.tui.screens.structure_view import StructureViewScreen

        return StructureViewScreen(state, router, _make_eval_rows(), 0)
    raise ValueError(name)


_ALL_SCREENS = [
    "home",
    "status",
    "study_list",
    "study_create",
    "study_detail",
    "db_settings",
    "calc_settings",
    "csp_search",
    "csp_resume",
    "phonon",
    "dedup",
    "eval",
    "eval_results",
    "results",
    "hull",
    "structure_view",
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


class TestResultsStructureView:

    _SIZE = (121, 38)

    @pytest.fixture
    def results_env(self, app_env):
        from ase.build import bulk

        from conftest import add_relaxed_structure

        state, app = app_env
        for idx, a in enumerate((4.05, 4.10, 4.15)):
            add_relaxed_structure(
                state.store, "smoke-run", bulk("Al", "fcc", a=a, cubic=True),
                -3.0 - idx * 0.1, f"smoke-run/{idx + 1}",
            )
        state.loop = None
        screen = _make_screen("results", state, app._router)
        app._router.push(screen)
        widget = app._router._stack[-1][1]
        widget.render(self._SIZE, focus=True)
        return state, app, screen, widget

    def _open(self, app, screen):
        from rapmat.tui.screens.structure_view import StructureViewScreen

        screen.keypress((), "enter")
        view = app._router.current
        assert isinstance(view, StructureViewScreen)
        app._router._stack[-1][1].render(self._SIZE, focus=True)
        return view

    def test_enter_opens_the_viewer(self, results_env):
        _state, app, screen, _widget = results_env
        view = self._open(app, screen)
        assert view._result is screen._table.get_focused_row()

    def test_enter_on_a_focused_row_opens_the_viewer(self, results_env):
        from rapmat.tui.screens.structure_view import StructureViewScreen

        _state, app, screen, widget = results_env
        row = screen._table._walker[screen._table._listbox.focus_position]
        row.keypress((80,), "enter")
        assert isinstance(app._router.current, StructureViewScreen)
        app._router._stack[-1][1].render(self._SIZE, focus=True)

    def test_three_is_no_longer_bound(self, results_env):
        _state, app, screen, _widget = results_env
        depth = app._router.depth
        screen.keypress((), "3")
        assert app._router.depth == depth

    def test_viewer_receives_the_whole_displayed_list(self, results_env):
        _state, app, screen, _widget = results_env
        view = self._open(app, screen)
        assert view._results == screen._get_display_results()

    def test_stepping_moves_the_table_focus(self, results_env):
        _state, app, screen, _widget = results_env
        view = self._open(app, screen)
        view.keypress((), "d")
        assert screen._table.get_focused_row() is view._result

    def test_esc_returns_to_the_last_structure_viewed(self, results_env):
        _state, app, screen, widget = results_env
        started_on = screen._table.get_focused_row()

        view = self._open(app, screen)
        view.keypress((), "d")
        view.keypress((), "d")
        last_seen = view._result

        app._router.pop()
        widget.render(self._SIZE, focus=True)

        assert app._router.current is screen
        assert screen._table.get_focused_row() is last_seen
        assert last_seen is not started_on

    def test_rows_without_atoms_are_skipped(self, results_env):
        _state, app, screen, _widget = results_env
        display = screen._get_display_results()
        display[1].structure.final_atoms = None
        display[1].structure.initial_atoms = None

        view = self._open(app, screen)
        assert display[1] not in view._results
        assert len(view._results) == 2

    def test_focused_row_without_atoms_reports_instead_of_opening(self, results_env):
        _state, app, screen, _widget = results_env
        focused = screen._table.get_focused_row()
        focused.structure.final_atoms = None
        focused.structure.initial_atoms = None

        depth = app._router.depth
        screen.keypress((), "enter")
        assert app._router.depth == depth


class _RunningThread:
    def is_alive(self) -> bool:
        return True


class TestDbSettingsCompact:
    @staticmethod
    def _screen(app_env):
        state, app = app_env
        screen = _make_screen("db_settings", state, app._router)
        screen.build()
        return state, screen

    @staticmethod
    def _status(screen) -> str:
        return screen._status_text.get_text()[0]

    def test_usage_section_reports_size(self, app_env):
        _, screen = self._screen(app_env)

        assert "Size:" in screen._usage_text.get_text()[0]
        assert "Reclaimable:" in screen._usage_text.get_text()[0]

    def test_compact_reports_nothing_to_reclaim(self, app_env):
        state, screen = self._screen(app_env)
        state.store.vacuum()

        screen._on_compact(None)

        assert "Nothing to reclaim" in self._status(screen)

    def test_compact_asks_before_rebuilding(self, app_env):
        from test_vacuum import _bloat

        state, screen = self._screen(app_env)
        _bloat(state.store._engine)
        body_before = screen._frame.body

        screen._on_compact(None)

        assert screen._frame.body is not body_before

    def test_compact_reports_what_it_freed(self, app_env):
        from test_vacuum import _bloat

        state, screen = self._screen(app_env)
        _bloat(state.store._engine)
        before = state.store.storage_stats().total_bytes

        state.store.vacuum()
        screen._on_compact_done(before)

        status = self._status(screen)
        assert "Compacted:" in status
        assert "freed" in status
        assert state.store.storage_stats().free_pages == 0

    def test_compact_surfaces_errors(self, app_env):
        _, screen = self._screen(app_env)

        screen._on_compact_error("disk full")

        assert "Compact failed: disk full" in self._status(screen)

    def test_compact_refuses_while_one_is_running(self, app_env):
        state, screen = self._screen(app_env)
        screen._task = _dummy_task(state)
        screen._task._progress.finished = False
        screen._task._thread = _RunningThread()

        screen._on_compact(None)

        assert "Already compacting" in self._status(screen)


def _widgets(root):
    import urwid

    stack, seen = [root], set()
    while stack:
        w = stack.pop()
        if id(w) in seen or not isinstance(w, urwid.Widget):
            continue
        seen.add(id(w))
        yield w
        for attr in ("_original_widget", "_w", "top_w", "bottom_w", "_body"):
            child = getattr(w, attr, None)
            if isinstance(child, urwid.Widget):
                stack.append(child)
        contents = getattr(w, "contents", None)
        if isinstance(contents, (list, urwid.MonitoredList)):
            for item in contents:
                stack.append(item[0] if isinstance(item, tuple) else item)


def _press(root, label):
    import urwid

    button = next(
        w for w in _widgets(root) if isinstance(w, urwid.Button) and w.label == label
    )
    button._emit("click")


def _text_of(root) -> str:
    import urwid

    return " ".join(
        w.text for w in _widgets(root)
        if isinstance(w, urwid.Text) and not isinstance(w, urwid.Edit)
    )


class _TaskLoop:
    def __init__(self):
        self.alarms = []

    def set_alarm_in(self, _delay, callback, data=None):
        self.alarms.append((callback, data))

    def draw_screen(self):
        pass

    def finish(self, task):
        task._thread.join(timeout=30)
        while self.alarms:
            callback, data = self.alarms.pop(0)
            callback(self, data)


class TestPhononConsistency:
    _SIZE = (121, 38)

    @staticmethod
    def _stored(**overrides):
        from rapmat.core.phonon_settings import PhononSettings

        fields = dict(
            calculator="MATTERSIM",
            calculator_settings="toml",
            supercell=(2, 2, 2),
            mesh=(10, 10, 10),
            displacement=0.02,
            symprec=1e-3,
            reduce_primitive=False,
        )
        fields.update(overrides)
        return PhononSettings(**fields)

    @pytest.fixture
    def env(self, app_env):
        from ase.build import bulk

        from conftest import add_relaxed_structure

        state, app = app_env
        for i, a in enumerate((4.05, 4.1), start=1):
            add_relaxed_structure(
                state.store, "smoke-run", bulk("Al", "fcc", a=a), -3.0 + i * 0.01,
                f"smoke-run/{i}",
            )
        state.loop = None
        return state, app

    def _screen(self, state, app, name="results"):
        screen = _make_screen(name, state, app._router)
        widget = screen.build()
        widget.render(self._SIZE, focus=True)
        return screen

    def _save(self, store, sid, settings, freq=-0.01):
        run = sid.rpartition("/")[0]
        store.save_phonon_result(sid, run, freq, settings=settings.model_dump(mode="json"))

    def _open(self, screen):
        from rapmat.tui.widgets.dialog import FormDialog

        screen.keypress((), "p")
        dialog = screen._main_frame.body
        assert isinstance(dialog, FormDialog)
        return dialog

    def _fake_compute(self, monkeypatch):
        from rapmat.core import phonon_stability
        from rapmat.core.phonon_stability import PhononRunSummary

        calls = []

        def fake(**kwargs):
            calls.append(kwargs)
            return PhononRunSummary(targets=1, computed=1)

        monkeypatch.setattr(
            phonon_stability, "compute_dynamical_stability_for_results", fake
        )
        return calls

    def test_prefilled_from_stored_settings(self, env):
        state, app = env
        self._save(state.store, "smoke-run/1", self._stored(calculator="UPET"))
        vals = self._open(self._screen(state, app))._form.get_values()

        assert vals["calculator"] == "UPET"
        assert vals["phonon_supercell"] == (2, 2, 2)
        assert vals["phonon_mesh"] == (10, 10, 10)
        assert vals["phonon_displacement"] == 0.02
        assert vals["reduce_prim"] is False

    def test_defaults_follow_the_study(self, env):
        from ase.build import bulk

        from conftest import add_relaxed_structure

        state, app = env
        state.store.create_study("mono", "Al", "monolayer", "NEQUIP-OAML")
        state.store.create_run(
            name="mono-run", study_id="mono", config={"formula": {"Al": 1}}
        )
        add_relaxed_structure(
            state.store, "mono-run", bulk("Al", "fcc", a=4.05), -3.0, "mono-run/1"
        )
        state.active_run = "mono-run"
        vals = self._open(self._screen(state, app))._form.get_values()

        assert vals["calculator"] == "NEQUIP-OAML"
        assert vals["phonon_supercell"] == (3, 3, 1)
        assert vals["phonon_mesh"] == (20, 20, 1)

    def test_cutoff_is_the_study_one(self, env):
        state, app = env
        state.store.set_run_config_value("smoke-run", "phonon_cutoff", -0.3)
        screen = self._screen(state, app)
        assert screen._phonon_cutoff == -0.3
        assert self._open(screen)._form.get_values()["phonon_cutoff"] == -0.3

        state.store.set_study_config_value("smoke-study", "phonon_cutoff", -0.1)
        assert self._screen(state, app)._phonon_cutoff == -0.1

    def test_different_settings_ask_before_deleting(self, env, monkeypatch):
        state, app = env
        calls = self._fake_compute(monkeypatch)
        self._save(state.store, "smoke-run/1", self._stored())
        screen = self._screen(state, app)
        dialog = self._open(screen)
        dialog._form.set_values({"phonon_supercell": (4, 4, 4)})
        state.loop = _TaskLoop()

        _press(dialog, "Run Phonons")
        confirm = screen._main_frame.body
        assert confirm is not dialog
        assert (
            "1 stored phonon result in 1 run of study 'smoke-study' was computed "
            "with different settings (supercell) and will be deleted"
        ) in _text_of(confirm)

        _press(confirm, "No")
        assert screen._main_frame.body is dialog
        assert dialog._form.get_values()["phonon_supercell"] == (4, 4, 4)
        assert state.store.get_phonon_settings(["smoke-run"])
        assert calls == []

        _press(dialog, "Run Phonons")
        _press(screen._main_frame.body, "Yes")
        state.loop.finish(screen._phonon_task)

        assert state.store.get_phonon_settings(["smoke-run"]) == {}
        assert calls[0]["settings"].supercell == (4, 4, 4)
        assert calls[0]["skip_ids"] == set()

    def test_same_settings_keep_the_stored_results(self, env, monkeypatch):
        state, app = env
        calls = self._fake_compute(monkeypatch)
        self._save(state.store, "smoke-run/1", self._stored())
        screen = self._screen(state, app)
        dialog = self._open(screen)
        dialog._form.set_values({"phonon_cutoff": -0.05})
        state.loop = _TaskLoop()

        _press(dialog, "Run Phonons")
        state.loop.finish(screen._phonon_task)

        assert calls[0]["skip_ids"] == {"smoke-run/1"}
        assert "smoke-run/1" in state.store.get_phonon_settings(["smoke-run"])
        assert state.store.get_study("smoke-study").config["phonon_cutoff"] == -0.05
        assert screen._phonon_cutoff == -0.05
        assert "Phonons for 1 structure: 1 computed." in screen._app_message

    def test_one_calculation_at_a_time(self, env):
        from rapmat.tui.widgets.dialog import FormDialog

        state, app = env
        screen = self._screen(state, app)
        screen._phonon_task = _dummy_task(state)
        screen._phonon_task._thread = _RunningThread()

        screen.keypress((), "p")
        assert not isinstance(screen._main_frame.body, FormDialog)
        assert screen._app_message == "A phonon calculation is already running."

    def test_clear_counts_what_it_deletes(self, env):
        state, app = env
        for sid in ("smoke-run/1", "smoke-run/2"):
            self._save(state.store, sid, self._stored())
        state.store.set_study_config_value("smoke-study", "phonon_cutoff", -0.2)
        screen = self._screen(state, app)
        _press(self._open(screen), "Clear results")

        confirm = screen._main_frame.body
        assert "Delete 2 stored phonon results in 1 run?" in _text_of(confirm)
        _press(confirm, "Yes")

        assert state.store.get_phonon_settings(["smoke-run"]) == {}
        assert screen._phonon_cutoff == -0.2

    def test_phase_analysis_saves_per_run_and_knows_monolayers(
        self, app_env, monkeypatch
    ):
        from ase.build import bulk

        from conftest import add_relaxed_structure
        from rapmat.core import phonon_stability as ps

        state, app = app_env
        store = state.store
        store.create_study("cu", "Cu", "monolayer", "MATTERSIM")
        for run in ("cu-a", "cu-b"):
            store.create_run(name=run, study_id="cu", config={"formula": {"Cu": 1}})
            add_relaxed_structure(
                store, run, bulk("Cu", "fcc", a=3.6), -3.0, f"{run}/1"
            )
        state.active_study = "cu"
        state.loop = None

        providers = []

        class _Provider:
            def __init__(self, *args, **kwargs):
                providers.append(kwargs)

            def __call__(self, _atoms):
                return object()

        monkeypatch.setattr(ps, "CalculatorProvider", _Provider)
        monkeypatch.setattr(
            ps, "calculate_phonons_with_freq", lambda *a, **k: (object(), -0.02)
        )
        monkeypatch.setattr(ps, "serialize_phonons", lambda _p: "BLOB")

        screen = self._screen(state, app, name="hull")
        dialog = self._open(screen)
        assert dialog._form.get_values()["phonon_supercell"] == (3, 3, 1)
        state.loop = _TaskLoop()
        _press(dialog, "Run Phonons")
        state.loop.finish(screen._phonon_task)

        assert providers[0]["monolayer"] is True
        stored = store.get_phonon_settings(["cu-a", "cu-b"])
        assert {sid: run for sid, (run, _t) in stored.items()} == {
            "cu-a/1": "cu-a",
            "cu-b/1": "cu-b",
        }

        state.loop = None
        screen._start_async_fetch()
        assert [r.min_phonon_freq for r in screen._results] == [-0.02, -0.02]


class TestPhononCancel:
    _SIZE = (121, 38)

    def test_esc_cancels_and_stays_on_the_view(self, app_env, monkeypatch):
        import threading

        from ase.build import bulk

        from conftest import add_relaxed_structure
        from rapmat.core import phonon_stability

        state, app = app_env
        add_relaxed_structure(
            state.store, "smoke-run", bulk("Al", "fcc", a=4.05), -3.0, "smoke-run/1"
        )
        state.loop = None
        started = threading.Event()

        def endless(**kwargs):
            started.set()
            while True:
                kwargs["progress_callback"](0, 1, "working")

        monkeypatch.setattr(
            phonon_stability, "compute_dynamical_stability_for_results", endless
        )

        screen = _make_screen("results", state, app._router)
        app._router.push(screen)
        screen._main_frame.render(self._SIZE, focus=True)
        depth = app._router.depth
        screen.keypress((), "p")
        state.loop = _TaskLoop()
        _press(screen._main_frame.body, "Run Phonons")
        assert started.wait(10)

        assert screen.esc_label() == "Cancel"
        assert screen.keypress((), "esc") is None
        assert app._router.depth == depth
        assert screen._phonon_task.cancelled

        state.loop.finish(screen._phonon_task)
        assert app._router.depth == depth
        assert screen._app_message == (
            "Phonon calculation cancelled. Completed results were kept."
        )
        assert screen.esc_label() == "Back"
        assert screen._body_pile.contents[-1][0] is screen._details_panel


class TestPhononScreenCancel:
    @pytest.fixture
    def run_worker(self, app_env, tmp_path, monkeypatch):
        from ase.build import bulk
        from ase.calculators.emt import EMT
        from ase.io import write

        from rapmat.calculators import factory
        from rapmat.tui.tasks import TaskProgress

        class _Provider:
            def __init__(self, *args, **kwargs):
                pass

            def set_calc_label(self, _label):
                pass

            def __call__(self, _atoms):
                return EMT()

        monkeypatch.setattr(factory, "CalculatorProvider", _Provider)

        cif = tmp_path / "cu.cif"
        atoms = bulk("Cu", "fcc", a=3.7)
        atoms.rattle(0.05, seed=1)
        write(cif, atoms)

        state, app = app_env
        screen = _make_screen("phonon", state, app._router)
        screen.build()

        def run(**overrides):
            vals = screen._form.get_values()
            vals.update(
                structure_file=str(cif),
                calculator="MATTERSIM",
                phonon_supercell=(2, 2, 2),
                phonon_mesh=(4, 4, 4),
                plot_file=str(tmp_path / "plot.png"),
                **overrides,
            )
            progress = TaskProgress()
            progress.cancelled = True
            screen._worker(progress, vals)

        return run

    def test_pre_relax_stops_early(self, run_worker, monkeypatch):
        from rapmat.core import relaxation

        steps = []
        real_relax = relaxation.structure_relax

        def counting_relax(*args, progress_callback=None, **kwargs):
            def _cb(step, max_steps, msg):
                steps.append(step)
                progress_callback(step, max_steps, msg)

            return real_relax(*args, progress_callback=_cb, **kwargs)

        monkeypatch.setattr(relaxation, "structure_relax", counting_relax)

        with pytest.raises(KeyboardInterrupt):
            run_worker(prerelax=True, steps_max=500, force_conv_crit=1e-6)
        assert steps == [1]

    def test_phonons_stop_at_the_next_displacement(self, run_worker, monkeypatch):
        from rapmat.core import phonon

        seen = []
        real = phonon.structure_calculate_phonons

        def counting(*args, progress_callback=None, **kwargs):
            def _cb(current, total, message=""):
                seen.append(message)
                progress_callback(current, total, message)

            return real(*args, progress_callback=_cb, **kwargs)

        monkeypatch.setattr(phonon, "structure_calculate_phonons", counting)

        with pytest.raises(KeyboardInterrupt):
            run_worker(prerelax=False)
        assert len(seen) == 1
