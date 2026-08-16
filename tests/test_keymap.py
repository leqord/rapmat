"""Tests for the keymap."""

import sys

import pytest
import urwid

if sys.platform == "win32":
    sys.modules.pop("urwid.display.curses", None)

from rapmat.tui.keymap import KeyBinding, dispatch, footer_hints
from rapmat.tui.widgets.status_bar import StatusBar


def _binding(keys, label="X", priority=50, enabled=None, case_sensitive=False,
             on_fire=None):
    fired = []

    def _action():
        fired.append(True)
        if on_fire:
            on_fire()

    b = KeyBinding(
        keys=keys, label=label, action=_action, enabled=enabled,
        priority=priority, case_sensitive=case_sensitive,
    )
    return b, fired


# ---------------------------------------------------------------------------
#  Dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_letter_case_insensitive(self):
        b, fired = _binding(("u",))
        assert dispatch([b], "U") is True
        assert dispatch([b], "u") is True
        assert len(fired) == 2

    def test_case_sensitive_letter(self):
        b, fired = _binding(("s",), case_sensitive=True)
        assert dispatch([b], "S") is False
        assert dispatch([b], "s") is True
        assert len(fired) == 1

    def test_named_keys_exact(self):
        b, fired = _binding(("f5",))
        assert dispatch([b], "f5") is True
        assert dispatch([b], "f") is False
        d, d_fired = _binding(("delete",))
        assert dispatch([d], "delete") is True

    def test_disabled_falls_through(self):
        b, fired = _binding(("c",), enabled=lambda: False)
        assert dispatch([b], "c") is False
        assert fired == []

    def test_unbound_key_falls_through(self):
        b, fired = _binding(("u",))
        assert dispatch([b], "z") is False

    def test_mouse_event_tuple_ignored(self):
        b, fired = _binding(("u",))
        assert dispatch([b], ("mouse press", 4, 10, 10)) is False
        assert fired == []


# ---------------------------------------------------------------------------
#  Footer derivation
# ---------------------------------------------------------------------------


class TestFooterHints:
    def test_priority_order_and_enabled_filter(self):
        b1, _ = _binding(("s",), label="Save", priority=20)
        b2, _ = _binding(("/",), label="Search", priority=10)
        b3, _ = _binding(("c",), label="Cutoff", priority=15,
                         enabled=lambda: False)
        hints = footer_hints([b1, b2, b3])
        assert hints == [("/", "Search", 10), ("s", "Save", 20)]

    def test_dynamic_label(self):
        hidden = [True]
        b = KeyBinding(
            keys=("e",),
            label=lambda: "Show Excl" if hidden[0] else "Hide Excl",
            action=lambda: None,
        )
        assert footer_hints([b])[0][1] == "Show Excl"
        hidden[0] = False
        assert footer_hints([b])[0][1] == "Hide Excl"

    def test_key_display_names(self):
        b = KeyBinding(keys=("delete",), label="Remove", action=lambda: None)
        f = KeyBinding(keys=("f5",), label="Run", action=lambda: None)
        assert footer_hints([b, f]) == [
            ("Del", "Remove", 50), ("F5", "Run", 50),
        ]


# ---------------------------------------------------------------------------
#  StatusBar width fitting
# ---------------------------------------------------------------------------


def _bar_text(bar: StatusBar, cols: int) -> str:
    canvas = bar.render((cols,))
    return b"".join(canvas.text).decode()


class TestStatusBarFitting:
    def test_all_hints_fit_wide(self):
        bar = StatusBar()
        bar.set_hints([("u", "Uncv", 30), ("s", "Save", 20), ("?", "Help", 0)])
        text = _bar_text(bar, 120)
        assert "[u] Uncv" in text
        assert "[s] Save" in text
        assert "[?] Help" in text

    def test_low_priority_dropped_whole_when_narrow(self):
        bar = StatusBar()
        hints = [
            ("?", "Help", 0),
            ("Esc", "Back", 0),
            ("/", "Search", 10),
            ("s", "Save", 20),
            ("u", "Unconverged", 30),
            ("t", "Thickness filter", 60),
        ]
        bar.set_hints(hints)
        text = _bar_text(bar, 40)

        assert "[?] Help" in text
        assert "[Esc] Back" in text

        assert "Thickness" not in text
        assert "[t]" not in text

    def test_message_takes_space_only_when_set(self):
        bar = StatusBar()
        bar.set_hints([("?", "Help", 0)])
        bar.set_message("saved!")
        text = _bar_text(bar, 40)
        assert text.rstrip().endswith("saved!")
        bar.clear_message()
        assert "saved!" not in _bar_text(bar, 40)

    def test_refit_on_resize(self):
        bar = StatusBar()
        bar.set_hints([("?", "Help", 0), ("t", "Thickness filter", 60)])
        wide = _bar_text(bar, 120)
        assert "[t] Thickness filter" in wide
        narrow = _bar_text(bar, 14)
        assert "[t]" not in narrow
        assert "[?] Help" in narrow


# ---------------------------------------------------------------------------
#  ScreenBase integration
# ---------------------------------------------------------------------------


def _make_app():
    from rapmat.storage.sqlite_store import SQLiteStore
    from rapmat.tui.app import RapmatApp
    from rapmat.tui.state import AppState

    store = SQLiteStore(":memory:")
    state = AppState(store=store)
    return RapmatApp(state, startup_error=None), state


class TestScreenBaseFooter:
    def test_home_footer_has_pinned_help_no_esc(self):
        app, state = _make_app()
        home = app._router.current
        home.refresh_footer()
        hints = state.status_bar._hints
        keys = [h[0] for h in hints]
        assert "?" in keys
        assert "Esc" not in keys
        assert "n" in keys and "q" in keys

    def test_screen_keypress_dispatches_bindings(self):
        app, state = _make_app()
        home = app._router.current
        assert home.keypress((), "i") is None
        assert app._router.current.title == "Status"
        assert home.keypress((), "z") == "z"

    def test_global_input_ignores_mouse_events(self):
        app, _state = _make_app()
        app._global_input(("mouse press", 4, 10, 10))  # must not raise anything


class _FakeRunningTask:
    is_running = True

    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class TestScreenBaseEsc:
    def test_esc_cancels_running_task(self):
        app, _state = _make_app()
        screen = app._router.current

        cancelling = []

        class _Panel:
            def set_cancelling(self):
                cancelling.append(True)

        task = _FakeRunningTask()
        screen._task = task
        screen._progress_panel = _Panel()

        assert screen.keypress((), "esc") is None
        assert task.cancelled is True
        assert cancelling == [True]

    def test_esc_without_task_falls_through(self):
        app, _state = _make_app()
        screen = app._router.current
        assert screen.keypress((), "esc") == "esc"

    def test_esc_label_reflects_running_task(self):
        from rapmat.tui.screens.base import ScreenBase

        app, state = _make_app()
        screen = ScreenBase(state, app._router)
        assert screen.esc_label() == "Back"
        screen._task = _FakeRunningTask()
        assert screen.esc_label() == "Cancel"


def _host_screen(app, state):
    from rapmat.tui.screens.base import ScreenBase

    class _HostScreen(ScreenBase):
        def __init__(self, st, router):
            super().__init__(st, router)
            self.body = urwid.SolidFill(" ")

        def _dialog_host_get(self):
            return self.body

        def _dialog_host_set(self, widget):
            self.body = widget

    return _HostScreen(state, app._router)


class TestScreenBaseDialogs:
    def test_show_dialog_opens_and_close_restores(self):
        app, state = _make_app()
        screen = _host_screen(app, state)
        base = screen.body
        sentinel = urwid.SolidFill("X")

        close = screen.show_dialog(lambda parent, close: sentinel)
        assert screen.body is sentinel
        close()
        assert screen.body is base

    def test_show_dialog_no_host_returns_none(self):
        app, state = _make_app()
        screen = _host_screen(app, state)
        screen.body = None
        assert screen.show_dialog(lambda parent, close: parent) is None

    def test_confirm_dialog_cancel_restores_without_firing(self):
        from rapmat.tui.widgets.dialog import ModalDialog

        app, state = _make_app()
        screen = _host_screen(app, state)
        base = screen.body
        fired = []

        screen.confirm_dialog("T", "M", lambda: fired.append(True))
        assert isinstance(screen.body, ModalDialog)
        screen.body._esc_handler()
        assert screen.body is base
        assert fired == []

    def test_confirm_dialog_yes_fires_and_restores(self):
        app, state = _make_app()
        screen = _host_screen(app, state)
        base = screen.body
        fired = []

        screen.confirm_dialog("T", "M", lambda: fired.append(True))
        dialog = screen.body
        dialog.render((60, 10), focus=True)
        dialog.keypress((60, 10), "enter")
        assert fired == [True]
        assert screen.body is base


# ---------------------------------------------------------------------------
#  Help overlay
# ---------------------------------------------------------------------------


class TestHelpOverlay:
    def test_question_mark_opens_and_esc_closes(self):
        from rapmat.tui.widgets.dialog import ModalDialog

        app, _state = _make_app()
        saved_body = app._frame.body
        app._global_input("?")
        assert isinstance(app._frame.body, ModalDialog)
        app._frame.body.keypress((80, 24), "esc")
        assert app._frame.body is saved_body

    def test_ignored_while_dialog_open(self):
        from rapmat.tui.widgets.dialog import ModalDialog

        app, _state = _make_app()
        app._global_input("?")
        dlg = app._frame.body
        assert isinstance(dlg, ModalDialog)
        app._global_input("?")
        assert app._frame.body is dlg

    def test_lists_current_screen_bindings(self):
        app, _state = _make_app()
        app._global_input("?")
        canvas = app._frame.body.render((100, 30))
        text = b"\n".join(canvas.text).decode()
        assert "New Run" in text or "Start a new CSP run" in text
        assert "This help" in text


# ---------------------------------------------------------------------------
#  Quit guard
# ---------------------------------------------------------------------------


class TestQuitGuard:
    def test_quit_exits_when_idle(self):
        app, _state = _make_app()
        with pytest.raises(urwid.ExitMainLoop):
            app._global_input("q")

    def test_quit_confirms_when_task_running(self):
        from rapmat.tui.widgets.dialog import ModalDialog

        app, _state = _make_app()

        class _FakeTask:
            is_running = True

            def cancel(self):
                pass

        screen = app._router.current
        screen._task = _FakeTask()
        app._global_input("q")
        assert isinstance(app._frame.body, ModalDialog)

        saved = app._frame.body
        saved.keypress((80, 24), "esc")
        assert not isinstance(app._frame.body, ModalDialog)

    def test_screen_keypress_runs_before_quit(self):
        app, _state = _make_app()
        home = app._router.current

        with pytest.raises(urwid.ExitMainLoop):
            home.keypress((), "q")


class TestHelpOverlayEnter:

    def _help_rows(self, screen, app):
        rows = [
            (b.key_display(), b.help_text())
            for b in screen.bindings() if b.is_enabled()
        ]
        bound = {k for k, _d in rows}
        if "Enter" not in bound:
            rows.append(("Enter", "Select"))
        return rows

    def test_results_help_describes_enter_once(self):
        from rapmat.tui.screens.results import ResultsScreen

        app, state = _make_app()
        rows = self._help_rows(ResultsScreen(state, app._router), app)
        enter = [d for k, d in rows if k == "Enter"]
        assert len(enter) == 1
        assert "3D" in enter[0] or "viewer" in enter[0]

    def test_other_screens_keep_the_generic_enter_row(self):
        app, _state = _make_app()
        rows = self._help_rows(app._router.current, app)
        assert ("Enter", "Select") in rows

    def test_app_help_overlay_renders_without_duplicate_enter(self):
        app, _state = _make_app()
        app._show_help()
        body = app._frame.body
        canvas = body.render((100, 30), focus=True)
        text = b"\n".join(canvas.text).decode()
        assert text.count("Enter") == 1


class TestBindingKeysAreUnique:

    def _screens(self):
        from rapmat.tui.screens.hull import PhaseAnalysisScreen
        from rapmat.tui.screens.results import ResultsScreen
        from rapmat.tui.screens.structure_view import StructureViewScreen

        app, state = _make_app()
        return [
            ResultsScreen(state, app._router),
            PhaseAnalysisScreen(state, app._router),
            StructureViewScreen(state, app._router, [], 0),
        ]

    def test_no_screen_binds_a_key_twice(self):
        for screen in self._screens():
            seen: dict[str, str] = {}
            for binding in screen.bindings():
                for key in binding.keys:
                    key = key.lower()
                    assert key not in seen, (
                        f"{type(screen).__name__} binds {key!r} to both "
                        f"{seen[key]!r} and {binding.label_text()!r}"
                    )
                    seen[key] = binding.label_text()

    def test_results_screens_expose_the_3d_view(self):
        for screen in self._screens()[:2]:
            keys = {k for b in screen.bindings() for k in b.keys}
            assert "enter" in keys, type(screen).__name__

    def test_results_screens_do_not_shadow_global_keys(self):
        for screen in self._screens()[:2]:
            keys = {k.lower() for b in screen.bindings() for k in b.keys}
            assert "?" not in keys
            assert "esc" not in keys
