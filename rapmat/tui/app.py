import sys
import threading

import urwid

# NOTE: fix urwid on win32
if sys.platform == "win32":
    sys.modules.pop("urwid.display.curses", None)

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState
from rapmat.tui.theme import PALETTE, apply_color_depth
from rapmat.tui.widgets.status_bar import StatusBar
from rapmat.utils import hardware

# ------------------------------------------------------------------ #
#  Application
# ------------------------------------------------------------------ #


class RapmatApp:
    def __init__(
        self,
        state: "AppState",
        startup_error: Exception | None = None,
    ) -> None:
        self._state = state
        self._startup_error = startup_error

        self._breadcrumb = urwid.Text(" Rapmat TUI", wrap="clip")

        self._hw_status = urwid.Text(
            hardware.header_markup(hardware.cached()), align="right"
        )

        header_cols = urwid.Columns(
            [
                ("weight", 1, self._breadcrumb),
                ("pack", self._hw_status),
            ]
        )
        header = urwid.AttrMap(header_cols, "header")

        self._status_bar = StatusBar()
        self._state.status_bar = self._status_bar

        self._frame = urwid.Frame(
            body=urwid.SolidFill(" "),
            header=header,
            footer=self._status_bar,
        )

        self._router = ScreenRouter(self._frame, self._breadcrumb)

        self._loop = urwid.MainLoop(
            self._frame,
            PALETTE,
            unhandled_input=self._global_input,
            pop_ups=True,
        )
        
        state.color_depth = apply_color_depth(self._loop.screen)

        state.loop = self._loop
        state.request_quit = self._request_quit

        from rapmat.tui.screens.home import HomeScreen

        self._router.push(HomeScreen(self._state, self._router))

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        if self._startup_error is not None:
            self._loop.set_alarm_in(0, self._show_startup_error)
        self._start_hw_detection()
        try:
            self._loop.run()
        finally:
            try:
                self._state.store.close()
            except Exception:
                pass

    def _start_hw_detection(self) -> None:
        """Detect torch device on a separate thread."""
        threading.Thread(target=hardware.detect, daemon=True).start()
        self._loop.set_alarm_in(0.1, self._poll_hw_status)

    def _poll_hw_status(self, loop, _data) -> None:
        info = hardware.cached()
        if info is None:
            loop.set_alarm_in(0.2, self._poll_hw_status)
            return
        self._hw_status.set_text(hardware.header_markup(info))

        refresh = getattr(self._router.current, "refresh_hw_label", None)
        if callable(refresh):
            refresh()
        loop.draw_screen()

    def _show_startup_error(self, _loop, _data) -> None:
        from rapmat.tui.widgets.dialog import ModalDialog

        exc = self._startup_error
        err_type = type(exc).__name__
        err_msg = str(exc) or "(no details)"
        message = (
            f"Could not connect to the configured database.\n\n"
            f"  {err_type}: {err_msg}\n\n"
            f"The TUI is running with a temporary in-memory store.\n"
            f"Data will NOT be persisted until the connection is fixed.\n\n"
            f"Open DB Settings to reconfigure, or Continue to proceed."
        )

        saved_body = self._frame.body

        def _open_settings():
            self._frame.body = saved_body
            from rapmat.tui.screens.db_settings import DbSettingsScreen

            self._router.push(DbSettingsScreen(self._state, self._router))

        def _continue():
            self._frame.body = saved_body

        dlg = ModalDialog.error(
            title="Database Connection Error",
            message=message,
            parent=saved_body,
            actions=[
                ("DB Settings", _open_settings),
                ("Continue", _continue),
            ],
            esc_action_index=1,
        )
        self._frame.body = dlg

    # ------------------------------------------------------------------ #
    #  Input handling
    # ------------------------------------------------------------------ #

    def _global_input(self, key) -> None:
        from rapmat.tui.widgets.dialog import ModalDialog

        if not isinstance(key, str):
            return
        
        if isinstance(self._frame.body, ModalDialog):
            # App-level dialog is open and it has its own keys
            return
        current = self._router.current
        if current is not None:
            result = current.keypress((), key)
            if result is None:
                return
        if key in ("q", "Q"):
            self._request_quit()
            return
        if key == "?":
            self._show_help()
            return
        if key == "esc":
            self._router.pop()

    def _request_quit(self) -> None:
        current = self._router.current
        busy = current is not None and getattr(
            current, "has_live_tasks", lambda: False
        )()
        if not busy:
            raise urwid.ExitMainLoop()

        from rapmat.tui.widgets.dialog import ModalDialog

        saved_body = self._frame.body

        def _on_close(ok: bool) -> None:
            self._frame.body = saved_body
            if ok:
                raise urwid.ExitMainLoop()

        self._frame.body = ModalDialog.confirm(
            "Quit",
            "A background task is still running.\nQuit anyway?",
            saved_body,
            _on_close,
        )

    def _show_help(self) -> None:
        from rapmat.tui.widgets.dialog import ModalDialog

        current = self._router.current
        rows: list[tuple[str, str]] = []
        if current is not None and hasattr(current, "bindings"):
            rows = [
                (b.key_display(), b.help_text())
                for b in current.bindings()
                if b.is_enabled()
            ]
        esc_label = "Back"
        if current is not None and hasattr(current, "esc_label"):
            esc_label = current.esc_label()
        bound = {key for key, _desc in rows}
        if "Enter" not in bound:
            rows.append(("Enter", "Select"))
        if esc_label:
            rows.append(("Esc", esc_label))
        if "q" not in bound:
            rows.append(("q", "Quit"))
        rows.append(("?", "This help"))

        saved_body = self._frame.body

        def _close(_btn: urwid.Button | None = None) -> None:
            self._frame.body = saved_body

        items: list[urwid.Widget] = [
            urwid.Columns(
                [
                    (12, urwid.Text(("form_label", f" {key}"))),
                    urwid.Text(desc),
                ],
                dividechars=1,
            )
            for key, desc in rows
        ]
        close_btn = urwid.AttrMap(
            urwid.Button("Close", on_press=_close), None, focus_map="btn_focus"
        )
        items.extend(
            [urwid.Divider(), urwid.Padding(close_btn, align=urwid.CENTER, width=11)]
        )

        dlg = ModalDialog("Help", urwid.Pile(items), saved_body)
        dlg._esc_handler = _close
        self._frame.body = dlg
