from typing import Callable

import urwid

from rapmat.tui.keymap import PRIORITY_PINNED, KeyBinding, dispatch, footer_hints
from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState
from rapmat.tui.tasks import BackgroundTask
from rapmat.tui.widgets.dialog import ModalDialog


class ScreenBase:
    """Shared scaffold for TUI screens.

    Implements the router's Screen protocol.
    Subclasses have to override:

    - ``build()``: required
    - ``bindings()``: to declare hotkeys, dispatch, footer hints and the help
    - ``esc_label()``: if Esc does something other than going back
    - ``_tasks()``: if they own tasks outside ``self._task``
    - ``_dialog_host_get()``/``_dialog_host_set()``: to use ``show_dialog()``
    """

    title: str = "Screen"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._task: BackgroundTask | None = None
        self._progress_panel = None

    @property
    def breadcrumb_title(self) -> str:
        return self.title

    def build(self) -> urwid.Widget:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Key bindings
    # ------------------------------------------------------------------ #

    def bindings(self) -> list[KeyBinding]:
        return []

    def esc_label(self) -> str:
        if self._task is not None and self._task.is_running:
            return "Cancel"
        return "Back"

    def keypress(self, size: tuple, key: str) -> str | None:
        if dispatch(self.bindings(), key):
            return None
        if key == "esc" and self._on_esc():
            return None
        return key

    def _on_esc(self) -> bool:
        if self._task is not None and self._task.is_running:
            self._task.cancel()
            if self._progress_panel is not None:
                self._progress_panel.set_cancelling()
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Footer
    # ------------------------------------------------------------------ #

    def extra_hints(self) -> list:
        return []

    def refresh_footer(self, message: str = "") -> None:
        bar = self._state.status_bar
        if not bar:
            return
        hints = footer_hints(self.bindings())
        hints.extend(self.extra_hints())
        hints.append(("?", "Help", PRIORITY_PINNED))
        if self.esc_label():
            hints.append(("Esc", self.esc_label(), PRIORITY_PINNED))
        bar.set_hints(hints)
        bar.set_message(message)

    # ------------------------------------------------------------------ #
    #  Modal dialogs
    # ------------------------------------------------------------------ #

    def _dialog_host_get(self) -> "urwid.Widget | None":
        """Current body widget that a modal dialog temporarily replaces."""
        return None

    def _dialog_host_set(self, widget: urwid.Widget) -> None:
        raise NotImplementedError

    def show_dialog(
        self,
        factory: Callable[[urwid.Widget, Callable[[], None]], urwid.Widget],
    ) -> "Callable[[], None] | None":
        parent = self._dialog_host_get()
        if parent is None:
            return None

        def close() -> None:
            self._dialog_host_set(parent)
            self.refresh_footer()

        self._dialog_host_set(factory(parent, close))
        self.refresh_footer()
        return close

    def confirm_dialog(
        self, title: str, message: str, on_confirm: Callable[[], None]
    ) -> None:
        def _factory(
            parent: urwid.Widget, close: Callable[[], None]
        ) -> urwid.Widget:
            def _on_close(confirmed: bool) -> None:
                close()
                if confirmed:
                    on_confirm()

            return ModalDialog.confirm(title, message, parent, _on_close)

        self.show_dialog(_factory)

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def on_resume(self) -> None:
        self.refresh_footer()

    def _tasks(self) -> list[BackgroundTask]:
        return [self._task] if self._task is not None else []

    def has_live_tasks(self) -> bool:
        return any(t.is_running for t in self._tasks())

    def on_leave(self) -> None:
        for task in self._tasks():
            task.cancel()

    # ------------------------------------------------------------------ #
    #  Background tasks
    # ------------------------------------------------------------------ #

    def run_task(
        self,
        fn,
        *,
        on_progress=None,
        on_log=None,
        on_complete=None,
        on_error=None,
    ) -> BackgroundTask:
        """Start a BackgroundTask in ``self._task`` so that
        ``on_leave`` cancels it."""
        task = BackgroundTask(
            fn=fn,
            loop=self._state.loop,
            on_progress=on_progress,
            on_log=on_log,
            on_complete=on_complete,
            on_error=on_error,
        )
        self._task = task
        task.start()
        return task
