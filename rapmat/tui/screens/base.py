import urwid

from rapmat.tui.keymap import PRIORITY_PINNED, KeyBinding, dispatch, footer_hints
from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState
from rapmat.tui.tasks import BackgroundTask


class ScreenBase:
    """Shared scaffold for TUI screens.

    Implements the router's Screen protocol.
    Subclasses have to override:

    - ``build()``: required
    - ``bindings()``: to declare hotkeys, dispatch, footer hints and the help
    - ``esc_label()``: if Esc does something other than going back
    - ``_tasks()``: if they own tasks outside ``self._task``
    """

    title: str = "Screen"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._task: BackgroundTask | None = None

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
        return "Back"

    def keypress(self, size: tuple, key: str) -> str | None:
        if dispatch(self.bindings(), key):
            return None
        return key

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
