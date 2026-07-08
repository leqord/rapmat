import uuid

import urwid

from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
from rapmat.tui.widgets.form import FormGroup, int_field
from rapmat.tui.widgets.progress import ProgressPanel


class CSPResumeScreen(ScreenBase):
    title = "Resume Run"

    @property
    def breadcrumb_title(self) -> str:
        run = self._state.active_run
        return f"Resume: {run}" if run else self.title

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._frame: urwid.Frame | None = None
        self._main_body: urwid.Widget | None = None
        self._progress_panel = ProgressPanel(title=" Resume Progress ")
        self._running = False

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._frame = self._build_frame()
        return self._frame

    def bindings(self) -> list[KeyBinding]:
        return [
            KeyBinding(
                ("f5",), "Resume", self._on_start,
                help="Resume the interrupted run", priority=10,
            ),
        ]

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _build_frame(self) -> urwid.Frame:
        run_name = self._state.active_run or ""
        meta = self._state.store.get_run_metadata(run_name) if run_name else None

        if meta is None:
            body = urwid.Filler(
                urwid.Text(("error", f"Run '{run_name}' not found.")),
                valign="top",
            )
            return urwid.Frame(body=body)

        cfg = meta.search_config
        from rapmat.utils.common import format_formula

        formula_str = format_formula(cfg.formula)
        seed_str = str(cfg.seed) if cfg.seed is not None else "-"

        info_text = urwid.Text(
            [
                ("form_label", "  Formula:    "),
                ("details", formula_str + "\n"),
                ("form_label", "  Domain:     "),
                ("details", str(cfg.domain) + "\n"),
                ("form_label", "  Calculator: "),
                ("details", str(cfg.calculator) + "\n"),
                ("form_label", "  Seed:       "),
                ("details", seed_str + "\n"),
                ("form_label", "  Status:     "),
                ("details", str(meta.run_status or "-")),
            ]
        )

        self._form = FormGroup(
            [int_field("workers", "Workers (CPU)", default=1)], label_width=20
        )
        self._error_text = urwid.Text("")

        resume_btn = urwid.AttrMap(
            urwid.Button("Resume [F5]", on_press=self._on_start),
            "menu_item",
            focus_map="btn_focus",
        )

        body = urwid.Pile(
            [
                ("pack", info_text),
                ("pack", urwid.Divider("-")),
                ("pack", self._form),
                ("pack", urwid.Divider()),
                ("pack", urwid.Columns([(18, resume_btn)], dividechars=1)),
                ("pack", self._error_text),
                ("pack", urwid.Divider()),
                ("weight", 1, self._progress_panel),
            ]
        )

        self._main_body = body

        self.refresh_footer()
        return urwid.Frame(body=body)

    # ------------------------------------------------------------------ #
    #  Submit
    # ------------------------------------------------------------------ #

    def _on_start(self, _btn=None) -> None:
        if self._running:
            return

        run_name = self._state.active_run
        if not run_name:
            return

        self._running = True
        self._error_text.set_text("")
        self._progress_panel.clear()
        self._progress_panel.add_log(f"Resuming run '{run_name}'...")

        vals = self._form.get_values()

        self.run_task(
            lambda prog: self._worker(prog, run_name, vals),
            on_progress=self._progress_panel.set_progress,
            on_log=self._progress_panel.add_log,
            on_complete=self._on_complete,
            on_error=self._on_error,
        )

    def _worker(self, progress, run_name: str, vals: dict) -> None:
        from rapmat.core.csp import execute_run

        store = self._state.store
        wid = uuid.uuid4().hex[:12]

        meta = store.get_run_metadata(run_name)
        if meta is None:
            progress.fail(f"Run '{run_name}' not found")
            return

        if not store.claim_run(run_name, wid):
            progress.fail(f"Run '{run_name}' is locked by another worker")
            return

        execute_run(
            run_name,
            store,
            meta.search_config,
            worker_id=wid,
            workers=max(1, vals.get("workers", 1)),
            progress_callback=progress.as_callback(),
            log_callback=progress.log,
        )

        self._state.invalidate()
        progress.finish()

    # ------------------------------------------------------------------ #
    #  Completion callbacks
    # ------------------------------------------------------------------ #

    def _on_complete(self) -> None:
        self._running = False
        self._progress_panel.set_finished(True, "Run resumed and completed!")
        self.confirm_dialog(
            "Resume Complete", "View results?", self._open_results
        )

    def _on_error(self, error: str) -> None:
        self._running = False
        self._progress_panel.set_finished(False, f"Error: {error}")

    def _dialog_host_get(self) -> "urwid.Widget | None":
        return self._frame.body if self._frame is not None else None

    def _dialog_host_set(self, widget: urwid.Widget) -> None:
        self._frame.body = widget

    def _open_results(self) -> None:
        if self._state.active_run:
            from rapmat.tui.screens.results import ResultsScreen

            self._router.push(ResultsScreen(self._state, self._router))

