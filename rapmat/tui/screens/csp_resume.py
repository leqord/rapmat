import uuid

import urwid

from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
from rapmat.tui.widgets.dialog import ModalDialog
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

    def esc_label(self) -> str:
        return "Cancel" if self._running else "Back"

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
        formula_str = "".join(
            f"{el}{n}" if n > 1 else el for el, n in cfg.formula.items()
        )
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
        import time

        from rapmat.core.csp import run_generation_loop, run_processing_loop
        from rapmat.utils.common import workdir_context

        store = self._state.store
        wid = uuid.uuid4().hex[:12]

        meta = store.get_run_metadata(run_name)
        if meta is None:
            progress.fail(f"Run '{run_name}' not found")
            return

        if not store.claim_run(run_name, wid):
            progress.fail(f"Run '{run_name}' is locked by another worker")
            return

        config = meta.search_config

        def _cb(current, total, msg, is_log=True):
            if progress.cancelled:
                raise KeyboardInterrupt("Cancelled by user")
            progress.update(current, total, msg)
            if is_log:
                progress.log(msg)

        cancel_flag = [False]

        try:
            with workdir_context(None) as workdir_path:
                progress.log(f"Working directory: {workdir_path}")
                pending_gen = store.get_pending_generation(run_name)
                if pending_gen:
                    progress.log(
                        f"Finishing generation phase ({len(pending_gen)} placeholders remaining)..."
                    )
                    store.set_run_status(run_name, "generating")
                    run_generation_loop(
                        run_name=run_name,
                        store=store,
                        config=config,
                        worker_id=wid,
                        workers=max(1, vals.get("workers", 1)),
                        progress_callback=_cb,
                        cancel_flag=cancel_flag,
                        log_callback=progress.log,
                    )

                    if progress.cancelled or cancel_flag[0]:
                        raise KeyboardInterrupt("Cancelled by user")

                    progress.log(
                        "Generation complete. Initializing calculator for processing..."
                    )
                    store.set_run_status(run_name, "processing")

                def _proc_cb(current, total, msg, is_log=True):
                    if progress.cancelled:
                        cancel_flag[0] = True
                        raise KeyboardInterrupt("Cancelled by user")
                    progress.update(current, total, msg)
                    if is_log:
                        progress.log(msg)

                progress.log("Running processing phase...")

                t0 = time.monotonic()
                run_processing_loop(
                    run_name=run_name,
                    store=store,
                    config=config,
                    workdir_path=workdir_path,
                    worker_id=wid,
                    progress_callback=_proc_cb,
                    cancel_flag=cancel_flag,
                )
                t1 = time.monotonic()
                progress.log(
                    f"Resumed run '{run_name}' computation finished in {t1 - t0:.2f} seconds."
                )

                store.release_run(run_name, "completed")
        except KeyboardInterrupt:
            store.release_run(run_name, "interrupted")
            raise
        except Exception:
            store.release_run(run_name, "failed")
            raise

        self._state.invalidate()
        progress.finish()

    # ------------------------------------------------------------------ #
    #  Completion callbacks
    # ------------------------------------------------------------------ #

    def _on_complete(self) -> None:
        self._running = False
        self._progress_panel.set_finished(True, "Run resumed and completed!")
        if self._frame and self._main_body:
            dlg = ModalDialog.confirm(
                "Resume Complete",
                "Run finished. View results?",
                parent=self._main_body,
                on_close=self._on_dialog_close,
            )
            self._frame.body = dlg

    def _on_error(self, error: str) -> None:
        self._running = False
        self._progress_panel.set_finished(False, f"Error: {error}")

    def _on_dialog_close(self, confirmed: bool) -> None:
        if self._frame:
            self._frame.body = self._main_body
        if confirmed and self._state.active_run:
            from rapmat.tui.screens.results import ResultsScreen

            self._router.push(ResultsScreen(self._state, self._router))

    # ------------------------------------------------------------------ #
    #  Key handling
    # ------------------------------------------------------------------ #

    def keypress(self, size: tuple, key: str) -> str | None:
        if super().keypress(size, key) is None:
            return None
        if key == "esc":
            if self._running:
                if self._task:
                    self._task.cancel()
                    self._progress_panel.set_cancelling()
                return None
            self._router.pop()
            return None
        return key
