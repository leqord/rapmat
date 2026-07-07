import uuid

import urwid

from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
from rapmat.tui.widgets.dialog import ModalDialog
from rapmat.tui.widgets.form import (FormGroup, dropdown_field, int_field,
                                     text_field)
from rapmat.tui.widgets.progress import ProgressPanel


class CSPSearchScreen(ScreenBase):
    title = "New CSP Run"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._frame: urwid.Frame | None = None
        self._main_body: urwid.Widget | None = None
        self._progress_panel = ProgressPanel(title=" Run Progress ")
        self._running = False

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._state.refresh_studies_if_needed()
        self._frame = self._build_frame()
        return self._frame

    def bindings(self) -> list[KeyBinding]:
        return [
            KeyBinding(
                ("f5",), "Start Run", self._on_start,
                help="Generate and relax structures", priority=10,
            ),
        ]

    def esc_label(self) -> str:
        return "Cancel" if self._running else "Back"

    # ------------------------------------------------------------------ #
    #  Form construction
    # ------------------------------------------------------------------ #

    def _study_options(self) -> list[str]:
        if not self._state.studies_cache:
            return ["- (no studies available)"]
        return [str(s.study_id) for s in self._state.studies_cache]

    def _build_form(self) -> FormGroup:
        options = self._study_options()
        default_idx = 0
        if self._state.active_study and self._state.active_study in options:
            default_idx = options.index(self._state.active_study)

        return FormGroup(
            [
                dropdown_field("study", "Study", options, default=default_idx),
                text_field(
                    "formula",
                    "Formula (e.g. Al2O3)",
                    default="",
                    validator=self._validate_formula,
                ),

                int_field("fu_min", "Formula units min", default=2),
                int_field("fu_max", "Formula units max", default=4),
                int_field("candidates", "Candidates/group", default=2),
                text_field("run_name", "Run name", default=""),
                int_field("seed", "Seed (0 = auto)", default=0),
                int_field("workers", "Workers (CPU)", default=1),
            ],
            label_width=26,
        )

    @staticmethod
    def _validate_formula(val):
        if not val or not val.strip():
            return "Formula is required"
        try:
            from rapmat.utils.common import parse_formula

            parse_formula(val.strip())
        except Exception:
            return "Invalid formula (e.g. Al2O3)"
        return None

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #


    def _build_frame(self) -> urwid.Frame:
        self._form = self._build_form()

        self._error_text = urwid.Text("")

        listbox = urwid.ListBox(
            urwid.SimpleListWalker(
                [
                    self._form,
                    urwid.Divider(),
                    self._error_text,
                ]
            )
        )

        form_box = urwid.ScrollBar(
            listbox,
            trough_char=urwid.ScrollBar.Symbols.LITE_SHADE,
        )

        start_btn = urwid.AttrMap(
            urwid.Button("Start Run [F5]", on_press=self._on_start),
            "menu_item",
            focus_map="btn_focus",
        )

        body = urwid.Pile(
            [
                ("weight", 3, form_box),
                ("pack", urwid.Columns([(20, start_btn)], dividechars=1)),
                ("pack", urwid.Divider()),
                ("weight", 2, self._progress_panel),
            ]
        )

        self._main_body = body

        self.refresh_footer()
        return urwid.Frame(body=body)

    # ------------------------------------------------------------------ #
    #  Submit handler
    # ------------------------------------------------------------------ #

    def _on_start(self, _btn=None) -> None:
        if self._running:
            return

        errs = self._form.validate()
        if errs:
            self._error_text.set_text(("form_error", "\n".join(errs)))
            return

        self._error_text.set_text("")
        vals = self._form.get_values()
        self._running = True
        self._progress_panel.clear()
        self._progress_panel.add_log("Preparing run...")

        self.run_task(
            lambda prog: self._worker(prog, vals),
            on_progress=self._progress_panel.set_progress,
            on_log=self._progress_panel.add_log,
            on_complete=self._on_complete,
            on_error=self._on_error,
        )

    def _worker(self, progress, vals) -> None:
        import time

        from rapmat.core.csp import run_generation_loop, run_processing_loop
        from rapmat.utils.common import parse_formula, workdir_context

        store = self._state.store
        fu_min = max(1, vals["fu_min"])
        fu_max = max(fu_min, vals["fu_max"])
        candidates = max(1, vals["candidates"])
        run_name = vals["run_name"].strip() or f"run-{uuid.uuid4().hex[:8]}"
        workers = max(1, vals["workers"])

        formula_str = vals.get("formula", "")
        formula = parse_formula(formula_str) if formula_str else {}

        import random as _random

        seed_val = vals.get("seed", 0)
        if seed_val == 0:
            seed_val = _random.randint(1, 2**32 - 1)
        progress.log(f"Using seed: {seed_val}")

        study_id = vals["study"]
        if study_id.startswith("-"):
            raise ValueError("You must select a valid Study to start a run.")

        study = store.get_study(study_id)
        if not study:
            raise ValueError(f"Study {study_id} no longer exists.")

        search_dim = 3 if study.domain == "bulk" else 2

        wid = uuid.uuid4().hex[:12]

        def _cb(current, total, msg, is_log=True):
            if progress.cancelled:
                raise KeyboardInterrupt("Cancelled by user")
            progress.update(current, total, msg)
            if is_log:
                progress.log(msg)

        progress.log(f"Queuing run '{run_name}'...")

        run_config = {
            "formula": formula,
            "formula_units": [fu_min, fu_max],
            "candidates_per_group": candidates,
            "seed": seed_val,
        }

        store.create_run(
            name=run_name,
            study_id=study_id,
            config=run_config,
            worker_id=wid,
        )

        spg_total = 230 if search_dim == 3 else 80
        placeholders = []
        idx = 0
        for fu in range(fu_min, fu_max + 1):
            for spg in range(1, spg_total + 1):
                for _ in range(candidates):
                    idx += 1
                    placeholders.append((f"{run_name}/{idx}", spg, fu))

        store.add_generation_placeholders(run_name, placeholders)

        cancel_flag = [False]

        progress.log(f"Starting generation phase for {run_name}...")
        try:
            with workdir_context(None) as workdir_path:
                progress.log(f"Working directory: {workdir_path}")

                meta = store.get_run_metadata(run_name)
                full_cfg = meta.search_config if meta else run_config

                run_generation_loop(
                    run_name=run_name,
                    store=store,
                    config=full_cfg,
                    worker_id=wid,
                    workers=workers,
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

                progress.log(f"Starting processing phase for {run_name}...")

                t0 = time.monotonic()
                run_processing_loop(
                    run_name=run_name,
                    store=store,
                    config=full_cfg,
                    workdir_path=workdir_path,
                    worker_id=wid,
                    progress_callback=_proc_cb,
                    cancel_flag=cancel_flag,
                )
                t1 = time.monotonic()
                progress.log(
                    f"Run '{run_name}' computation finished in {t1 - t0:.2f} seconds."
                )

                store.release_run(run_name, "completed")
        except KeyboardInterrupt:
            store.release_run(run_name, "interrupted")
            raise
        except Exception:
            store.release_run(run_name, "failed")
            raise

        self._state.active_run = run_name
        self._state.invalidate()
        progress.finish()

    # ------------------------------------------------------------------ #
    #  Completion callbacks
    # ------------------------------------------------------------------ #

    def _on_complete(self) -> None:
        self._running = False
        self._progress_panel.set_finished(True, "Run completed successfully!")
        if self._frame and self._main_body:
            dlg = ModalDialog.confirm(
                "Run Complete",
                "CSP run finished. View results?",
                parent=self._main_body,
                on_close=self._on_dialog_close,
            )
            self._frame.body = dlg

    def _on_error(self, error: str) -> None:
        self._running = False
        self._progress_panel.set_finished(False, f"Error: {error}")
        self._progress_panel.add_log(f"ERROR: {error}")

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
