import uuid

import urwid

from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
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
        from rapmat.core.csp import execute_run
        from rapmat.utils.common import parse_formula

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

        meta = store.get_run_metadata(run_name)
        full_cfg = meta.search_config if meta else run_config

        execute_run(
            run_name,
            store,
            full_cfg,
            worker_id=wid,
            workers=workers,
            progress_callback=progress.as_callback(),
            log_callback=progress.log,
        )

        self._state.active_run = run_name
        self._state.invalidate()
        progress.finish()

    # ------------------------------------------------------------------ #
    #  Completion callbacks
    # ------------------------------------------------------------------ #

    def _on_complete(self) -> None:
        self._running = False
        self._progress_panel.set_finished(True, "Run completed successfully!")
        self.confirm_dialog(
            "Run Complete", "View results?", self._open_results
        )

    def _on_error(self, error: str) -> None:
        self._running = False
        self._progress_panel.set_finished(False, f"Error: {error}")
        self._progress_panel.add_log(f"ERROR: {error}")

    def _dialog_host_get(self) -> "urwid.Widget | None":
        return self._frame.body if self._frame is not None else None

    def _dialog_host_set(self, widget: urwid.Widget) -> None:
        self._frame.body = widget

    def _open_results(self) -> None:
        if self._state.active_run:
            from rapmat.tui.screens.results import ResultsScreen

            self._router.push(ResultsScreen(self._state, self._router))

