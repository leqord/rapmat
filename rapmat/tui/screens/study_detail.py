from typing import TYPE_CHECKING

import urwid

from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
from rapmat.tui.widgets.config_grid import build_config_grid
from rapmat.tui.widgets.table import SortableTable

if TYPE_CHECKING:
    from rapmat.core.entities import RunMetadata

_RUN_COLS = [
    ("Run Name", 28),
    ("Formula", 14),
    ("Type", 14),
    ("Structures", 16),
    ("Status", 12),
]


def _classify_run(run: "RunMetadata", study_elements: list[str]) -> str:
    formula = run.search_config.formula
    run_elements = set(formula.keys())
    if len(run_elements) == 1 and run_elements <= set(study_elements):
        return "endpoint"
    return "intermediate"


def _formula_str(run: "RunMetadata") -> str:
    formula = run.search_config.formula
    return "".join(f"{el}{n}" if n > 1 else el for el, n in formula.items())


class StudyDetailScreen(ScreenBase):
    title = "Study Detail"

    @property
    def breadcrumb_title(self) -> str:
        study = self._state.active_study
        return f"{study}" if study else self.title

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._table: SortableTable | None = None
        self._placeholder: urwid.WidgetPlaceholder | None = None
        self._details_content: urwid.WidgetPlaceholder | None = None
        self._details_panel: urwid.Widget | None = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._placeholder = urwid.WidgetPlaceholder(self._build_widget())
        return self._placeholder

    def on_resume(self) -> None:
        if self._placeholder is not None:
            self._placeholder.original_widget = self._build_widget()
        super().on_resume()

    def bindings(self) -> list[KeyBinding]:
        return [
            KeyBinding(
                ("h",), "Phase Analysis", self._go_phase_analysis,
                help="Convex hull", priority=10,
            ),
            KeyBinding(
                ("r",), "Resume Run", self._resume_focused,
                help="Resume the run", priority=20,
            ),
            KeyBinding(
                ("n",), "New Run", self._go_new_run,
                help="Start a new run", priority=30,
            ),
            KeyBinding(
                ("d",), "Dedup", self._go_dedup_focused,
                help="Duplicate analysis", priority=40,
            ),
            KeyBinding(
                ("u",), "Unlock Run", self._unlock_focused,
                help="Release the focused run's worker lock", priority=50,
            ),
            KeyBinding(
                ("delete",), "Remove", self._delete_focused,
                help="Delete the run", priority=60,
            ),
        ]

    def extra_hints(self) -> list:
        return [("Enter", "Open Run", 5)]

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _build_widget(self) -> urwid.Widget:
        study_id = self._state.active_study or ""
        study = self._state.store.get_study(study_id) if study_id else None

        if study is None:
            return urwid.Filler(
                urwid.Text(("error", f"Study '{study_id}' not found.")), valign="top"
            )

        from rapmat.utils.common import parse_system

        elements = parse_system(study.system)
        ts = study.timestamp[:16].replace("T", " ")

        info_text = urwid.Text(
            [
                ("form_label", "  System:     "),
                ("details", study.system + "\n"),
                ("form_label", "  Domain:     "),
                ("details", study.domain + "\n"),
                ("form_label", "  Calculator: "),
                ("details", study.calculator + "\n"),
                ("form_label", "  Created:    "),
                ("details", ts),
            ]
        )

        runs = self._state.store.get_study_runs(study_id)
        run_rows = []
        for run in sorted(runs, key=lambda r: r.timestamp):
            counts = self._state.store.count_by_status(run.name)
            relaxed = counts.get("relaxed", 0)
            generated = counts.get("generated", 0)
            total = relaxed + generated
            run_type = _classify_run(run, elements)

            st = run.run_status or "pending"
            wid = run.worker_id
            if wid and st in ("generating", "processing"):
                st = f"active({wid[:4]})"

            d = run.model_dump()
            d["_formula"] = _formula_str(run)
            d["_type"] = run_type
            d["_structures"] = f"{relaxed} / {total}"
            d["_status"] = st
            run_rows.append(d)

        self._table = SortableTable(
            columns=_RUN_COLS,
            row_data=run_rows,
            format_row=lambda r: [
                r["name"],
                r["_formula"],
                r["_type"],
                r["_structures"],
                r["_status"],
            ],
            attr_fn=lambda r: "warning" if r.get("worker_id") else "body",
            on_focus_change=self._on_run_focus_change,
        )
        urwid.connect_signal(self._table, "select", self._on_run_select)

        self._details_content = urwid.WidgetPlaceholder(urwid.Text("No run selected."))
        self._details_panel = urwid.LineBox(
            self._details_content,
            title="Run Configuration",
        )

        n_elements = len(elements)
        endpoint_elements: set[str] = set()
        for run in runs:
            formula = run.search_config.formula
            if len(formula) == 1:
                endpoint_elements.update(formula.keys())
        missing = set(elements) - endpoint_elements

        if n_elements < 2:
            status_text = urwid.Text(
                (
                    "details",
                    "  Single-element system. Phase analysis shows energy ranking.",
                )
            )
        elif missing:
            status_text = urwid.Text(
                ("unconv", f"  Missing pure-element runs: {', '.join(sorted(missing))}")
            )
        elif n_elements == 2:
            status_text = urwid.Text(
                ("success", "  All endpoints present. Phase analysis available.")
            )
        else:
            status_text = urwid.Text(
                (
                    "success",
                    "  All endpoints present. Phase analysis available (table only).",
                )
            )

        body = urwid.Pile(
            [
                ("pack", info_text),
                ("pack", status_text),
                ("pack", urwid.Divider("-")),
                ("weight", 1, self._table),
                ("pack", urwid.Divider()),
                ("pack", self._details_panel),
                ("pack", urwid.Divider()),
            ]
        )
        if self._table:
            self._on_run_focus_change(self._table.get_focused_row())
        self.refresh_footer()
        return urwid.Padding(body, left=1, right=1)

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #

    def _on_run_focus_change(self, run: dict | None) -> None:
        if getattr(self, "_details_content", None) is None:
            return

        if run is None:
            self._details_content.original_widget = urwid.Text(
                [("details", "No run selected.")]
            )
        else:
            config = run.get("config", {})
            self._details_content.original_widget = build_config_grid(config)

    def _on_run_select(self, _table, run: dict) -> None:
        self._state.active_run = run["name"]
        from rapmat.tui.screens.results import ResultsScreen

        self._router.push(ResultsScreen(self._state, self._router))

    def _on_unlock_run(self, run_name: str) -> None:
        self._state.store.release_run(run_name, "pending")
        if self._placeholder:
            self._placeholder.original_widget = self._build_widget()

    def _open_delete_modal(self, run_name: str) -> None:
        if self._placeholder is None:
            return

        run_data = self._state.store.get_run_metadata(run_name)
        status = run_data.run_status if run_data else "pending"
        worker_id = run_data.worker_id if run_data else None
        is_active = worker_id and status in ("generating", "processing")

        from rapmat.tui.widgets.dialog import ModalDialog

        current_body = self._placeholder.original_widget

        def _on_close(confirmed: bool) -> None:
            if self._placeholder is not None:
                self._placeholder.original_widget = current_body
                if confirmed:
                    self._state.store.delete_run(run_name)
                    self._placeholder.original_widget = self._build_widget()

        msg = f"Are you sure you want to permanently delete run '{run_name}' and all its structures?"
        if is_active:
            msg += f"\n\nWARNING: This run appears to be claimed and being processed right now by the {worker_id[:4]} worker"

        dlg = ModalDialog.confirm(
            title="Delete Run",
            message=msg,
            parent=current_body,
            on_close=_on_close,
        )
        self._placeholder.original_widget = dlg

    def _focused_run_name(self) -> str | None:
        if self._table is None:
            return None
        run = self._table.get_focused_row()
        return run["name"] if run else None

    def _go_new_run(self) -> None:
        from rapmat.tui.screens.csp_search import CSPSearchScreen

        self._router.push(CSPSearchScreen(self._state, self._router))

    def _resume_focused(self) -> None:
        run_name = self._focused_run_name()
        if run_name:
            self._state.active_run = run_name
            from rapmat.tui.screens.csp_resume import CSPResumeScreen

            self._router.push(CSPResumeScreen(self._state, self._router))

    def _unlock_focused(self) -> None:
        run_name = self._focused_run_name()
        if run_name:
            self._on_unlock_run(run_name)

    def _go_phase_analysis(self) -> None:
        from rapmat.tui.screens.hull import PhaseAnalysisScreen

        self._router.push(PhaseAnalysisScreen(self._state, self._router))

    def _go_dedup_focused(self) -> None:
        run_name = self._focused_run_name()
        if run_name:
            self._state.active_run = run_name
            from rapmat.tui.screens.dedup import DedupScreen

            self._router.push(DedupScreen(self._state, self._router))

    def _delete_focused(self) -> None:
        run_name = self._focused_run_name()
        if run_name:
            self._open_delete_modal(run_name)
