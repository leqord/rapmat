import json
from dataclasses import replace

import urwid

from rapmat.core.entities import ResultRow, Structure
from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.screens.base_results import BaseResultsScreen
from rapmat.tui.state import AppState
from rapmat.tui.widgets.calc_fields import (
    calculator_fields,
    parse_toml_config,
    setup_calculator_signals,
)
from rapmat.tui.widgets.form import (FormGroup, checkbox_field,
                                     float_field, int_field, tuple_field)
from rapmat.tui.widgets.progress import ProgressPanel


# ------------------------------------------------------------------ #
#  Results screen
# ------------------------------------------------------------------ #

_RESULT_COLS_BASE = [
    ("ID", 8),
    ("Formula", 10),
    ("MLIP eV/at", 12),
    ("Ref eV/at", 12),
    ("Δ eV/at", 10),
    ("MLIP#", 6),
    ("Ref#", 6),
]

_DYN_COLS = [
    ("MLIP Dyn", 9),
    ("Ref Dyn", 9),
]


class EvalResultsScreen(BaseResultsScreen):
    """Evaluation table with filter support and live metric recompute.
    """

    title = "Eval Results"

    def __init__(
        self,
        state: "AppState",
        router: "ScreenRouter",
        *,
        eval_rows: list,
        phonon_cutoff: float,
        stable_only: bool,
        run_name: str,
    ) -> None:
        super().__init__(state, router)

        self._hide_excluded = False
        self._eval_rows = eval_rows
        self._phonon_cutoff = phonon_cutoff
        self._stable_only = stable_only
        self._run_name = run_name
        self._ranking: dict = {}
        self._stability: dict | None = None
        self._rank_map: dict[str, tuple[int, int | None]] = {}
        self._metrics_text: urwid.Text | None = None
        self._n_display: int = 0
        self._n_total: int = 0

    @property
    def breadcrumb_title(self) -> str:
        return f"Eval Results: {self._run_name}"

    # ------------------------------------------------------------------ #
    #  Data + metric recompute
    # ------------------------------------------------------------------ #

    def _fetch_data(self, progress_callback=None) -> None:
        self._results = list(self._eval_rows)
        self._show_duplicate_col = any(
            r.duplicate is not None for r in self._results
        )
        self._show_thickness = any(
            r.thickness is not None for r in self._results
        )
        self._show_dynamical_stability = any(
            r.min_phonon_freq is not None or r.ref_phonon_freq is not None
            for r in self._results
        )
        self._recompute_view(self._get_display_results())

    def _recompute_view(self, display: list) -> None:
        """Recompute metrics and the in-view ranks."""
        from rapmat.core.evaluation import (comparison_from_result_rows,
                                            compute_ranking_metrics,
                                            compute_stability_metrics)

        comparison = comparison_from_result_rows(display)
        self._ranking = compute_ranking_metrics(
            comparison, self._phonon_cutoff, self._stable_only
        )
        self._stability = compute_stability_metrics(
            comparison, self._phonon_cutoff
        )
        self._rank_map = self._compute_rank_map(display)
        self._n_display = len(display)
        self._n_total = len(self._results)
        if self._metrics_text is not None:
            self._metrics_text.set_text(self._metrics_markup())

    @staticmethod
    def _compute_rank_map(display: list) -> dict:
        """Rank within the visible set."""
        mlip_rank = {
            r.structure_id: i
            for i, r in enumerate(
                sorted(display, key=lambda x: x.energy_per_atom), 1
            )
        }
        ref_rows = [r for r in display if r.ref_energy_per_atom is not None]
        ref_rank = {
            r.structure_id: i
            for i, r in enumerate(
                sorted(ref_rows, key=lambda x: x.ref_energy_per_atom), 1
            )
        }
        return {sid: (mlip_rank[sid], ref_rank.get(sid)) for sid in mlip_rank}

    def _rebuild_table(self) -> None:
        if self._table is None:
            return

        self._recompute_view(self._get_display_results())
        super()._rebuild_table()

    # ------------------------------------------------------------------ #
    #  Header
    # ------------------------------------------------------------------ #

    def _header_widget(self) -> urwid.Widget:
        self._metrics_text = urwid.Text(self._metrics_markup())
        return urwid.Pile([self._metrics_text, urwid.Divider("-")])

    def _metrics_markup(self) -> list:
        header_parts: list = [
            ("form_label", "Showing "),
            ("details", f"{self._n_display}/{self._n_total}"),
        ]

        metric_parts: list = []

        r = self._ranking or {}
        if r.get("kendall_tau") is not None:
            metric_parts.extend([
                ("form_label", "Kendall τ: "),
                ("success" if r["kendall_tau"] > 0.7
                 else "unconv" if r["kendall_tau"] > 0.4 else "error",
                 f"{r['kendall_tau']:.4f}"),
                ("details", f" (p={r['p_value']:.2e}, n={r['n_structures']})"),
            ])
        if r.get("mae_epa") is not None:
            if metric_parts:
                metric_parts.append(("details", "  |  "))
            metric_parts.extend([
                ("form_label", "MAE: "),
                ("details", f"{r['mae_epa']:.4f} eV/at"),
            ])

        s = self._stability
        if s is not None:
            if metric_parts:
                metric_parts.append(("details", "  |  "))
            metric_parts.extend([
                ("form_label", "F1: "),
                ("success" if s["f1"] > 0.8
                 else "unconv" if s["f1"] > 0.5 else "error",
                 f"{s['f1']:.2f}"),
                ("details", f" (P={s['precision']:.2f}, R={s['recall']:.2f})"),
            ])

        if metric_parts:
            header_parts.append(("details", "  |  "))
            header_parts.extend(metric_parts)
        else:
            header_parts.append(("details", "  No metrics available"))
        return header_parts

    # ------------------------------------------------------------------ #
    #  Table
    # ------------------------------------------------------------------ #

    def _columns_def(self) -> list:
        cols = list(_RESULT_COLS_BASE)
        if self._show_thickness:
            cols.append(("Thick", 9))
        if self._show_dynamical_stability:
            cols.extend(_DYN_COLS)
        if self._show_duplicate_col:
            cols.append(("Dup", 5))
        return cols

    def _fmt_dyn(self, val: float | None) -> str:
        if val is None:
            return "N/A"
        cutoff = self._phonon_cutoff if self._phonon_cutoff is not None else -0.15
        return "Yes" if val >= cutoff else "No"

    def _format_row(self, r) -> list:
        full_id = str(r.structure_id)
        short_id = full_id.split("/")[-1] if "/" in full_id else full_id

        mlip = r.energy_per_atom
        ref = r.ref_energy_per_atom
        mr, rr = self._rank_map.get(r.structure_id, (None, None))

        row = [
            short_id,
            r.formula or "",
            f"{mlip:.4f}",
            "" if ref is None else f"{ref:.4f}",
            "" if ref is None else f"{ref - mlip:+.4f}",
            "" if mr is None else str(mr),
            "" if rr is None else str(rr),
        ]
        if self._show_thickness:
            t = r.thickness
            row.append("" if t is None else f"{t:.2f}")
        if self._show_dynamical_stability:
            row.append(self._fmt_dyn(r.min_phonon_freq))
            row.append(self._fmt_dyn(r.ref_phonon_freq))
        if self._show_duplicate_col:
            dup = r.duplicate
            row.append("Yes" if dup is True else ("No" if dup is False else ""))
        return row

    def _attr_fn(self, r) -> str:
        mr, rr = self._rank_map.get(r.structure_id, (None, None))
        if mr is not None and rr is not None:
            diff = abs(mr - rr)
            if diff >= 3:
                return "error"
            if diff >= 1:
                return "unconv"
        return "body"

    def _get_extra_details(self, r) -> list:
        ref = r.ref_energy_per_atom
        if ref is None:
            return []
        mlip = r.energy_per_atom
        extras = [
            ("details", f"Ref Energy/A: {ref:.6f} eV"),
            ("details", f"Δ (Ref-MLIP): {ref - mlip:+.6f} eV/at"),
        ]
        mr, rr = self._rank_map.get(r.structure_id, (None, None))
        if mr is not None and rr is not None:
            rank_str = f"{mr} (unchanged)" if mr == rr else f"{mr} -> {rr}"
            extras.append(("details", f"Rank (MLIP -> Ref): {rank_str}"))
        if r.ref_phonon_freq is not None:
            extras.append(
                ("details", f"Ref min freq: {r.ref_phonon_freq:.4f} THz")
            )
        return extras

    # ------------------------------------------------------------------ #
    #  Footer / keys
    # ------------------------------------------------------------------ #

    def bindings(self) -> list[KeyBinding]:
        drop = {"p", "o", "x", "e"}
        gates = {
            "t": lambda: self._show_thickness,
            "d": lambda: self._show_duplicate_col,
            "y": lambda: self._show_dynamical_stability,
        }
        out: list[KeyBinding] = []
        for b in super().bindings():
            key = b.key_text()
            if key in drop:
                continue
            if key in gates:
                b = replace(b, enabled=gates[key])
            out.append(b)
        return out


# ------------------------------------------------------------------ #
#  Main evaluation form screen
# ------------------------------------------------------------------ #


class EvalScreen(ScreenBase):
    title = "Evaluation"

    @property
    def breadcrumb_title(self) -> str:
        return f"Evaluation: {self._run_name}" if self._run_name else self.title

    def __init__(
        self,
        state: "AppState",
        router: "ScreenRouter",
        run_name: str,
        filtered_ids: list[str] | None = None,
    ) -> None:
        super().__init__(state, router)
        self._run_name = run_name
        self._filtered_ids = filtered_ids
        self._widget: urwid.WidgetPlaceholder | None = None
        self._main_body: urwid.Widget | None = None
        self._progress_panel = ProgressPanel(title=" Evaluation Progress ")
        self._running = False
        self._eval_rows: list["ResultRow"] = []
        self._records: list["Structure"] = []
        self._eval_vals: dict | None = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._state.refresh_runs_if_needed()
        self._widget = self._build_frame()
        return self._widget

    def bindings(self) -> list[KeyBinding]:
        return [
            KeyBinding(
                ("f5",), "Evaluate", self._on_start,
                help="Run the evaluation", priority=10,
            ),
            KeyBinding(
                ("delete",), "Clear Cache", self._on_clear_cache,
                help="Clear the cached reference evaluations for this run",
                priority=20,
            ),
        ]

    def _dialog_host_get(self) -> "urwid.Widget | None":
        return self._widget.original_widget if self._widget is not None else None

    def _dialog_host_set(self, widget: urwid.Widget) -> None:
        self._widget.original_widget = widget

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _build_frame(self) -> urwid.WidgetPlaceholder:
        self._form = FormGroup(
            [
                *calculator_fields(calc_label="Ref. calculator"),
                int_field("top_n", "Top N (0 - all)", default=0),
                checkbox_field(
                    "cached_only", "Cached only", default=False
                ),
                checkbox_field("run_phonons", "Phonons", default=False),
                checkbox_field("stable_only", "Tau over dyn. stable only", default=False),
                tuple_field(
                    "phonon_supercell", "Phonon supercell", size=3, default=(3, 3, 3)
                ),
                tuple_field("phonon_mesh", "Phonon mesh", size=3, default=(20, 20, 20)),
                float_field("phonon_displacement", "Phonon displacement", default=1e-2),
                float_field("phonon_cutoff", "Phonon cutoff", default=-0.15),
            ],
            label_width=22,
            groups=[
                ("Reference Calculator", ["calculator", "calculator_config"]),
                ("Phonon Settings", [
                    "run_phonons", "stable_only",
                    "phonon_supercell", "phonon_mesh",
                    "phonon_displacement", "phonon_cutoff",
                ]),
                ("Filter", ["top_n", "cached_only"]),
            ],
        )

        setup_calculator_signals(self._form)

        self._error_text = urwid.Text("")

        start_btn = urwid.AttrMap(
            urwid.Button("Evaluate [F5]", on_press=self._on_start),
            "menu_item",
            focus_map="btn_focus",
        )

        clear_btn = urwid.AttrMap(
            urwid.Button("Clear Cache [Del]", on_press=self._on_clear_cache),
            "menu_item",
            focus_map="btn_focus",
        )

        listbox_form = urwid.ListBox(
            urwid.SimpleListWalker(
                [
                    self._form,
                    urwid.Divider(),
                    urwid.Columns([(18, start_btn), (22, clear_btn)], dividechars=1),
                    self._error_text,
                ]
            )
        )

        form_area = urwid.ScrollBar(
            listbox_form,
            trough_char=urwid.ScrollBar.Symbols.LITE_SHADE,
        )

        body = urwid.Pile(
            [
                ("weight", 3, form_area),
                ("weight", 2, self._progress_panel),
            ]
        )

        self._main_body = body

        self.refresh_footer()

        return urwid.WidgetPlaceholder(urwid.Frame(body=body))

    # ------------------------------------------------------------------ #
    #  Submit & Actions
    # ------------------------------------------------------------------ #

    def _on_clear_cache(self, _btn=None) -> None:
        if self._running:
            return

        run_name = self._run_name

        if not run_name:
            self._error_text.set_text(("form_error", "No active run selected"))
            return

        def _confirmed() -> None:
            self._state.store.clear_evaluations(run_name)
            self._error_text.set_text(
                ("success", f"Cache cleared for run '{run_name}'")
            )

        self.confirm_dialog(
            "Clear Reference Cache",
            (
                f"Are you sure you want to clear the evaluation cache for ALL structures and ALL calculators in the run '{run_name}'?\n\n"
            ),
            _confirmed,
        )

    def _on_start(self, _btn=None) -> None:
        if self._running:
            return

        self._running = True
        self._error_text.set_text("")
        self._progress_panel.clear()

        vals = self._form.get_values()

        run_name = self._run_name
        if not run_name:
            self._error_text.set_text(("form_error", "No active run selected"))
            self._running = False
            return

        vals["run_name"] = run_name

        calc_config_dict, toml_err = parse_toml_config(vals)
        if toml_err:
            self._error_text.set_text(("form_error", toml_err))
            self._running = False
            return

        vals["calculator_config_dict"] = calc_config_dict
        self._eval_vals = vals

        self.run_task(
            lambda prog: self._worker(prog, vals),
            on_progress=self._progress_panel.set_progress,
            on_log=self._progress_panel.add_log,
            on_complete=self._on_complete,
            on_error=self._on_error,
        )

    def _worker(self, progress, vals: dict) -> None:
        from rapmat.calculators import Calculators
        from rapmat.calculators.factory import load_calculator
        from rapmat.core.evaluation import (eval_rows_from_cache, run_eval_loop,
                                            select_eval_records)
        from rapmat.utils.common import workdir_context

        class _TaskCalcCallback:
            def on_status(self, message: str) -> None:
                progress.log(message)

        store = self._state.store
        run_name = vals["run_name"]
        calculator_name = vals["calculator"]
        top_n = vals["top_n"]
        cached_only = vals.get("cached_only", False)
        run_phonons = vals["run_phonons"]

        progress.log(f"Loading structures for '{run_name}'...")
        records = store.get_structures(run_name, status="relaxed")

        initial_count = len(records)

        if self._filtered_ids is not None:
            valid_ids = set(self._filtered_ids)
            records = [r for r in records if r.id in valid_ids]
            progress.log(
                f"Excluded {initial_count - len(records)} structures hidden by Results filters"
            )
        else:
            records = [r for r in records if r.converged]
            progress.log(
                f"Excluded {initial_count - len(records)} unconverged structures"
            )

        records = select_eval_records(records, top_n)

        if not records:
            progress.fail("No relaxed structures found")
            return

        self._records = records
        config_dict = {"run_phonons": run_phonons}
        config_dict["calculator_config"] = vals.get("calculator_config_dict", {})
        if run_phonons:
            config_dict["phonon_supercell"] = vals.get("phonon_supercell", (3, 3, 3))
            config_dict["phonon_mesh"] = vals.get("phonon_mesh", (20, 20, 20))
            config_dict["phonon_displacement"] = vals.get("phonon_displacement", 1e-2)

        config_json = json.dumps(config_dict, sort_keys=True)

        pending = [
            r
            for r in records
            if not store.has_evaluation(r.id, calculator_name, config_json)
        ]

        if cached_only and pending:
            progress.log(
                f"Cached only: skipping {len(pending)} structure(s) without cached evaluations"
            )
            pending = []

        progress.log(f"{len(records)} structures, {len(pending)} to evaluate")

        if pending:
            with workdir_context(None) as wdir:
                progress.log(f"Working directory: {wdir}")
                calculator = load_calculator(
                    Calculators(calculator_name),
                    wdir,
                    config=vals.get("calculator_config_dict", {}),
                    callback=_TaskCalcCallback(),
                )

                run_eval_loop(
                    pending,
                    store,
                    run_name,
                    calculator,
                    calculator_name,
                    config_json,
                    run_phonons=run_phonons,
                    phonon_displacement=vals.get("phonon_displacement", 1e-2),
                    phonon_supercell=vals.get("phonon_supercell", (3, 3, 3)),
                    phonon_mesh=vals.get("phonon_mesh", (20, 20, 20)),
                    progress_callback=progress.as_callback(),
                    log_callback=progress.log,
                )

        evals = store.get_evaluations(run_name, calculator=calculator_name)
        eval_map = {
            ev.structure_id: ev
            for ev in evals
            if ev.config_json == config_json
        }

        rows = eval_rows_from_cache(records, eval_map, run_name)
        used = len(rows)
        no_cache = len(records) - used
        if no_cache:
            progress.log(
                f"Metrics over {used} cached structures, "
                f"{no_cache} of {len(records)} had no cached evaluation"
            )
        else:
            progress.log(f"Metrics over {used} cached structures")

        self._eval_rows = rows
        progress.finish()

    # ------------------------------------------------------------------ #
    #  Completion
    # ------------------------------------------------------------------ #

    def _on_complete(self) -> None:
        self._running = False
        self._progress_panel.set_finished(True, "Evaluation complete!")

        if not self._eval_rows:
            self._error_text.set_text(("unconv", "No evaluation results."))
            return

        vals = self._eval_vals or {}
        phonon_cutoff = vals.get("phonon_cutoff", -0.15)
        stable_only = vals.get("stable_only", False)

        self._router.push(
            EvalResultsScreen(
                self._state,
                self._router,
                eval_rows=self._eval_rows,
                phonon_cutoff=phonon_cutoff,
                stable_only=stable_only,
                run_name=self._run_name,
            )
        )

    def _on_error(self, error: str) -> None:
        self._running = False
        self._progress_panel.set_finished(False, f"Error: {error}")

