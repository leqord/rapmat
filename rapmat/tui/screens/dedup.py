from pathlib import Path

import urwid

from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
from rapmat.tui.theme import DIALOG_REMAP
from rapmat.tui.widgets.dialog import ModalDialog
from rapmat.tui.widgets.form import (FormGroup, checkbox_field, dropdown_field,
                                     float_field, int_field)
from rapmat.tui.widgets.progress import ProgressPanel
from rapmat.tui.widgets.table import SortableTable

_WATERFALL_COLS = [("Stage", 24), ("Kept", 10), ("Change", 10), ("Notes", 30)]
_PERCENTILE_COLS = [
    ("Survival %", 14),
    ("Threshold", 14),
    ("Kept", 18),
]


class DedupScreen(ScreenBase):
    title = "Dedup Analysis"

    @property
    def breadcrumb_title(self) -> str:
        run = self._state.active_run
        return f"Dedup: {run}" if run else self.title

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._frame: urwid.Frame | None = None
        self._main_body: urwid.Widget | None = None
        self._progress_panel = ProgressPanel(title=" Dedup Progress ")
        self._running = False
        self._result_data: dict | None = None
        self._overlay_open = False
        self._applying = False
        self._applied = False
        self._results_overlay: urwid.Widget | None = None
        self._apply_status: urwid.Text | None = None
        self._apply_bar: urwid.ProgressBar | None = None
        self._apply_text: urwid.Text | None = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._state.refresh_runs_if_needed()
        self._frame = self._build_frame()
        return self._frame

    def bindings(self) -> list[KeyBinding]:
        if self._applying:
            return []
        if self._overlay_open:
            def overlay_active() -> bool:
                return (
                    self._frame is not None
                    and self._frame.body is self._results_overlay
                )
            return [
                KeyBinding(
                    ("a",), "Apply to DB", self._apply_to_db,
                    help="Write the duplicate labels to the database",
                    enabled=overlay_active, priority=10,
                ),
                KeyBinding(
                    ("p",), "Save plot", self._save_plot,
                    help="Save the deduplication analysis plot",
                    enabled=overlay_active, priority=20,
                ),
            ]
        return [
            KeyBinding(
                ("f5",), "Analyze", self._on_start,
                help="Run duplicate analysis", priority=10,
            ),
            KeyBinding(
                ("c",), "Clear DB", self._confirm_clear_duplicates,
                help="Remove all duplicate labels for the selected run",
                enabled=lambda: not self._running
                and self._frame is not None
                and self._frame.body is self._main_body,
                priority=20,
            ),
        ]

    def esc_label(self) -> str:
        if self._overlay_open and not self._running:
            return "Close"
        return super().esc_label()

    def _dialog_host_get(self) -> "urwid.Widget | None":
        return self._frame.body if self._frame is not None else None

    def _dialog_host_set(self, widget: urwid.Widget) -> None:
        self._frame.body = widget

    def refresh_footer(self, message: str = "") -> None:
        if self._applying:
            bar = self._state.status_bar
            if bar:
                bar.set_hints([])
                bar.set_message(message or "Applying to database... please wait")
            return
        super().refresh_footer(message)

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _run_options(self) -> list[str]:
        names = [r.name for r in self._state.runs_cache]
        if self._state.active_run and self._state.active_run not in names:
            names.insert(0, self._state.active_run)
        return names if names else ["(no runs)"]

    def _build_frame(self) -> urwid.Frame:
        from rapmat.core.dedup_analysis import (DEFAULT_SOAP_L_MAX,
                                                DEFAULT_SOAP_N_MAX,
                                                DEFAULT_SOAP_R_CUT,
                                                DEFAULT_SOAP_SIGMA,
                                                METRIC_BY_CHOICE, METRICS)

        self._metrics = METRICS
        self._metric_by_choice = METRIC_BY_CHOICE

        run_opts = self._run_options()
        default_idx = 0
        if self._state.active_run and self._state.active_run in run_opts:
            default_idx = run_opts.index(self._state.active_run)

        self._form = FormGroup(
            [
                dropdown_field("run_name", "Run", run_opts, default=default_idx),
                dropdown_field("stage", "Stage", ["relaxed", ], default=0),
                float_field("dedup_threshold", "Threshold", default=1e-2),
                dropdown_field(
                    "metric", "Metric",
                    list(METRIC_BY_CHOICE), default=0,
                ),
                checkbox_field("pymatgen_dedup", "Pymatgen dedup", default=False),
                float_field("pymatgen_ltol", "Pymatgen ltol", default=0.2),
                float_field("pymatgen_stol", "Pymatgen stol", default=0.3),
                float_field("pymatgen_angle", "Pymatgen angle tol", default=5.0),
                checkbox_field("force_dedup", "Force dedup", default=False),
                float_field("force_cosine", "Force cosine thresh", default=0.95),
                float_field("soap_r_cut", "SOAP r_cut", default=DEFAULT_SOAP_R_CUT),
                int_field("soap_n_max", "SOAP n_max", default=DEFAULT_SOAP_N_MAX),
                int_field("soap_l_max", "SOAP l_max", default=DEFAULT_SOAP_L_MAX),
                float_field("soap_sigma", "SOAP sigma", default=DEFAULT_SOAP_SIGMA),
            ],
            label_width=22,
            groups=[
                ("General", ["run_name", "stage", "dedup_threshold", "metric"]),
                ("Pymatgen", [
                    "pymatgen_dedup", "pymatgen_ltol",
                    "pymatgen_stol", "pymatgen_angle",
                ]),
                ("Forces", ["force_dedup", "force_cosine"]),
                ("SOAP Descriptor", [
                    "soap_r_cut", "soap_n_max", "soap_l_max", "soap_sigma",
                ]),
            ],
        )

        self._metric_hint = urwid.Text(
            ("details", f"  {METRICS['euclidean'].hint}")
        )
        metric_widget = self._form.get_widget("metric")
        if metric_widget is not None:
            urwid.connect_signal(metric_widget, "change", self._on_metric_change)

        self._error_text = urwid.Text("")

        start_btn = urwid.AttrMap(
            urwid.Button("Analyze [F5]", on_press=self._on_start),
            "menu_item",
            focus_map="btn_focus",
        )
        clear_btn = urwid.AttrMap(
            urwid.Button(
                "Clear DB [c]",
                on_press=lambda _b: self._confirm_clear_duplicates(),
            ),
            "menu_item",
            focus_map="btn_focus",
        )

        listbox_form = urwid.ListBox(
            urwid.SimpleListWalker(
                [
                    self._form,
                    self._metric_hint,
                    urwid.Divider(),
                    urwid.Columns(
                        [(18, start_btn), (18, clear_btn)], dividechars=1
                    ),
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
        return urwid.Frame(body=body)

    def _on_metric_change(self, _widget, option: str) -> None:
        metric = self._metric_by_choice.get(option, "euclidean")
        self._metric_hint.set_text(
            ("details", f"  {self._metrics[metric].hint}")
        )

    # ------------------------------------------------------------------ #
    #  Submit
    # ------------------------------------------------------------------ #

    def _validate(self, vals: dict) -> list[str]:
        errors = self._form.validate()
        threshold = vals["dedup_threshold"]
        if threshold <= 0:
            errors.append("Threshold must be > 0")
        elif vals["metric"] == "cosine" and threshold >= 2:
            errors.append(
                f"Threshold must be < 2 for the {self._metrics['cosine'].short} metric"
            )
        return errors

    def _on_start(self, _btn=None) -> None:
        if self._running or self._applying:
            return

        vals = self._form.get_values()
        vals["metric"] = self._metric_by_choice.get(vals["metric"], "euclidean")
        errors = self._validate(vals)
        if errors:
            self._error_text.set_text(("form_error", " " + "; ".join(errors)))
            return

        self._running = True
        self._applied = False
        self._error_text.set_text("")
        self._progress_panel.clear()

        # Close any existing overlay
        if self._overlay_open:
            self._close_overlay()

        self.run_task(
            lambda prog: self._worker(prog, vals),
            on_progress=self._progress_panel.set_progress,
            on_log=self._progress_panel.add_log,
            on_complete=self._on_complete,
            on_error=self._on_error,
        )

    def _worker(self, progress, vals: dict) -> None:
        from rapmat.core.dedup_analysis import (DedupAnalysisError,
                                                run_dedup_analysis)

        try:
            analysis = run_dedup_analysis(
                self._state.store,
                vals["run_name"],
                stage=vals["stage"],
                threshold=vals["dedup_threshold"],
                metric=vals["metric"],
                soap_r_cut=vals["soap_r_cut"],
                soap_n_max=int(vals["soap_n_max"]),
                soap_l_max=int(vals["soap_l_max"]),
                soap_sigma=vals["soap_sigma"],
                use_pymatgen=vals["pymatgen_dedup"],
                ltol=vals["pymatgen_ltol"],
                stol=vals["pymatgen_stol"],
                angle_tol=vals["pymatgen_angle"],
                use_forces=vals["force_dedup"],
                force_cosine_threshold=vals["force_cosine"],
                progress_callback=progress.as_callback(
                    raise_on_cancel=False, default_is_log=False
                ),
            )
        except DedupAnalysisError as exc:
            progress.fail(str(exc))
            return

        self._result_data = analysis.to_result_dict()
        progress.update(4, 4, "Done")
        progress.finish()

    # ------------------------------------------------------------------ #
    #  Completion - show results overlay
    # ------------------------------------------------------------------ #

    def _on_complete(self) -> None:
        self._running = False
        self._progress_panel.set_finished(True, "Analysis complete!")

        d = self._result_data
        if d is None:
            return

        self._show_results_overlay(d)

    def _show_results_overlay(self, d: dict) -> None:
        if self._frame is None:
            return

        sim = d["sim"]
        dropped_pct = 100 * sim.final_dropped / max(sim.total, 1)

        # -- Header ------------------------------------ #
        summary = urwid.Text(
            [
                ("success", f"  {d['n_structs']}"),
                ("details", " structures -> "),
                ("success", f"{sim.kept}"),
                ("details", " unique  ("),
                ("unconv", f"{sim.final_dropped} duplicates, {dropped_pct:.1f}%"),
                ("details", ")"),
            ]
        )

        # -- Distance stats  ---------------------------- #
        below_pct = 100 * d["below_thresh"] / max(d["n_pairs"], 1)
        dist_stats = urwid.Text(
            [
                ("form_label", "  Pairs: "),
                ("details", f"{d['n_pairs']:,}"),
                ("form_label", "  Min: "),
                ("details", f"{d['min_dist']:.6f}"),
                ("form_label", "  Max: "),
                ("details", f"{d['max_dist']:.6f}"),
                ("form_label", "  Mean: "),
                ("details", f"{d['mean_dist']:.6f}"),
                ("form_label", "  Med: "),
                ("details", f"{d['median_dist']:.6f}"),
                ("form_label", "  Std: "),
                ("details", f"{d['std_dist']:.6f}"),
                ("form_label", f"  Below {d['threshold']}: "),
                ("unconv" if d["below_thresh"] > 0 else "details",
                 f"{d['below_thresh']} ({below_pct:.1f}%)"),
            ]
        )

        # -- Waterfall table ------------------------------------------ #
        waterfall_rows = [
            {
                "stage": "Initial",
                "kept": str(sim.total),
                "change": "",
                "notes": f"All {d['stage']}",
            },
            {
                "stage": "Stage 1: Vector (L2)",
                "kept": str(
                    sim.total
                    - sim.dropped_by_vector
                    - sim.rescued_by_pymatgen
                    - sim.rescued_by_forces
                ),
                "change": f"-{sim.dropped_by_vector + sim.rescued_by_pymatgen + sim.rescued_by_forces}",
                "notes": f"threshold < {d['threshold']}",
            },
        ]
        if d["use_pymatgen"]:
            after_vec = (
                sim.total
                - sim.dropped_by_vector
                - sim.rescued_by_pymatgen
                - sim.rescued_by_forces
            )
            waterfall_rows.append(
                {
                    "stage": "Stage 2: Pymatgen",
                    "kept": str(after_vec + sim.rescued_by_pymatgen),
                    "change": (
                        f"+{sim.rescued_by_pymatgen}"
                        if sim.rescued_by_pymatgen
                        else "0"
                    ),
                    "notes": f"{sim.pymatgen_mismatches}/{sim.pymatgen_comparisons} collisions",
                }
            )
        if d["use_forces"]:
            waterfall_rows.append(
                {
                    "stage": "Stage 3: Forces",
                    "kept": str(sim.kept),
                    "change": (
                        f"+{sim.rescued_by_forces}" if sim.rescued_by_forces else "0"
                    ),
                    "notes": f"{sim.force_mismatches}/{sim.force_comparisons} disagreements",
                }
            )
        waterfall_rows.append(
            {
                "stage": "Final",
                "kept": str(sim.kept),
                "change": "",
                "notes": f"{sim.final_dropped} dropped, {sim.kept} unique",
            }
        )

        n_waterfall_rows = len(waterfall_rows)
        waterfall_table = SortableTable(
            columns=_WATERFALL_COLS,
            row_data=waterfall_rows,
            format_row=lambda r: [r["stage"], r["kept"], r["change"], r["notes"]],
        )

        # -- Percentile table ----------------------------------------- #
        perc_rows = []
        for p, thresh, kept in d["percentiles"]:
            perc_rows.append(
                {
                    "pct": f"{p}%",
                    "threshold": f"{thresh:.4f}",
                    "kept": f"{kept} / {sim.total}",
                }
            )

        percentile_table = SortableTable(
            columns=_PERCENTILE_COLS,
            row_data=perc_rows,
            format_row=lambda r: [r["pct"], r["threshold"], r["kept"]],
        )

        # -- Collision summary ------------------------- #
        collision_widgets = []
        if d["use_pymatgen"] and sim.pymatgen_comparisons > 0:
            rate = 100 * sim.pymatgen_mismatches / sim.pymatgen_comparisons
            agrees = sim.pymatgen_comparisons - sim.pymatgen_mismatches
            collision_widgets.append(
                urwid.Text(
                    [
                        ("form_label", "  Pymatgen: "),
                        ("details", f"{sim.pymatgen_comparisons} comparisons, "),
                        ("success", f"{agrees} agree"),
                        ("details", ", "),
                        ("unconv" if sim.pymatgen_mismatches > 0 else "details",
                         f"{sim.pymatgen_mismatches} disagree ({rate:.1f}%)"),
                    ]
                )
            )
        if d["use_forces"] and sim.force_comparisons > 0:
            rate = 100 * sim.force_mismatches / sim.force_comparisons
            agrees = sim.force_comparisons - sim.force_mismatches
            collision_widgets.append(
                urwid.Text(
                    [
                        ("form_label", "  Forces:   "),
                        ("details", f"{sim.force_comparisons} comparisons, "),
                        ("success", f"{agrees} agree"),
                        ("details", ", "),
                        ("unconv" if sim.force_mismatches > 0 else "details",
                         f"{sim.force_mismatches} disagree ({rate:.1f}%)"),
                    ]
                )
            )

        # -- Scrollable --------------------------------- #
        body_widgets = [
            summary,
            urwid.Divider(),
            urwid.Text(("section", " Distance Statistics")),
            urwid.Divider("-"),
            dist_stats,
            urwid.Divider(),
            urwid.Text(("section", " Dedup Simulation Waterfall")),
            urwid.Divider("-"),
        ]

        waterfall_height = n_waterfall_rows + 1
        body_widgets.append(urwid.BoxAdapter(waterfall_table, waterfall_height))

        body_widgets.extend(
            [
                urwid.Divider(),
                urwid.Text(("section", " Survival Thresholds")),
                urwid.Divider("-"),
            ]
        )


        perc_height = len(perc_rows) + 1
        body_widgets.append(urwid.BoxAdapter(percentile_table, perc_height))

        if collision_widgets:
            body_widgets.extend(
                [
                    urwid.Divider(),
                    urwid.Text(("section", " Collision Summary")),
                    urwid.Divider("-"),
                ]
            )
            body_widgets.extend(collision_widgets)

        scrollable_body = urwid.ListBox(urwid.SimpleListWalker(body_widgets))

        # -- Pinned footer with buttons ------------------------ #
        apply_btn = urwid.AttrMap(
            urwid.Button("Apply to DB", on_press=lambda _: self._apply_to_db()),
            "menu_item",
            focus_map="btn_focus",
        )
        plot_btn = urwid.AttrMap(
            urwid.Button("Save Plot", on_press=lambda _: self._save_plot()),
            "menu_item",
            focus_map="btn_focus",
        )
        close_btn = urwid.AttrMap(
            urwid.Button("Close", on_press=lambda _: self._close_overlay()),
            "menu_item",
            focus_map="btn_focus",
        )

        footer_buttons = urwid.Columns(
            [
                (18, apply_btn),
                (15, plot_btn),
                (12, close_btn),
            ],
            dividechars=2,
        )
        self._apply_status = urwid.Text(
            ("success", "Applied") if self._applied else ""
        )
        footer = urwid.Pile(
            [
                self._apply_status,
                urwid.Divider("-"),
                footer_buttons,
            ]
        )

        run_name = d["run_name"]
        stage = d["stage"]
        metric = self._metrics[d["metric"]].short
        overlay_body = urwid.Frame(
            body=scrollable_body,
            footer=footer,
        )

        inner = urwid.LineBox(
            urwid.Padding(overlay_body, left=1, right=1),
            title=f"Deduplication: {run_name} ({stage}, {metric}) ",
        )

        overlay = urwid.Overlay(
            urwid.AttrMap(inner, DIALOG_REMAP),
            self._main_body,
            align=urwid.CENTER,
            width=(urwid.RELATIVE, 90),
            valign=urwid.MIDDLE,
            height=(urwid.RELATIVE, 85),
            min_width=60,
            min_height=20,
        )

        self._overlay_open = True
        self._results_overlay = overlay
        self._frame.body = overlay
        self.refresh_footer()

    def _close_overlay(self) -> None:
        if self._frame is None or self._main_body is None:
            return
        self._overlay_open = False
        self._frame.body = self._main_body
        self.refresh_footer()

    def _on_error(self, error: str) -> None:
        self._running = False
        self._progress_panel.set_finished(False, f"Error: {error}")

    # ------------------------------------------------------------------ #
    #  Save plot
    # ------------------------------------------------------------------ #

    def _save_plot(self) -> None:
        if self._result_data is None or "distances" not in self._result_data:
            return
        from rapmat.core.dedup_analysis import plot_distance_histogram

        d = self._result_data
        plot_path = Path(f"dedup_{d['run_name']}_{d['stage']}.png")
        try:
            plot_distance_histogram(
                d["distances"],
                threshold=d["threshold"],
                save_path=plot_path,
                title=f"Pairwise Distance Distribution - {d['run_name']} ({d['stage']})",
                axis_label=self._metrics[d["metric"]].axis,
            )
            if self._state.status_bar:
                self._state.status_bar.set_message(f"Plot saved to {plot_path}")
        except Exception as e:
            if self._state.status_bar:
                self._state.status_bar.set_message(f"Plot error: {e}")

    # ------------------------------------------------------------------ #
    #  Apply dedup results to DB
    # ------------------------------------------------------------------ #

    def _apply_to_db(self) -> None:
        if self._applying:
            return
        if self._result_data is None:
            return
        sim = self._result_data.get("sim")
        if sim is None:
            return
        if self._frame is None or self._results_overlay is None:
            return

        store = self._state.store
        dropped_ids = sim.dropped_ids
        kept_ids = sim.kept_ids
        total = len(dropped_ids) + len(kept_ids)

        self._applying = True

        self._apply_bar = urwid.ProgressBar(
            "progress", "pg_done", current=0, done=max(total, 1)
        )
        self._apply_text = urwid.Text(f"0 / {total}", align="center")
        inner = urwid.LineBox(
            urwid.Padding(
                urwid.Pile(
                    [
                        ("pack", self._apply_bar),
                        ("pack", urwid.Divider()),
                        ("pack", self._apply_text),
                    ]
                ),
                left=2,
                right=2,
            ),
            title=" Applying to database... ",
        )
        progress_overlay = urwid.Overlay(
            urwid.AttrMap(inner, DIALOG_REMAP),
            self._results_overlay,
            align=urwid.CENTER,
            width=(urwid.RELATIVE, 50),
            valign=urwid.MIDDLE,
            height=urwid.PACK,
            min_width=40,
        )
        self._frame.body = progress_overlay
        self.refresh_footer()

        if self._state.loop is not None:
            self._state.loop.draw_screen()

        def _do_apply(prog) -> None:
            store.mark_duplicates(
                dropped_ids,
                kept_ids,
                progress_callback=lambda d, t: prog.update(d, t, f"{d} / {t}"),
            )

        self.run_task(
            _do_apply,
            on_progress=self._on_apply_progress,
            on_complete=lambda: self._on_apply_complete(
                len(dropped_ids), len(kept_ids)
            ),
            on_error=self._on_apply_error,
        )

    def _on_apply_progress(self, current: int, total: int, message: str = "") -> None:
        if self._apply_bar is not None:
            self._apply_bar.done = max(total, 1)
            self._apply_bar.set_completion(current)
        if self._apply_text is not None:
            self._apply_text.set_text(message or f"{current} / {total}")

    def _on_apply_complete(self, n_dropped: int, n_kept: int) -> None:
        self._applying = False
        self._applied = True
        if self._apply_status is not None:
            self._apply_status.set_text(("success", " Applied to DB"))
        msg = (
            f"Applied: {n_dropped} marked duplicate, "
            f"{n_kept} marked unique"
        )

        if self._frame is not None and self._results_overlay is not None:

            def _close() -> None:
                if self._frame is not None and self._results_overlay is not None:
                    self._frame.body = self._results_overlay
                self.refresh_footer()

            dlg = ModalDialog.info(
                title=" Applied ",
                message=msg,
                parent=self._results_overlay,
                on_close=_close,
            )
            self._frame.body = dlg
            self.refresh_footer()

        if self._state.status_bar:
            self._state.status_bar.set_message(msg)

    def _on_apply_error(self, error: str) -> None:
        self._applying = False

        if self._frame is not None and self._results_overlay is not None:

            def _close() -> None:
                if self._frame is not None and self._results_overlay is not None:
                    self._frame.body = self._results_overlay
                self.refresh_footer()

            dlg = ModalDialog.error(
                title=" Apply Failed ",
                message=f"Could not apply to database:\n\n  {error}",
                parent=self._results_overlay,
                actions=[("OK", _close)],
            )
            self._frame.body = dlg
            self.refresh_footer()

        if self._state.status_bar:
            self._state.status_bar.set_message(f"Apply failed: {error}")

    # ------------------------------------------------------------------ #
    #  Clear duplicate labels
    # ------------------------------------------------------------------ #

    def _confirm_clear_duplicates(self) -> None:
        if self._running or self._applying:
            return
        run_name = self._form.get_values().get("run_name")
        if not run_name or run_name == "(no runs)":
            if self._state.status_bar:
                self._state.status_bar.set_message("No run selected.")
            return

        self.confirm_dialog(
            "Clear duplicate labels",
            f"Remove all duplicate labels for run '{run_name}'?",
            lambda: self._do_clear_duplicates(run_name),
        )

    def _do_clear_duplicates(self, run_name: str) -> None:
        try:
            self._state.store.clear_run_duplicates(run_name)
            msg = f"Cleared duplicate labels for '{run_name}'."
        except Exception as exc:
            msg = f"Clear failed: {exc}"
        if self._state.status_bar:
            self._state.status_bar.set_message(msg)

    # ------------------------------------------------------------------ #
    #  Key handling
    # ------------------------------------------------------------------ #

    def keypress(self, size: tuple, key: str) -> str | None:
        if self._applying:
            return None
        return super().keypress(size, key)

    def _on_esc(self) -> bool:
        if super()._on_esc():
            return True
        if self._overlay_open:
            self._close_overlay()
            return True
        return False
