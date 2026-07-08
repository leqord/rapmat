from dataclasses import replace
from pathlib import Path

from rapmat.core.config import SearchConfig
from rapmat.core.entities import ResultRow
from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base_results import (
    BaseResultsScreen,
    _dyn_stability,
    _flags_str,
)
from rapmat.tui.state import AppState
from rapmat.tui.widgets.dialog import ModalDialog

_SORT_OPTIONS: list[tuple[str, str]] = [
    ("Formation energy", "formation"),
    ("Energy above hull", "eah"),
    ("Composition (x)", "composition"),
    ("Energy/atom", "energy"),
]


def sort_result_rows(results: list[ResultRow], sort_key: str) -> None:
    """Sort phase analysis rows in place and recalculate ``index``.
    """
    inf = float("inf")

    def num(row: ResultRow, attr: str) -> float:
        v = getattr(row, attr)
        return inf if v is None else v

    if sort_key == "composition":
        results.sort(key=lambda r: num(r, "composition_frac"))
    elif sort_key == "eah":
        results.sort(
            key=lambda r: (num(r, "energy_above_hull"), num(r, "formation_energy"))
        )
    elif sort_key == "energy":
        results.sort(key=lambda r: num(r, "effective_per_atom"))
    else:  # "formation"
        results.sort(
            key=lambda r: (num(r, "formation_energy"), num(r, "energy_above_hull"))
        )

    for i, row in enumerate(results):
        row.index = i + 1


class PhaseAnalysisScreen(BaseResultsScreen):
    title = "Phase Analysis"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._study_id: str = ""
        self._study_system: str = ""
        self._system_size: int = 0
        self._use_enthalpy: bool = False
        self._show_all: bool = True
        self._hull_cutoff: float = 0.0
        self._sort_key: str = "formation"

    def bindings(self) -> list[KeyBinding]:
        base = [
            replace(b, enabled=lambda: self._show_thickness)
            if b.key_text() == "t"
            else b
            for b in super().bindings()
        ]
        return base + [
            KeyBinding(
                ("a",),
                lambda: "Show Best" if self._show_all else "Show All",
                self._action_toggle_scope,
                help="Toggle between all structures and on-hull",
                priority=12,
            ),
            KeyBinding(
                ("c",), "Cutoff", self._open_cutoff_modal,
                help="Set the energy-above-hull cutoff filter",
                enabled=lambda: not self._show_all, priority=14,
            ),
            KeyBinding(
                ("g",), "Plot", self._open_save_plot_modal,
                help="Save the binary hull plot",
                enabled=lambda: self._system_size == 2, priority=22,
            ),
        ]

    # ------------------------------------------------------------------ #
    #  Data fetch
    # ------------------------------------------------------------------ #

    def _apply_fetch_result(self, box: dict) -> None:
        self._study_id = box.get("study_id", "")
        self._study_system = box.get("system", "")
        self._system_size = box.get("system_size", 0)
        self._use_enthalpy = box.get("use_enthalpy", False)

        study = box.get("study")
        study_cfg = study.search_config if study else SearchConfig()
        self._phonon_cutoff = study_cfg.phonon_cutoff
        self._pressure_gpa = study_cfg.pressure_gpa

        sd = box.get("sd", [])
        self._results = list(sd)
        sort_result_rows(self._results, self._sort_key)

        self.title = (
            f"Phase Analysis: {self._study_system} | {len(self._results)} structures"
        )
        is_bulk = study_cfg.domain == "bulk"
        self._show_thickness = (not is_bulk) and any(
            r.thickness is not None for r in self._results
        )
        self._show_dynamical_stability = any(
            r.min_phonon_freq is not None or r.dynamical_stability is not None
            for r in self._results
        )
        self._show_duplicate_col = any(
            r.duplicate is not None for r in self._results
        )

        self._show_flags_col = bool(self._results)

    def _fetch_data(self, progress_callback=None) -> None:
        def _log(msg: str) -> None:
            if progress_callback is not None:
                progress_callback(0, 1, msg, True)

        _log("Resolving study...")
        study_id = self._state.active_study
        if not study_id:
            return

        store = self._state.store
        study = store.get_study(str(study_id))
        if not study:
            return

        box: dict = {}
        symprec = study.search_config.symprec
        self._symprec = symprec
        system = study.system
        from rapmat.utils.common import parse_system
        from rapmat.core.hull import (
            build_energy_ranking,
            build_phase_diagram,
            collect_study_structures,
            hull_input,
        )

        elements = parse_system(system)
        system_size = len(elements)
        _log(
            f"System: {system} "
            f"({system_size} element{'s' if system_size != 1 else ''})"
        )

        structures, _system, use_enthalpy = collect_study_structures(
            store, str(study_id), symprec=symprec
        )

        kept = hull_input(
            structures,
            hide_unconverged=self._hide_unconverged,
            hide_duplicates=self._hide_duplicates,
            hide_excluded=self._hide_excluded,
        )

        if system_size < 2:
            _log("Building energy ranking...")
            sd = build_energy_ranking(
                kept,
                use_enthalpy=use_enthalpy,
                show_all=self._show_all,
                hull_cutoff=self._hull_cutoff,
            )
        else:
            _log("Building phase diagram...")
            _, sd = build_phase_diagram(
                kept,
                system,
                use_enthalpy=use_enthalpy,
                show_all=self._show_all,
                hull_cutoff=self._hull_cutoff,
            )

        if progress_callback is not None:
            progress_callback(1, 1, f"Done - {len(sd)} structures")

        box.update(
            {
                "study_id": str(study_id),
                "study": study,
                "system": system,
                "system_size": system_size,
                "sd": sd,
                "use_enthalpy": use_enthalpy,
            }
        )
        self._apply_fetch_result(box)

    def _columns_def(self) -> list[tuple[str, int]]:
        epa_label = "H/A" if self._use_enthalpy else "E/A"
        form_label = "H_form" if self._use_enthalpy else "E_form"
        cols: list[tuple[str, int]] = [("#", 4), ("ID", 28)]
        if self._system_size < 2:
            cols += [
                ("Formula", 14),
                ("Final SG", 16),
                (epa_label, 10),
            ]
        elif self._system_size == 2:
            cols += [
                ("Formula", 14),
                ("Final SG", 16),
                ("x", 8),
                (epa_label, 10),
                (form_label, 10),
                ("EAH", 10),
            ]
        else:
            cols += [
                ("Formula", 14),
                ("Final SG", 16),
                (epa_label, 10),
                (form_label, 10),
                ("EAH", 10),
            ]
        if self._show_flags_col:
            cols.append(("Flags", 6))
        if self._show_thickness:
            cols.append(("Thick(A)", 9))
        if self._show_dynamical_stability:
            cols.append(("Dyn", 5))
        return cols

    def _format_row(self, result: ResultRow) -> list[str]:
        formula = result.formula or result.reduced_formula or "N/A"
        epa = result.display_epa
        spg = result.final_spg or ""

        lead = [str(result.index), str(result.structure_id)]

        if self._system_size < 2:
            row = lead + [formula, spg, f"{epa:.4f}"]
        elif self._system_size == 2:
            x = f"{result.composition_frac or 0.0:.3f}"
            e_form = f"{result.formation_energy or 0.0:.4f}"
            eah = f"{result.energy_above_hull or 0.0:.4f}"
            row = lead + [formula, spg, x, f"{epa:.4f}", e_form, eah]
        else:
            e_form = f"{result.formation_energy or 0.0:.4f}"
            eah = f"{result.energy_above_hull or 0.0:.4f}"
            row = lead + [formula, spg, f"{epa:.4f}", e_form, eah]

        if self._show_flags_col:
            row.append(_flags_str(result))
        if self._show_thickness:
            t = result.thickness
            row.append("" if t is None else f"{t:.2f}")
        if self._show_dynamical_stability:
            dyn = _dyn_stability(result, self._phonon_cutoff)
            row.append("Yes" if dyn is True else ("No" if dyn is False else ""))
        return row

    def _refresh_after_membership_change(self) -> None:

        self._start_async_fetch()

    def _extra_option_fields(self) -> list:
        from rapmat.tui.widgets.form import dropdown_field

        keys = [k for _, k in _SORT_OPTIONS]
        default_idx = keys.index(self._sort_key) if self._sort_key in keys else 0
        return [
            dropdown_field(
                "sort",
                "Sort by",
                options=[label for label, _ in _SORT_OPTIONS],
                default=default_idx,
            )
        ]

    def _apply_extra_options(self, vals: dict) -> str | None:
        label = vals.get("sort")
        key = next((k for lbl, k in _SORT_OPTIONS if lbl == label), self._sort_key)
        self._sort_key = key
        sort_result_rows(self._results, self._sort_key)
        return f"Sorted by {label.lower()}." if label else None

    def _get_symprec(self) -> float:
        study = self._state.store.get_study(self._study_id)
        return study.search_config.symprec if study else SearchConfig().symprec

    def _persist_symprec(self, value: float) -> None:
        self._state.store.set_study_config_value(self._study_id, "symprec", value)

    def _get_extra_details(self, result: ResultRow) -> list:
        extra: list = []
        quantity = "Enthalpy" if self._use_enthalpy else "Energy"
        if self._system_size >= 2:
            if result.formation_energy is not None:
                extra.append(
                    (
                        "details",
                        f"Formation {quantity}: {result.formation_energy:.4f} eV/A\n",
                    )
                )
            if result.energy_above_hull is not None:
                extra.append(
                    (
                        "details",
                        f"{quantity} Above Hull: {result.energy_above_hull:.4f} eV/A\n",
                    )
                )
            if result.is_stable is not None:
                extra.append(
                    (
                        "details",
                        f"Hull Stable: {'Yes' if result.is_stable else 'No'}\n",
                    )
                )
        return extra

    def _save_subdir(self) -> str | None:
        return self._study_id or None

    def _save_ident(self, result: ResultRow) -> str:
        return f"{self._study_id}_{result.index}_{result.structure_id}"

    def _on_phonon_complete(self, phonon_cutoff: float) -> None:

        self._state.store.set_study_config_value(
            self._study_id, "phonon_cutoff", phonon_cutoff
        )

    def _phonon_clear_target(self) -> list[str]:
        runs = self._state.store.get_study_runs(self._study_id)
        return [r.name for r in runs if r.name]

    def _action_toggle_scope(self) -> None:
        if self._show_all:
            self._open_cutoff_modal()
        else:
            self._show_all = True
            self._start_async_fetch()

    def _open_cutoff_modal(self) -> None:
        def _factory(parent, close):
            def _on_save(val_str: str) -> None:
                close()
                try:
                    cutoff = float(val_str)
                    self._hull_cutoff = max(0.0, cutoff)
                    self._show_all = False
                    self._start_async_fetch()
                except ValueError:
                    self._show_message("Invalid cutoff: must be a number.")

            return ModalDialog.input_text(
                title="Hull Cutoff",
                message="Enter Energy Above Hull cutoff (eV/A):",
                parent=parent,
                on_save=_on_save,
                on_cancel=close,
                default=f"{self._hull_cutoff:.2f}",
            )

        self.show_dialog(_factory)

    def _open_save_plot_modal(self) -> None:
        if self._system_size != 2:
            return

        def _factory(parent, close):
            def _do_save(path_str: str) -> None:
                close()
                plot_path = Path(path_str)
                try:
                    from rapmat.core.hull import plot_binary_hull

                    plot_binary_hull(
                        self._results,
                        self._study_system,
                        save_path=plot_path,
                        show=False,
                        use_enthalpy=self._use_enthalpy,
                    )
                    self._show_message(f"Plot saved to {plot_path}")
                except Exception as exc:
                    self._show_message(f"Save failed: {exc}")

            return ModalDialog.input_text(
                title="Save Hull Plot",
                message="Enter file path for the hull plot:",
                parent=parent,
                on_save=_do_save,
                on_cancel=close,
                default="hull.png",
            )

        self.show_dialog(_factory)
