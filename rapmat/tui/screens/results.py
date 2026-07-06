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


class ResultsScreen(BaseResultsScreen):
    title = "Results"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._run_name: str = ""

    def bindings(self) -> list[KeyBinding]:
        return super().bindings() + [
            KeyBinding(
                ("v",), "Eval", self._open_eval,
                help="Evaluate using a reference calculator", priority=15,
            ),
        ]

    def _fetch_data(self, progress_callback=None) -> None:
        run_name = self._state.active_run or ""
        self._run_name = run_name

        store = self._state.store

        meta = store.get_run_metadata(run_name)
        cfg = meta.search_config if meta else SearchConfig()
        self._pressure_gpa = cfg.pressure_gpa
        self._phonon_cutoff = cfg.phonon_cutoff
        self._symprec = self._get_symprec()

        records = store.get_structures(
            run_name, status="relaxed", symprec=self._symprec,
            progress_callback=progress_callback,
        )

        if self._pressure_gpa > 0:
            records.sort(
                key=lambda r: (
                    r.enthalpy_per_atom
                    if r.enthalpy_per_atom is not None
                    else r.energy_per_atom
                )
            )
        else:
            records.sort(key=lambda r: r.energy_per_atom)

        self._results = [
            ResultRow(structure=rec, index=idx + 1, run_name=run_name)
            for idx, rec in enumerate(records)
        ]

        self.title = (
            f"Results: {run_name} {self._pressure_gpa} GPa {len(self._results)} relaxed"
        )
        is_bulk = cfg.domain == "bulk"
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
        self._show_flags_col = any(
            r.duplicate is True or r.excluded for r in self._results
        )

    def _columns_def(self) -> list[tuple[str, int]]:
        epa_label = "H/A" if self._pressure_gpa > 0 else "E/A"
        cols = [
            ("#", 4),
            ("ID", 10),
            ("Formula", 14),
            ("Final SG", 14),
            (epa_label, 10),
        ]
        cols.append(("Fmax", 8))
        if self._show_thickness:
            cols.append(("Thick(A)", 9))
        if self._show_dynamical_stability:
            cols.append(("Dyn.", 4))
        if self._show_flags_col:
            cols.append(("Flags", 6))
        cols.append(("Status", 8))
        return cols

    def _format_row(self, result: ResultRow) -> list[str]:
        full_id = str(result.structure_id)
        short_id = full_id.split("/")[-1] if "/" in full_id else full_id
        if len(short_id) > 10:
            short_id = short_id[:10]

        if self._pressure_gpa > 0:
            h = result.enthalpy_per_atom
            epa_val = h if h is not None else result.energy_per_atom
        else:
            epa_val = result.energy_per_atom

        row = [
            str(result.index),
            short_id,
            result.formula or "N/A",
            result.final_spg or "N/A",
            f"{epa_val:.4f}",
        ]

        fmax = result.fmax
        row.append(f"{fmax:.3f}" if fmax is not None else "N/A")

        if self._show_thickness:
            t = result.thickness
            row.append("" if t is None else f"{t:.2f}")
        if self._show_dynamical_stability:
            dyn = _dyn_stability(result, self._phonon_cutoff)
            row.append("Yes" if dyn is True else ("No" if dyn is False else "N/A"))
        if self._show_flags_col:
            row.append(_flags_str(result))

        conv = result.converged
        row.append("OK" if conv is True or conv is None else "Unconv")
        return row

    def _get_symprec(self) -> float:
        meta = self._state.store.get_run_metadata(self._run_name)
        return meta.search_config.symprec if meta else SearchConfig().symprec

    def _persist_symprec(self, value: float) -> None:
        self._state.store.set_run_config_value(self._run_name, "symprec", value)

    def _on_phonon_complete(self, phonon_cutoff: float) -> None:
        self._state.store.set_run_config_value(
            self._run_name, "phonon_cutoff", phonon_cutoff
        )

    def _phonon_clear_target(self) -> list[str]:
        return [self._run_name] if self._run_name else []

    def _open_eval(self) -> None:
        from rapmat.tui.screens.eval import EvalScreen

        display_results = list(self._get_display_results())
        filtered_ids = [r.structure_id for r in display_results]

        self._router.push(
            EvalScreen(self._state, self._router, self._run_name, filtered_ids)
        )
