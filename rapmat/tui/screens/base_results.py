import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import urwid

from rapmat.core.entities import ResultRow
from rapmat.utils.structure import DEFAULT_SYMPREC
from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
from rapmat.tui.tasks import BackgroundTask
from rapmat.tui.widgets.dialog import FormDialog, ModalDialog
from rapmat.tui.widgets.dropdown import DropdownSelect
from rapmat.tui.widgets.progress import ProgressPanel
from rapmat.tui.widgets.search import SubmitSearchEdit
from rapmat.tui.widgets.table import SortableTable


def _dyn_stability(result: "ResultRow", phonon_cutoff: float | None) -> Optional[bool]:
    min_freq = result.min_phonon_freq
    if min_freq is not None and phonon_cutoff is not None:
        try:
            f = float(min_freq)
            if not math.isnan(f):
                return f >= phonon_cutoff
        except (TypeError, ValueError):
            pass
    return result.dynamical_stability


@dataclass
class SaveOptions:
    fmt: str
    directory: str
    cell_mode: str
    save_all: bool
    version: str = "relaxed"
    symprec: float = 1e-3
    table_export: str | None = "txt"
    save_dispersion: bool = False


def _yes_no(val: bool | None, na: str = "N/A") -> str:
    if val is True:
        return "Yes"
    if val is False:
        return "No"
    return na


def _row_attr(result: "ResultRow") -> str:
    """Grey out rows.
    """
    if result.excluded:
        return "unconv"
    if not result.converged:
        return "unconv"
    if result.duplicate is True:
        return "unconv"
    return "body"


def _flags_str(result: "ResultRow") -> str:
    """Flags: R(eference endpoint) H(ull) D(uplicate) E(xcluded)."""
    chars = []
    if result.is_reference:
        chars.append("R")
    if result.is_stable:
        chars.append("H")
    if result.duplicate is True:
        chars.append("D")
    if result.excluded:
        chars.append("E")
    return "".join(chars)


class _ResultsFooter(urwid.WidgetWrap):
    def __init__(self, screen: "BaseResultsScreen") -> None:
        self._screen = screen
        self._search = SubmitSearchEdit(screen.apply_search, screen.exit_search)
        self._pile = urwid.Pile([])
        super().__init__(self._pile)

    def show_search(self) -> None:
        self._search.set_edit_text("")
        self._pile.contents = [(self._search, self._pile.options())]

    def show_status(self, message: str = "") -> None:
        self._pile.contents = []
        self._screen.refresh_footer(message)


class _SaveDialog(ModalDialog):
    def __init__(
        self,
        parent: urwid.Widget,
        on_save,
        num_filtered: int = 1,
        default_dir: str = "",
        has_initial: bool = False,
        default_symprec: float = 1e-3,
        has_phonon: bool = False,
    ) -> None:
        self._scope_dropdown = DropdownSelect(
            "Scope",
            options=["Focused structure only", f"All {num_filtered} filtered structures"],
            default=0,
        )

        self._version_dropdown = DropdownSelect(
            "Stage",
            options=["Relaxed", "Candidate"],
            default=0,
        )

        self._fmt_dropdown = DropdownSelect(
            "Format",
            options=["cif", "xyz"],
            default=0,
        )
        if not default_dir:
            default_dir = str(Path.cwd())
        self._dir_edit = urwid.Edit(caption="Directory: ", edit_text=default_dir)
        self._cell_dropdown = DropdownSelect(
            "Cell",
            options=["As-is", "Conventional", "Primitive"],
            default=0,
        )
        self._symprec_edit = urwid.Edit(
            caption="Symprec: ", edit_text=f"{default_symprec:g}",
        )
        self._table_export_dropdown = DropdownSelect(
            "Results table",
            options=["txt", "csv", "None"],
            default=0,
        )
        self._has_phonon = has_phonon
        self._dispersion_dropdown = DropdownSelect(
            "Dispersion plot",
            options=["No", "Yes"],
            default=0,
        )

        def _ok(_btn: urwid.Button) -> None:
            save_all = self._scope_dropdown.selected == 1
            version = "initial" if self._version_dropdown.value == "Candidate" else "relaxed"
            fmt = self._fmt_dropdown.value
            directory = self._dir_edit.edit_text.strip() or default_dir
            cell_mode = self._cell_dropdown.value
            try:
                symprec = float(self._symprec_edit.edit_text.strip())
            except ValueError:
                symprec = default_symprec
            table_export = self._table_export_dropdown.value
            if table_export == "None":
                table_export = None
            save_dispersion = (
                self._has_phonon and self._dispersion_dropdown.value == "Yes"
            )
            self._emit("close", True)
            on_save(
                SaveOptions(
                    fmt=fmt, directory=directory, cell_mode=cell_mode,
                    save_all=save_all, version=version, symprec=symprec,
                    table_export=table_export, save_dispersion=save_dispersion,
                )
            )

        def _cancel(_btn: urwid.Button) -> None:
            self._emit("close", False)

        ok_btn = urwid.AttrMap(
            urwid.Button("Save", on_press=_ok), None, focus_map="btn_focus"
        )
        cancel_btn = urwid.AttrMap(
            urwid.Button("Cancel", on_press=_cancel), None, focus_map="btn_focus"
        )

        body_widgets = [
            self._scope_dropdown,
            urwid.Divider(),
        ]

        if has_initial:
            body_widgets.extend([
                self._version_dropdown,
                urwid.Divider(),
            ])

        body_widgets.extend([
            self._fmt_dropdown,
            urwid.Divider(),
            self._dir_edit,
            urwid.Divider(),
            self._cell_dropdown,
            self._symprec_edit,
            urwid.Divider(),
            self._table_export_dropdown,
        ])

        if has_phonon:
            body_widgets.append(self._dispersion_dropdown)

        body_widgets.extend([
            urwid.Divider(),
            urwid.Columns(
                [("weight", 1, ok_btn), ("weight", 1, cancel_btn)],
                dividechars=2,
            ),
        ])

        super().__init__(
            "Save Structure", urwid.Pile(body_widgets), parent, min_width=45
        )
        self._esc_handler = lambda: self._emit("close", False)


class BaseResultsScreen(ScreenBase):
    title = "Base Results"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)

        self._pressure_gpa: float = 0.0
        self._thickness_cutoff: float | None = None
        self._phonon_cutoff: float | None = None
        self._symprec: float | None = None

        self._results: list[ResultRow] = []

        self._hide_unconverged: bool = True
        self._hide_thick: bool = False
        self._hide_duplicates: bool = False
        self._hide_excluded: bool = True
        self._hide_dyn_unstable: bool = False
        self._search_query: str = ""

        self._show_thickness: bool = False
        self._show_dynamical_stability: bool = False
        self._show_duplicate_col: bool = False
        self._show_flags_col: bool = False

        self._app_message: str = ""

        self._main_frame: urwid.Frame | None = None
        self._table: SortableTable | None = None
        self._details_content: urwid.WidgetPlaceholder | None = None
        self._footer: _ResultsFooter | None = None
        self._details_panel: urwid.Widget | None = None
        self._body_pile: urwid.Pile | None = None
        self._phonon_task: "BackgroundTask | None" = None

        self._outer_placeholder: urwid.WidgetPlaceholder | None = None
        self._loading_task: "BackgroundTask | None" = None

    def build(self) -> urwid.Widget:
        self._outer_placeholder = urwid.WidgetPlaceholder(urwid.SolidFill(" "))
        self.refresh_footer()
        self._start_async_fetch()
        return self._outer_placeholder

    def _loading_title(self) -> str:
        return f" Loading {self.title} "

    def _start_async_fetch(self) -> None:
        panel = ProgressPanel(title=self._loading_title())
        panel.set_progress(0, 1, "Loading structures from database...")

        loading_widget = urwid.Filler(urwid.BoxAdapter(panel, 10), valign="middle")
        if self._outer_placeholder is not None:
            self._outer_placeholder.original_widget = loading_widget

        loop = self._state.loop

        if loop is None:
            self._fetch_data()
            self._main_frame = self._build_frame()
            if self._outer_placeholder is not None:
                self._outer_placeholder.original_widget = self._main_frame
            return

        def _worker(progress) -> None:
            self._fetch_data(
                progress_callback=progress.as_callback(default_is_log=False)
            )

        def _on_progress(current: int, total: int, message: str) -> None:
            panel.set_progress(current, total, message)

        def _on_log(line: str) -> None:
            panel.add_log(line)

        def _on_complete() -> None:
            self._main_frame = self._build_frame()
            if self._outer_placeholder is not None:
                self._outer_placeholder.original_widget = self._main_frame

            current = self._router.current
            if current is not self and current is not None:
                current.on_resume()

        def _on_error(error: str) -> None:
            panel.set_finished(False, f"Error: {error}")

        self._loading_task = BackgroundTask(
            fn=_worker,
            loop=loop,
            on_progress=_on_progress,
            on_log=_on_log,
            on_complete=_on_complete,
            on_error=_on_error,
        )
        self._loading_task.start()

    def _tasks(self) -> list[BackgroundTask]:
        return [
            t for t in (self._loading_task, self._phonon_task) if t is not None
        ]

    def _fetch_data(self, progress_callback=None) -> None:
        pass

    def _columns_def(self) -> list[tuple[str, int]]:
        return []

    def _format_row(self, result: "ResultRow") -> list[str]:
        return []

    def _header_widget(self) -> "urwid.Widget | None":
        """Optional widget rendered above the table.
        """
        return None

    def _get_symprec(self) -> float:
        return DEFAULT_SYMPREC

    def _persist_symprec(self, value: float) -> None:
        """Persist an adjusted labeling symprec. Subclasses opt in."""

    def _get_extra_details(self, result: "ResultRow") -> list:
        return []

    def _save_subdir(self) -> str | None:
        """Name of the ``saved_<...>`` subfolder offered by default on save."""
        return getattr(self, "_run_name", None) or self._state.active_run

    def _save_ident(self, result: "ResultRow") -> str:
        """Filename for a saved structure: ``structure_<ident>.<fmt>``."""
        if result.run_name:
            return f"{result.run_name}_{result.index}"
        return str(result.index)

    def _on_phonon_complete(self, phonon_cutoff: float) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  Phonon results
    # ------------------------------------------------------------------ #

    def _phonon_clear_target(self) -> list[str]:
        """Run names whose persisted phonon results a clear/recompute wipes."""
        return []

    def _reset_inmemory_phonon(self) -> None:
        for r in self._results:
            r.structure.min_phonon_freq = None
            r.dynamical_stability = None

    def _wipe_phonon_results(self) -> None:
        """Delete persisted phonon results for this view (both DB and in-memory)."""
        store = self._state.store
        for run_name in self._phonon_clear_target():
            store.clear_run_phonon_results(run_name)
        self._reset_inmemory_phonon()

    def _invalidate_phonon_before_run(self) -> None:
        self._wipe_phonon_results()

    def _clear_phonon_results(self) -> None:
        try:
            self._wipe_phonon_results()
        except Exception as exc:
            self._show_message(f"Clear failed: {exc}")
            return
        self._show_dynamical_stability = False
        self._phonon_cutoff = None
        self._rebuild_table()
        if self._table is not None:
            self._update_details(self._table.get_focused_row())
        self._show_message("Phonon results cleared.")

    def _attr_fn(self, result: "ResultRow") -> str:
        return _row_attr(result)

    def _dialog_host_get(self) -> "urwid.Widget | None":
        return self._main_frame.body if self._main_frame is not None else None

    def _dialog_host_set(self, widget: urwid.Widget) -> None:
        self._main_frame.body = widget

    def _build_frame(self) -> urwid.Frame:
        cols = self._columns_def()
        display = self._get_display_results()

        self._table = SortableTable(
            columns=cols,
            row_data=display,
            format_row=self._format_row,
            attr_fn=self._attr_fn,
            on_focus_change=self._on_focus_change,
        )
        urwid.connect_signal(self._table, "select", self._on_row_select)

        self._details_content = urwid.WidgetPlaceholder(
            urwid.Text("No structure selected.")
        )
        self._details_panel = urwid.LineBox(
            self._details_content,
            title="Structure Details",
        )

        self._footer = _ResultsFooter(self)

        body_items: list = []
        header = self._header_widget()
        if header is not None:
            body_items.append(("pack", header))
        body_items.append(("weight", 1, urwid.AttrMap(self._table, "body")))
        body_items.append(("pack", self._details_panel))
        self._body_pile = urwid.Pile(body_items)

        frame = urwid.Frame(
            body=self._body_pile,
            footer=self._footer,
        )

        self._update_details(self._table.get_focused_row())
        self.refresh_footer()

        return frame

    def _get_display_results(self) -> list["ResultRow"]:
        res = self._results
        if self._hide_unconverged:
            res = [r for r in res if r.converged]
        if self._hide_thick and self._thickness_cutoff is not None:
            res = [
                r
                for r in res
                if r.thickness is not None
                and r.thickness <= self._thickness_cutoff
            ]
        if self._hide_duplicates:
            res = [r for r in res if r.duplicate is not True]
        if self._hide_excluded:
            res = [r for r in res if not r.excluded]
        if self._hide_dyn_unstable:
            res = [
                r
                for r in res
                if _dyn_stability(r, self._phonon_cutoff) is True
            ]
        if self._search_query:
            q = self._search_query
            res = [r for r in res if q in r.search_text()]
        return res

    def _rebuild_table(self) -> None:
        if self._table is None:
            return
        new_cols = self._columns_def()
        self._table.update_columns(new_cols)
        self._table.set_data(self._get_display_results())

    def _update_details(self, result: "ResultRow | None") -> None:
        if getattr(self, "_details_content", None) is None:
            return

        if result is None:
            if (
                self._hide_unconverged
                and self._results
                and not any(r.converged for r in self._results)
            ):
                self._details_content.original_widget = urwid.Text(
                    [
                        (
                            "details",
                            "All structures are unconverged. Press [u] to show them.",
                        )
                    ]
                )
            else:
                self._details_content.original_widget = urwid.Text(
                    [("details", "No structure selected.")]
                )
            return

        atoms = result.atoms

        if atoms is not None:
            cell_lengths = atoms.get_cell().lengths()

            cells = []

            def add_cell(label, val):
                cells.append(
                    urwid.Text(
                        [("form_label", f"{label}: "), ("details", str(val))],
                        align="left",
                    )
                )

            if result.structure_id:
                add_cell("ID", result.structure_id)

            add_cell("Atoms", len(atoms))
            add_cell(
                "Cell (Å)",
                f"{cell_lengths[0]:.3f}, {cell_lengths[1]:.3f}, {cell_lengths[2]:.3f}",
            )
            add_cell("Initial SG", result.initial_spg or "N/A")
            add_cell("Final SG", result.final_spg or "N/A")

            epa = result.energy_per_atom

            if self._pressure_gpa > 0:
                h = result.enthalpy_per_atom
                if h is not None:
                    add_cell("Enthalpy/A", f"{h:.4f} eV")
                add_cell("Energy/A", f"{epa:.4f} eV")
                vol = result.volume
                if vol is not None:
                    add_cell("Volume", f"{vol:.3f} Å³")
                add_cell("Pressure", f"{self._pressure_gpa} GPa")
            else:
                add_cell("Energy/A", f"{epa:.4f} eV")

            fmax = result.fmax
            if fmax is not None:
                add_cell("Fmax", f"{fmax:.3f}")

            converged = result.converged
            if converged is not None:
                add_cell("Converged", bool(converged))

            if self._show_thickness:
                t = result.thickness
                if t is not None:
                    add_cell("Thickness (Å)", f"{t:.2f}")

            for extra in self._get_extra_details(result):
                markup, text = extra
                text = text.rstrip("\n")
                if ":" in text:
                    label, val = text.split(":", 1)
                    cells.append(
                        urwid.Text(
                            [("form_label", f"{label}: "), ("details", val.strip())],
                            align="left",
                        )
                    )
                else:
                    cells.append(urwid.Text([(markup, text)], align="left"))

            if self._show_dynamical_stability:
                dyn = _dyn_stability(result, self._phonon_cutoff)
                add_cell("Dyn. Stability", _yes_no(dyn))

            min_freq = result.min_phonon_freq
            if min_freq is not None:
                try:
                    f = float(min_freq)
                    if not math.isnan(f):
                        add_cell("Min freq (THz)", f"{f:.4f}")
                except (TypeError, ValueError):
                    pass

            dup = result.duplicate
            if dup is not None:
                add_cell("Duplicate", "Yes" if dup else "No")

            if result.excluded:
                add_cell("Excluded", "Yes")

            grid = urwid.GridFlow(cells, cell_width=35, h_sep=2, v_sep=1, align="left")
            self._details_content.original_widget = grid
        else:
            self._details_content.original_widget = urwid.Text(
                [("details", "No structure data available for this row.")]
            )

    def apply_search(self, query: str) -> None:
        self._search_query = query.strip().lower()
        self._rebuild_table()
        if self._search_query:
            self._show_message(f"Filtered: {self._search_query!r}")
        else:
            self._show_message("")

    def exit_search(self) -> None:
        if self._main_frame is None or self._footer is None:
            return
        self._footer.show_status(self._app_message)
        self._main_frame.focus_position = "body"

    def _show_message(self, msg: str) -> None:
        self._app_message = msg
        if self._footer:
            self._footer.show_status(msg)

    def _action_toggle_unconverged(self) -> None:
        self._hide_unconverged = not self._hide_unconverged
        self._refresh_after_membership_change()
        self._show_message(
            "Hiding unconverged." if self._hide_unconverged else "Showing unconverged."
        )

    def _action_toggle_show_excluded(self) -> None:
        self._hide_excluded = not self._hide_excluded
        self._refresh_after_membership_change()
        self._show_message(
            "Hiding excluded." if self._hide_excluded else "Showing excluded."
        )

    def _refresh_after_membership_change(self) -> None:
        """Refresh the view after a change to what is in the hull set.
        """
        self._rebuild_table()

    def _action_toggle_duplicates(self) -> None:
        if not self._show_duplicate_col:
            self._show_message("No dedup data. Run dedup analysis and apply first.")
            return
        self._hide_duplicates = not self._hide_duplicates
        self._refresh_after_membership_change()
        self._show_message(
            "Hiding duplicates." if self._hide_duplicates else "Showing duplicates."
        )

    def _action_toggle_excluded(self) -> None:
        if self._table is None:
            return
        result = self._table.get_focused_row()
        if result is None:
            self._show_message("No structure selected.")
            return
        new_val = result.excluded is not True
        try:
            self._state.store.set_structure_excluded(result.structure_id, new_val)
        except Exception as exc:
            self._show_message(f"Exclude failed: {exc}")
            return
        result.structure.excluded = new_val
        self._show_flags_col = True
        self._refresh_after_membership_change()
        self._show_message(
            "Excluded from hull." if new_val else "Restored to hull."
        )

    def _action_toggle_unstable(self) -> None:
        if not self._show_dynamical_stability:
            self._show_message("No phonon data. Run phonon calculation first.")
            return
        self._hide_dyn_unstable = not self._hide_dyn_unstable
        self._rebuild_table()
        self._show_message(
            "Showing only stable." if self._hide_dyn_unstable else "Showing all."
        )

    def _action_thickness(self) -> None:
        if not self._show_thickness:
            self._show_message("No thickness data available for this run.")
            return

        from rapmat.tui.widgets.form import FormGroup, float_field

        default_val = (
            self._thickness_cutoff if self._thickness_cutoff is not None else 0.0
        )
        form = FormGroup(
            fields=[
                float_field("cutoff", "Max thickness (Å)", default=default_val),
            ],
            label_width=18,
        )

        def _factory(parent, close):
            def _on_apply() -> None:
                vals = dlg.validated_values()
                if vals is None:
                    return
                cutoff = float(vals.get("cutoff", 0.0))
                close()
                if cutoff > 0:
                    self._thickness_cutoff = cutoff
                    self._hide_thick = True
                    self._rebuild_table()
                    self._show_message(f"Hiding thickness > {cutoff:.2f} Å.")
                else:
                    self._thickness_cutoff = None
                    self._hide_thick = False
                    self._rebuild_table()
                    self._show_message("Showing all thicknesses.")

            def _on_clear() -> None:
                close()
                self._thickness_cutoff = None
                self._hide_thick = False
                self._rebuild_table()
                self._show_message("Showing all thicknesses.")

            dlg = FormDialog(
                "Thickness Filter",
                form,
                parent,
                [("Apply", _on_apply), ("Clear", _on_clear), ("Cancel", close)],
                section="Thickness Filter",
                on_cancel=close,
            )
            return dlg

        self.show_dialog(_factory)

    def _relabel_spg(self) -> None:
        """Re-derive spacegroup labels in memory at the current symprec.
        """
        if self._symprec is None:
            return
        for result in self._results:
            result.structure.symprec = self._symprec

    def _extra_option_fields(self) -> list:
        """Extra fields appended to the Options dialog."""
        return []

    def _apply_extra_options(self, vals: dict) -> str | None:
        """Apply extra options values."""
        return None

    def _action_options(self) -> None:
        from rapmat.tui.widgets.form import FormGroup, float_field

        current = self._symprec if self._symprec is not None else self._get_symprec()
        form = FormGroup(
            fields=[
                float_field("symprec", "Symprec (SG labels)", default=current),
                *self._extra_option_fields(),
            ],
            label_width=20,
        )

        def _factory(parent, close):
            def _on_apply() -> None:
                vals = dlg.validated_values()
                if vals is None:
                    return
                value = float(vals.get("symprec", current))
                close()
                self._symprec = value
                self._relabel_spg()
                extra_msg = self._apply_extra_options(vals)
                self._rebuild_table()
                if self._table is not None:
                    self._update_details(self._table.get_focused_row())
                self._persist_symprec(value)
                self._show_message(
                    extra_msg or f"SG labels recomputed at symprec={value:g}."
                )

            dlg = FormDialog(
                "Options",
                form,
                parent,
                [("Apply", _on_apply), ("Cancel", close)],
                section="Options",
                on_cancel=close,
            )
            return dlg

        self.show_dialog(_factory)

    def _action_save(self) -> None:
        if self._table is None:
            return
        result = self._table.get_focused_row()
        if result is None:
            self._show_message("No structure selected.")
            return
        if result.atoms is None:
            self._show_message("No structure data available to save.")
            return

        filtered_results = self._get_display_results()
        num_filtered = len(filtered_results)
        has_initial = any(r.initial_atoms is not None for r in self._results)

        label = self._save_subdir()
        default_dir = (
            str(Path.cwd() / f"saved_{label}") if label else str(Path.cwd())
        )

        def _factory(parent, close):
            def _on_save(opts: SaveOptions) -> None:
                close()
                if not opts.save_all:
                    self._do_save(result, opts, quiet=False)
                else:
                    success_count = 0
                    for res in filtered_results:
                        if res.atoms is None:
                            continue
                        if self._do_save(res, opts, quiet=True):
                            success_count += 1
                    self._show_message(
                        f"Saved {success_count}/{num_filtered} structures "
                        f"to {opts.directory}"
                    )
                if opts.table_export:
                    self._export_results_table(
                        filtered_results, opts.directory, opts.table_export,
                    )

            save_dlg = _SaveDialog(
                parent, _on_save, num_filtered=num_filtered,
                default_dir=default_dir, has_initial=has_initial,
                default_symprec=(
                    self._symprec if self._symprec is not None else self._get_symprec()
                ),
                has_phonon=self._show_dynamical_stability,
            )
            urwid.connect_signal(
                save_dlg, "close", lambda _w, ok: close() if not ok else None
            )
            return save_dlg

        self.show_dialog(_factory)

    def _action_export_settings(self) -> None:
        if self._table is None:
            return
        result = self._table.get_focused_row()
        if result is None:
            self._show_message("No structure selected.")
            return
        if result.atoms is None:
            self._show_message("No structure data available.")
            return

        from rapmat.tui.widgets.form import (FormGroup, checkbox_field,
                                             text_field)

        ident = self._save_ident(result).replace("/", "_")
        run_name = getattr(self, "_run_name", None)
        meta = self._state.store.get_run_metadata(run_name) if run_name else None

        form = FormGroup(
            [
                text_field(
                    "path", "File",
                    default=str(Path.cwd() / f"vasp_{ident}.toml"),
                ),
                checkbox_field(
                    "monolayer", "Monolayer",
                    default=bool(meta and meta.domain == "monolayer"),
                ),
            ],
            label_width=12,
        )

        def _factory(parent, close):
            def _on_submit() -> None:
                vals = dlg.validated_values()
                if vals is None:
                    return
                close()
                self._write_settings_toml(
                    result, vals["path"], vals["monolayer"]
                )

            dlg = FormDialog(
                "Export VASP settings",
                form,
                parent,
                [("Export", _on_submit), ("Cancel", close)],
                on_cancel=close,
                width=70,
                min_width=50,
            )
            return dlg

        self.show_dialog(_factory)

    def _write_settings_toml(
        self, result: "ResultRow", path: str, monolayer: bool
    ) -> None:
        from rapmat.calculators.vasp_auto import (export_toml,
                                                  omat24_vasp_params,
                                                  resolve_potcar_version)

        try:
            # NOTE:  what would actually be used
            potcar_version, _note = resolve_potcar_version()
            params = omat24_vasp_params(
                result.atoms,
                monolayer=monolayer,
                potcar_version=potcar_version,
            )
            out_path = Path(path).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                export_toml(params, f"{result.formula} [{result.short_id}]"),
                encoding="utf-8",
            )
        except Exception as exc:
            self._show_message(f"Export failed: {exc}")
            return
        self._show_message(f"Wrote {out_path}")

    def _do_save(
        self,
        result: "ResultRow",
        opts: SaveOptions,
        *,
        quiet: bool = False,
    ) -> bool:
        from rapmat.core.export import save_structure

        if opts.version == "initial" and result.initial_atoms is not None:
            atoms = result.initial_atoms
        else:
            atoms = result.atoms
        if atoms is None:
            return False
        ident = self._save_ident(result).replace("/", "_")
        prefix = "initial_" if opts.version == "initial" else ""
        try:
            out_path = save_structure(
                atoms, opts.directory, ident, opts.fmt,
                cell_mode=opts.cell_mode, symprec=opts.symprec, prefix=prefix,
            )
            if opts.save_dispersion:
                self._save_dispersion_plot(
                    result, Path(opts.directory) / f"dispersion_{ident}.png"
                )
            if not quiet:
                self._show_message(f"Saved: {out_path}")
            return True
        except Exception as exc:
            if not quiet:
                self._show_message(f"Save failed: {exc}")
            return False

    def _save_dispersion_plot(self, result: "ResultRow", out_path: Path) -> None:
        """Regenerate a phonon dispersion plot from previously phonopy data.
        """
        structure_id = result.structure_id
        if not structure_id:
            return
        rec = self._state.store.get_phonon_result(structure_id)
        if not rec or not rec.params_gz:
            return
        from rapmat.core.phonon import (deserialize_phonons,
                                        save_phonon_dispersion)

        phonons = deserialize_phonons(rec.params_gz)
        save_phonon_dispersion(phonons, out_path)

    def _export_results_table(
        self,
        results: list["ResultRow"],
        directory: str,
        fmt: str,
    ) -> None:
        from rapmat.core.export import export_results_table

        headers = [name for name, _width in self._columns_def()]
        rows = [self._format_row(r) for r in results]
        try:
            out_path = export_results_table(headers, rows, directory, fmt)
            self._show_message(f"Table exported: {out_path}")
        except Exception as exc:
            self._show_message(f"Table export failed: {exc}")

    def _action_enter_search(self) -> None:
        if self._footer is None or self._main_frame is None:
            return
        self._footer.show_search()
        self._main_frame.focus_position = "footer"

    def _on_focus_change(self, result: "ResultRow | None") -> None:
        self._update_details(result)

    def _on_row_select(self, _table, result: "ResultRow") -> None:
        self._update_details(result)

    def bindings(self) -> list[KeyBinding]:
        return [
            KeyBinding(
                ("/",), "Search", self._action_enter_search,
                help="Filter rows by a search string", priority=10,
            ),
            KeyBinding(
                ("s",), "Save", self._action_save,
                help="Save structures",
                priority=20,
            ),
            KeyBinding(
                ("o",), "Opts", self._action_options,
                help="Options", priority=25,
            ),
            KeyBinding(
                ("t",), "Thk", self._action_thickness,
                help="Filter by thickness", priority=30,
            ),
            KeyBinding(
                ("p",), "Phon", self._action_phonon,
                help="Dynamical stability calculation",
                priority=35,
            ),
            KeyBinding(
                ("d",), "Dup", self._action_toggle_duplicates,
                help="Toggle duplicates", priority=40,
            ),
            KeyBinding(
                ("e",),
                lambda: "Show Excl" if self._hide_excluded else "Hide Excl",
                self._action_toggle_show_excluded,
                help="Toggle excluded", priority=45,
            ),
            KeyBinding(
                ("x",), "Excl", self._action_toggle_excluded,
                help="Exclude/restore the focused structure",
                priority=50,
            ),
            KeyBinding(
                ("y",), "Dyn", self._action_toggle_unstable,
                help="Show only dynamically stable", priority=55,
            ),
            KeyBinding(
                ("u",), "Uncv", self._action_toggle_unconverged,
                help="Toggle unconverged", priority=70,
            ),
            KeyBinding(
                ("i",), "INCAR", self._action_export_settings,
                help="Export auto-generated DFT (VASP) settings",
                priority=75,
            ),
        ]

    def _on_esc(self) -> bool:
        if self._search_query:
            self._search_query = ""
            self._rebuild_table()
            self._show_message("")
            return True

        return super()._on_esc()

    def _action_phonon(self) -> None:
        if self._main_frame is None:
            return
        if not any(r.converged for r in self._results):
            self._show_message(
                "No converged structures available for phonon calculation."
            )
            return

        from rapmat.tui.widgets.calc_fields import (
            CALCULATOR_FIELD_KEYS,
            calculator_fields,
            parse_toml_config,
            phonon_fields,
            remember_vasp_command,
            setup_calculator_signals,
            validate_calculator,
        )
        from rapmat.tui.widgets.form import (FormGroup, dropdown_field,
                                             int_field)

        form = FormGroup(
            fields=[
                int_field("top_n", "Top N structures", default=5),
                dropdown_field(
                    "apply_to",
                    "Apply to",
                    options=["Filtered view", "All converged"],
                    default=0,
                ),
                *calculator_fields(),
                *phonon_fields(include_symprec=True),
            ],
            label_width=18,
            groups=[
                ("Scope", ["top_n", "apply_to"]),
                ("Calculator", CALCULATOR_FIELD_KEYS),
                ("Phonon Settings", [
                    "phonon_supercell", "phonon_mesh",
                    "phonon_displacement", "phonon_cutoff",
                    "phonon_symprec", "reduce_prim",
                ]),
            ],
        )

        setup_calculator_signals(form)

        def _factory(parent, close):
            def _on_submit() -> None:
                vals = dlg.validated_values()
                if vals is None:
                    return

                calc_err = validate_calculator(vals) or parse_toml_config(vals)[1]
                if calc_err:
                    dlg.set_error(calc_err)
                    return
                vals["calculator_config_dict"] = parse_toml_config(vals)[0]
                remember_vasp_command(vals)

                close()
                self._start_phonon_task(vals)

            def _on_clear() -> None:
                close()
                self._confirm_clear_phonons()

            dlg = FormDialog(
                "Dynamical stability",
                form,
                parent,
                [
                    ("Run Phonons", _on_submit),
                    ("Clear results", _on_clear),
                    ("Cancel", close),
                ],
                on_cancel=close,
                width=60,
                min_width=50,
            )
            return dlg

        self.show_dialog(_factory)

    def _confirm_clear_phonons(self) -> None:
        if not self._show_dynamical_stability:
            self._show_message("No phonon results to clear.")
            return

        self.confirm_dialog(
            "Clear phonon results",
            "Delete all stored phonon data for this view?",
            self._clear_phonon_results,
        )

    def _start_phonon_task(self, vals: dict) -> None:
        if self._main_frame is None or self._body_pile is None:
            return

        self._invalidate_phonon_before_run()

        from rapmat.calculators import Calculators
        from rapmat.core.phonon_stability import \
            compute_dynamical_stability_for_results
        from rapmat.tui.tasks import BackgroundTask
        from rapmat.tui.widgets.progress import ProgressPanel

        top_n = int(vals.get("top_n", 5))
        calc_name = vals.get("calculator", Calculators.MATTERSIM.value)
        supercell = vals.get("phonon_supercell", (3, 3, 3))
        mesh = vals.get("phonon_mesh", (20, 20, 20))
        displacement = float(vals.get("phonon_displacement", 0.01))
        cutoff = float(vals.get("phonon_cutoff", -0.15))
        reduce_prim = bool(vals.get("reduce_prim", True))
        phonon_symprec = float(vals.get("phonon_symprec", DEFAULT_SYMPREC))

        from rapmat.tui.widgets.calc_fields import (calculator_run_config,
                                                    is_auto_settings)

        calc_config = calculator_run_config(vals)
        auto_settings = is_auto_settings(vals)
        run_name = getattr(self, "_run_name", None)
        meta = self._state.store.get_run_metadata(run_name) if run_name else None
        monolayer = bool(meta and meta.domain == "monolayer")

        try:
            calc_enum = Calculators(calc_name)
        except ValueError:
            calc_enum = Calculators.MATTERSIM

        panel = ProgressPanel(title=" Phonon Calculation ")
        panel.set_progress(0, top_n)
        panel.add_log(f"Starting phonon calculation for top {top_n} structures...")

        self._body_pile.contents[-1] = (
            urwid.BoxAdapter(panel, 12),
            self._body_pile.options("pack"),
        )

        apply_to = vals.get("apply_to", "Filtered view")
        if apply_to == "Filtered view":
            results_snapshot = list(self._get_display_results())
        else:
            results_snapshot = list(self._results)
        store = self._state.store
        phonon_cutoff = (
            self._phonon_cutoff if self._phonon_cutoff is not None else cutoff
        )

        def _worker(progress) -> None:
            compute_dynamical_stability_for_results(
                results=results_snapshot,
                phonon_top=top_n,
                phonon_cutoff=phonon_cutoff,
                phonon_supercell=supercell,
                phonon_mesh=mesh,
                phonon_displacement=displacement,
                phonon_calculator=calc_enum,
                store=store,
                calculator_config=calc_config,
                progress_callback=progress.as_callback(),
                symprec=phonon_symprec,
                reduce_primitive=reduce_prim,
                run_name=run_name,
                auto_settings=auto_settings,
                monolayer=monolayer,
            )

        def _on_progress(current: int, total: int, message: str) -> None:
            panel.set_progress(current, total, message)

        def _on_log(line: str) -> None:
            panel.add_log(line)

        def _on_complete() -> None:
            panel.set_finished(True, "Phonon calculation complete.")
            self._show_dynamical_stability = True
            self._phonon_cutoff = phonon_cutoff
            self._on_phonon_complete(phonon_cutoff)
            self._rebuild_table()

            if self._body_pile is not None and self._details_panel is not None:
                self._body_pile.contents[-1] = (
                    self._details_panel,
                    self._body_pile.options("pack"),
                )
            self._show_message("Phonon calculation complete.")

        def _on_error(error: str) -> None:
            panel.set_finished(False, f"Error: {error}")
            self._show_message(f"Failed: {error}")

        loop = self._state.loop
        if loop is None:
            self._show_message("Cannot start background task: no event loop.")
            return

        self._phonon_task = BackgroundTask(
            fn=_worker,
            loop=loop,
            on_progress=_on_progress,
            on_log=_on_log,
            on_complete=_on_complete,
            on_error=_on_error,
        )
        self._phonon_task.start()
