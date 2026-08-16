from pathlib import Path

import urwid

from rapmat.tui.keymap import KeyBinding
from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
from rapmat.tui.widgets.calc_fields import (
    CALCULATOR_FIELD_KEYS,
    calculator_fields,
    calculator_run_config,
    is_auto_settings,
    parse_toml_config,
    phonon_fields,
    remember_vasp_command,
    setup_calculator_signals,
    validate_calculator,
)
from rapmat.tui.widgets.form import (FormGroup, checkbox_field, dropdown_field,
                                     text_field)
from rapmat.tui.widgets.progress import ProgressPanel


class PhononDispersionScreen(ScreenBase):
    title = "Phonon Dispersion"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._frame: urwid.Frame | None = None
        self._main_body: urwid.Widget | None = None
        self._progress_panel = ProgressPanel(title=" Phonon Progress ")
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
                ("f5",), "Calculate", self._on_start,
                help="Compute the phonon dispersion", priority=10,
            ),
        ]

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _build_frame(self) -> urwid.Frame:
        self._form = FormGroup(
            [
                text_field("structure_file", "Structure file", default=""),
                dropdown_field(
                    "domain", "Domain", ["bulk", "monolayer"], default=0
                ),
                *calculator_fields(
                    include_convergence=True,
                    force_conv_default=1e-3,
                    steps_max_default=10000,
                ),
                *phonon_fields(
                    supercell_default=(4, 4, 4),
                    displacement_default=3e-2,
                    include_reduce_prim=False,
                    include_symprec=True,
                ),
                checkbox_field("prerelax", "Pre-relax", default=False),
                checkbox_field("reduce_prim", "Reduce to primitive", default=False),
                text_field("plot_file", "Plot file", default="phonon_plot.png"),
            ],
            label_width=22,
            groups=[
                ("Input", ["structure_file", "domain"]),
                ("Calculator", [
                    *CALCULATOR_FIELD_KEYS,
                    "force_conv_crit", "steps_max",
                ]),
                ("Phonon Settings", [
                    "phonon_supercell", "phonon_mesh",
                    "phonon_displacement", "phonon_cutoff", "phonon_symprec",
                ]),
                ("Options", ["prerelax", "reduce_prim", "plot_file"]),
            ],
        )

        setup_calculator_signals(
            self._form, convergence_toggle_key="prerelax",
        )

        self._error_text = urwid.Text("")
        self._summary_text = urwid.Text("")

        start_btn = urwid.AttrMap(
            urwid.Button("Calculate [F5]", on_press=self._on_start),
            "menu_item",
            focus_map="btn_focus",
        )

        listbox = urwid.ListBox(
            urwid.SimpleListWalker(
                [
                    self._form,
                    urwid.Divider(),
                    urwid.Columns([(20, start_btn)], dividechars=1),
                    self._error_text,
                ]
            )
        )

        form_area = urwid.ScrollBar(
            listbox,
            trough_char=urwid.ScrollBar.Symbols.LITE_SHADE,
        )

        body = urwid.Pile(
            [
                ("weight", 3, form_area),
                ("weight", 2, self._progress_panel),
                ("pack", self._summary_text),
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

        vals = self._form.get_values()
        structure_file = vals["structure_file"].strip()
        if not structure_file:
            self._error_text.set_text(("form_error", "Structure file is required"))
            return

        if not Path(structure_file).is_file():
            self._error_text.set_text(
                ("form_error", f"File not found: {structure_file}")
            )
            return

        calc_err = validate_calculator(vals) or parse_toml_config(vals)[1]
        if calc_err:
            self._error_text.set_text(("form_error", calc_err))
            return

        vals["calculator_config_dict"] = parse_toml_config(vals)[0]
        remember_vasp_command(vals, log=self._progress_panel.add_log)

        self._running = True
        self._error_text.set_text("")
        self._summary_text.set_text("")
        self._progress_panel.clear()

        self.run_task(
            lambda prog: self._worker(prog, vals),
            on_progress=self._progress_panel.set_progress,
            on_log=self._progress_panel.add_log,
            on_complete=self._on_complete,
            on_error=self._on_error,
        )

    def _worker(self, progress, vals: dict) -> None:
        import warnings

        from ase import Atoms
        from ase.io import read as read_ase_structure

        from rapmat.calculators import Calculators, LogCalcCallback
        from rapmat.calculators.factory import CalculatorProvider
        from rapmat.core.phonon import (structure_calculate_phonons,
                                        structure_has_imag_phonon_freq)
        from rapmat.utils.common import workdir_context

        def _calc_status(message: str) -> None:
            progress.log(message)
            progress.update(1, 5, message)

        structure_file = vals["structure_file"].strip()
        calculator_name = vals["calculator"]
        supercell = vals["phonon_supercell"]
        qpoint_mesh = vals["phonon_mesh"]
        displacement = vals["phonon_displacement"]
        imag_cutoff = vals["phonon_cutoff"]
        prerelax = vals["prerelax"]
        reduce_prim = vals["reduce_prim"]
        plot_file = vals["plot_file"].strip() or "phonon_plot.png"

        progress.log(f"Reading structure from {structure_file}...")
        progress.update(0, 5, "Reading structure")

        structure = read_ase_structure(structure_file)
        if not isinstance(structure, Atoms):
            structure = structure[-1]

        with workdir_context(None) as wdir:
            progress.log(f"Working directory: {wdir}")
            progress.update(1, 5, "Loading calculator")
            progress.log(f"Loading calculator {calculator_name}...")
            calculator_for = CalculatorProvider(
                Calculators(calculator_name),
                wdir,
                config=calculator_run_config(vals),
                callback=LogCalcCallback(_calc_status),
                auto_settings=is_auto_settings(vals),
                monolayer=vals.get("domain") == "monolayer",
                log_callback=progress.log,
            )
            structure.calc = calculator_for(structure)

            if prerelax:
                progress.update(2, 5, "Pre-relaxing")
                progress.log("Pre-relaxing structure...")
                from rapmat.core.relaxation import structure_relax

                cancel_flag = [False]

                def _phony_check():
                    if progress.cancelled:
                        cancel_flag[0] = True

                converged, relaxed = structure_relax(
                    structure,
                    force_conv_crit=vals.get("force_conv_crit", 1e-3),
                    steps_max=vals.get("steps_max", 10000),
                    cancel_flag=cancel_flag,
                )
                if progress.cancelled or cancel_flag[0]:
                    raise KeyboardInterrupt("Cancelled by user")

                if converged:
                    structure = relaxed
                else:
                    progress.log("WARNING: Pre-relax did not converge")

            if reduce_prim:
                progress.update(2, 5, "Reducing to primitive cell")
                progress.log("Reducing to primitive cell...")
                from rapmat.utils.structure import (DEFAULT_SYMPREC,
                                                    standardize_atoms)

                try:
                    structure = standardize_atoms(
                        structure,
                        to_primitive=True,
                        symprec=vals.get("phonon_symprec", DEFAULT_SYMPREC),
                    )
                    # NOTE: re-derived
                    structure.calc = calculator_for(structure)
                except Exception as e:
                    progress.log(f"WARNING: Could not reduce cell: {e}")

            progress.update(3, 5, "Computing phonons")
            progress.log("Computing phonon dispersion...")
            phonons = structure_calculate_phonons(
                structure,
                displacement,
                supercell,
                qpoint_mesh,
                progress_callback=progress.update,
                calculator_for=calculator_for,
            )

            is_unstable = structure_has_imag_phonon_freq(phonons, imag_cutoff)

            progress.update(4, 5, "Saving plot")
            progress.log("Generating band structure plot...")

            plot_path = Path(plot_file).resolve()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = phonons.auto_band_structure(plot=True)
            fig.savefig(plot_path)

            self._result = {
                "calculator": calculator_name,
                "stable": not is_unstable,
                "plot_path": str(plot_path),
                "displacement": displacement,
                "supercell": supercell,
                "qpoint_mesh": qpoint_mesh,
            }

        progress.update(5, 5, "Done")
        progress.finish()

    # ------------------------------------------------------------------ #
    #  Completion
    # ------------------------------------------------------------------ #

    def _on_complete(self) -> None:
        self._running = False
        self._progress_panel.set_finished(True, "Phonon calculation complete!")

        r = getattr(self, "_result", None)
        if r:
            stable_str = "Yes" if r["stable"] else "No (imaginary frequencies detected)"
            self._summary_text.set_text(
                [
                    ("section", "\n Phonon Dispersion Result\n"),
                    ("form_label", "  Calculator: "),
                    ("details", r["calculator"] + "\n"),
                    ("form_label", "  Stable:     "),
                    ("success" if r["stable"] else "error", stable_str + "\n"),
                    ("form_label", "  Supercell:  "),
                    ("details", str(r["supercell"]) + "\n"),
                    ("form_label", "  Q-mesh:     "),
                    ("details", str(r["qpoint_mesh"]) + "\n"),
                    ("form_label", "  Plot:       "),
                    ("details", r["plot_path"]),
                ]
            )

    def _on_error(self, error: str) -> None:
        self._running = False
        self._progress_panel.set_finished(False, f"Error: {error}")

