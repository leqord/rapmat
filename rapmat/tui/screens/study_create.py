import urwid

from rapmat.tui.router import ScreenRouter
from rapmat.tui.screens.base import ScreenBase
from rapmat.tui.state import AppState
from rapmat.tui.widgets.calc_fields import (
    calculator_fields,
    parse_toml_config,
    setup_calculator_signals,
)
from rapmat.tui.widgets.dialog import ModalDialog
from rapmat.tui.widgets.form import (FormGroup, checkbox_field, dropdown_field,
                                     float_field, text_field)


def _validate_system(v: str) -> str | None:
    from rapmat.utils.common import parse_system

    try:
        parse_system(v)
        return None
    except Exception as exc:
        return str(exc)


def _validate_thickness(v: str) -> str | None:
    v = v.strip()
    if not v:
        return None
    try:
        float(v)
        return None
    except ValueError:
        return "Must be a number or empty"


class StudyCreateScreen(ScreenBase):
    title = "New Study"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._form: FormGroup | None = None
        self._error_text: urwid.Text | None = None
        self._frame: urwid.Frame | None = None
        self._main_body: urwid.Widget | None = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._form = FormGroup(
            fields=[
                # --- System ---
                text_field(
                    key="system",
                    label="System",
                    default="Al-O",
                    validator=_validate_system,
                ),
                text_field(
                    key="name",
                    label="Study Name",
                    default="",
                    validator=lambda v: "Required" if not v.strip() else None,
                ),
                dropdown_field(
                    key="domain",
                    label="Domain",
                    options=["bulk", "monolayer"],
                    default=0,
                ),
                float_field("pressure", "Pressure (GPa)", default=0.0),
                text_field(
                    key="thickness_cutoff",
                    label="Thickness (Å)",
                    default="",
                    validator=_validate_thickness,
                ),
                # --- Relaxation ---
                *calculator_fields(include_convergence=True),
                # --- Etc ---
                checkbox_field("sanity_pymatgen", "Sanity check", default=False),
                float_field("symprec", "Default symprec", default=1e-2),
            ],
            label_width=22,
            groups=[
                ("System", [
                    "system", "name", "domain", "pressure", "thickness_cutoff",
                ]),
                ("Relaxation", [
                    "calculator",
                    "calculator_config",
                    "force_conv_crit",
                    "steps_max",
                ]),
                ("Misc", [
                    "sanity_pymatgen", "symprec",
                ]),
            ],
        )

        self._error_text = urwid.Text("")

        submit_btn = urwid.AttrMap(
            urwid.Button("Create Study", on_press=self._on_submit),
            None,
            focus_map="btn_focus",
        )
        cancel_btn = urwid.AttrMap(
            urwid.Button("Cancel", on_press=lambda _: self._router.pop()),
            None,
            focus_map="btn_focus",
        )
        btn_row = urwid.Columns(
            [("weight", 1, submit_btn), ("weight", 1, cancel_btn)],
            dividechars=2,
        )

        body = urwid.Pile(
            [
                ("pack", urwid.Divider()),
                ("pack", self._form),
                ("pack", urwid.Divider()),
                ("pack", self._error_text),
                ("pack", urwid.Divider()),
                ("pack", btn_row),
                ("pack", urwid.Divider()),
            ]
        )

        listbox = urwid.ListBox(urwid.SimpleListWalker([body]))
        scrollable = urwid.ScrollBar(
            listbox,
            trough_char=urwid.ScrollBar.Symbols.LITE_SHADE,
        )
        self._main_body = urwid.Padding(scrollable, left=2, right=2)
        self._frame = urwid.Frame(body=self._main_body)

        setup_calculator_signals(self._form)

        domain_widget = self._form.get_widget("domain")
        if domain_widget:
            urwid.connect_signal(domain_widget, "change", self._on_domain_change)
            self._form.set_field_disabled("thickness_cutoff", disabled=True)

        self.refresh_footer()
        return self._frame

    def _on_domain_change(self, _widget, new_value: str) -> None:
        if self._form:
            self._form.set_field_disabled(
                "thickness_cutoff", disabled=(new_value != "monolayer")
            )

    def extra_hints(self) -> list:
        return [("Tab", "Navigate", 10), ("Enter", "Submit", 20)]

    def esc_label(self) -> str:
        return "Cancel"

    # ------------------------------------------------------------------ #
    #  Submit
    # ------------------------------------------------------------------ #

    def _on_submit(self, _btn) -> None:
        if self._form is None or self._error_text is None:
            return

        errors = self._form.validate()
        if errors:
            self._error_text.set_text(("form_error", "  " + "; ".join(errors)))
            return

        vals = self._form.get_values()
        system_raw = vals["system"].strip()
        name = vals["name"].strip()
        domain = vals["domain"]
        calculator = vals["calculator"]
        calc_config_dict, toml_err = parse_toml_config(vals)
        if toml_err:
            self._error_text.set_text(("form_error", f"  {toml_err}"))
            return

        try:
            from rapmat.utils.common import format_system, parse_system

            elements = parse_system(system_raw)
            normalized_system = format_system(elements)
        except Exception as exc:
            self._error_text.set_text(("form_error", f"  Invalid system: {exc}"))
            return

        thickness_val = vals.get("thickness_cutoff", "").strip()
        thickness_cutoff = None
        if thickness_val and thickness_val.lower() != "none":
            try:
                thickness_cutoff = float(thickness_val)
            except ValueError:
                pass

        try:
            self._state.store.create_study(
                study_id=name,
                system=normalized_system,
                domain=domain,
                calculator=calculator,
                config={
                    "calculator_config": calc_config_dict,
                    "thickness_cutoff": thickness_cutoff,
                    "pressure_gpa": vals["pressure"],
                    "force_conv_crit": vals["force_conv_crit"],
                    "steps_max": vals["steps_max"],
                    "sanity_pymatgen": vals["sanity_pymatgen"],
                    "sanity_pymatgen_tol": 0.5,
                    "symprec": vals["symprec"],
                },
            )
        except Exception as exc:
            self._error_text.set_text(("form_error", f"  Error: {exc}"))
            return

        self._state.invalidate()

        if self._frame is not None and self._main_body is not None:
            dlg = ModalDialog.info(
                "Study Created",
                f"Study '{name}' created.\nSystem: {normalized_system}\nDomain: {domain}",
                parent=self._main_body,
                on_close=self._on_info_close,
            )
            self._frame.body = dlg

    def _on_info_close(self) -> None:
        self._router.pop()

