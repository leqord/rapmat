
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import urwid

from rapmat.calculators import REQUIRES_EXTERNAL_CONFIG, Calculators
from rapmat.tui.widgets.form import (
    FormGroup,
    _FieldSpec,
    checkbox_field,
    dropdown_field,
    float_field,
    int_field,
    text_field,
    tuple_field,
)

if TYPE_CHECKING:
    pass

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #


def _calc_options() -> list[str]:
    return [c.value for c in Calculators]


# ------------------------------------------------------------------ #
#  Field factory
# ------------------------------------------------------------------ #


def calculator_fields(
    *,
    calc_label: str = "Calculator",
    include_convergence: bool = False,
    force_conv_default: float = 5e-3,
    steps_max_default: int = 2000,
) -> list[_FieldSpec]:
    fields: list[_FieldSpec] = [
        dropdown_field("calculator", calc_label, _calc_options(), default=0),
        text_field("calculator_config", "Config TOML Path", default=""),
    ]
    if include_convergence:
        fields.extend(
            [
                float_field(
                    "force_conv_crit", "Force conv. crit", default=force_conv_default
                ),
                int_field("steps_max", "Max steps", default=steps_max_default),
            ]
        )
    return fields


def phonon_fields(
    *,
    supercell_default: tuple = (3, 3, 3),
    mesh_default: tuple = (20, 20, 20),
    displacement_default: float = 1e-2,
    cutoff_default: float = -0.15,
    include_reduce_prim: bool = True,
    include_symprec: bool = False,
    symprec_default: float = 1e-3,
) -> list[_FieldSpec]:
    fields: list[_FieldSpec] = [
        tuple_field(
            "phonon_supercell", "Supercell", size=3, default=supercell_default
        ),
        tuple_field(
            "phonon_mesh", "Q-point mesh", size=3, default=mesh_default
        ),
        float_field(
            "phonon_displacement", "Displacement", default=displacement_default
        ),
        float_field(
            "phonon_cutoff", "Imag freq cutoff", default=cutoff_default
        ),
    ]
    if include_symprec:
        fields.append(
            float_field("phonon_symprec", "Symprec", default=symprec_default)
        )
    if include_reduce_prim:
        fields.append(
            checkbox_field("reduce_prim", "Reduce to primitive", default=True)
        )
    return fields


# ------------------------------------------------------------------ #
#  Signal wiring
# ------------------------------------------------------------------ #


def _needs_external_config(calculator_value: str) -> bool:
    try:
        return Calculators(calculator_value) in REQUIRES_EXTERNAL_CONFIG
    except ValueError:
        return False


def setup_calculator_signals(
    form: FormGroup,
    *,
    disable_config_for_mlips: bool = True,
    convergence_toggle_key: str | None = None,
) -> None:
    if disable_config_for_mlips:
        calc_widget = form.get_widget("calculator")
        if calc_widget is not None:
            urwid.connect_signal(
                calc_widget,
                "change",
                lambda _w, val: form.set_field_disabled(
                    "calculator_config",
                    disabled=not _needs_external_config(val),
                ),
            )
            form.set_field_disabled("calculator_config", disabled=True)

    if convergence_toggle_key:
        toggle_widget = form.get_widget(convergence_toggle_key)
        if toggle_widget is not None:
            def _on_toggle(_w, new_state):
                enabled = bool(new_state)
                form.set_field_disabled("force_conv_crit", disabled=not enabled)
                form.set_field_disabled("steps_max", disabled=not enabled)

            urwid.connect_signal(toggle_widget, "change", _on_toggle)

            form.set_field_disabled("force_conv_crit", disabled=True)
            form.set_field_disabled("steps_max", disabled=True)


# ------------------------------------------------------------------ #
#  TOML config validation
# ------------------------------------------------------------------ #


def parse_toml_config(vals: dict) -> tuple[dict, str | None]:
    calc_config_path = vals.get("calculator_config", "").strip()
    if not calc_config_path:
        return {}, None

    import tomllib

    config_file = Path(calc_config_path)
    if not config_file.is_file():
        return {}, f"Config file not found: {calc_config_path}"
    try:
        with open(config_file, "rb") as f:
            return tomllib.load(f), None
    except Exception as e:
        return {}, f"Invalid TOML in config: {e}"
