
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


SETTINGS_TOML = "TOML file"
SETTINGS_AUTO = "Auto (OMat24)"
_SETTINGS_OPTIONS = [SETTINGS_TOML, SETTINGS_AUTO]

CALCULATOR_FIELD_KEYS = [
    "calculator",
    "calculator_settings",
    "calculator_config",
    "vasp_command",
]


def _calc_options() -> list[str]:
    return [c.value for c in Calculators]


def is_auto_settings(vals: dict) -> bool:
    return vals.get("calculator_settings") == SETTINGS_AUTO


# ------------------------------------------------------------------ #
#  Field factory
# ------------------------------------------------------------------ #


def calculator_fields(
    *,
    calc_label: str = "Calculator",
    calc_default: str | None = None,
    include_convergence: bool = False,
    force_conv_default: float = 5e-3,
    steps_max_default: int = 2000,
) -> list[_FieldSpec]:
    from rapmat.app_config import resolve_vasp_command

    options = _calc_options()
    calc_index = options.index(calc_default) if calc_default in options else 0

    fields: list[_FieldSpec] = [
        dropdown_field("calculator", calc_label, options, default=calc_index),
        dropdown_field(
            "calculator_settings",
            "Settings",
            _SETTINGS_OPTIONS,
            default=_SETTINGS_OPTIONS.index(SETTINGS_AUTO),
        ),
        text_field("calculator_config", "Config TOML Path", default=""),
        text_field("vasp_command", "VASP command", default=resolve_vasp_command()),
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


def _sync_calculator_fields(form: FormGroup) -> None:
    vals = form.get_values()
    external = _needs_external_config(vals.get("calculator", ""))
    auto = is_auto_settings(vals)

    form.set_field_disabled("calculator_settings", disabled=not external)
    form.set_field_disabled("vasp_command", disabled=not external)
    form.set_field_disabled(
        "calculator_config", disabled=not external or auto
    )


def setup_calculator_signals(
    form: FormGroup,
    *,
    disable_config_for_mlips: bool = True,
    convergence_toggle_key: str | None = None,
) -> None:
    if disable_config_for_mlips:
        for key in ("calculator", "calculator_settings"):
            widget = form.get_widget(key)
            if widget is not None:
                urwid.connect_signal(
                    widget, "change", lambda _w, _val: _sync_calculator_fields(form)
                )
        _sync_calculator_fields(form)

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
    if is_auto_settings(vals):
        return {}, None

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


def calculator_run_config(vals: dict) -> dict:
    config = dict(vals.get("calculator_config_dict", {}))
    command = vals.get("vasp_command", "").strip()
    if command and _needs_external_config(vals.get("calculator", "")):
        config["command"] = command
    return config


def validate_calculator(vals: dict) -> str | None:
    if not _needs_external_config(vals.get("calculator", "")):
        return None

    if vals.get("vasp_command", "").strip():
        return None

    # NOTE: ASE also accepts a Python driver script, but it's is not a shell command
    import os

    if os.environ.get("VASP_SCRIPT", "").strip():
        return None

    return (
        "VASP command is required "
        "(or set ASE_VASP_COMMAND / VASP_COMMAND / VASP_SCRIPT)"
    )


def remember_vasp_command(vals: dict, log=None) -> None:
    if not _needs_external_config(vals.get("calculator", "")):
        return

    from rapmat.app_config import persist_vasp_command

    command = vals.get("vasp_command", "").strip()
    if persist_vasp_command(command) and log:
        log(f"Saved '{command}' as the default VASP command.")
