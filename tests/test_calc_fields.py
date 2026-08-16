"""Tests for the shared calculator form fields."""

import pytest

from rapmat.tui.widgets import calc_fields
from rapmat.tui.widgets.calc_fields import (CALCULATOR_FIELD_KEYS,
                                            SETTINGS_AUTO, SETTINGS_TOML,
                                            calculator_fields,
                                            calculator_run_config,
                                            is_auto_settings,
                                            parse_toml_config,
                                            setup_calculator_signals,
                                            validate_calculator)
from rapmat.tui.widgets.form import FormGroup


@pytest.fixture(autouse=True)
def no_saved_command(monkeypatch):
    monkeypatch.setattr(
        calc_fields, "resolve_vasp_command", lambda: "", raising=False
    )
    monkeypatch.setattr(
        "rapmat.app_config.resolve_vasp_command", lambda: ""
    )


def _form() -> FormGroup:
    form = FormGroup(calculator_fields(), label_width=20)
    setup_calculator_signals(form)
    return form


def _select(form: FormGroup, key: str, value: str) -> None:
    widget = form.get_widget(key)
    widget._pick(None, widget.options.index(value))


# ------------------------------------------------------------------ #
#  Field set
# ------------------------------------------------------------------ #


class TestFieldSet:
    def test_all_keys_present(self):
        keys = {spec.key for spec in calculator_fields()}
        assert set(CALCULATOR_FIELD_KEYS) <= keys

    def test_defaults_to_auto_mode(self):
        assert _form().get_values()["calculator_settings"] == SETTINGS_AUTO

    def test_calculator_defaults_to_an_mlip(self):
        assert _form().get_values()["calculator"] == "MATTERSIM"

    def test_calc_default_can_be_overridden(self):
        fields = calculator_fields(calc_default="VASP")
        form = FormGroup(fields, label_width=20)
        assert form.get_values()["calculator"] == "VASP"

    def test_unknown_calc_default_falls_back(self):
        fields = calculator_fields(calc_default="NOT-A-CALCULATOR")
        form = FormGroup(fields, label_width=20)
        assert form.get_values()["calculator"] == "MATTERSIM"


# ------------------------------------------------------------------ #
#  Enable logic
# ------------------------------------------------------------------ #


class TestEnableLogic:
    def test_mlip_disables_everything_external(self):
        form = _form()
        for key in ("calculator_settings", "calculator_config", "vasp_command"):
            assert form.is_field_disabled(key), key

    def test_vasp_toml_enables_path_and_command(self):
        form = _form()
        _select(form, "calculator", "VASP")
        _select(form, "calculator_settings", SETTINGS_TOML)
        assert not form.is_field_disabled("calculator_config")
        assert not form.is_field_disabled("vasp_command")
        assert not form.is_field_disabled("calculator_settings")

    def test_vasp_auto_locks_the_toml_path(self):
        form = _form()
        _select(form, "calculator", "VASP")
        assert form.get_values()["calculator_settings"] == SETTINGS_AUTO
        assert form.is_field_disabled("calculator_config")
        assert not form.is_field_disabled("vasp_command")

    def test_switching_back_to_toml_unlocks_the_path(self):
        form = _form()
        _select(form, "calculator", "VASP")
        _select(form, "calculator_settings", SETTINGS_TOML)
        _select(form, "calculator_settings", SETTINGS_AUTO)
        _select(form, "calculator_settings", SETTINGS_TOML)
        assert not form.is_field_disabled("calculator_config")

    def test_leaving_vasp_disables_everything_again(self):
        form = _form()
        _select(form, "calculator", "VASP")
        _select(form, "calculator", "MATTERSIM")
        assert form.is_field_disabled("calculator_config")
        assert form.is_field_disabled("vasp_command")


# ------------------------------------------------------------------ #
#  Values
# ------------------------------------------------------------------ #


class TestIsAutoSettings:
    def test_detects_auto(self):
        assert is_auto_settings({"calculator_settings": SETTINGS_AUTO})

    def test_toml_is_not_auto(self):
        assert not is_auto_settings({"calculator_settings": SETTINGS_TOML})

    def test_missing_key_is_not_auto(self):
        assert not is_auto_settings({})


class TestParseTomlConfig:
    def test_auto_mode_ignores_the_path(self):
        config, err = parse_toml_config(
            {
                "calculator_settings": SETTINGS_AUTO,
                "calculator_config": "/nonexistent/path.toml",
            }
        )
        assert config == {}
        assert err is None

    def test_toml_mode_still_reports_a_missing_file(self):
        _config, err = parse_toml_config(
            {
                "calculator_settings": SETTINGS_TOML,
                "calculator_config": "/nonexistent/path.toml",
            }
        )
        assert err and "not found" in err

    def test_reads_a_real_file(self, tmp_path):
        path = tmp_path / "vasp.toml"
        path.write_text("encut = 520\n", encoding="utf-8")
        config, err = parse_toml_config({"calculator_config": str(path)})
        assert err is None
        assert config == {"encut": 520}


class TestValidateCalculator:
    @pytest.fixture(autouse=True)
    def no_vasp_env(self, monkeypatch):
        for name in ("ASE_VASP_COMMAND", "VASP_COMMAND", "VASP_SCRIPT"):
            monkeypatch.delenv(name, raising=False)

    def test_blank_command_is_rejected_for_vasp(self):
        err = validate_calculator({"calculator": "VASP", "vasp_command": "  "})
        assert err and "VASP command is required" in err

    def test_missing_key_is_rejected_for_vasp(self):
        assert validate_calculator({"calculator": "VASP"}) is not None

    def test_command_satisfies_it(self):
        assert validate_calculator(
            {"calculator": "VASP", "vasp_command": "srun vasp_std"}
        ) is None

    def test_mlips_are_unaffected(self):
        assert validate_calculator(
            {"calculator": "MATTERSIM", "vasp_command": ""}
        ) is None

    def test_vasp_script_satisfies_it(self, monkeypatch):
        monkeypatch.setenv("VASP_SCRIPT", "/opt/run_vasp.py")
        assert validate_calculator(
            {"calculator": "VASP", "vasp_command": ""}
        ) is None


class TestCalculatorRunConfig:
    def test_folds_in_the_vasp_command(self):
        config = calculator_run_config(
            {
                "calculator": "VASP",
                "calculator_config_dict": {"encut": 520},
                "vasp_command": "srun vasp_std",
            }
        )
        assert config == {"encut": 520, "command": "srun vasp_std"}

    def test_ignores_the_command_for_mlips(self):
        config = calculator_run_config(
            {
                "calculator": "MATTERSIM",
                "calculator_config_dict": {},
                "vasp_command": "srun vasp_std",
            }
        )
        assert config == {}

    def test_blank_command_is_omitted(self):
        config = calculator_run_config(
            {"calculator": "VASP", "calculator_config_dict": {}, "vasp_command": "  "}
        )
        assert config == {}

    def test_does_not_mutate_the_cache_relevant_dict(self):
        cache_dict = {"encut": 520}
        vals = {
            "calculator": "VASP",
            "calculator_config_dict": cache_dict,
            "vasp_command": "srun vasp_std",
        }
        calculator_run_config(vals)
        assert cache_dict == {"encut": 520}
        assert vals["calculator_config_dict"] == {"encut": 520}
