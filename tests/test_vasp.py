"""Tests for VASP calculator initialization, config parsing, and CLI integration.

All tests run without invoking VASP -- the ASE ``Vasp()`` constructor only
stores parameters and does not launch any external process.
"""

from pathlib import Path

import pytest
from ase.calculators.vasp import Vasp

from rapmat.calculators import Calculators
from rapmat.calculators.factory import load_calculator
from rapmat.calculators.vasp import build_calculator_vasp

# ------------------------------------------------------------------ #
#  3. build_calculator_vasp + factory routing
# ------------------------------------------------------------------ #


class TestBuildCalculatorVasp:
    def test_empty_config_returns_vasp(self):
        calc = build_calculator_vasp({})
        assert isinstance(calc, Vasp)

    def test_params_forwarded_xc(self):
        calc = build_calculator_vasp({"xc": "PBE"})
        assert isinstance(calc, Vasp)
        assert calc.parameters.get("xc") is not None

    def test_params_forwarded_encut(self):
        calc = build_calculator_vasp({"encut": 500})
        assert isinstance(calc, Vasp)
        assert calc.parameters["encut"] == 500

    def test_params_forwarded_ediff(self):
        calc = build_calculator_vasp({"ediff": 1e-6})
        assert isinstance(calc, Vasp)
        assert calc.parameters["ediff"] == pytest.approx(1e-6)

    def test_params_forwarded_prec(self):
        calc = build_calculator_vasp({"prec": "Accurate"})
        assert isinstance(calc, Vasp)
        assert calc.parameters["prec"] == "Accurate"

    def test_params_forwarded_ismear(self):
        calc = build_calculator_vasp({"ismear": 0})
        assert isinstance(calc, Vasp)
        assert calc.parameters["ismear"] == 0

    def test_directory_from_argument(self, tmp_path):
        target = tmp_path / "vasp_work"
        calc = build_calculator_vasp({}, directory=target)
        assert isinstance(calc, Vasp)
        assert calc.directory == str(target)

    def test_directory_from_config_takes_precedence(self, tmp_path):
        config_dir = str(tmp_path / "from_config")
        arg_dir = tmp_path / "from_arg"
        calc = build_calculator_vasp({"directory": config_dir}, directory=arg_dir)
        assert calc.directory == config_dir

    def test_directory_not_set_when_none(self):
        calc = build_calculator_vasp({})
        assert isinstance(calc, Vasp)

        assert calc.directory == "."

    def test_multiple_params_combined(self, tmp_path):
        calc = build_calculator_vasp(
            {
                "xc": "PBE",
                "encut": 500,
                "ediff": 1e-5,
                "prec": "Accurate",
                "ismear": 0,
                "sigma": 0.05,
            },
            directory=tmp_path / "work",
        )
        assert isinstance(calc, Vasp)
        assert calc.parameters["encut"] == 500
        assert calc.parameters["ediff"] == pytest.approx(1e-5)
        assert calc.parameters["prec"] == "Accurate"
        assert calc.parameters["sigma"] == pytest.approx(0.05)
        assert calc.directory == str(tmp_path / "work")

    def test_config_dict_not_mutated(self):
        config = {"encut": 500}
        build_calculator_vasp(config, directory=Path("/tmp/x"))
        assert "directory" not in config


class TestFactoryVasp:
    def test_factory_routes_to_vasp(self):
        calc = load_calculator(Calculators.VASP, config={"encut": 500})
        assert isinstance(calc, Vasp)
        assert calc.parameters["encut"] == 500

    def test_factory_vasp_no_config(self):
        calc = load_calculator(Calculators.VASP)
        assert isinstance(calc, Vasp)

    def test_factory_vasp_with_directory(self, tmp_path):
        calc = load_calculator(
            Calculators.VASP,
            output_dir_path=tmp_path / "out",
            config={"xc": "PBE"},
        )
        assert isinstance(calc, Vasp)
        assert calc.directory == str(tmp_path / "out")

    def test_factory_vasp_config_none_gives_empty(self):
        calc = load_calculator(Calculators.VASP, config=None)
        assert isinstance(calc, Vasp)

    def test_factory_vasp_complex_config(self):
        config = {
            "xc": "PBE",
            "encut": 600,
            "ediff": 1e-6,
            "prec": "Accurate",
            "kpts": [4, 4, 4],
            "ismear": 0,
            "sigma": 0.05,
        }
        calc = load_calculator(Calculators.VASP, config=config)
        assert isinstance(calc, Vasp)
        assert calc.parameters["encut"] == 600
