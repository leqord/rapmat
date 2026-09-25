import os
from pathlib import Path

import pytest
from ase.calculators.vasp import Vasp

from rapmat import app_config
from rapmat.calculators import Calculators, cleanup_calculator_files
from rapmat.calculators.factory import load_calculator
from rapmat.calculators.vasp import (build_calculator_vasp, preflight_command,
                                     with_default_command)


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


class TestVaspTxt:
    def test_txt_is_relative_to_the_directory(self, tmp_path):
        calc = build_calculator_vasp({}, directory=tmp_path)
        assert calc.txt == "vasp.out"

    def test_explicit_txt_is_preserved(self, tmp_path):
        calc = build_calculator_vasp({"txt": "-"}, directory=tmp_path)
        assert calc.txt == "-"


class TestCleanupCalculatorFiles:
    def test_lock_files_are_removed(self, tmp_path):
        calc = build_calculator_vasp({}, directory=tmp_path)
        (tmp_path / "vasp1.lock").write_text("x", encoding="utf-8")

        cleanup_calculator_files(calc)

        assert not (tmp_path / "vasp1.lock").exists()

    def test_stale_results_are_removed(self, tmp_path):
        calc = build_calculator_vasp({}, directory=tmp_path)
        for name in ("OUTCAR", "vasprun.xml", "CONTCAR"):
            (tmp_path / name).write_text("x", encoding="utf-8")

        cleanup_calculator_files(calc)

        assert not any((tmp_path / n).exists() for n in ("OUTCAR", "CONTCAR"))

    def test_inputs_are_left_alone(self, tmp_path):
        calc = build_calculator_vasp({}, directory=tmp_path)
        for name in ("INCAR", "POSCAR", "KPOINTS"):
            (tmp_path / name).write_text("x", encoding="utf-8")

        cleanup_calculator_files(calc)

        assert all((tmp_path / n).exists() for n in ("INCAR", "POSCAR", "KPOINTS"))

    def test_missing_directory_does_not_raise(self, tmp_path):
        cleanup_calculator_files(build_calculator_vasp({}, directory=tmp_path / "nope"))

    def test_non_vasp_calculator_is_ignored(self, tmp_path):
        from ase.calculators.emt import EMT

        (tmp_path / "OUTCAR").write_text("x", encoding="utf-8")
        calc = EMT()
        calc.directory = str(tmp_path)

        cleanup_calculator_files(calc)

        assert (tmp_path / "OUTCAR").exists()


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


@pytest.fixture
def no_vasp_command(tmp_path, monkeypatch):
    monkeypatch.setattr(app_config, "APP_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(app_config, "_SETTINGS_FILE", tmp_path / "settings.toml")

    monkeypatch.setattr("ase.calculators.vasp.vasp.cfg", os.environ)
    for name in ("ASE_VASP_COMMAND", "VASP_COMMAND", "VASP_SCRIPT"):
        monkeypatch.delenv(name, raising=False)


class TestWithDefaultCommand:
    def test_uses_the_saved_command(self, no_vasp_command):
        app_config.persist_vasp_command("mpirun -np 4 vasp_std")
        assert with_default_command({"encut": 500}) == {
            "encut": 500,
            "command": "mpirun -np 4 vasp_std",
        }

    def test_uses_the_environment(self, no_vasp_command, monkeypatch):
        monkeypatch.setenv("ASE_VASP_COMMAND", "srun vasp_std")
        assert with_default_command({})["command"] == "srun vasp_std"

    def test_explicit_command_wins(self, no_vasp_command):
        app_config.persist_vasp_command("saved")
        assert with_default_command({"command": "explicit"}) == {
            "command": "explicit"
        }

    def test_nothing_to_add(self, no_vasp_command):
        assert with_default_command({"encut": 500}) == {"encut": 500}

    def test_config_dict_not_mutated(self, no_vasp_command):
        app_config.persist_vasp_command("saved")
        config = {"encut": 500}
        with_default_command(config)
        assert config == {"encut": 500}


class TestPreflightCommand:
    def test_missing_command_raises(self, no_vasp_command):
        with pytest.raises(RuntimeError, match="No VASP command"):
            preflight_command(build_calculator_vasp({}))

    def test_explicit_command_passes(self, no_vasp_command):
        preflight_command(build_calculator_vasp({"command": "vasp_std"}))

    def test_vasp_script_passes(self, no_vasp_command, monkeypatch):
        monkeypatch.setenv("VASP_SCRIPT", "/opt/run_vasp.py")
        preflight_command(build_calculator_vasp({}))

    def test_non_vasp_calculator_is_ignored(self, no_vasp_command):
        from ase.calculators.emt import EMT

        preflight_command(EMT())
