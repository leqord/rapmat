"""Tests for CalculatorProvider."""

from unittest.mock import patch

import pytest
from ase.build import bulk, mx2
from ase.calculators.vasp import Vasp

from rapmat.calculators import Calculators
from rapmat.calculators.factory import CalculatorProvider
from rapmat.core.config import SearchConfig
from rapmat.core.evaluation import evaluation_config_key


# ------------------------------------------------------------------ #
#  Sharing vs per-structure
# ------------------------------------------------------------------ #


class TestStaticConfig:
    def test_shares_one_instance(self):
        provider = CalculatorProvider(Calculators.VASP, config={"encut": 500})
        first = provider(bulk("Si", "diamond", a=5.43))
        second = provider(bulk("Cu", "fcc", a=3.6))
        assert first is second

    def test_built_lazily(self):
        with patch("rapmat.calculators.factory.load_calculator") as loader:
            provider = CalculatorProvider(Calculators.VASP)
            loader.assert_not_called()
            provider(bulk("Si", "diamond", a=5.43))
            loader.assert_called_once()

    def test_reset_forces_a_rebuild(self):
        provider = CalculatorProvider(Calculators.VASP, config={"encut": 500})
        atoms = bulk("Si", "diamond", a=5.43)
        first = provider(atoms)
        provider.reset()
        assert provider(atoms) is not first

    def test_auto_ignored_for_non_vasp(self):
        provider = CalculatorProvider(
            Calculators.MATTERSIM, auto_settings=True
        )
        assert provider.auto is False

    def test_accepts_a_plain_string_name(self):
        assert CalculatorProvider("VASP", auto_settings=True).auto is True


class TestAutoSettings:
    def test_reports_auto(self):
        assert CalculatorProvider(Calculators.VASP, auto_settings=True).auto

    def test_fresh_calculator_per_structure(self):
        provider = CalculatorProvider(Calculators.VASP, auto_settings=True)
        first = provider(bulk("Si", "diamond", a=5.43))
        second = provider(bulk("Si", "diamond", a=5.43))
        assert first is not second

    def test_settings_differ_between_structures(self):
        provider = CalculatorProvider(Calculators.VASP, auto_settings=True)
        si = provider(bulk("Si", "diamond", a=5.43))
        fe = provider(bulk("Fe", "bcc", a=2.87))
        assert si.input_params["kpts"] != fe.input_params["kpts"]

    def test_supercell_gets_its_own_mesh_and_magmom(self):
        provider = CalculatorProvider(Calculators.VASP, auto_settings=True)
        unit = bulk("Fe", "bcc", a=2.87)
        supercell = unit.repeat((2, 2, 2))

        unit_calc = provider(unit)
        super_calc = provider(supercell)

        assert len(super_calc.list_float_params["magmom"]) == len(supercell)
        assert (
            super_calc.input_params["kpts"][0] < unit_calc.input_params["kpts"][0]
        )

    def test_base_config_is_preserved(self):
        provider = CalculatorProvider(
            Calculators.VASP,
            config={"command": "mpirun -np 4 vasp_std"},
            auto_settings=True,
        )
        calc = provider(bulk("Si", "diamond", a=5.43))
        assert calc.command == "mpirun -np 4 vasp_std"
        assert calc.string_params["algo"] == "Normal"

    def test_monolayer_flag_reaches_the_mesh(self):
        atoms = mx2("MoS2", vacuum=1.0)
        flat = CalculatorProvider(
            Calculators.VASP, auto_settings=True, monolayer=True
        )
        solid = CalculatorProvider(
            Calculators.VASP, auto_settings=True, monolayer=False
        )
        assert flat(atoms).input_params["kpts"][2] == 1
        assert solid(atoms).input_params["kpts"][2] > 1

    def test_logs_the_settings(self):
        lines = []
        provider = CalculatorProvider(
            Calculators.VASP, auto_settings=True, log_callback=lines.append
        )
        provider(bulk("Si", "diamond", a=5.43))
        assert len(lines) == 1
        assert "Si2" in lines[0]
        assert "k-mesh" in lines[0]

    def test_returns_a_usable_vasp_calculator(self):
        provider = CalculatorProvider(Calculators.VASP, auto_settings=True)
        assert isinstance(provider(bulk("Si", "diamond", a=5.43)), Vasp)


# ------------------------------------------------------------------ #
#  Evaluation cache key
# ------------------------------------------------------------------ #


class TestEvaluationConfigKey:
    def test_auto_and_toml_do_not_collide(self):
        toml = evaluation_config_key(
            calculator_config={}, calculator_settings="toml"
        )
        auto = evaluation_config_key(
            calculator_config={}, calculator_settings="auto"
        )
        assert toml != auto

    def test_toml_key_is_unchanged_from_before_auto_existed(self):
        import json

        legacy = json.dumps(
            {"run_phonons": False, "calculator_config": {"encut": 520}},
            sort_keys=True,
        )
        assert (
            evaluation_config_key(calculator_config={"encut": 520}) == legacy
        )

    def test_default_is_toml(self):
        assert evaluation_config_key(
            calculator_config={}
        ) == evaluation_config_key(
            calculator_config={}, calculator_settings="toml"
        )

    def test_phonon_settings_only_when_enabled(self):
        without = evaluation_config_key(
            calculator_config={}, run_phonons=False, phonon_mesh=(9, 9, 9)
        )
        assert "phonon_mesh" not in without

    def test_phonon_settings_change_the_key(self):
        a = evaluation_config_key(
            calculator_config={}, run_phonons=True, phonon_mesh=(20, 20, 20)
        )
        b = evaluation_config_key(
            calculator_config={}, run_phonons=True, phonon_mesh=(9, 9, 9)
        )
        assert a != b


# ------------------------------------------------------------------ #
#  Config persistence: study -> run -> resume
# ------------------------------------------------------------------ #


class TestSearchConfigMode:
    def test_defaults_to_toml(self):
        assert SearchConfig().calculator_settings == "toml"

    def test_legacy_study_config_validates(self):
        cfg = SearchConfig.model_validate({"calculator_config": {"encut": 500}})
        assert cfg.calculator_settings == "toml"

    def test_survives_the_study_run_merge(self):
        cfg = SearchConfig.from_stored(
            {"calculator_settings": "auto", "calculator_config": {}},
            {"formula": {"Si": 1}, "seed": 7},
            domain="monolayer",
            calculator="VASP",
        )
        assert cfg.calculator_settings == "auto"
        assert cfg.domain == "monolayer"
        assert cfg.seed == 7


@pytest.mark.parametrize("mode", ["toml", "auto"])
def test_mode_round_trips_through_the_store(tmp_path, mode):
    from rapmat.storage import SQLiteStore

    store = SQLiteStore.from_path(tmp_path / "modes_db")
    store.create_study(
        study_id=f"study-{mode}",
        system="Si",
        domain="monolayer",
        calculator="VASP",
        config={"calculator_settings": mode, "calculator_config": {}},
    )
    store.create_run(name=f"run-{mode}", study_id=f"study-{mode}")

    meta = store.get_run_metadata(f"run-{mode}")
    assert meta.search_config.calculator_settings == mode
    assert meta.domain == "monolayer"
