import json

import pytest

from rapmat.calculators import Calculators
from rapmat.core.phonon_settings import (DEFAULT_PHONON_CUTOFF,
                                         PhononSettings, default_phonon_grid,
                                         load_stored_phonons,
                                         prevailing_settings,
                                         resolve_phonon_cutoff,
                                         save_phonon_cutoff, split_stored,
                                         stale_differences)


def _settings(**overrides) -> PhononSettings:
    fields = dict(
        calculator=Calculators.VASP,
        calculator_settings="toml",
        calculator_config={"encut": 520},
        supercell=(3, 3, 3),
        mesh=(20, 20, 20),
        displacement=0.01,
        symprec=1e-3,
        reduce_primitive=True,
    )
    fields.update(overrides)
    return PhononSettings(**fields)


_PHYSICAL_CHANGES = [
    ("calculator", Calculators.MATTERSIM),
    ("calculator_settings", "auto"),
    ("calculator_config", {"encut": 400}),
    ("supercell", (2, 2, 2)),
    ("mesh", (10, 10, 10)),
    ("displacement", 0.02),
    ("symprec", 1e-2),
    ("reduce_primitive", False),
]


class TestIdentity:
    def test_ignores_path_machine_keys_and_key_order(self):
        a = _settings(calculator_config={"encut": 520, "ediff": 1e-6}, config_path="a.toml")
        b = _settings(
            calculator_config={
                "ediff": 1e-6,
                "encut": 520,
                "command": "srun vasp_std",
                "directory": "/scratch/x",
            },
            config_path="b.toml",
        )
        assert a.identity() == b.identity()
        assert a.differences(b) == []

    @pytest.mark.parametrize("field, value", _PHYSICAL_CHANGES)
    def test_every_physical_field_counts(self, field, value):
        changed = _settings(**{field: value})
        assert changed.identity() != _settings().identity()
        assert _settings().differences(changed) == [field]

    def test_the_field_set_is_known(self):
        assert set(PhononSettings.model_fields) == {
            name for name, _ in _PHYSICAL_CHANGES
        } | {"config_path"}

    def test_survives_the_stored_json(self):
        s = _settings(config_path="vasp.toml")
        text = json.dumps(s.model_dump(mode="json"), sort_keys=True)
        back = PhononSettings.from_json(text)
        assert back == s
        assert back.identity() == s.identity()


class TestFromJson:
    @pytest.mark.parametrize("text", [None, "", "not json", "[]"])
    def test_unreadable_gives_none(self, text):
        assert PhononSettings.from_json(text) is None

    def test_legacy_settings_give_none(self):
        legacy = {
            "supercell": [3, 3, 3],
            "mesh": [20, 20, 20],
            "displacement": 0.01,
            "symprec": 1e-3,
            "calculator": "MATTERSIM",
        }
        assert PhononSettings.from_json(json.dumps(legacy)) is None


class TestRunConfig:
    def test_vasp_gets_the_command(self):
        s = _settings()
        assert s.run_config("srun vasp_std") == {"encut": 520, "command": "srun vasp_std"}
        assert s.calculator_config == {"encut": 520}

    def test_mlips_do_not(self):
        s = _settings(calculator=Calculators.UPET, calculator_config={})
        assert s.run_config("srun vasp_std") == {}


def test_default_grid_leaves_the_vacuum_alone():
    assert default_phonon_grid("bulk") == ((3, 3, 3), (20, 20, 20))
    assert default_phonon_grid("monolayer") == ((3, 3, 1), (20, 20, 1))


class TestStoredPhonons:
    def test_split_and_differences(self):
        current = _settings()
        stored = {
            "r1/1": ("r1", current),
            "r1/2": ("r1", _settings(supercell=(2, 2, 2))),
            "r2/1": ("r2", None),
        }
        matching, stale = split_stored(stored, current)
        assert matching == {"r1/1"}
        assert stale == {"r1/2": "r1", "r2/1": "r2"}
        assert stale_differences(stored, stale, current) == ["supercell", "not recorded"]

    def test_prevailing_is_the_most_common_known(self):
        common = _settings(supercell=(2, 2, 2))
        stored = {
            "a": ("r", common),
            "b": ("r", common.model_copy(update={"config_path": "x.toml"})),
            "c": ("r", _settings()),
            "d": ("r", None),
            "e": ("r", None),
        }
        chosen = prevailing_settings(stored)
        assert chosen.identity() == common.identity()
        assert chosen.config_path == "x.toml"

    @pytest.mark.parametrize("stored", [{}, {"a": ("r", None)}])
    def test_nothing_known(self, stored):
        assert prevailing_settings(stored) is None

    def test_loaded_from_the_store(self, store):
        from ase.build import bulk

        from conftest import add_relaxed_structure

        store.create_study("s", "Cu", "bulk", "MATTERSIM")
        store.create_run(name="r", study_id="s")
        add_relaxed_structure(store, "r", bulk("Cu", "fcc", a=3.6), -3.0, "r/1")
        add_relaxed_structure(store, "r", bulk("Cu", "fcc", a=3.7), -3.0, "r/2")
        s = _settings()
        store.save_phonon_result("r/1", "r", -0.01, settings=s.model_dump(mode="json"))
        store.save_phonon_result("r/2", "r", -0.02)

        stored = load_stored_phonons(store, ["r"])
        assert stored["r/1"][0] == "r"
        assert stored["r/1"][1].identity() == s.identity()
        assert stored["r/2"] == ("r", None)


class TestCutoff:
    @pytest.fixture
    def study(self, store):
        store.create_study("s", "Cu", "bulk", "MATTERSIM")
        store.create_run(name="r1", study_id="s")
        store.create_run(name="r2", study_id="s")
        return store

    def test_default(self, study):
        assert resolve_phonon_cutoff(study, "s", ["r1"]) == DEFAULT_PHONON_CUTOFF

    def test_legacy_run_value(self, study):
        study.set_run_config_value("r1", "phonon_cutoff", -0.3)
        assert resolve_phonon_cutoff(study, "s", ["r1"]) == -0.3

    def test_ambiguous_legacy_values_give_the_default(self, study):
        study.set_run_config_value("r1", "phonon_cutoff", -0.3)
        study.set_run_config_value("r2", "phonon_cutoff", -0.2)
        assert resolve_phonon_cutoff(study, "s", ["r1", "r2"]) == DEFAULT_PHONON_CUTOFF

    def test_study_value_wins(self, study):
        study.set_run_config_value("r1", "phonon_cutoff", -0.3)
        study.set_study_config_value("s", "phonon_cutoff", -0.1)
        assert resolve_phonon_cutoff(study, "s", ["r1"]) == -0.1

    def test_saving_moves_it_to_the_study(self, study):
        study.set_run_config_value("r1", "phonon_cutoff", -0.3)
        save_phonon_cutoff(study, "s", ["r1", "r2"], -0.05)

        assert study.get_study("s").config["phonon_cutoff"] == -0.05
        assert all("phonon_cutoff" not in m.config for m in study.get_study_runs("s"))
        assert study.get_run_metadata("r1").search_config.phonon_cutoff == -0.05

    def test_a_run_without_a_study_keeps_its_own(self, store):
        store.create_study("s", "Cu", "bulk", "MATTERSIM")
        store.create_run(name="r", study_id="s")
        save_phonon_cutoff(store, None, ["r"], -0.4)
        assert resolve_phonon_cutoff(store, None, ["r"]) == -0.4
