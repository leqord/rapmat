"""POTCAR resolution.
"""

from pathlib import Path

import pytest
from ase.build import bulk
from ase.calculators.vasp import Vasp

from rapmat.calculators.vasp import preflight_potcars
from rapmat.calculators.vasp_auto import (omat24_vasp_params,
                                          resolve_potcar_version)


@pytest.fixture
def pp_tree(tmp_path, monkeypatch):
    monkeypatch.delenv("VASP_PP_PATH", raising=False)
    monkeypatch.delenv("VASP_PP_VERSION", raising=False)

    def _make(*folders, symbols=("Si",)):
        for folder in folders:
            for symbol in symbols:
                target = tmp_path / folder / symbol
                target.mkdir(parents=True, exist_ok=True)
                (target / "POTCAR").write_text("fake\n", encoding="utf-8")
        monkeypatch.setenv("VASP_PP_PATH", str(tmp_path))
        return tmp_path

    return _make


@pytest.fixture
def si():
    return bulk("Si", "diamond", a=5.43)


class TestPotcarResolution:
    def test_auto_needs_the_versioned_pbe_directory(self, pp_tree, si):
        pp_tree("potpaw_PBE.54")
        calc = Vasp(**omat24_vasp_params(si))
        calc.initialize(si)
        assert "potpaw_PBE.54" in calc.ppp_list[0]

    def test_bare_potpaw_is_not_enough(self, pp_tree, si):
        pp_tree("potpaw")
        calc = Vasp(**omat24_vasp_params(si))
        with pytest.raises(RuntimeError, match="No pseudopotential"):
            calc.initialize(si)

    def test_unversioned_pbe_is_not_enough_either(self, pp_tree, si):
        pp_tree("potpaw_PBE")
        calc = Vasp(**omat24_vasp_params(si))
        with pytest.raises(RuntimeError, match="No pseudopotential"):
            calc.initialize(si)

    def test_setup_suffix_is_part_of_the_directory_name(self, pp_tree):
        pp_tree("potpaw_PBE.54", symbols=("Fe_pv",))
        atoms = bulk("Fe", "bcc", a=2.87)
        calc = Vasp(**omat24_vasp_params(atoms))
        calc.initialize(atoms)
        assert Path(calc.ppp_list[0]).parent.name == "Fe_pv"


class TestResolvePotcarVersion:
    def test_prefers_the_omat24_version(self, pp_tree):
        pp_tree("potpaw_PBE.54", "potpaw_PBE.64", "potpaw_PBE")
        version, note = resolve_potcar_version()
        assert version == "54"
        assert note is None

    def test_falls_back_to_an_unversioned_tree(self, pp_tree):
        pp_tree("potpaw_PBE")
        version, note = resolve_potcar_version()
        assert version == ""
        assert "OMat24 specifies potpaw_PBE.54" in note

    def test_falls_back_to_64(self, pp_tree):
        pp_tree("potpaw_PBE.64")
        version, note = resolve_potcar_version()
        assert version == "64"
        assert "potpaw_PBE.64" in note

    def test_defers_when_the_user_set_a_version(self, pp_tree, monkeypatch):
        pp_tree("potpaw_PBE.54")
        monkeypatch.setenv("VASP_PP_VERSION", "52")
        assert resolve_potcar_version() == (None, None)

    def test_defers_without_a_pp_path(self, monkeypatch):
        monkeypatch.delenv("VASP_PP_PATH", raising=False)
        monkeypatch.delenv("VASP_PP_VERSION", raising=False)
        assert resolve_potcar_version() == (None, None)

    def test_defers_when_no_pbe_tree_exists(self, pp_tree):
        pp_tree("potpaw_LDA")
        assert resolve_potcar_version() == (None, None)

    def test_omitting_the_version_leaves_it_to_ase(self, si):
        params = omat24_vasp_params(si, potcar_version=None)
        assert "pp_version" not in params
        assert params["xc"] == "PBE"


class TestProviderUsesDetection:
    def test_unversioned_tree_resolves_end_to_end(self, pp_tree, si):
        from rapmat.calculators.factory import CalculatorProvider

        pp_tree("potpaw_PBE")
        notes = []
        provider = CalculatorProvider(
            "VASP", auto_settings=True, log_callback=notes.append
        )
        calc = provider(si)
        preflight_potcars(calc, si)

        assert Path(calc.ppp_list[0]).parent.parent.name == "potpaw_PBE"
        assert any("OMat24 specifies" in n for n in notes)

    def test_no_note_when_the_tree_matches_omat24(self, pp_tree, si):
        from rapmat.calculators.factory import CalculatorProvider

        pp_tree("potpaw_PBE.54")
        notes = []
        provider = CalculatorProvider(
            "VASP", auto_settings=True, log_callback=notes.append
        )
        provider(si)
        assert not any("OMat24 specifies" in n for n in notes)


class TestPreflight:
    def test_raises_with_guidance(self, pp_tree, si):
        pp_tree("potpaw")
        calc = Vasp(**omat24_vasp_params(si))
        with pytest.raises(RuntimeError, match="potpaw_PBE.54"):
            preflight_potcars(calc, si)

    def test_silent_when_potcars_resolve(self, pp_tree, si):
        pp_tree("potpaw_PBE.54")
        preflight_potcars(Vasp(**omat24_vasp_params(si)), si)

    def test_ignores_non_vasp_calculators(self, si):
        from ase.calculators.emt import EMT

        preflight_potcars(EMT(), si)


class TestFailureIsolation:
    def test_one_bad_structure_does_not_stop_the_batch(self, tmp_path):
        from ase.calculators.emt import EMT
        from conftest import add_relaxed_structure

        from rapmat.core.evaluation import run_eval_loop
        from rapmat.storage import SQLiteStore
        from rapmat.storage.status import StructureStatus

        store = SQLiteStore.from_path(tmp_path / "iso_db")
        store.create_study("s", system="Cu", domain="bulk",
                           calculator="VASP", config={})
        store.create_run(name="r", study_id="s")
        for i in range(1, 4):
            add_relaxed_structure(
                store, "r", bulk("Cu", "fcc", a=3.6), -3.5, f"r/{i}"
            )

        pending = store.get_structures("r", status=StructureStatus.RELAXED)
        assert len(pending) == 3

        class Exploding(EMT):
            def __init__(self, doomed: bool) -> None:
                super().__init__()
                self._doomed = doomed

            def calculate(self, atoms=None, *args, **kwargs):
                if self._doomed:
                    raise RuntimeError("VASP exited abnormally")
                return super().calculate(atoms, *args, **kwargs)

        seen = []

        def calculator_for(atoms):
            seen.append(atoms)

            return Exploding(doomed=len(seen) == 3)

        logged = []
        run_eval_loop(
            pending, store, "r", calculator_for, "VASP", "{}",
            log_callback=logged.append,
        )

        stored = {ev.structure_id for ev in store.get_evaluations("r")}
        assert stored == {"r/1", "r/3"}, stored
        assert any("Failed to evaluate structure r/2" in line for line in logged)
        store.close()
