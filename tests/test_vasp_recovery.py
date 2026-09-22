import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase.build import bulk

from rapmat.calculators import Calculators
from rapmat.calculators.factory import (CalculatorProvider,
                                        apply_provider_overrides,
                                        provider_last_directory,
                                        provider_overrides)
from rapmat.calculators.vasp_recovery import (PATHOLOGICAL, RECOVERY_KEYS,
                                              SYMMETRY, TETRAHEDRON, _LADDER,
                                              classify_vasp_abort,
                                              format_deviations,
                                              is_command_missing,
                                              next_override, read_vasp_out)

# TODO: copied verbatim, check later if these violate VASP copyrights
BZINTS_ABORT = """|     VERY BAD NEWS! internal error in subroutine BZINTS: Tetrahedron         |
|     method fails (number of k-points < 4) 3                                 |
|       ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----       |"""

IBZKPT_ABORT = """|     IBZKPT: not all point group operations associated with the symmetry     |
|     operators of the crystalline lattice are members of the point group     |
|     of the reciprocal lattice.                                              |"""

POSMAP_ABORT = """|     VERY BAD NEWS! internal error in subroutine POSMAP: symmetry            |
|     equivalent atom not found, you might try decreasing or increasing       |
|     SYMPREC by an order of magnitude. 2                                     |"""

PRICELV_ABORT = """|     PRICELV: current lattice and primitive lattice are incommensurate       |
|     within SYMPREC.                                                         |"""

RHOSYG_ABORT = """|     VERY BAD NEWS! internal error in subroutineRHOSYG:stars are not         |
|     distinct, try to increase SYMPREC to e.g. 1E-4 2                        |"""

INVGRP_ABORT = """|     VERY BAD NEWS! internal error in subroutine INVGRP: inverse of          |
|     rotation matrix was not found (increase SYMPREC) 2                      |"""

USF_ABORT = """|     internal error in: us.F  at line: 1344                                  |
|     internal ERROR SETYLM_AUG: 27 -2 81 4 26 9 -3                           |"""

HEALTHY_BZINTS = " BZINTS: Fermi energy:    0.003448;  144.000000 electrons"
HEALTHY_IBZKPT = " Subroutine IBZKPT returns following result:"

MPIRUN_MISSING = (
    "vasp in /tmp/tmploecczxirapmatmaterials returned an error: 127 "
    "stderr /bin/sh: 1: mpirun: not found"
)

IEEE_STDERR = "returned an error: 1 stderr Warning: ieee_invalid is signaling"


class TestClassifier:
    @pytest.mark.parametrize(
        "payload,expected",
        [
            (BZINTS_ABORT, TETRAHEDRON),
            (IBZKPT_ABORT, SYMMETRY),
            (POSMAP_ABORT, SYMMETRY),
            (PRICELV_ABORT, SYMMETRY),
            (RHOSYG_ABORT, SYMMETRY),
            (INVGRP_ABORT, SYMMETRY),
            (USF_ABORT, PATHOLOGICAL),
        ],
    )
    def test_each_abort_class_is_recognised(self, payload, expected):
        assert classify_vasp_abort(payload) == expected

    def test_healthy_bzints_line_is_not_an_abort(self):
        assert classify_vasp_abort(HEALTHY_BZINTS) is None

    def test_healthy_ibzkpt_line_is_not_an_abort(self):
        assert classify_vasp_abort(HEALTHY_IBZKPT) is None

    def test_pricelv_carries_no_internal_error_prefix(self):
        # NOTE: anchoring the table on that phrase would silently miss it
        assert "internal error in subroutine" not in PRICELV_ABORT
        assert classify_vasp_abort(PRICELV_ABORT) == SYMMETRY

    def test_rhosyg_without_a_space_is_recognised(self):
        assert "subroutineRHOSYG" in RHOSYG_ABORT
        assert classify_vasp_abort(RHOSYG_ABORT) == SYMMETRY

    def test_a_missing_command_is_not_an_abort(self):
        assert classify_vasp_abort(MPIRUN_MISSING) is None
        assert is_command_missing(MPIRUN_MISSING)

    def test_an_ieee_warning_alone_classifies_nothing(self):
        assert classify_vasp_abort(IEEE_STDERR) is None

    def test_empty_text_classifies_nothing(self):
        assert classify_vasp_abort("") is None


class TestReadVaspOut:
    def test_none_directory_reads_empty(self):
        assert read_vasp_out(None) == ""

    def test_missing_directory_reads_empty(self, tmp_path):
        assert read_vasp_out(tmp_path / "nope") == ""

    def test_missing_file_reads_empty(self, tmp_path):
        assert read_vasp_out(tmp_path) == ""

    def test_only_the_tail_is_read(self, tmp_path):
        (tmp_path / "vasp.out").write_text(
            "x" * 5000 + BZINTS_ABORT, encoding="utf-8"
        )
        text = read_vasp_out(tmp_path, tail_bytes=200)
        assert len(text) <= 200
        assert classify_vasp_abort(text) == TETRAHEDRON


class TestLadder:
    def test_symmetry_has_a_single_rung(self):
        assert next_override(SYMMETRY, {}) == {"isym": 0}
        assert next_override(SYMMETRY, {"isym": 0}) is None

    def test_tetrahedron_tries_isym_before_ismear(self):
        assert next_override(TETRAHEDRON, {}) == {"isym": 0}
        assert next_override(TETRAHEDRON, {"isym": 0}) == {"ismear": 0}
        assert next_override(TETRAHEDRON, {"isym": 0, "ismear": 0}) is None

    def test_pathological_is_never_retried(self):
        assert next_override(PATHOLOGICAL, {}) is None

    def test_an_unclassified_abort_is_never_retried(self):
        assert next_override(None, {}) is None

    def test_ladder_never_touches_a_physics_flag(self):
        forbidden = {
            "encut", "setups", "pp_version", "xc", "ldau_luj", "ldau", "ldaul",
            "ldauu", "ldauj", "magmom", "ediff", "kpts", "gamma", "algo",
            "prec", "lasph", "ispin", "nsw", "ibrion", "lwave", "lcharg",
            "sigma", "nelm", "nbands", "command", "directory", "txt",
        }
        for rungs in _LADDER.values():
            for patch_ in rungs:
                assert set(patch_) <= RECOVERY_KEYS
                assert not set(patch_) & forbidden

    def test_symprec_is_never_an_actual_rung(self):
        # NOTE: loosening SYMPREC makes VASP symmetrize a different structure
        for rungs in _LADDER.values():
            for patch_ in rungs:
                assert "symprec" not in patch_


class TestFormatDeviations:
    def test_empty_is_empty(self):
        assert format_deviations({}) == ""

    def test_sorted_and_upper_cased(self):
        assert format_deviations({"isym": 0, "ismear": 0}) == "ISMEAR=0, ISYM=0"


class TestProviderOverrides:
    def test_a_forbidden_key_is_rejected(self):
        provider = CalculatorProvider(Calculators.VASP)
        with pytest.raises(ValueError, match="encut"):
            provider.apply_overrides({"encut": 400})

    def test_toml_overrides_do_not_poison_the_cache(self, tmp_path):
        provider = CalculatorProvider(
            Calculators.VASP, tmp_path, config={"encut": 500}
        )
        provider.set_calc_label("r/1")
        pristine = provider(bulk("Si", "diamond", a=5.43))

        provider.apply_overrides({"isym": 0})
        recovered = provider(bulk("Si", "diamond", a=5.43))

        assert recovered is not pristine
        assert recovered.int_params["isym"] == 0
        assert pristine.int_params["isym"] is None

        provider.set_calc_label("r/2")
        assert provider.overrides == {}
        back_to_normal = provider(bulk("Si", "diamond", a=5.43))
        assert back_to_normal is pristine
        assert back_to_normal.int_params["isym"] is None

    def test_overrides_are_sticky_within_a_structure(self, tmp_path):
        provider = CalculatorProvider(Calculators.VASP, tmp_path)
        provider.set_calc_label("r/1")
        provider.apply_overrides({"isym": 0})

        for _ in range(3):
            calc = provider(bulk("Si", "diamond", a=5.43))
            assert calc.int_params["isym"] == 0

    def test_repeating_a_label_keeps_the_overrides(self, tmp_path):
        provider = CalculatorProvider(Calculators.VASP, tmp_path)
        provider.set_calc_label("r/1")
        provider.apply_overrides({"isym": 0})
        provider.set_calc_label("r/1")
        assert provider.overrides == {"isym": 0}

    def test_a_non_string_label_normalises_like_the_allocator(self, tmp_path):
        provider = CalculatorProvider(Calculators.VASP, tmp_path)
        provider.set_calc_label(1)
        provider.apply_overrides({"isym": 0})
        provider.set_calc_label("1")
        assert provider.overrides == {"isym": 0}

    def test_overrides_beat_the_auto_params(self, tmp_path):
        provider = CalculatorProvider(
            Calculators.VASP, tmp_path, auto_settings=True, monolayer=True
        )
        provider.set_calc_label("r/1")
        provider.apply_overrides({"isym": 0})
        calc = provider(bulk("Si", "diamond", a=5.43))

        assert calc.int_params["isym"] == 0
        assert calc.int_params["ismear"] == 0

    def test_pinned_keys_are_reported(self, tmp_path):
        provider = CalculatorProvider(
            Calculators.VASP, tmp_path, config={"ismear": -5}
        )
        assert provider.pinned_recovery_keys == frozenset({"ismear"})

    def test_deviations_include_the_run_level_one(self, tmp_path):
        provider = CalculatorProvider(
            Calculators.VASP, tmp_path, auto_settings=True, monolayer=True
        )
        provider.set_calc_label("r/1")
        provider.apply_overrides({"isym": 0})
        assert provider.deviations == {"ismear": 0, "isym": 0}

    def test_shims_tolerate_a_plain_callable(self):
        plain = (lambda atoms: None)
        assert apply_provider_overrides(plain, {"isym": 0}) is False
        assert provider_overrides(plain) == {}
        assert provider_last_directory(plain) is None


def _scripted_vasp_class(script):
    from ase.calculators.calculator import CalculationFailed

    from rapmat.calculators.vasp import RapmatVasp

    calls = []

    class FakeVasp(RapmatVasp):
        def initialize(self, atoms):
            ...

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=None):
            target = Path(self.directory)
            target.mkdir(parents=True, exist_ok=True)
            index = len(calls)
            payload = script[index] if index < len(script) else None

            calls.append({
                "directory": str(target),
                "int": dict(self.int_params),
                "float": dict(self.float_params),
                "exp": dict(self.exp_params),
                "input": dict(self.input_params),
                "dict": dict(self.dict_params),
                "list_float": dict(self.list_float_params),
                "string": dict(self.string_params),
            })

            for name in ("INCAR", "POSCAR", "OUTCAR"):
                (target / name).write_text("x", encoding="utf-8")
            (target / "vasp.out").write_text(payload or "OK", encoding="utf-8")

            if payload is not None:
                raise CalculationFailed(
                    f"vasp in {target} {IEEE_STDERR}"
                )

            self.atoms = atoms.copy()
            self.results = {
                "energy": -4.2,
                "free_energy": -4.2,
                "forces": np.zeros((len(atoms), 3)),
            }

    return FakeVasp, calls


def _store_with(tmp_path, n=1, atoms_factory=None):
    from conftest import add_relaxed_structure

    from rapmat.storage import SQLiteStore

    factory = atoms_factory or (lambda i: bulk("Cu", "fcc", a=3.6 + 0.1 * i))
    store = SQLiteStore.from_path(tmp_path / "recovery_db")
    store.create_study(
        "s", system="Cu", domain="bulk", calculator="VASP", config={}
    )
    store.create_run(name="r", study_id="s")
    for i in range(1, n + 1):
        add_relaxed_structure(store, "r", factory(i), -3.5, f"r/{i}")
    return store


def _run_loop(tmp_path, script, *, n=1, auto=False, config=None, logged=None):
    from rapmat.core.evaluation import run_eval_loop
    from rapmat.storage.status import StructureStatus

    store = _store_with(tmp_path, n=n)
    pending = store.get_structures("r", status=StructureStatus.RELAXED)
    fake, calls = _scripted_vasp_class(script)

    with patch("rapmat.calculators.vasp.RapmatVasp", fake):
        provider = CalculatorProvider(
            Calculators.VASP,
            tmp_path / "calc",
            config=config,
            auto_settings=auto,
        )
        summary = run_eval_loop(
            pending, store, "r", provider, "VASP", "{}",
            log_callback=None if logged is None else logged.append,
        )

    return store, calls, summary


class TestEndToEndRecovery:
    def test_a_symmetry_abort_recovers_with_isym_zero(self, tmp_path):
        store, calls, summary = _run_loop(tmp_path, [RHOSYG_ABORT])

        assert len(calls) == 2
        assert calls[1]["int"]["isym"] == 0
        assert {ev.structure_id for ev in store.get_evaluations("r")} == {"r/1"}
        assert summary.evaluated == 1
        assert summary.failed == 0

    def test_a_tetrahedron_abort_falls_through_to_ismear(self, tmp_path):
        store, calls, _ = _run_loop(
            tmp_path, [BZINTS_ABORT, BZINTS_ABORT]
        )

        assert len(calls) == 3
        assert calls[1]["int"]["isym"] == 0
        assert calls[2]["int"]["isym"] == 0
        assert calls[2]["int"]["ismear"] == 0

        stored = store.get_evaluations("r")[0]
        assert stored.deviations == "ISMEAR=0, ISYM=0"

    def test_recovery_gives_up_after_two_rungs(self, tmp_path):
        logged = []
        store, calls, summary = _run_loop(
            tmp_path,
            [BZINTS_ABORT, BZINTS_ABORT, BZINTS_ABORT],
            logged=logged,
        )

        assert len(calls) == 3
        assert store.get_evaluations("r") == []
        assert summary.failed == 1
        assert any("Failed to evaluate structure r/1" in x for x in logged)

    def test_a_failure_does_not_stop_the_batch(self, tmp_path):
        store, _calls, summary = _run_loop(
            tmp_path,
            [USF_ABORT],
            n=2,
        )

        assert {ev.structure_id for ev in store.get_evaluations("r")} == {"r/2"}
        assert summary.evaluated == 1
        assert summary.failed == 1

    def test_pathological_geometry_is_not_retried(self, tmp_path):
        _store, calls, _ = _run_loop(tmp_path, [USF_ABORT])
        assert len(calls) == 1

    def test_a_missing_command_is_not_retried(self, tmp_path):
        _store, calls, _ = _run_loop(tmp_path, [MPIRUN_MISSING])
        assert len(calls) == 1

    def test_a_pinned_value_is_not_overridden(self, tmp_path):
        logged = []
        _store, calls, _ = _run_loop(
            tmp_path, [BZINTS_ABORT, BZINTS_ABORT],
            config={"isym": 2},
            logged=logged,
        )

        assert len(calls) == 1
        assert any("pinned" in line for line in logged)

    def test_physics_params_are_identical_across_attempts(self, tmp_path):
        _store, calls, _ = _run_loop(tmp_path, [RHOSYG_ABORT], auto=True)

        before, after = calls[0], calls[1]
        for group in ("float", "exp", "input", "dict", "list_float", "string"):
            assert before[group] == after[group], group

        assert {k: v for k, v in before["int"].items() if k != "isym"} == \
               {k: v for k, v in after["int"].items() if k != "isym"}
        assert before["int"]["isym"] is None
        assert after["int"]["isym"] == 0

    def test_evidence_survives_the_retry(self, tmp_path):
        _store, calls, _ = _run_loop(tmp_path, [RHOSYG_ABORT])

        first, second = Path(calls[0]["directory"]), Path(calls[1]["directory"])
        assert first != second
        assert (first / "vasp.out").is_file()
        assert classify_vasp_abort(
            (first / "vasp.out").read_text(encoding="utf-8")
        ) == SYMMETRY

    def test_deviations_land_in_the_store(self, tmp_path):
        store, _calls, summary = _run_loop(tmp_path, [POSMAP_ABORT])

        assert store.get_evaluations("r")[0].deviations == "ISYM=0"
        assert summary.deviated == {"r/1": "ISYM=0"}

    def test_a_clean_structure_stores_no_deviation(self, tmp_path):
        store, calls, summary = _run_loop(tmp_path, [])

        assert len(calls) == 1
        assert store.get_evaluations("r")[0].deviations is None
        assert summary.deviated == {}


@pytest.fixture
def warnings_logged():
    from rapmat.utils.console import get_logger

    logger = get_logger("rapmat.task")
    records = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Collector(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


class TestWarnings:
    @staticmethod
    def _messages(records):
        return [
            r.getMessage() for r in records if r.levelno == logging.WARNING
        ]

    def test_a_recovery_warns_at_warning_level(self, tmp_path, warnings_logged):
        _run_loop(tmp_path, [RHOSYG_ABORT])

        messages = self._messages(warnings_logged)
        assert any("symmetry abort" in m and "ISYM=0" in m for m in messages)

    def test_a_recovery_warning_names_the_protocol(
        self, tmp_path, warnings_logged
    ):
        _run_loop(tmp_path, [RHOSYG_ABORT])

        messages = self._messages(warnings_logged)
        assert any("deviates from the OMat24 protocol" in m for m in messages)

    def test_the_summary_warns_when_anything_deviated(
        self, tmp_path, warnings_logged
    ):
        _run_loop(tmp_path, [RHOSYG_ABORT])

        messages = self._messages(warnings_logged)
        assert any("1 of 1 evaluations used deviated settings" in m
                   for m in messages)

    def test_nothing_is_warned_for_a_clean_run(self, tmp_path, warnings_logged):
        _run_loop(tmp_path, [])

        assert self._messages(warnings_logged) == []

    def test_a_missing_command_warns(self, tmp_path, warnings_logged):
        _run_loop(tmp_path, [MPIRUN_MISSING])

        messages = self._messages(warnings_logged)
        assert any("not runnable" in m for m in messages)

    def test_the_monolayer_deviation_warns_before_any_structure_runs(
        self, tmp_path, warnings_logged
    ):
        CalculatorProvider(
            Calculators.VASP,
            tmp_path,
            auto_settings=True,
            monolayer=True,
        )

        messages = self._messages(warnings_logged)
        assert any("ISMEAR=0" in m and "tetrahedron" in m for m in messages)

    def test_a_warning_line_is_logged_at_warning_level(self, warnings_logged):
        from rapmat.tui.tasks import TaskProgress

        TaskProgress().warn("something deviated")
        TaskProgress().log("a routine progress line")

        assert self._messages(warnings_logged) == [
            "WARNING: something deviated"
        ]

    def test_task_progress_picks_the_level_from_the_prefix(self):
        from rapmat.tui.tasks import TaskProgress

        progress = TaskProgress()
        progress.warn("something deviated")

        assert progress.log_lines == ["WARNING: something deviated"]


class TestSummary:
    def test_a_clean_run_says_so(self, tmp_path):
        _store, _calls, summary = _run_loop(tmp_path, [])
        assert summary.describe() == "1 evaluations, no settings deviations"

    def test_deviations_are_counted_by_kind(self):
        from rapmat.core.evaluation import EvalLoopSummary

        summary = EvalLoopSummary(
            evaluated=3,
            deviated={"r/1": "ISYM=0", "r/2": "ISYM=0", "r/3": "ISMEAR=0"},
        )
        text = summary.describe()

        assert "3 of 3 evaluations used deviated settings" in text
        assert "ISYM=0 (2)" in text
        assert "ISMEAR=0 (1)" in text

    def test_two_summaries_do_not_share_their_deviation_map(self):
        from rapmat.core.evaluation import EvalLoopSummary

        first, second = EvalLoopSummary(), EvalLoopSummary()
        first.deviated["r/1"] = "ISYM=0"
        assert second.deviated == {}

    def test_the_run_level_deviation_is_named(self):
        from rapmat.core.evaluation import EvalLoopSummary

        summary = EvalLoopSummary(evaluated=2, run_deviations="ISMEAR=0")
        assert "run-level deviation: ISMEAR=0" in summary.describe()


class TestPhononDisplacementRecovery:
    def test_recovery_reaches_the_displacement_supercells(self, tmp_path):
        from rapmat.calculators.vasp_recovery import evaluate_with_recovery

        fake, calls = _scripted_vasp_class([POSMAP_ABORT])
        cell = bulk("Cu", "fcc", a=3.6, cubic=True)

        with patch("rapmat.calculators.vasp.RapmatVasp", fake):
            provider = CalculatorProvider(Calculators.VASP, tmp_path / "calc")
            provider.set_calc_label("r/1")
            forces = evaluate_with_recovery(
                cell,
                provider,
                lambda c: c.get_forces(),
                label="displacement 1/6",
            )

        assert len(calls) == 2
        assert calls[1]["int"]["isym"] == 0
        assert provider.overrides == {"isym": 0}
        assert forces.shape == (len(cell), 3)
