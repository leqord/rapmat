"""Integration test for the main processing loop.

The calculator factory is mocked (to avoid loading MatterSim/NequIP).
"""

from unittest.mock import patch

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from conftest import add_generated_candidate

from rapmat.core.csp import execute_run, run_processing_loop
from rapmat.storage import SQLiteStore
from rapmat.storage.status import RunStatus, StructureStatus


@pytest.fixture
def loop_env(tmp_path):
    """Store with 3 pre-generated Cu candidates ready for the processing loop."""
    store = SQLiteStore.from_path(tmp_path / "loop_db")

    run_name = "loop-run"
    config = {
        "formula": {"Cu": 1},
        "calculator": "MATTERSIM",
        "calculator_config": {},
        "domain": "bulk",
        "thickness_cutoff": None,
        "dedup_threshold": 1e-2,
        "symprec": 1e-3,
    }
    store.create_study(
        study_id=f"study-{run_name}",
        system="Cu",
        domain="bulk",
        calculator="MATTERSIM",
        config=config,
    )
    store.create_run(name=run_name, study_id=f"study-{run_name}")

    for idx, (struct_type, a) in enumerate(
        [("fcc", 3.7), ("fcc", 3.8), ("bcc", 2.87)], start=1
    ):
        atoms = bulk("Cu", struct_type, a=a)
        add_generated_candidate(store, run_name, f"{run_name}/{idx}", atoms)

    return {
        "store": store,
        "run_name": run_name,
        "config": config,
        "workdir": tmp_path,
    }


@patch("rapmat.calculators.factory.load_calculator")
def test_processing_loop_end_to_end(mock_load_calc, loop_env):
    """Full pipeline: relax -> filter -> dedup -> store, using EMT."""
    mock_load_calc.return_value = EMT()

    run_processing_loop(
        run_name=loop_env["run_name"],
        store=loop_env["store"],
        config=loop_env["config"],
        workdir_path=loop_env["workdir"],
    )

    store = loop_env["store"]
    run_name = loop_env["run_name"]

    assert len(store.get_unrelaxed_candidates(run_name)) == 0

    counts = store.count_by_status(run_name)
    total = sum(counts.values())
    assert total == 3

    assert counts.get("error", 0) == 0

    relaxed = store.get_structures(run_name, status="relaxed")
    assert len(relaxed) >= 1

    for r in relaxed:
        assert abs(r.energy_per_atom) < 100.0
        assert r.fmax >= 0
        assert r.converged is True
        assert r.final_atoms is not None
        assert r.final_spg != ""

@patch("rapmat.calculators.factory.load_calculator")
def test_dedup_flag_disabled_keeps_duplicates(mock_load_calc, tmp_path):
    """When dedup=False, candidate/relaxed dedup blocks are bypassed."""
    store = SQLiteStore.from_path(tmp_path / "no_dedup_db")

    run_name = "no-dedup-run"
    config = {
        "formula": {"Cu": 1},
        "calculator": "MATTERSIM",
        "calculator_config": {},
        "domain": "bulk",
        "thickness_cutoff": None,
        "dedup": False,
        "dedup_threshold": 5.0,
        "symprec": 1e-3,
    }
    store.create_study(
        study_id=f"study-{run_name}",
        system="Cu",
        domain="bulk",
        calculator="MATTERSIM",
        config=config,
    )
    store.create_run(name=run_name, study_id=f"study-{run_name}")

    cu = bulk("Cu", "fcc", a=3.615)

    add_generated_candidate(store, run_name, f"{run_name}/1", cu)

    cu2 = bulk("Cu", "fcc", a=3.616)
    add_generated_candidate(store, run_name, f"{run_name}/2", cu2)

    mock_load_calc.return_value = EMT()

    run_processing_loop(
        run_name=run_name,
        store=store,
        config=config,
        workdir_path=tmp_path,
    )

    counts = store.count_by_status(run_name)
    assert counts.get("relaxed", 0) >= 2
    assert counts.get("discarded", 0) == 0
    assert counts.get("error", 0) == 0


# ---------------------------------------------------------------------------
# Claimed run lifecycle
# ---------------------------------------------------------------------------


@patch("rapmat.calculators.factory.load_calculator")
def test_execute_run_resume_skips_generation(mock_load_calc, loop_env):
    mock_load_calc.return_value = EMT()
    store = loop_env["store"]
    run_name = loop_env["run_name"]

    execute_run(run_name, store, loop_env["config"], worker_id="w-test")

    meta = store.get_run_metadata(run_name)
    assert meta.run_status == str(RunStatus.COMPLETED)
    assert meta.worker_id is None
    assert store.get_unrelaxed_candidates(run_name) == []
    assert store.count_by_status(run_name).get("relaxed", 0) >= 1


@patch("rapmat.core.csp._generate_one_structure")
@patch("rapmat.calculators.factory.load_calculator")
def test_execute_run_generates_then_processes(mock_load_calc, mock_gen, tmp_path):
    store = SQLiteStore.from_path(tmp_path / "gen_db")
    run_name = "gen-run"
    config = {
        "formula": {"Cu": 1},
        "calculator": "MATTERSIM",
        "domain": "bulk",
        "symprec": 1e-3,
    }
    store.create_study(
        study_id=f"study-{run_name}", system="Cu", domain="bulk",
        calculator="MATTERSIM", config=config,
    )
    store.create_run(name=run_name, study_id=f"study-{run_name}")
    store.add_generation_placeholders(
        run_name, [(f"{run_name}/1", 225, 1), (f"{run_name}/2", 225, 1)]
    )

    cu = bulk("Cu", "fcc", a=3.615)
    mock_gen.side_effect = lambda struct_id, *a, **k: (
        StructureStatus.GENERATED, struct_id, cu.copy()
    )
    mock_load_calc.return_value = EMT()

    execute_run(run_name, store, config, worker_id="w-test")

    assert mock_gen.call_count == 2
    meta = store.get_run_metadata(run_name)
    assert meta.run_status == str(RunStatus.COMPLETED)
    assert store.get_pending_generation(run_name) == []
    assert store.count_by_status(run_name).get("relaxed", 0) >= 1


@patch("rapmat.calculators.factory.load_calculator")
def test_execute_run_failure_releases_failed(mock_load_calc, loop_env):
    mock_load_calc.side_effect = RuntimeError("calculator boom")
    store = loop_env["store"]
    run_name = loop_env["run_name"]

    with pytest.raises(RuntimeError, match="calculator boom"):
        execute_run(run_name, store, loop_env["config"], worker_id="w-test")

    meta = store.get_run_metadata(run_name)
    assert meta.run_status == str(RunStatus.FAILED)
    assert meta.worker_id is None

