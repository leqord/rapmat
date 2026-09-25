import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from conftest import add_generated_candidate

from rapmat import app_config
from rapmat.core.csp import (_placeholder_seed, execute_run,
                             run_generation_loop, run_processing_loop)
from rapmat.core.generation_worker import generate_one_structure
from rapmat.storage import SQLiteStore
from rapmat.storage.status import RunStatus, StructureStatus


@pytest.fixture
def loop_env(tmp_path):
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


@pytest.fixture
def vasp_loop_env(loop_env, tmp_path, monkeypatch):
    monkeypatch.setattr(app_config, "APP_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(app_config, "_SETTINGS_FILE", tmp_path / "settings.toml")
    monkeypatch.setattr("ase.calculators.vasp.vasp.cfg", os.environ)
    for name in ("ASE_VASP_COMMAND", "VASP_COMMAND", "VASP_SCRIPT"):
        monkeypatch.delenv(name, raising=False)

    loop_env["config"] = {**loop_env["config"], "calculator": "VASP"}
    return loop_env


@patch("rapmat.calculators.factory.load_calculator")
def test_vasp_run_uses_the_saved_command(mock_load_calc, vasp_loop_env):
    mock_load_calc.return_value = EMT()
    app_config.persist_vasp_command("mpirun -np 4 vasp_std")

    execute_run(
        vasp_loop_env["run_name"],
        vasp_loop_env["store"],
        vasp_loop_env["config"],
        worker_id="w-test",
    )

    config = mock_load_calc.call_args.kwargs["config"]
    assert config["command"] == "mpirun -np 4 vasp_std"
    meta = vasp_loop_env["store"].get_run_metadata(vasp_loop_env["run_name"])
    assert meta.run_status == str(RunStatus.COMPLETED)


def test_vasp_run_without_command_fails_before_relaxing(vasp_loop_env):
    store = vasp_loop_env["store"]
    run_name = vasp_loop_env["run_name"]

    with pytest.raises(RuntimeError, match="No VASP command"):
        execute_run(run_name, store, vasp_loop_env["config"], worker_id="w-test")

    meta = store.get_run_metadata(run_name)
    assert meta.run_status == str(RunStatus.FAILED)
    # NOTE: nothing is marked ERROR, so the run can be resumed
    assert len(store.get_unrelaxed_candidates(run_name)) == 3
    assert store.count_by_status(run_name).get("error", 0) == 0


@pytest.fixture
def seed_env(tmp_path):
    store = SQLiteStore.from_path(tmp_path / "seed_db")
    run_name = "seed-run"
    config = {
        "formula": {"Cu": 1},
        "calculator": "MATTERSIM",
        "domain": "bulk",
        "seed": 1000,
    }
    store.create_study(
        study_id=f"study-{run_name}", system="Cu", domain="bulk",
        calculator="MATTERSIM", config=config,
    )
    store.create_run(name=run_name, study_id=f"study-{run_name}")
    store.add_generation_placeholders(
        run_name, [(f"{run_name}/{i}", 225, 1) for i in range(1, 7)]
    )
    return store, run_name, config


def _record_seeds(mock_gen) -> dict:
    cu = bulk("Cu", "fcc", a=3.615)
    seeds = {}

    def _generate(struct_id, *args, seed=None, **kwargs):
        seeds[struct_id] = seed
        return (StructureStatus.GENERATED, struct_id, cu.copy())

    mock_gen.side_effect = _generate
    return seeds


@pytest.mark.parametrize("workers", [1, 3])
@patch("concurrent.futures.ProcessPoolExecutor", ThreadPoolExecutor)
@patch("rapmat.core.csp._generate_one_structure")
def test_fresh_generation_seeds(mock_gen, workers, seed_env):
    store, run_name, config = seed_env
    seeds = _record_seeds(mock_gen)

    run_generation_loop(run_name, store, config, workers=workers)

    assert seeds == {f"{run_name}/{i}": 1000 + i for i in range(1, 7)}


@pytest.mark.parametrize("workers", [1, 3])
@patch("concurrent.futures.ProcessPoolExecutor", ThreadPoolExecutor)
@patch("rapmat.core.csp._generate_one_structure")
def test_resumed_generation_keeps_the_seeds(mock_gen, workers, seed_env):
    store, run_name, config = seed_env
    seeds = _record_seeds(mock_gen)

    store.update_generated_structure(f"{run_name}/1", bulk("Cu", "fcc", a=3.615))
    store.discard_generation_placeholder(f"{run_name}/3")

    run_generation_loop(run_name, store, config, workers=workers)

    assert seeds == {f"{run_name}/{i}": 1000 + i for i in (2, 4, 5, 6)}


class TestPlaceholderSeed:
    def test_uses_the_ordinal_from_the_id(self):
        assert _placeholder_seed(1000, "run/42", position=1) == 1042

    def test_run_name_with_a_slash(self):
        assert _placeholder_seed(1000, "a/b/7", position=1) == 1007

    def test_wraps_to_32_bits(self):
        assert _placeholder_seed(2**32 - 1, "run/2", position=1) == 1

    def test_no_run_seed(self):
        assert _placeholder_seed(None, "run/3", position=1) is None

    def test_non_standard_id_uses_the_position(self):
        assert _placeholder_seed(1000, "custom-id", position=5) == 1005


def test_unexpected_generation_error_is_an_error_status():
    with patch("pyxtal.pyxtal", side_effect=ValueError("bad cell")):
        status, struct_id, atoms = generate_one_structure(
            "run/1", 225, 1, ["Cu"], [1], 3, None, seed=1
        )

    assert status == StructureStatus.ERROR
    assert struct_id == "run/1"
    assert atoms is None
