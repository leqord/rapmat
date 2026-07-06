import math

import numpy as np
import pytest
from ase.build import bulk

try:
    from rapmat.storage import SOAPDescriptor
except ImportError:
    pytest.skip("rapmat.storage not available", allow_module_level=True)

from conftest import (
    add_generated_candidate,
    phonon_params_blob,
    read_batch_config,
)


def test_soap_descriptor_basics():
    atoms = bulk("Si", "diamond", a=5.43)
    desc = SOAPDescriptor(species=["Si"], n_max=4, l_max=3)
    vec = desc.compute(atoms)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] == desc.dimension()
    assert vec.shape[0] > 0
    assert np.linalg.norm(vec) > 0


def test_run_lifecycle(store):
    """Create a run, add candidates, update to relaxed, query back."""
    atoms = bulk("Si", "diamond", a=5.43)

    store.create_study("test-study", "Si", "bulk", "mattersim")
    run_id = store.create_run(name="test-run", study_id="test-study")
    store.update_run_config(run_id, {"formula": {"Si": 1}})
    assert run_id == "test-run"

    meta = store.get_run_metadata("test-run")
    assert meta is not None
    assert meta.name == "test-run"
    assert meta.domain == "bulk"
    assert meta.config == {
        "formula": {"Si": 1},
        "system": "Si",
        "domain": "bulk",
        "calculator": "mattersim",
    }

    struct_id = add_generated_candidate(store, "test-run", "test-run/1", atoms)
    assert struct_id == "test-run/1"

    unrelaxed = store.get_unrelaxed_candidates(run_id)
    assert len(unrelaxed) == 1
    assert unrelaxed[0].id == struct_id

    store.update_structure(
        struct_id,
        "relaxed",
        atoms=atoms,
        metadata={"energy_per_atom": -5.0, "fmax": 0.01, "converged": True},
    )

    relaxed = store.get_structures("test-run", status="relaxed")
    assert len(relaxed) == 1
    assert relaxed[0].final_spg != ""
    assert relaxed[0].initial_spg != ""
    assert relaxed[0].converged is True

    unrelaxed = store.get_unrelaxed_candidates("test-run")
    assert len(unrelaxed) == 0


def test_add_generation_placeholders_batch(store):
    """One placeholders call inserts multiple candidates and they are readable."""
    atoms = bulk("Si", "diamond", a=5.43)

    store.create_study("batch-study", "Si", "bulk", "mattersim")
    run_id = store.create_run(name="batch-run", study_id="batch-study")
    store.update_run_config(run_id, {"formula": {"Si": 1}})

    n = store.add_generation_placeholders("batch-run", [
        ("batch-run/1", 1, 1),
        ("batch-run/2", 1, 1),
        ("batch-run/3", 1, 1),
    ])
    assert n == 3

    for i in (1, 2, 3):
        store.update_generated_structure(f"batch-run/{i}", atoms)

    unrelaxed = store.get_unrelaxed_candidates(run_id)
    assert len(unrelaxed) == 3
    ids = {r.id for r in unrelaxed}
    assert ids == {"batch-run/1", "batch-run/2", "batch-run/3"}
    assert store.count() == 3


def test_phonon_min_freq_persistence(store):
    """save_phonon_result persists min_phonon_freq, get_structures returns it."""
    atoms = bulk("Si", "diamond", a=5.43)

    store.create_study("phonon-study", "Si", "bulk", "mattersim")
    run_id = store.create_run(name="phonon-run", study_id="phonon-study")
    store.update_run_config(run_id, {"formula": {"Si": 1}})
    sid = add_generated_candidate(store, "phonon-run", "phonon-run/1", atoms)
    store.update_structure(sid, "relaxed", atoms=atoms, metadata={"converged": True})

    relaxed = store.get_structures("phonon-run", status="relaxed")
    assert len(relaxed) == 1
    assert relaxed[0].min_phonon_freq is None

    store.save_phonon_result("phonon-run/1", "phonon-run", -0.05)
    relaxed = store.get_structures("phonon-run", status="relaxed")
    assert len(relaxed) == 1
    assert relaxed[0].min_phonon_freq == -0.05

    store.clear_run_phonon_results("phonon-run")
    relaxed = store.get_structures("phonon-run", status="relaxed")
    assert len(relaxed) == 1
    val = relaxed[0].min_phonon_freq
    assert val is None or (isinstance(val, float) and math.isnan(val))


def test_phonon_result_persistence_and_clear(store):
    """The blob lives in phonon_params."""
    atoms = bulk("Si", "diamond", a=5.43)

    store.create_study("pb-study", "Si", "bulk", "mattersim")
    run_id = store.create_run(name="pb-run", study_id="pb-study")
    store.update_run_config(run_id, {"formula": {"Si": 1}})
    sid = add_generated_candidate(store, "pb-run", "pb-run/1", atoms)
    store.update_structure(sid, "relaxed", atoms=atoms, metadata={"converged": True})

    assert store.get_phonon_result("pb-run/1") is None

    store.save_phonon_result(
        "pb-run/1",
        "pb-run",
        -0.05,
        params_gz="BLOB_DATA",
        settings={
            "supercell": [2, 2, 2],
            "mesh": [10, 10, 10],
            "displacement": 0.03,
            "symprec": 1e-3,
            "calculator": "mattersim",
        },
    )

    rec = store.get_phonon_result("pb-run/1")
    assert rec is not None
    assert rec.params_gz == "BLOB_DATA"
    assert rec.displacement == 0.03
    assert rec.calculator == "mattersim"


    assert phonon_params_blob(store, "pb-run/1") == "BLOB_DATA"

    relaxed = store.get_structures("pb-run", status="relaxed")
    assert relaxed[0].min_phonon_freq == -0.05

    store.clear_run_phonon_results("pb-run")
    assert store.get_phonon_result("pb-run/1") is None
    assert phonon_params_blob(store, "pb-run/1") is None
    relaxed = store.get_structures("pb-run", status="relaxed")
    val = relaxed[0].min_phonon_freq
    assert val is None or (isinstance(val, float) and math.isnan(val))


def test_initial_and_final_atoms_preserved(store):
    """Both initial and final atoms survive the add / update cycle."""
    store.create_study("dual-study", "Si", "bulk", "mattersim")
    run_id = store.create_run(name="dual-run", study_id="dual-study")
    store.update_run_config(run_id, {"formula": {"Si": 1}})

    initial_atoms = bulk("Si", "diamond", a=5.43)
    relaxed_atoms = bulk("Si", "diamond", a=5.40)

    sid = add_generated_candidate(store, "dual-run", "dual-run/1", initial_atoms)

    unrelaxed = store.get_unrelaxed_candidates("dual-run")
    assert len(unrelaxed) == 1
    assert len(unrelaxed[0].atoms) == len(initial_atoms)

    store.update_structure(
        sid,
        "relaxed",
        atoms=relaxed_atoms,
        metadata={"converged": True, "energy_per_atom": -5.0},
    )

    records = store.get_structures("dual-run", status="relaxed")
    assert len(records) == 1
    rec = records[0]

    assert rec.initial_atoms is not None
    assert rec.final_atoms is not None
    np.testing.assert_allclose(
        rec.initial_atoms.cell.lengths(), initial_atoms.cell.lengths(), atol=1e-6
    )
    np.testing.assert_allclose(
        rec.final_atoms.cell.lengths(), relaxed_atoms.cell.lengths(), atol=1e-6
    )
    np.testing.assert_allclose(
        rec.atoms.cell.lengths(), relaxed_atoms.cell.lengths(), atol=1e-6
    )


def test_spg_recomputation(store):
    """Different symprec values should produce different SPG results for the same atoms."""
    store.create_study("spg-study", "Si", "bulk", "mattersim")
    run_id = store.create_run(name="spg-run", study_id="spg-study")
    store.update_run_config(run_id, {"formula": {"Si": 1}})

    atoms = bulk("Si", "diamond", a=5.43)
    sid = add_generated_candidate(store, "spg-run", "spg-run/1", atoms)
    store.update_structure(
        sid,
        "relaxed",
        atoms=atoms,
        metadata={"converged": True, "energy_per_atom": -5.0},
    )

    records_default = store.get_structures("spg-run", status="relaxed", symprec=1e-3)
    assert len(records_default) == 1
    spg_default = records_default[0].final_spg
    assert spg_default != ""
    assert spg_default != "N/A"
    assert "(" in spg_default and ")" in spg_default

    records_tight = store.get_structures("spg-run", status="relaxed", symprec=1e-10)
    assert len(records_tight) == 1
    assert records_tight[0].final_spg != ""

    assert records_default[0].initial_spg != ""


def test_derived_fields_roundtrip(store):
    """formula/volume/energy_total/enthalpy/thickness are derived at read time."""
    from ase.units import GPa

    atoms = bulk("Si", "diamond", a=5.43)
    epa = -5.4
    n = len(atoms)

    def make_run(study_id, run_name, domain, config=None):
        store.create_study(study_id, "Si", domain, "mattersim")
        store.create_run(name=run_name, study_id=study_id)
        if config:
            store.update_run_config(run_name, config)
        add_generated_candidate(store, run_name, f"{run_name}/1", atoms)
        store.update_structure(
            f"{run_name}/1", "relaxed", atoms=atoms,
            metadata={"energy_per_atom": epa, "fmax": 0.01, "converged": True},
        )

    make_run("d-bulk", "d-bulk-run", "bulk")
    rec = store.get_structures("d-bulk-run", status="relaxed")[0]
    assert rec.formula == atoms.get_chemical_formula()
    assert rec.volume == pytest.approx(atoms.get_volume())
    assert rec.energy_total == pytest.approx(epa * n)
    assert rec.enthalpy_per_atom is None
    assert rec.thickness is None

    pressure_gpa = 5.0
    make_run("d-press", "d-press-run", "bulk", {"pressure_gpa": pressure_gpa})
    rec = store.get_structures("d-press-run", status="relaxed")[0]
    expected_h = epa + pressure_gpa * GPa * atoms.get_volume() / n
    assert rec.enthalpy_per_atom == pytest.approx(expected_h)

    make_run("d-mono", "d-mono-run", "monolayer")
    rec = store.get_structures("d-mono-run", status="relaxed")[0]
    assert rec.thickness is not None
    assert rec.thickness > 0


def test_get_structures_progress_callback(store):
    """get_structures reports per-record progress via the optional callback."""
    store.create_study("prog-study", "Si", "bulk", "mattersim")
    store.create_run(name="prog-run", study_id="prog-study")

    atoms = bulk("Si", "diamond", a=5.43)
    n_records = 3
    for i in range(1, n_records + 1):
        sid = add_generated_candidate(store, "prog-run", f"prog-run/{i}", atoms)
        store.update_structure(
            sid, "relaxed", atoms=atoms,
            metadata={"energy_per_atom": -5.0, "fmax": 0.01, "converged": True},
        )

    calls = []
    store.get_structures(
        "prog-run", status="relaxed",
        progress_callback=lambda cur, total, msg="", *a: calls.append((cur, total)),
    )
    assert len(calls) == n_records
    assert [c[0] for c in calls] == list(range(1, n_records + 1))
    assert all(total == n_records for _, total in calls)


def test_mark_duplicates_progress_callback(store):
    """mark_duplicates reports monotonic progress ending at (total, total)
    while still applying the flags correctly."""
    store.create_study("dup-study", "Si", "bulk", "mattersim")
    store.create_run(name="dup-run", study_id="dup-study")

    ids = [f"dup-run/{i}" for i in range(1502)]
    store.add_generation_placeholders("dup-run", [(sid, 1, 1) for sid in ids])

    dropped = ids[:1001]
    kept = ids[1001:]
    total = len(dropped) + len(kept)

    calls: list[tuple[int, int]] = []
    store.mark_duplicates(
        dropped_ids=dropped,
        kept_ids=kept,
        progress_callback=lambda done, tot: calls.append((done, tot)),
    )

    assert calls, "progress_callback was never invoked"
    assert all(tot == total for _done, tot in calls)

    done_values = [done for done, _tot in calls]
    assert done_values == sorted(done_values)
    assert done_values[-1] == total

    flag_by_id = {r.id: r.duplicate for r in store.get_structures("dup-run")}
    assert all(flag_by_id[sid] is True for sid in dropped)
    assert all(flag_by_id[sid] is False for sid in kept)


def test_clear_run_duplicates(store):
    """clear_run_duplicates resets the duplicate flag on every structure."""
    store.create_study("dc-study", "Si", "bulk", "mattersim")
    store.create_run(name="dc-run", study_id="dc-study")

    ids = [f"dc-run/{i}" for i in range(5)]
    store.add_generation_placeholders("dc-run", [(sid, 1, 1) for sid in ids])

    store.mark_duplicates(dropped_ids=ids[:2], kept_ids=ids[2:])
    flags = {r.id: r.duplicate for r in store.get_structures("dc-run")}
    assert flags[ids[0]] is True
    assert flags[ids[2]] is False

    store.clear_run_duplicates("dc-run")
    flags = {r.id: r.duplicate for r in store.get_structures("dc-run")}
    assert all(v is None for v in flags.values())


def test_set_structure_excluded_roundtrip(store):
    """set_structure_excluded persists per structure and defaults to False."""
    store.create_study("ex-study", "Si", "bulk", "mattersim")
    store.create_run(name="ex-run", study_id="ex-study")

    ids = [f"ex-run/{i}" for i in range(3)]
    store.add_generation_placeholders("ex-run", [(sid, 1, 1) for sid in ids])

    flags = {r.id: r.excluded for r in store.get_structures("ex-run")}
    assert all(v is False for v in flags.values())

    store.set_structure_excluded(ids[1], True)
    flags = {r.id: r.excluded for r in store.get_structures("ex-run")}
    assert flags[ids[0]] is False
    assert flags[ids[1]] is True

    store.set_structure_excluded(ids[1], False)
    flags = {r.id: r.excluded for r in store.get_structures("ex-run")}
    assert flags[ids[1]] is False


# ------------------------------------------------------------------ #
#  Status enums round-trip through storage exactly
# ------------------------------------------------------------------ #


def test_status_roundtrip_all_enums(store):
    """Every RunStatus/StructureStatus value survives storage exatly.
    """
    from rapmat.storage.status import RunStatus, StructureStatus

    store.create_study("rt-study", "Si", "bulk", "mattersim")
    store.create_run(name="rt-run", study_id="rt-study")

    for status in RunStatus:
        store.set_run_status("rt-run", status)
        assert store.get_run_metadata("rt-run").run_status == status.value

    store.add_generation_placeholders("rt-run", [("rt-run/1", 1, 1)])
    assert store.count_by_status("rt-run") == {StructureStatus.GENERATING.value: 1}

    atoms = bulk("Si", "diamond", a=5.43)
    store.update_generated_structure("rt-run/1", atoms)
    assert store.count_by_status("rt-run") == {StructureStatus.GENERATED.value: 1}

    for status in (StructureStatus.RELAXED, StructureStatus.DISCARDED,
                   StructureStatus.ERROR):
        store.update_structure("rt-run/1", status, atoms=atoms, metadata={})
        assert store.count_by_status("rt-run") == {status.value: 1}


def test_schema_reapplies_on_reconnect(store_factory):
    """Schema is idempotent across reconnects to the same database, and existing
    data survives (all backends)."""
    from rapmat.storage.status import RunStatus

    store = store_factory("reopen_db")
    store.create_study("re-study", "Si", "bulk", "mattersim")
    store.create_run(name="re-run", study_id="re-study")
    store.set_run_status("re-run", RunStatus.COMPLETED)
    store.close()

    store2 = store_factory("reopen_db")
    meta = store2.get_run_metadata("re-run")
    assert meta is not None
    assert meta.run_status == "completed"
    store2.set_run_status("re-run", RunStatus.PENDING)
    assert store2.get_run_metadata("re-run").run_status == "pending"


def test_set_run_config_value_no_leak(store):
    """A per-run override writes only run keys (no leaks to study) and wins on read."""
    store.create_study("s", "Si", "bulk", "mattersim", config={"symprec": 1e-3})
    store.create_run(name="r", study_id="s", config={"formula": {"Si": 1}})

    store.set_run_config_value("r", "symprec", 1e-5)

    assert read_batch_config(store, "r") == {"formula": {"Si": 1}, "symprec": 1e-5}
    assert store.get_run_metadata("r").search_config.symprec == 1e-5


def test_set_study_config_value_leaves_runs_untouched(store):
    """The study default is written to the study, runs inherit it via the merge."""
    store.create_study("s", "Si", "bulk", "mattersim", config={})
    store.create_run(name="r", study_id="s", config={"formula": {"Si": 1}})

    store.set_study_config_value("s", "phonon_cutoff", -0.1)

    assert store.get_study("s").config.get("phonon_cutoff") == -0.1
    assert read_batch_config(store, "r") == {"formula": {"Si": 1}}
    assert store.get_run_metadata("r").search_config.phonon_cutoff == -0.1
