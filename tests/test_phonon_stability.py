import numpy as np
import pytest
from ase.build import bulk

from rapmat.calculators import Calculators
from rapmat.core import phonon_stability as ps
from rapmat.core.entities import ResultRow, Structure
from rapmat.core.phonon_settings import PhononSettings


def test_serialize_deserialize_phonons_roundtrip():
    import base64
    import gzip

    from phonopy import Phonopy
    from phonopy.interface.phonopy_yaml import PhonopyYaml
    from phonopy.structure.atoms import PhonopyAtoms

    from rapmat.core.phonon import (
        deserialize_phonons,
        get_mesh_min_frequency,
        serialize_phonons,
    )

    atoms = bulk("Si", "diamond", a=5.43)
    pa = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell().array,
    )
    mesh = [8, 8, 8]
    ph = Phonopy(pa, supercell_matrix=np.diag((2, 2, 2)), primitive_matrix="auto")
    ph.generate_displacements(distance=0.03)
    scells = ph.supercells_with_displacements
    rng = np.random.default_rng(0)
    ph.forces = np.array([rng.standard_normal((len(c), 3)) * 0.01 for c in scells])
    ph.produce_force_constants()
    ph.run_mesh(mesh)
    ref_min = get_mesh_min_frequency(ph)

    blob = serialize_phonons(ph)
    assert isinstance(blob, str) and blob

    text = gzip.decompress(base64.b64decode(blob)).decode("utf-8")
    assert "displacements:" in text
    assert "\nforce_constants:" not in text

    py = PhonopyYaml(settings={"force_constants": True})
    py.set_phonon_info(ph)
    fc_blob = base64.b64encode(gzip.compress(str(py).encode("utf-8"))).decode("ascii")
    assert len(blob) < len(fc_blob)

    ph2 = deserialize_phonons(blob)
    assert ph2.force_constants is not None
    assert len(ph2.primitive) == len(ph.primitive)
    ph2.run_mesh(mesh)
    min2 = get_mesh_min_frequency(ph2)

    ph3 = deserialize_phonons(serialize_phonons(ph2))
    ph3.run_mesh(mesh)
    np.testing.assert_allclose(get_mesh_min_frequency(ph3), min2, atol=1e-9)

    np.testing.assert_allclose(min2, ref_min, atol=5e-3)

    ph2.auto_band_structure(plot=False)


def _settings(**overrides):
    fields = dict(
        calculator=Calculators.MATTERSIM,
        calculator_settings="toml",
        supercell=(1, 1, 1),
        mesh=(1, 1, 1),
        displacement=0.01,
        symprec=1e-3,
        reduce_primitive=False,
    )
    fields.update(overrides)
    return PhononSettings(**fields)


def _converged_row(atoms):
    return ResultRow(
        structure=Structure(
            id="", status="relaxed", converged=True, final_atoms=atoms
        )
    )


def test_phonon_progress_does_not_reset_between_structures(monkeypatch):

    monkeypatch.setattr(ps, "CalculatorProvider", lambda *a, **k: (lambda _atoms: object()))

    def fake_calc(atoms, *, progress_callback=None, **kwargs):

        if progress_callback is not None:
            for k in range(4):
                progress_callback(0, 0, f"Processing deformed structure {k + 1}/4")
        return None, -0.1

    monkeypatch.setattr(ps, "calculate_phonons_with_freq", fake_calc)

    calls = []

    def cb(current, total, message, is_log=True):
        calls.append((current, total))

    results = [_converged_row(bulk("Cu", "fcc", a=3.6)) for _ in range(3)]

    ps.compute_dynamical_stability_for_results(
        results=results,
        phonon_top=3,
        settings=_settings(),
        store=None,
        progress_callback=cb,
    )

    currents = [c for c, _t in calls]
    totals = [t for _c, t in calls]

    assert currents == sorted(currents), currents

    assert all(t == 3 for t in totals), totals

    assert currents[-1] == 3


class _Provider:
    made: list = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        _Provider.made.append(self)

    def __call__(self, _atoms):
        return object()


def _row(sid, run="r", domain="bulk", **kw):
    return ResultRow(
        structure=Structure(
            id=sid, run=run, status="relaxed", converged=True,
            final_atoms=bulk("Cu", "fcc", a=3.6), domain=domain, **kw,
        )
    )


@pytest.fixture
def fake_calc(monkeypatch):
    _Provider.made = []
    monkeypatch.setattr(ps, "CalculatorProvider", _Provider)
    computed = []

    def calc(atoms, **kwargs):
        computed.append(atoms)
        return object(), -0.05

    monkeypatch.setattr(ps, "calculate_phonons_with_freq", calc)
    monkeypatch.setattr(ps, "serialize_phonons", lambda _p: "BLOB")
    return computed


def test_stored_results_are_skipped(fake_calc):
    rows = [_row("r/1"), _row("r/2"), _row("r/3")]
    summary = ps.compute_dynamical_stability_for_results(
        rows, 3, _settings(), skip_ids={"r/1", "r/3"}
    )
    assert (summary.targets, summary.skipped, summary.computed) == (3, 2, 1)
    assert [r.min_phonon_freq for r in rows] == [None, -0.05, None]


def test_nothing_to_compute_loads_no_calculator(fake_calc):
    summary = ps.compute_dynamical_stability_for_results(
        [_row("r/1")], 5, _settings(), skip_ids={"r/1"}
    )
    assert summary.skipped == 1
    assert _Provider.made == []


def test_rows_without_atoms_do_not_take_a_slot(fake_calc):
    empty = ResultRow(
        structure=Structure(id="r/0", run="r", status="relaxed", converged=True)
    )
    summary = ps.compute_dynamical_stability_for_results(
        [empty, _row("r/1")], 1, _settings()
    )
    assert summary.computed == 1


def test_monolayer_comes_from_the_rows(fake_calc):
    ps.compute_dynamical_stability_for_results(
        [_row("r/1", domain="monolayer")], 1, _settings()
    )
    assert _Provider.made[0].kwargs["monolayer"] is True


def test_mixed_domains_are_refused(fake_calc):
    with pytest.raises(ValueError, match="mix domains"):
        ps.compute_dynamical_stability_for_results(
            [_row("r/1"), _row("r/2", domain="monolayer")], 2, _settings()
        )


def test_the_command_reaches_the_calculator_but_is_not_stored(fake_calc, store):
    settings = _settings(calculator=Calculators.VASP, calculator_config={"encut": 520})
    store.create_study("s", "Cu", "bulk", "VASP")
    store.create_run(name="r", study_id="s")
    from conftest import add_relaxed_structure

    add_relaxed_structure(store, "r", bulk("Cu", "fcc", a=3.6), -3.0, "r/1")
    rows = [ResultRow(structure=s) for s in store.get_structures("r", status="relaxed")]

    ps.compute_dynamical_stability_for_results(
        rows, 1, settings, store=store, calculator_command="srun vasp_std"
    )

    assert _Provider.made[0].kwargs["config"] == {"encut": 520, "command": "srun vasp_std"}
    stored = store.get_phonon_result("r/1")
    assert "srun" not in stored.settings_json
    assert PhononSettings.from_json(stored.settings_json).identity() == settings.identity()


def test_phase_analysis_rows_are_saved_under_their_runs(fake_calc, store):
    from conftest import add_relaxed_structure

    store.create_study("s", "Cu", "bulk", "MATTERSIM")
    for run in ("a", "b"):
        store.create_run(name=run, study_id="s")
        add_relaxed_structure(store, run, bulk("Cu", "fcc", a=3.6), -3.0, f"{run}/1")

    rows = [
        ResultRow(structure=s)
        for run in ("a", "b")
        for s in store.get_structures(run, status="relaxed")
    ]
    assert all(r.run_name == "" for r in rows)

    summary = ps.compute_dynamical_stability_for_results(rows, 5, _settings(), store=store)

    assert (summary.computed, summary.unsaved) == (2, 0)
    stored = store.get_phonon_settings(["a", "b"])
    assert {sid: run for sid, (run, _text) in stored.items()} == {"a/1": "a", "b/1": "b"}
    assert all(PhononSettings.from_json(text) == _settings() for _run, text in stored.values())


def test_a_row_without_a_run_is_counted_as_unsaved(fake_calc, store):
    summary = ps.compute_dynamical_stability_for_results(
        [_row("x/1", run=None)], 1, _settings(), store=store
    )
    assert (summary.computed, summary.unsaved) == (1, 1)


def test_a_failure_keeps_the_other_results(fake_calc, monkeypatch):
    def calc(atoms, **kwargs):
        if len(fake_calc) == 1:
            fake_calc.append(atoms)
            raise RuntimeError("boom")
        fake_calc.append(atoms)
        return object(), -0.05

    monkeypatch.setattr(ps, "calculate_phonons_with_freq", calc)
    rows = [_row("r/1"), _row("r/2"), _row("r/3")]
    summary = ps.compute_dynamical_stability_for_results(rows, 3, _settings())

    assert (summary.computed, summary.failed) == (2, 1)
    assert [r.min_phonon_freq for r in rows] == [-0.05, None, -0.05]
    assert "2 computed, 1 failed" in summary.describe()
