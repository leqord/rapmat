"""Tests for phonon dynamical stability."""

import numpy as np
from ase.build import bulk

from rapmat.calculators import Calculators
from rapmat.core import phonon_stability as ps
from rapmat.core.entities import ResultRow, Structure


def test_serialize_deserialize_phonons_roundtrip():
    """The force-sets blob reproduces force constants on load.
    """
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
        phonon_cutoff=-0.15,
        phonon_supercell=(1, 1, 1),
        phonon_mesh=(1, 1, 1),
        phonon_displacement=0.01,
        phonon_calculator=Calculators.MATTERSIM,
        store=None,
        progress_callback=cb,
        reduce_primitive=False,
    )

    currents = [c for c, _t in calls]
    totals = [t for _c, t in calls]

    assert currents == sorted(currents), currents

    assert all(t == 3 for t in totals), totals

    assert currents[-1] == 3
