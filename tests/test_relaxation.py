import numpy as np
import pytest
from ase.build import bulk, fcc111
from ase.calculators.emt import EMT

from rapmat.core.relaxation import structure_relax


def test_relax_converges_emt():
    atoms = bulk("Cu", "fcc", a=3.7)
    atoms.calc = EMT()

    converged, relaxed = structure_relax(
        atoms,
        force_conv_crit=0.05,
        steps_max=200,
    )

    assert converged is True
    forces = relaxed.get_forces()
    fmax = float(np.max(np.linalg.norm(forces, axis=1)))
    assert fmax < 0.05


def test_relax_energy_decreases():
    atoms = bulk("Al", "fcc", a=4.3)
    atoms.calc = EMT()
    e_before = atoms.get_potential_energy()

    converged, relaxed = structure_relax(
        atoms,
        force_conv_crit=0.05,
        steps_max=200,
    )

    e_after = relaxed.get_potential_energy()
    assert e_after < e_before


def test_relax_no_calculator_raises():
    atoms = bulk("Cu", "fcc", a=3.615)
    with pytest.raises(RuntimeError, match="No calculator"):
        structure_relax(atoms)


def test_relax_force_break_aborts():
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms = bulk("Cu", "fcc", a=3.615)

    huge_forces = np.ones((len(atoms), 3)) * 1e7
    stress = np.zeros(6)
    atoms.calc = SinglePointCalculator(
        atoms, energy=-10.0, forces=huge_forces, stress=stress
    )

    converged, _ = structure_relax(
        atoms,
        forces_break=1e6,
        steps_max=200,
    )

    assert converged is False


def test_relax_mask_preserves_z_cell():
    slab = fcc111("Al", size=(2, 2, 2), vacuum=10.0)
    slab.calc = EMT()
    c_before = float(slab.cell[2, 2])

    _, relaxed = structure_relax(
        slab,
        mask=[1, 1, 0, 0, 0, 1],
        force_conv_crit=0.05,
        steps_max=50,
    )

    c_after = float(relaxed.cell[2, 2])
    assert c_after == pytest.approx(c_before, abs=1e-10)
