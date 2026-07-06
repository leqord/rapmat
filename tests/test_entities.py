"""Tests for the typed domain entities and computed properties."""

import numpy as np
import pytest
from ase.build import bulk
from ase.units import GPa

from rapmat.core.entities import Candidate, ResultRow, Structure


def _si():
    return bulk("Si", "diamond", a=5.43)


def test_atoms_alias_prefers_final():
    a = _si()
    b = bulk("Cu", "fcc", a=3.6)
    assert Structure(id="s", final_atoms=b, initial_atoms=a).atoms is b
    assert Structure(id="s", initial_atoms=a).atoms is a
    assert Structure(id="s").atoms is None


def test_formula_volume_energy_total():
    atoms = _si()
    n = len(atoms)
    s = Structure(id="s", final_atoms=atoms, energy_per_atom=-5.4)
    assert s.formula == atoms.get_chemical_formula()
    assert s.volume == pytest.approx(atoms.get_volume())
    assert s.n_atoms == n
    assert s.energy_total == pytest.approx(-5.4 * n)


def test_enthalpy_only_under_pressure():
    atoms = _si()
    n = len(atoms)
    epa = -5.4
    no_p = Structure(id="s", final_atoms=atoms, energy_per_atom=epa)
    assert no_p.enthalpy_per_atom is None

    under_p = Structure(
        id="s", final_atoms=atoms, energy_per_atom=epa, pressure_gpa=5.0
    )
    expected = epa + 5.0 * GPa * atoms.get_volume() / n
    assert under_p.enthalpy_per_atom == pytest.approx(expected)


def test_thickness_only_non_bulk():
    atoms = _si()
    assert Structure(id="s", final_atoms=atoms, domain="bulk").thickness is None
    mono = Structure(id="s", final_atoms=atoms, domain="monolayer")
    assert mono.thickness is not None
    assert mono.thickness > 0


def test_spacegroup_uses_symprec_field_and_relabels():
    atoms = _si()
    s = Structure(id="s", initial_atoms=atoms, final_atoms=atoms, symprec=1e-3)
    label = s.final_spg
    assert label != "" and "(" in label and ")" in label
    assert s.initial_spg != ""

    s.symprec = 1e-10
    assert isinstance(s.final_spg, str)


def test_spacegroup_empty_without_atoms():
    assert Structure(id="s").final_spg == ""


def test_forces_from_atoms_info():
    atoms = _si()
    assert Structure(id="s", final_atoms=atoms).forces is None
    atoms.info["forces"] = np.zeros((len(atoms), 3))
    assert Structure(id="s", final_atoms=atoms).forces is not None


def test_candidate_holds_atoms_and_gen_fields():
    atoms = _si()
    c = Candidate(id="run/1", atoms=atoms, gen_spg=227, gen_fu=2)
    assert c.id == "run/1"
    assert c.atoms is atoms
    assert c.gen_spg == 227
    assert c.gen_fu == 2


def test_resultrow_delegates_to_structure():
    atoms = _si()
    s = Structure(id="run/1", final_atoms=atoms, energy_per_atom=-5.4, fmax=0.01,
                  converged=True)
    row = ResultRow(structure=s, index=1, run_name="run")
    assert row.structure_id == "run/1"
    assert row.formula == atoms.get_chemical_formula()
    assert row.energy_per_atom == -5.4
    assert row.fmax == 0.01
    assert row.converged is True
    assert row.atoms is atoms

    assert row.display_epa == -5.4


def test_resultrow_display_epa_prefers_effective():
    s = Structure(id="x", final_atoms=_si(), energy_per_atom=-5.4)
    row = ResultRow(structure=s, effective_per_atom=-4.2)
    assert row.display_epa == -4.2


def test_resultrow_relabel_via_structure_symprec():
    atoms = _si()
    s = Structure(id="x", initial_atoms=atoms, final_atoms=atoms, symprec=1e-3)
    row = ResultRow(structure=s)
    first = row.final_spg
    assert first != ""

    row.structure.symprec = 1e-10
    assert isinstance(row.final_spg, str)


def test_resultrow_search_text():
    s = Structure(id="run/7", final_atoms=_si())
    row = ResultRow(structure=s, run_name="myrun")
    text = row.search_text()
    assert "run/7" in text
    assert "myrun" in text
    assert text == text.lower()


def test_resultrow_dynamical_stability_is_settable():
    row = ResultRow(structure=Structure(id="x", final_atoms=_si()))
    assert row.dynamical_stability is None
    row.dynamical_stability = True
    assert row.dynamical_stability is True
