"""Tests for convex hull construction and formation energy math.

Reference energies:
    Al = -3.0 eV/A,  Cu = -4.0 eV/A

AlCu  (2 atoms, epa=-5.0):
    e_ref = 1*(-3.0) + 1*(-4.0) = -7.0
    formation_energy = (-10.0 - (-7.0)) / 2 = -1.5 eV/A   -> ON hull

Al3Cu (4 atoms, epa=-3.0):
    e_ref = 3*(-3.0) + 1*(-4.0) = -13.0
    formation_energy = (-12.0 - (-13.0)) / 4 = +0.25 eV/A  -> ABOVE hull
"""

import pytest
from ase import Atoms
from ase.build import bulk
from conftest import add_relaxed_structure

from rapmat.core.hull import (build_phase_diagram, collect_study_structures,
                              get_composition_fraction, get_reference_energies,
                              hull_input)
from rapmat.storage import SQLiteStore


def _phase_diagram(
    store,
    study_id,
    *,
    hide_unconverged=True,
    hide_duplicates=False,
    hide_excluded=True,
    show_all=True,
):
    """Fetch -> quality-filter -> build, exactly as the hull screen does."""
    structures, system, use_enthalpy = collect_study_structures(store, study_id)
    kept = hull_input(
        structures,
        hide_unconverged=hide_unconverged,
        hide_duplicates=hide_duplicates,
        hide_excluded=hide_excluded,
    )
    return build_phase_diagram(kept, system, use_enthalpy=use_enthalpy, show_all=show_all)


# ------------------------------------------------------------------ #
#  Hull: excluded flag, reference flag, quality filters
# ------------------------------------------------------------------ #


def test_excluding_structure_recomputes_hull(hull_store):
    """Excluding an on-hull structure drops it from the hull and the rows."""
    _, before = _phase_diagram(hull_store, "test-study")
    by_before = {d.structure_id: d for d in before}
    assert by_before["alcu-on/1"].is_stable is True
    al3cu_eah_before = by_before["al3cu-off/1"].energy_above_hull

    hull_store.set_structure_excluded("alcu-on/1", True)

    _, after = _phase_diagram(hull_store, "test-study")
    by_after = {d.structure_id: d for d in after}
    assert "alcu-on/1" not in by_after

    assert by_after["al3cu-off/1"].energy_above_hull < al3cu_eah_before

    _, shown = _phase_diagram(hull_store, "test-study", hide_excluded=False)
    by_shown = {d.structure_id: d for d in shown}
    assert by_shown["alcu-on/1"].is_stable is True


def test_excluding_reference_falls_back_to_next_lowest(hull_store):
    """Excluding the chosen endpoint reference picks the next-lowest one."""
    al2 = bulk("Al", "fcc", a=4.10)
    add_relaxed_structure(hull_store, "al-run", al2, -2.5, "al-run/2")

    structs, system, _ = collect_study_structures(hull_store, "test-study")
    assert get_reference_energies(hull_input(structs), system)["Al"] == pytest.approx(
        -3.0
    )

    hull_store.set_structure_excluded("al-run/1", True)
    structs2, system2, _ = collect_study_structures(hull_store, "test-study")
    refs = get_reference_energies(hull_input(structs2), system2)
    assert refs["Al"] == pytest.approx(-2.5)


def test_reference_flag_marks_endpoints(hull_store):
    """The lowest-energy pure-element structures carry is_reference."""
    _, data = _phase_diagram(hull_store, "test-study")
    by_id = {d.structure_id: d for d in data}
    assert by_id["al-run/1"].is_reference is True
    assert by_id["cu-run/1"].is_reference is True
    assert by_id["alcu-on/1"].is_reference is False


def test_hide_duplicates_controls_hull(hull_store):
    """Showing duplicates keeps them in the hull, hiding removes them entirely."""
    hull_store.mark_duplicates(["alcu-on/1"], [])

    _, shown = _phase_diagram(hull_store, "test-study", hide_duplicates=False)
    by_shown = {d.structure_id: d for d in shown}
    assert by_shown["alcu-on/1"].is_stable is True

    _, hidden = _phase_diagram(hull_store, "test-study", hide_duplicates=True)
    assert "alcu-on/1" not in {d.structure_id for d in hidden}


def test_hull_input_excludes_unconverged_by_default(hull_store):
    """Unconverged structures stay out of the hull unless shown."""
    bad = Atoms(
        symbols=["Al", "Cu"],
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
        cell=[3, 3, 3],
        pbc=True,
    )
    add_relaxed_structure(
        hull_store, "alcu-on", bad, -10.0, "alcu-on/unconv", converged=False
    )

    _, data = _phase_diagram(hull_store, "test-study")
    assert "alcu-on/unconv" not in {d.structure_id for d in data}

    _, data_inc = _phase_diagram(hull_store, "test-study", hide_unconverged=False)
    by_id = {d.structure_id: d for d in data_inc}
    assert "alcu-on/unconv" in by_id
    assert by_id["alcu-on/unconv"].is_stable is True


def test_sort_result_rows_orders_and_reindexes():
    from rapmat.core.entities import ResultRow, Structure
    from rapmat.tui.screens.hull import sort_result_rows

    def row(sid, form, eah):
        return ResultRow(
            structure=Structure(id=sid),
            formation_energy=form,
            energy_above_hull=eah,
        )

    rows = [
        row("a", 0.25, 0.10),
        row("b", -1.5, 0.30),
        row("c", -1.0, 0.0),
        row("d", None, None),
    ]

    sort_result_rows(rows, "formation")
    assert [r.structure_id for r in rows] == ["b", "c", "a", "d"]
    assert [r.index for r in rows] == [1, 2, 3, 4]

    sort_result_rows(rows, "eah")
    assert [r.structure_id for r in rows] == ["c", "a", "b", "d"]
    assert [r.index for r in rows] == [1, 2, 3, 4]


# ------------------------------------------------------------------ #
#  get_composition_fraction
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "formula,element,expected",
    [
        ({"Al": 2, "O": 3}, "Al", 0.4),
        ({"Al": 2, "O": 3}, "O", 0.6),
        ({"Al": 2, "O": 3}, "Si", 0.0),
        ({"Si": 1}, "Si", 1.0),
        ({}, "Al", 0.0),
    ],
)
def test_get_composition_fraction(formula, element, expected):
    assert get_composition_fraction(formula, element) == pytest.approx(expected)


# ------------------------------------------------------------------ #
#  get_reference_energies
# ------------------------------------------------------------------ #


def test_get_reference_energies(hull_store):
    structs, system, _ = collect_study_structures(hull_store, "test-study")
    refs = get_reference_energies(structs, system)
    assert refs["Al"] == pytest.approx(-3.0)
    assert refs["Cu"] == pytest.approx(-4.0)


def test_get_reference_energies_picks_minimum(hull_store):
    """If multiple pure-element structures exist, pick the lowest epa."""
    al2 = bulk("Al", "fcc", a=4.10)
    add_relaxed_structure(hull_store, "al-run", al2, -2.5, "al-run/2")

    structs, system, _ = collect_study_structures(hull_store, "test-study")
    refs = get_reference_energies(structs, system)
    assert refs["Al"] == pytest.approx(-3.0)


def test_get_reference_energies_missing_endpoint(tmp_path):
    """ValueError when a pure-element structure is missing."""
    store = SQLiteStore.from_path(tmp_path / "missing_ep")
    store.create_study("s", system="Al-Cu", domain="bulk", calculator="mock")

    store.create_run(
        "al-only",
        config={"formula": {"Al": 1}, "calculator": "mock"},
        study_id="s",
    )
    add_relaxed_structure(
        store, "al-only", bulk("Al", "fcc", a=4.05), -3.0, "al-only/1"
    )

    structs, system, _ = collect_study_structures(store, "s")
    with pytest.raises(ValueError, match="pure-Cu"):
        get_reference_energies(structs, system)


# ------------------------------------------------------------------ #
#  build_phase_diagram
# ------------------------------------------------------------------ #


def test_build_phase_diagram_stable_and_unstable(hull_store):
    pd_obj, data = _phase_diagram(hull_store, "test-study")

    by_formula = {d.reduced_formula: d for d in data}

    # AlCu should be stable (on the hull)
    alcu = by_formula["AlCu"]
    assert alcu.is_stable is True
    assert alcu.energy_above_hull < 1e-6
    assert alcu.formation_energy == pytest.approx(-1.5, abs=1e-4)
    assert alcu.composition_frac == pytest.approx(0.5, abs=1e-4)

    # Al3Cu should be unstable (above hull)
    al3cu = by_formula["Al3Cu"]
    assert al3cu.is_stable == False
    assert al3cu.energy_above_hull > 0.1
    assert al3cu.formation_energy == pytest.approx(0.25, abs=1e-4)
    assert al3cu.composition_frac == pytest.approx(0.25, abs=1e-4)


def test_build_phase_diagram_formation_energy_signs(hull_store):
    """Pure-element endpoints should have zero formation energy."""
    _, data = _phase_diagram(hull_store, "test-study")

    by_formula = {d.reduced_formula: d for d in data}

    assert by_formula["Al"].formation_energy == pytest.approx(0.0, abs=1e-6)
    assert by_formula["Cu"].formation_energy == pytest.approx(0.0, abs=1e-6)


def test_build_phase_diagram_show_all_vs_best(hull_store):
    """show_all=True returns every structure, False keeps only on-hull ones."""
    alcu2 = Atoms(
        symbols=["Al", "Cu"],
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
        cell=[3, 3, 3],
        pbc=True,
    )
    add_relaxed_structure(hull_store, "alcu-on", alcu2, -4.0, "alcu-on/2")

    _, data_best = _phase_diagram(hull_store, "test-study", show_all=False)
    alcu_best = [d for d in data_best if d.reduced_formula == "AlCu"]
    assert len(alcu_best) == 1
    assert alcu_best[0].energy_per_atom == pytest.approx(-5.0)

    _, data_all = _phase_diagram(hull_store, "test-study", show_all=True)
    alcu_all = [d for d in data_all if d.reduced_formula == "AlCu"]
    assert len(alcu_all) == 2


def test_build_phase_diagram_no_intermediates(tmp_path):
    """Only pure-element endpoints should raise ValueError."""
    store = SQLiteStore.from_path(tmp_path / "no_inter")
    store.create_study("s", system="Al-Cu", domain="bulk", calculator="mock")

    store.create_run(
        "al-r",
        config={"formula": {"Al": 1}, "calculator": "mock"},
        study_id="s",
    )
    add_relaxed_structure(store, "al-r", bulk("Al", "fcc", a=4.05), -3.0, "al-r/1")

    store.create_run(
        "cu-r",
        config={"formula": {"Cu": 1}, "calculator": "mock"},
        study_id="s",
    )
    add_relaxed_structure(store, "cu-r", bulk("Cu", "fcc", a=3.615), -4.0, "cu-r/1")

    structs, system, _ = collect_study_structures(store, "s")
    with pytest.raises(ValueError, match="intermediate"):
        build_phase_diagram(structs, system)


def test_collect_study_structures_study_not_found(tmp_path):
    store = SQLiteStore.from_path(tmp_path / "empty")
    with pytest.raises(ValueError, match="not found"):
        collect_study_structures(store, "nonexistent")


def test_build_phase_diagram_data_sorted_by_composition(hull_store):
    """Returned structure_data should be sorted by composition_frac."""
    _, data = _phase_diagram(hull_store, "test-study")
    fracs = [d.composition_frac for d in data]
    assert fracs == sorted(fracs)


def test_build_phase_diagram_uses_enthalpy_under_pressure(tmp_path):
    """When the study has pressure_gpa > 0, hull should use enthalpy not energy."""
    from ase.units import GPa

    pressure_gpa = 10.0
    p_evA3 = pressure_gpa * GPa

    store = SQLiteStore.from_path(tmp_path / "pressure_hull")
    store.create_study(
        "p-study",
        system="Al-Cu",
        domain="bulk",
        calculator="mock",
        config={"pressure_gpa": pressure_gpa},
    )

    # Pure Al endpoint
    store.create_run("p-al", config={"formula": {"Al": 1}}, study_id="p-study")
    al = bulk("Al", "fcc", a=4.05)
    add_relaxed_structure(store, "p-al", al, -3.0, "p-al/1")

    # Pure Cu endpoint
    store.create_run("p-cu", config={"formula": {"Cu": 1}}, study_id="p-study")
    cu = bulk("Cu", "fcc", a=3.615)
    add_relaxed_structure(store, "p-cu", cu, -4.0, "p-cu/1")

    # AlCu: low energy but a huge cell (V/N = 500 A^3) -> high enthalpy
    store.create_run(
        "p-alcu", config={"formula": {"Al": 1, "Cu": 1}}, study_id="p-study"
    )
    alcu = Atoms(
        symbols=["Al", "Cu"],
        positions=[[0, 0, 0], [5, 5, 5]],
        cell=[10, 10, 10],
        pbc=True,
    )
    add_relaxed_structure(store, "p-alcu", alcu, -5.0, "p-alcu/1")

    # Al3Cu: higher energy but compact cell (V/N = 6.75 A^3) -> low enthalpy
    store.create_run(
        "p-al3cu", config={"formula": {"Al": 3, "Cu": 1}}, study_id="p-study"
    )
    al3cu = Atoms(
        symbols=["Al", "Al", "Al", "Cu"],
        positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]],
        cell=[3, 3, 3],
        pbc=True,
    )
    add_relaxed_structure(store, "p-al3cu", al3cu, -3.0, "p-al3cu/1")

    structs, system, use_enthalpy = collect_study_structures(store, "p-study")
    pd_obj, data = build_phase_diagram(
        structs, system, use_enthalpy=use_enthalpy, show_all=True
    )

    assert use_enthalpy is True

    by_formula = {d.reduced_formula: d for d in data}

    expected_alcu = -5.0 + p_evA3 * alcu.get_volume() / len(alcu)
    expected_al3cu = -3.0 + p_evA3 * al3cu.get_volume() / len(al3cu)
    assert by_formula["AlCu"].effective_per_atom == pytest.approx(expected_alcu)
    assert by_formula["Al3Cu"].effective_per_atom == pytest.approx(expected_al3cu)
