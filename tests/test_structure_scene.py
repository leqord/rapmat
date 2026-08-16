"""ASE -> Scene tests."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from rapmat.core import structure_scene
from rapmat.core.render3d import MIN_EXTENT
from rapmat.core.structure_scene import build_scene, next_style, style_label

WATER = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])


def _nacl():
    return bulk("NaCl", "rocksalt", a=5.64, cubic=True) * (2, 2, 2)


def test_primitive_cell_has_bonds():
    scene = build_scene(bulk("Cu", "fcc", a=3.615))
    assert scene.n_atoms == 1
    assert scene.n_bonds == 6


def test_nacl_supercell_atom_and_bond_count():
    scene = build_scene(_nacl())
    assert scene.n_atoms == 64
    assert scene.n_bonds == 192


def test_bond_starts_sit_on_atoms():
    scene = build_scene(_nacl())
    for start in scene.bond_a:
        assert np.isclose(scene.pos, start).all(axis=1).any()


def test_every_bond_ends_on_a_drawn_atom():
    scene = build_scene(_nacl())
    for end in scene.bond_b:
        assert np.isclose(scene.pos, end).all(axis=1).any()


def test_boundary_images_are_added_outside_the_cell():
    atoms = _nacl()
    scene = build_scene(atoms)
    assert len(scene.pos) > scene.n_atoms == len(atoms)

    frac = np.linalg.solve(np.asarray(atoms.get_cell()).T, scene.pos.T).T
    assert ((frac < -1e-6) | (frac > 1 + 1e-6)).any()


def test_no_images_added_when_bonds_are_off():
    scene = build_scene(_nacl(), show_bonds=False)
    assert len(scene.pos) == scene.n_atoms


def test_image_atoms_inherit_the_neighbour_style_and_colour():
    scene = build_scene(_nacl(), style="spacefill")
    extra = len(scene.pos) - scene.n_atoms
    assert extra > 0
    assert len(scene.radius) == len(scene.color) == len(scene.pos)
    palette = {tuple(c) for c in scene.color[: scene.n_atoms]}
    for ghost in scene.color[scene.n_atoms:]:
        assert tuple(ghost) in palette


def test_cell_wireframe_has_twelve_edges():
    scene = build_scene(_nacl())
    assert scene.has_cell is True
    assert scene.edges.shape == (12, 2, 3)


def test_molecule_has_no_cell():
    scene = build_scene(WATER)
    assert scene.n_atoms == 3
    assert scene.has_cell is False
    assert scene.can_supercell is False
    assert scene.edges.shape == (0, 2, 3)


def test_molecule_supercell_is_ignored_not_raised():
    assert build_scene(WATER, supercell=2).n_atoms == 3


def test_zero_volume_cell_counts_as_no_cell():
    flat = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[[2, 0, 0], [0, 2, 0], [0, 0, 0]])
    scene = build_scene(flat)
    assert scene.has_cell is False
    assert scene.can_supercell is False


def test_supercell_multiplies_atoms_and_grows_the_box():
    one = build_scene(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    two = build_scene(bulk("NaCl", "rocksalt", a=5.64, cubic=True), supercell=2)
    assert two.n_atoms == one.n_atoms * 8
    assert two.extent > one.extent


def test_single_atom_scene():
    scene = build_scene(Atoms("H", positions=[[0, 0, 0]]))
    assert scene.n_atoms == 1
    assert scene.n_bonds == 0
    assert scene.extent >= MIN_EXTENT


def test_empty_and_none_give_an_empty_scene():
    for atoms in (None, Atoms()):
        scene = build_scene(atoms)
        assert scene.n_atoms == 0
        assert scene.n_bonds == 0
        assert scene.extent == MIN_EXTENT


def test_show_bonds_false_is_not_capped():
    scene = build_scene(_nacl(), show_bonds=False)
    assert scene.n_bonds == 0
    assert scene.bonds_capped is False


def test_bonds_capped_above_the_limit(monkeypatch):
    monkeypatch.setattr(structure_scene, "MAX_BOND_ATOMS", 4)
    scene = build_scene(_nacl())
    assert scene.bonds_capped is True
    assert scene.n_bonds == 0


def test_supercell_disabled_above_the_scene_limit(monkeypatch):
    monkeypatch.setattr(structure_scene, "MAX_SCENE_ATOMS", 10)
    scene = build_scene(_nacl())
    assert scene.can_supercell is False
    assert scene.n_atoms == 64


def test_style_radii_ordering():
    kw = dict(show_bonds=False)
    ball = build_scene(_nacl(), style="ball", **kw).radius
    space = build_scene(_nacl(), style="spacefill", **kw).radius
    wire = build_scene(_nacl(), style="wireframe", **kw).radius
    assert (space > ball).all()
    assert (ball > wire).all()


def test_unknown_style_falls_back_to_ball():
    assert np.array_equal(
        build_scene(_nacl(), style="nonsense", show_bonds=False).radius,
        build_scene(_nacl(), style="ball", show_bonds=False).radius,
    )


def test_next_style_cycles():
    assert next_style("ball") == "spacefill"
    assert next_style("spacefill") == "wireframe"
    assert next_style("wireframe") == "ball"
    assert next_style("bogus") == "ball"
    assert style_label("ball") == "ball-and-stick"


def test_non_finite_positions_raise():
    bad = bulk("Cu", "fcc", a=3.615) * (2, 1, 1)
    pos = bad.get_positions()
    pos[0, 0] = np.nan
    bad.set_positions(pos)
    with pytest.raises(ValueError):
        build_scene(bad)


def test_colors_and_radii_are_finite_and_positive():
    scene = build_scene(_nacl())
    assert np.isfinite(scene.color).all()
    assert (scene.radius > 0).all()
    assert scene.color.min() >= 0 and scene.color.max() <= 255


def test_extent_and_center_are_finite():
    for atoms in (WATER, _nacl(), bulk("Cu", "fcc", a=3.615)):
        scene = build_scene(atoms)
        assert np.isfinite(scene.center).all()
        assert np.isfinite(scene.extent) and scene.extent >= MIN_EXTENT
