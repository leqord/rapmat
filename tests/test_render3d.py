"""Rasterizer tests."""

import math

import numpy as np
import pytest

from rapmat.core.render3d import (
    BG,
    MIN_EXTENT,
    Scene,
    empty_scene,
    render_frame,
    rot_matrix,
)

_RED = np.array([255.0, 0.0, 0.0])
_BLUE = np.array([0.0, 0.0, 255.0])


def _scene(*, with_bonds=True, with_edges=True, radius=1.0) -> Scene:
    pos = np.array([[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
    corners = np.array(
        [[i, j, k] for i in (-2.0, 2.0) for j in (-2.0, 2.0) for k in (-2.0, 2.0)]
    )
    idx = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
           (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    edges = (
        np.array([[corners[a], corners[b]] for a, b in idx])
        if with_edges
        else np.zeros((0, 2, 3))
    )
    nb = 1 if with_bonds else 0
    return Scene(
        pos=pos,
        radius=np.full(2, radius),
        color=np.array([_RED, _BLUE]),
        bond_a=pos[:nb],
        bond_b=pos[1:1 + nb],
        bond_ca=np.array([_RED])[:nb],
        bond_cb=np.array([_BLUE])[:nb],
        edges=edges,
        center=np.zeros(3),
        extent=4.0,
        n_atoms=2,
        has_cell=with_edges,
        can_supercell=True,
        bonds_capped=False,
    )


def _render(scene, w=40, h=24, **kw):
    kw.setdefault("yaw", 0.3)
    kw.setdefault("pitch", -0.2)
    return render_frame(scene, w, h, **kw)


def _painted(fb) -> int:
    return int((np.abs(fb.astype(int) - BG.astype(int)) > 1).any(axis=-1).sum())


def test_shape_and_dtype():
    fb = _render(_scene())
    assert fb.shape == (24, 40, 3)
    assert fb.dtype == np.uint8


def test_paints_something():
    assert _painted(_render(_scene())) > 0


def test_bonds_toggle_changes_output():
    on = _render(_scene(), show_bonds=True)
    off = _render(_scene(), show_bonds=False)
    assert not np.array_equal(on, off)


def test_cell_toggle_changes_output():
    on = _render(_scene(), show_cell=True)
    off = _render(_scene(), show_cell=False)
    assert not np.array_equal(on, off)


def test_scene_without_edges_ignores_show_cell():
    scene = _scene(with_edges=False)
    assert np.array_equal(
        _render(scene, show_cell=True), _render(scene, show_cell=False)
    )


def test_rotation_changes_output():
    assert not np.array_equal(
        _render(_scene(), yaw=0.0, pitch=0.0),
        _render(_scene(), yaw=1.2, pitch=0.0),
    )


def test_zoom_changes_footprint():
    small = _painted(_render(_scene(), zoom=0.5))
    large = _painted(_render(_scene(), zoom=2.0))
    assert 0 < small < large


@pytest.mark.parametrize("w,h", [(1, 2), (2, 1), (1, 1), (0, 0), (0, 10), (10, 0)])
def test_degenerate_sizes_do_not_raise(w, h):
    fb = _render(_scene(), w=w, h=h)
    assert fb.shape == (h, w, 3)
    assert fb.dtype == np.uint8


def test_negative_size_is_clamped():
    assert render_frame(_scene(), -5, -5, yaw=0.0, pitch=0.0).shape == (0, 0, 3)


def test_empty_scene_is_all_background():
    fb = _render(empty_scene())
    assert fb.shape == (24, 40, 3)
    assert np.array_equal(np.unique(fb.reshape(-1, 3), axis=0), BG.astype(np.uint8)[None])


def test_single_atom_does_not_divide_by_zero():
    scene = empty_scene()
    scene.pos = np.zeros((1, 3))
    scene.radius = np.array([0.4])
    scene.color = _RED[None]
    scene.n_atoms = 1
    scene.extent = MIN_EXTENT
    fb = _render(scene)
    assert np.isfinite(fb.astype(float)).all()
    assert _painted(fb) > 0


def test_zero_extent_is_clamped_not_divided():
    scene = _scene()
    scene.extent = 0.0
    assert np.isfinite(_render(scene).astype(float)).all()


def test_rot_matrix_is_orthonormal():
    for yaw, pitch in [(0.0, 0.0), (0.7, -0.3), (math.pi, math.pi / 2)]:
        r = rot_matrix(yaw, pitch)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert math.isclose(float(np.linalg.det(r)), 1.0, abs_tol=1e-12)


def test_output_stays_in_range():
    fb = _render(_scene(radius=3.0), zoom=4.0)
    assert fb.min() >= 0 and fb.max() <= 255


def test_wireframe_thin_radii_still_render():
    assert _painted(_render(_scene(radius=0.12))) > 0
