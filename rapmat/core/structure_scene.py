"""ASE Atoms -> Scene.
"""

from typing import Literal, Optional

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.neighborlist import natural_cutoffs, neighbor_list

from rapmat.core.render3d import MIN_EXTENT, Scene, empty_scene

Style = Literal["ball", "spacefill", "wireframe"]

STYLES: tuple[Style, ...] = ("ball", "spacefill", "wireframe")
STYLE_LABELS: dict[str, str] = {
    "ball": "ball-and-stick",
    "spacefill": "spacefill",
    "wireframe": "wireframe",
}

BOND_MULT = 1.2
MAX_BOND_ATOMS = 4000
MAX_SCENE_ATOMS = 20000

_MAX_Z_RADIUS = len(covalent_radii) - 1
_MAX_Z_COLOR = len(jmol_colors) - 1
_FALLBACK_COLOR = np.array([255.0, 105.0, 180.0])
_MIN_RADIUS = 0.5

_EDGE_INDEX = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
               (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def next_style(style: str) -> Style:
    try:
        return STYLES[(STYLES.index(style) + 1) % len(STYLES)]
    except ValueError:
        return STYLES[0]


def style_label(style: str) -> str:
    return STYLE_LABELS.get(style, style)


def build_scene(
    atoms: Optional[Atoms],
    *,
    style: str = "ball",
    show_bonds: bool = True,
    supercell: int = 1,
) -> Scene:
    if atoms is None or len(atoms) == 0:
        return empty_scene()

    has_cell = bool(atoms.cell.rank == 3 and abs(atoms.get_volume()) > 1e-6)
    can_supercell = has_cell and len(atoms) * 8 <= MAX_SCENE_ATOMS
    if supercell > 1 and can_supercell:
        atoms = atoms * (supercell, supercell, supercell)

    pos = atoms.get_positions()
    if not np.isfinite(pos).all():
        raise ValueError("structure has non-finite coordinates")

    numbers = atoms.get_atomic_numbers()
    rcov = covalent_radii[np.clip(numbers, 0, _MAX_Z_RADIUS)]
    rcov = np.where(rcov > 0.05, rcov, _MIN_RADIUS)

    color = jmol_colors[np.clip(numbers, 0, _MAX_Z_COLOR)] * 255.0
    unknown = (numbers <= 0) | (numbers > _MAX_Z_COLOR)
    if unknown.any():
        color = color.copy()
        color[unknown] = _FALLBACK_COLOR

    if style == "spacefill":
        radius = rcov * 1.55
    elif style == "wireframe":
        radius = np.full(len(atoms), 0.12)
    else:
        radius = rcov * 0.42

    bonds_capped = bool(show_bonds and len(atoms) > MAX_BOND_ATOMS)
    if show_bonds and not bonds_capped:
        bond_a, bond_b, bond_ca, bond_cb, ghosts = _bonds(atoms, pos, color)
        if len(ghosts[0]):
            pos = np.vstack([pos, ghosts[0]])
            radius = np.concatenate([radius, radius[ghosts[1]]])
            color = np.vstack([color, color[ghosts[1]]])
    else:
        bond_a = bond_b = bond_ca = bond_cb = np.zeros((0, 3))

    if has_cell:
        cell = np.asarray(atoms.get_cell())
        corners = np.array(
            [[i, j, k] for i in (0.0, 1.0) for j in (0.0, 1.0) for k in (0.0, 1.0)]
        ) @ cell
        edges = np.array([[corners[a], corners[b]] for a, b in _EDGE_INDEX])
    else:
        corners = None
        edges = np.zeros((0, 2, 3))

    lo = (pos - radius[:, None]).min(0)
    hi = (pos + radius[:, None]).max(0)
    if corners is not None:
        lo = np.minimum(lo, corners.min(0))
        hi = np.maximum(hi, corners.max(0))

    return Scene(
        pos=pos,
        radius=radius,
        color=color,
        bond_a=bond_a,
        bond_b=bond_b,
        bond_ca=bond_ca,
        bond_cb=bond_cb,
        edges=edges,
        center=(lo + hi) / 2.0,
        extent=max(float(np.linalg.norm(hi - lo)) / 2.0 + 0.15, MIN_EXTENT),
        n_atoms=len(atoms),
        has_cell=has_cell,
        can_supercell=can_supercell,
        bonds_capped=bonds_capped,
    )


def _bonds(atoms: Atoms, pos: np.ndarray, color: np.ndarray):
    i, j, d, s = neighbor_list("ijDS", atoms, natural_cutoffs(atoms, mult=BOND_MULT))
    keep = i < j
    same = i == j
    if same.any():
        lex = (
            (s[:, 0] > 0)
            | ((s[:, 0] == 0) & (s[:, 1] > 0))
            | ((s[:, 0] == 0) & (s[:, 1] == 0) & (s[:, 2] > 0))
        )
        keep = keep | (same & lex)

    i, j, d, s = i[keep], j[keep], d[keep], s[keep]
    ends = pos[i] + d

    crossing = np.flatnonzero((s != 0).any(axis=1))
    if len(crossing):
        _, unique = np.unique(np.round(ends[crossing], 4), axis=0, return_index=True)
        crossing = crossing[np.sort(unique)]
    ghosts = (ends[crossing], j[crossing])

    return pos[i], ends, color[i], color[j], ghosts
