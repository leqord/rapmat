"""Software rasterizer.
"""

import dataclasses
import math

import numpy as np

LIGHT = np.array([-0.38, 0.55, 0.74])
LIGHT /= np.linalg.norm(LIGHT)
HALF = LIGHT + np.array([0.0, 0.0, 1.0])
HALF /= np.linalg.norm(HALF)

BG = np.array([9.0, 10.0, 14.0])
CELL_COLOR = np.array([118.0, 126.0, 148.0])

MIN_EXTENT = 0.5
MIN_ATOM_RADIUS_PX = 0.35


@dataclasses.dataclass(slots=True)
class Scene:
    pos: np.ndarray             # (N, 3)
    radius: np.ndarray          # (N,)
    color: np.ndarray           # (N, 3)
    bond_a: np.ndarray          # (M, 3)
    bond_b: np.ndarray          # (M, 3)
    bond_ca: np.ndarray         # (M, 3)
    bond_cb: np.ndarray         # (M, 3)
    edges: np.ndarray           # (E, 2, 3)
    center: np.ndarray          # (3,)
    extent: float               # halfdiagonal, >= MIN_EXTENT
    n_atoms: int
    has_cell: bool
    can_supercell: bool
    bonds_capped: bool

    @property
    def n_bonds(self) -> int:
        return int(self.bond_a.shape[0])


def empty_scene() -> Scene:
    return Scene(
        pos=np.zeros((0, 3)),
        radius=np.zeros(0),
        color=np.zeros((0, 3)),
        bond_a=np.zeros((0, 3)),
        bond_b=np.zeros((0, 3)),
        bond_ca=np.zeros((0, 3)),
        bond_cb=np.zeros((0, 3)),
        edges=np.zeros((0, 2, 3)),
        center=np.zeros(3),
        extent=MIN_EXTENT,
        n_atoms=0,
        has_cell=False,
        can_supercell=False,
        bonds_capped=False,
    )


def rot_matrix(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return rx @ ry


def shade(base: np.ndarray, nx, ny, nz) -> np.ndarray:
    diff = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    spec = np.clip(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0, 1) ** 28
    return base * (0.17 + 0.80 * diff)[..., None] + (72.0 * spec)[..., None]


def render_frame(
    scene: Scene,
    width: int,
    height: int,
    *,
    yaw: float,
    pitch: float,
    zoom: float = 1.0,
    show_bonds: bool = True,
    show_cell: bool = True,
) -> np.ndarray:
    w, h = max(int(width), 0), max(int(height), 0)
    if w == 0 or h == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    fb = np.repeat(BG[None, :], w * h, 0).reshape(h, w, 3)
    zb = np.full((h, w), -1e18)
    if scene.n_atoms == 0:
        return np.clip(fb, 0, 255).astype(np.uint8)

    rot = rot_matrix(yaw, pitch)
    scale = min(w, h) / (2.0 * max(scene.extent, MIN_EXTENT)) * zoom

    def project(points: np.ndarray):
        v = (points - scene.center) @ rot.T
        return w * 0.5 + v[..., 0] * scale, h * 0.5 - v[..., 1] * scale, v[..., 2] * scale

    sx, sy, sz = project(scene.pos)
    rpx = scene.radius * scale

    if show_bonds and scene.n_bonds:
        ax, ay, az = project(scene.bond_a)
        bx, by, bz = project(scene.bond_b)
        rb = max(0.9, 0.13 * scale)
        for k in range(scene.n_bonds):
            _segment(fb, zb, w, h, ax[k], ay[k], az[k], bx[k], by[k], bz[k],
                     rb, scene.bond_ca[k], scene.bond_cb[k], lit=True)

    for a in np.argsort(sz):
        r = rpx[a]
        if r < MIN_ATOM_RADIUS_PX:
            continue
        x0 = max(0, int(sx[a] - r) - 1)
        x1 = min(w, int(sx[a] + r) + 2)
        y0 = max(0, int(sy[a] - r) - 1)
        y1 = min(h, int(sy[a] + r) + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx = xx - sx[a]
        dy = yy - sy[a]
        d2 = dx * dx + dy * dy
        inside = d2 <= r * r
        if not inside.any():
            continue
        dz = np.sqrt(np.maximum(r * r - d2, 0.0))
        z = sz[a] + dz
        win = inside & (z > zb[y0:y1, x0:x1])
        if not win.any():
            continue
        col = shade(scene.color[a], dx / r, -dy / r, dz / r)
        fb[y0:y1, x0:x1][win] = col[win]
        zb[y0:y1, x0:x1][win] = z[win]

    if show_cell and len(scene.edges):
        ex, ey, ez = project(scene.edges)
        for k in range(len(scene.edges)):
            _segment(fb, zb, w, h, ex[k, 0], ey[k, 0], ez[k, 0],
                     ex[k, 1], ey[k, 1], ez[k, 1], 0.75,
                     CELL_COLOR, CELL_COLOR, lit=False)

    hit = zb > -1e17
    if hit.any():
        zv = zb[hit]
        lo, hi = float(zv.min()), float(zv.max())
        if hi - lo > 1e-6:
            t = (zb - lo) / (hi - lo)
            fog = np.clip(0.52 + 0.48 * t, 0, 1)[..., None]
            fb = np.where(hit[..., None], fb * fog, fb)

    return np.clip(fb, 0, 255).astype(np.uint8)


def _segment(fb, zb, w, h, x0, y0, z0, x1, y1, z1, rad, c0, c1, lit=True) -> None:
    lo_x = max(0, int(min(x0, x1) - rad) - 1)
    hi_x = min(w, int(max(x0, x1) + rad) + 2)
    lo_y = max(0, int(min(y0, y1) - rad) - 1)
    hi_y = min(h, int(max(y0, y1) + rad) + 2)
    if lo_x >= hi_x or lo_y >= hi_y:
        return
    dxs, dys = x1 - x0, y1 - y0
    len2 = dxs * dxs + dys * dys
    if len2 < 1e-9:
        return
    yy, xx = np.mgrid[lo_y:hi_y, lo_x:hi_x]
    t = np.clip(((xx - x0) * dxs + (yy - y0) * dys) / len2, 0.0, 1.0)
    px = xx - (x0 + t * dxs)
    py = yy - (y0 + t * dys)
    d2 = px * px + py * py
    inside = d2 <= rad * rad
    if not inside.any():
        return
    bulge = np.sqrt(np.maximum(rad * rad - d2, 0.0))
    z = z0 + t * (z1 - z0) + bulge
    win = inside & (z > zb[lo_y:hi_y, lo_x:hi_x])
    if not win.any():
        return
    base = np.where(t[..., None] < 0.5, c0, c1)
    if lit:
        col = shade(base, px / rad, -py / rad, bulge / rad)
    else:
        col = base * (0.55 + 0.45 * (bulge / rad))[..., None]
    fb[lo_y:hi_y, lo_x:hi_x][win] = col[win]
    zb[lo_y:hi_y, lo_x:hi_x][win] = z[win]
