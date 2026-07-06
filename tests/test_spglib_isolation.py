"""The spglib isolation worker must keep an spglib crash from killing the app.
"""

import pytest
from ase import Atoms
from ase.build import bulk

CRASH_CELL = [
    [5.060992592611191, -2.91858316609063, -1.5009230925800822e-07],
    [-0.002850400826870777, 5.84214487932445, -1.0248449050582182e-07],
    [-6.631926870001668e-07, -1.74270983099958e-07, 10.187112283575818],
]
CRASH_SCALED = [
    [0.0006151380022971649, 0.49969315135289377, 0.7499996627562331],
    [0.4996300561017988, 0.50044957184241, 0.2499999877090111],
    [0.9993883032604252, 0.49995839111009877, 0.2500004733773378],
    [0.49917202231385155, 0.00021336071283227287, 0.2500000852858078],
    [0.5004397683128793, 0.999781964602942, 0.7499998368828699],
    [0.5007502127575957, 0.4999231442007631, 0.7499999406796289],
    [0.6672783244234669, 0.33311026126679577, 0.7499991567941263],
    [0.3327175685950119, 0.6668724498015284, 0.24999985877111522],
    [0.6660618707989279, 0.3335422910349363, 0.25000050595134504],
    [0.33394673543439235, 0.6664554140732817, 0.7500004917928743],
]
CRASH_NUMBERS = [8, 8, 8, 8, 8, 8, 13, 13, 13, 13]


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setenv("RAPMAT_SPGLIB_ISOLATE", "1")
    import rapmat.utils.spg_isolated as si

    with si._worker_lock:
        if si._worker is not None:
            si._worker._kill()
        si._worker = None
    yield si
    with si._worker_lock:
        if si._worker is not None:
            si._worker._kill()
            si._worker = None


def test_isolation_normal_structure(isolated):
    """A normal structure resolves normally."""
    from rapmat.utils.structure import format_spg

    s = format_spg(bulk("Si", "diamond", a=5.43), symprec=1e-3)
    assert isinstance(s, str) and s and s != "N/A"


def test_isolation_survives_crashing_cell(isolated):
    """The cell and symprec combination that crashes spglib."""
    from rapmat.utils.structure import format_spg

    atoms = Atoms(
        numbers=CRASH_NUMBERS, scaled_positions=CRASH_SCALED, cell=CRASH_CELL, pbc=True
    )
    s = format_spg(atoms, symprec=1e-2)
    assert isinstance(s, str) and s

    s2 = format_spg(bulk("Si", "diamond", a=5.43), symprec=1e-3)
    assert isinstance(s2, str) and s2 != "N/A"
