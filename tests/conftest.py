"""Pytest configuration and fixtures."""

import os

import spglib

spglib.OLD_ERROR_HANDLING = False

os.environ["RAPMAT_SPGLIB_ISOLATE"] = "0"

import sys

import torch

if sys.platform == "win32":
    sys.modules.pop("urwid.display.curses", None)

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

import json

from rapmat.storage import SQLiteStore
from rapmat.storage.base import StructureStore


def add_generated_candidate(
    store: StructureStore,
    run_name: str,
    struct_id: str,
    atoms: Atoms,
    spg: int = 1,
    fu: int = 1,
) -> str:
    """Insert a single 'generated' candidate via the placeholder pipeline."""
    store.add_generation_placeholders(run_name, [(struct_id, spg, fu)])
    store.update_generated_structure(struct_id, atoms)
    return struct_id


def add_relaxed_structure(
    store: StructureStore,
    run_name: str,
    atoms: Atoms,
    energy_per_atom: float,
    struct_id: str,
    converged: bool = True,
) -> None:
    """Add a structure and immediately mark it relaxed with the given energy."""
    add_generated_candidate(store, run_name, struct_id, atoms)

    store.update_structure(
        struct_id,
        "relaxed",
        atoms=atoms,
        metadata={
            "energy_per_atom": energy_per_atom,
            "fmax": 0.01,
            "converged": converged,
        },
    )


# ------------------------------------------------------------------ #
#  Common atom fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def cu_fcc():
    return bulk("Cu", "fcc", a=3.615)


@pytest.fixture
def cu_bcc():
    return bulk("Cu", "bcc", a=2.87)


@pytest.fixture
def al_fcc():
    return bulk("Al", "fcc", a=4.05)


# ------------------------------------------------------------------ #
#  Store factory fixture (dual-backend)
# ------------------------------------------------------------------ #


def _make_sqlite(path):
    return SQLiteStore.from_path(path)


_BACKENDS = {"sqlite": _make_sqlite}


@pytest.fixture(params=list(_BACKENDS))
def store_factory(request, tmp_path):
    """Dual-backend factory."""
    maker = _BACKENDS[request.param]
    created: list[StructureStore] = []

    def make(name: str = "db") -> StructureStore:
        store = maker(tmp_path / name)
        created.append(store)
        return store

    yield make
    for s in created:
        try:
            s.close()
        except Exception:
            pass


@pytest.fixture
def store(store_factory):
    """A single store, auto-closed."""
    return store_factory()


# ------------------------------------------------------------------ #
#  Raw access helpers
# ------------------------------------------------------------------ #


def read_batch_config(store: StructureStore, run_name: str) -> dict:
    """The run's raw stored batch config, not merged with the study."""
    rows = store._read(
        "SELECT batch_config_json FROM run WHERE name = ?", (run_name,)
    )
    return json.loads(rows[0]["batch_config_json"])


def phonon_params_blob(store: StructureStore, structure_id: str):
    rows = store._read(
        "SELECT params_gz FROM phonon_params WHERE structure = ?", (structure_id,)
    )
    return rows[0]["params_gz"] if rows else None


def force_heartbeat(store: StructureStore, run_name: str, ts: str) -> None:
    """Force a run's heartbeat to an arbitrary timestamp."""
    store._write("UPDATE run SET heartbeat = ? WHERE name = ?", (ts, run_name))


def read_run_field(store: StructureStore, run_name: str, *fields: str) -> dict:
    cols = ", ".join(fields)
    rows = store._read(f"SELECT {cols} FROM run WHERE name = ?", (run_name,))
    return rows[0]


# ------------------------------------------------------------------ #
#  Pre-populated hull store
# ------------------------------------------------------------------ #


@pytest.fixture(params=list(_BACKENDS))
def hull_store(request, tmp_path):
    """Store pre-populated with a synthetic Al-Cu binary system.

    Reference energies (eV/A):
        Al = -3.0,  Cu = -4.0

    Intermediate structures:
        AlCu   (epa=-5.0, 2 atoms)  -> formation_energy = -1.5  (ON hull)
        Al3Cu  (epa=-3.3, 4 atoms)  -> formation_energy = -0.05 (ABOVE hull)
    """
    store = _BACKENDS[request.param](tmp_path / "hull_db")

    store.create_study("test-study", system="Al-Cu", domain="bulk", calculator="mock")

    # Pure-Al endpoint
    store.create_run(
        "al-run",
        config={"formula": {"Al": 1}, "calculator": "mock"},
        study_id="test-study",
    )
    add_relaxed_structure(store, "al-run", bulk("Al", "fcc", a=4.05), -3.0, "al-run/1")

    # Pure-Cu endpoint
    store.create_run(
        "cu-run",
        config={"formula": {"Cu": 1}, "calculator": "mock"},
        study_id="test-study",
    )
    add_relaxed_structure(store, "cu-run", bulk("Cu", "fcc", a=3.615), -4.0, "cu-run/1")

    # AlCu intermediate    
    # AlCu ON hull
    store.create_run(
        "alcu-on",
        config={"formula": {"Al": 1, "Cu": 1}, "calculator": "mock"},
        study_id="test-study",
    )
    alcu = Atoms(
        symbols=["Al", "Cu"],
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
        cell=[3, 3, 3],
        pbc=True,
    )
    add_relaxed_structure(store, "alcu-on", alcu, -5.0, "alcu-on/1")

    # Al3Cu intermediate (above hull)
    # epa = -3.0 gives formation_energy = (-12.0 - (-13.0)) / 4 = +0.25 eV/A    
    # Al3Cu ABOVE hull
    store.create_run(
        "al3cu-off",
        config={"formula": {"Al": 3, "Cu": 1}, "calculator": "mock"},
        study_id="test-study",
    )
    al3cu = Atoms(
        symbols=["Al", "Al", "Al", "Cu"],
        positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]],
        cell=[3, 3, 3],
        pbc=True,
    )
    add_relaxed_structure(store, "al3cu-off", al3cu, -3.0, "al3cu-off/1")

    yield store
    try:
        store.close()
    except Exception:
        pass


# ------------------------------------------------------------------ #
#  CLI testing helpers
# ------------------------------------------------------------------ #


@contextmanager
def mock_pyxtal_generation(atoms_list=None):
    """Mock PyXtal generation for resumable-generation testing.
    """
    from unittest.mock import MagicMock

    if atoms_list is None:
        atoms_list = [bulk("Cu", "fcc", a=3.615)]

    _idx = [0]

    class _MockPyxtal:
        def __init__(self):
            self.valid = True

        def from_random(self, **kwargs):
            pass

        def to_ase(self):
            atoms = atoms_list[_idx[0] % len(atoms_list)]
            _idx[0] += 1
            return atoms

    class _CompCompatError(Exception):
        pass

    mock_msg = MagicMock()
    mock_msg.Comp_CompatibilityError = _CompCompatError

    mock_module = MagicMock()
    mock_module.pyxtal = _MockPyxtal
    mock_module.msg = mock_msg

    with patch.dict("sys.modules", {"pyxtal": mock_module, "pyxtal.msg": mock_msg}):
        yield


@contextmanager
def mock_calculator_factory():
    """Mock calculator factory to return EMT instead of loading real calculators."""
    from rapmat.calculators.factory import load_calculator

    def mock_load(calculator_enum, config=None):
        return EMT()

    with patch("rapmat.calculators.factory.load_calculator", side_effect=mock_load):
        yield
