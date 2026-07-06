"""Typed domain entities.
"""

from __future__ import annotations

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field

from rapmat.core.config import SearchConfig
from rapmat.storage.models import Evaluation, Structure, Study

__all__ = [
    "Candidate",
    "Evaluation",
    "PhononResult",
    "ResultRow",
    "RunMetadata",
    "Structure",
    "Study",
]


class Candidate(BaseModel):
    """A generation placeholder or an unrelaxed candidate."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    atoms: Atoms | None = None
    gen_spg: int | None = None
    gen_fu: int | None = None


class RunMetadata(BaseModel):
    """A run's metadata: run config + study config."""

    name: str
    domain: str = ""
    config: dict = Field(default_factory=dict)
    timestamp: str = ""
    study_id: str | None = None
    run_status: str | None = None
    worker_id: str | None = None

    @property
    def search_config(self) -> SearchConfig:
        return SearchConfig.model_validate(self.config)


class PhononResult(BaseModel):
    """A stored phonon result for one structure."""

    min_phonon_freq: float | None = None
    params_gz: str = ""
    supercell: str | None = None
    mesh: str | None = None
    displacement: float | None = None
    symprec: float | None = None
    calculator: str | None = None


class ResultRow(BaseModel):
    """A row in a results/phase-analysis table.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    structure: Structure
    index: int = 0
    run_name: str = ""
    dynamical_stability: bool | None = None

    effective_per_atom: float | None = None
    formation_energy: float | None = None
    energy_above_hull: float | None = None
    is_stable: bool | None = None
    is_reference: bool = False
    composition_frac: float | None = None
    reduced_formula: str | None = None

    ref_energy_per_atom: float | None = None
    ref_phonon_freq: float | None = None

    @property
    def structure_id(self) -> str:
        return self.structure.id

    @property
    def formula(self) -> str:
        return self.structure.formula

    @property
    def energy_per_atom(self) -> float:
        return self.structure.energy_per_atom

    @property
    def enthalpy_per_atom(self) -> float | None:
        return self.structure.enthalpy_per_atom

    @property
    def volume(self) -> float | None:
        return self.structure.volume

    @property
    def fmax(self) -> float:
        return self.structure.fmax

    @property
    def converged(self) -> bool:
        return self.structure.converged

    @property
    def thickness(self) -> float | None:
        return self.structure.thickness

    @property
    def duplicate(self) -> bool | None:
        return self.structure.duplicate

    @property
    def excluded(self) -> bool:
        return self.structure.excluded

    @property
    def min_phonon_freq(self) -> float | None:
        return self.structure.min_phonon_freq

    @property
    def initial_spg(self) -> str:
        return self.structure.initial_spg

    @property
    def final_spg(self) -> str:
        return self.structure.final_spg

    @property
    def atoms(self) -> Atoms | None:
        return self.structure.atoms

    @property
    def initial_atoms(self) -> Atoms | None:
        return self.structure.initial_atoms

    @property
    def display_epa(self) -> float:
        """Energy shown in the table: energy effective value (E/A or H/A), else raw E/atom."""
        # NOTE: misleading naming, change later
        if self.effective_per_atom is not None:
            return self.effective_per_atom
        return self.structure.energy_per_atom

    def search_text(self) -> str:
        parts = [
            self.structure_id, self.formula, self.final_spg,
            self.initial_spg, self.run_name,
        ]
        return " ".join(str(p).lower() for p in parts if p)
