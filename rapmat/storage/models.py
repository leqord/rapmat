"""SQLAlchemy 2.0 declarative models.
"""

from __future__ import annotations

from typing import Optional

from ase import Atoms
from ase.units import GPa
from sqlalchemy import Float, ForeignKey, Index, Integer, MetaData, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from rapmat.storage.types import AtomsJSON, IntBool, JSONDict, OptIntBool
from rapmat.utils.structure import calculate_thickness, format_spg

_NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=_NAMING)


class Study(Base):
    """A search space definition."""

    __tablename__ = "study"

    study_id: Mapped[str] = mapped_column("id", Text, primary_key=True)
    system: Mapped[str] = mapped_column(Text, nullable=True)
    domain: Mapped[str] = mapped_column(Text, nullable=True)
    calculator: Mapped[str] = mapped_column(Text, nullable=True)
    config: Mapped[dict] = mapped_column("config_json", JSONDict, default=dict, nullable=True)
    timestamp: Mapped[str] = mapped_column(Text, default="", nullable=True)

    def __init__(self, **kw):
        kw.setdefault("config", {})
        kw.setdefault("timestamp", "")
        super().__init__(**kw)

    @property
    def search_config(self):
        from rapmat.core.config import SearchConfig

        return SearchConfig.from_stored(
            self.config,
            {},
            system=self.system,
            domain=self.domain,
            calculator=self.calculator,
        )

    def to_dict(self) -> dict:
        """Plain dict representation."""
        return {
            "study_id": self.study_id,
            "system": self.system,
            "domain": self.domain,
            "calculator": self.calculator,
            "config": self.config,
            "timestamp": self.timestamp,
        }


class Run(Base):
    """A run row."""

    __tablename__ = "run"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    batch_config: Mapped[dict] = mapped_column(
        "batch_config_json", JSONDict, default=dict, nullable=True
    )
    timestamp: Mapped[Optional[str]] = mapped_column(Text)
    study_id: Mapped[Optional[str]] = mapped_column(
        "study", ForeignKey("study.id", ondelete="CASCADE")
    )
    run_status: Mapped[Optional[str]] = mapped_column(Text)
    worker_id: Mapped[Optional[str]] = mapped_column(Text)
    heartbeat: Mapped[Optional[str]] = mapped_column(Text)


class Structure(Base):
    """A stored structure."""

    __tablename__ = "structure"
    __table_args__ = (
        Index("idx_struct_run", "run"),
        Index("idx_struct_status", "status"),
        Index("idx_struct_run_status", "run", "status"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    run: Mapped[Optional[str]] = mapped_column(
        ForeignKey("run.name", ondelete="CASCADE")
    )
    status: Mapped[str] = mapped_column(Text, nullable=True)
    gen_spg: Mapped[Optional[int]] = mapped_column(Integer)
    gen_fu: Mapped[Optional[int]] = mapped_column(Integer)
    energy_per_atom: Mapped[float] = mapped_column(Float, nullable=True)
    fmax: Mapped[float] = mapped_column(Float, nullable=True)
    converged: Mapped[bool] = mapped_column(IntBool, nullable=True)
    duplicate: Mapped[Optional[bool]] = mapped_column(OptIntBool)
    excluded: Mapped[bool] = mapped_column(IntBool, nullable=True)
    initial_atoms: Mapped[Optional[Atoms]] = mapped_column(
        "initial_atoms_json", AtomsJSON
    )
    final_atoms: Mapped[Optional[Atoms]] = mapped_column(
        "final_atoms_json", AtomsJSON
    )

    descriptor = None
    min_phonon_freq = None
    pressure_gpa = 0.0
    domain = "bulk"
    symprec = 1e-3

    _TRANSIENT = ("descriptor", "min_phonon_freq", "pressure_gpa", "domain", "symprec")
    _PY_DEFAULTS = {
        "status": "",
        "energy_per_atom": 0.0,
        "fmax": 0.0,
        "converged": False,
        "excluded": False,
    }

    def __init__(self, **kw):
        transient = {k: kw.pop(k) for k in list(kw) if k in self._TRANSIENT}
        super().__init__(**{**self._PY_DEFAULTS, **kw})
        for k, v in transient.items():
            setattr(self, k, v)

    # Computed properties

    @property
    def atoms(self) -> Atoms | None:
        return self.final_atoms if self.final_atoms is not None else self.initial_atoms

    @property
    def n_atoms(self) -> int:
        atoms = self.atoms
        return len(atoms) if atoms is not None else 0

    @property
    def formula(self) -> str:
        atoms = self.atoms
        return atoms.get_chemical_formula() if atoms is not None else ""

    @property
    def volume(self) -> float | None:
        atoms = self.atoms
        return atoms.get_volume() if atoms is not None else None

    @property
    def energy_total(self) -> float:
        n = self.n_atoms
        return self.energy_per_atom * n if n else 0.0

    @property
    def enthalpy_per_atom(self) -> float | None:
        volume = self.volume
        n = self.n_atoms
        if self.pressure_gpa > 0 and volume is not None and n:
            return self.energy_per_atom + self.pressure_gpa * GPa * volume / n
        return None

    @property
    def thickness(self) -> float | None:
        atoms = self.atoms
        if self.domain != "bulk" and atoms is not None:
            return calculate_thickness(atoms)
        return None

    @property
    def forces(self):
        """Per-atom forces, smuggled through ``atoms.info`` by the CSP loop."""
        atoms = self.atoms
        return atoms.info.get("forces") if atoms is not None else None

    @property
    def initial_spg(self) -> str:
        return format_spg(self.initial_atoms, symprec=self.symprec)

    @property
    def final_spg(self) -> str:
        return format_spg(self.final_atoms, symprec=self.symprec)


class Evaluation(Base):
    """Evaluation of a structure."""

    __tablename__ = "evaluation"
    __table_args__ = (
        Index("idx_eval_run", "run"),
        Index("idx_eval_struct", "structure"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    structure_id: Mapped[str] = mapped_column(
        "structure", ForeignKey("structure.id", ondelete="CASCADE"), nullable=True
    )
    run: Mapped[Optional[str]] = mapped_column(
        ForeignKey("run.name", ondelete="CASCADE")
    )
    calculator: Mapped[str] = mapped_column(Text, nullable=True)

    config_json: Mapped[str] = mapped_column(Text, nullable=True)
    energy_per_atom: Mapped[float] = mapped_column(Float, nullable=True)
    energy_total: Mapped[float] = mapped_column(Float, nullable=True)
    min_phonon_freq: Mapped[Optional[float]] = mapped_column(Float)


class Phonon(Base):
    """A structure's related phonon row."""

    __tablename__ = "phonon"
    __table_args__ = (Index("idx_phonon_run", "run"),)

    structure_id: Mapped[str] = mapped_column(
        "structure",
        ForeignKey("structure.id", ondelete="CASCADE"),
        primary_key=True,
    )
    run: Mapped[Optional[str]] = mapped_column(
        ForeignKey("run.name", ondelete="CASCADE")
    )
    min_phonon_freq: Mapped[Optional[float]] = mapped_column(Float)
    supercell: Mapped[Optional[str]] = mapped_column(Text)
    mesh: Mapped[Optional[str]] = mapped_column(Text)
    displacement: Mapped[Optional[float]] = mapped_column(Float)
    symprec: Mapped[Optional[float]] = mapped_column(Float)
    calculator: Mapped[Optional[str]] = mapped_column(Text)


class PhononParams(Base):
    """gzip-b64 phonopy blob"""

    __tablename__ = "phonon_params"
    __table_args__ = (Index("idx_phonon_params_run", "run"),)

    structure_id: Mapped[str] = mapped_column(
        "structure",
        ForeignKey("structure.id", ondelete="CASCADE"),
        primary_key=True,
    )
    run: Mapped[Optional[str]] = mapped_column(
        ForeignKey("run.name", ondelete="CASCADE")
    )
    params_gz: Mapped[Optional[str]] = mapped_column(Text)
