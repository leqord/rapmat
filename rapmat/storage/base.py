from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

from ase import Atoms

from rapmat.core.entities import (
    Candidate,
    Evaluation,
    PhononResult,
    RunMetadata,
    Structure,
    Study,
)
from rapmat.storage.status import StructureStatus

# ------------------------------------------------------------------ #
#  Store ABC
# ------------------------------------------------------------------ #


class StructureStore(ABC):
    @abstractmethod
    def create_run(
        self,
        name: str,
        study_id: str,
        config: Optional[dict] = None,
        worker_id: Optional[str] = None,
    ) -> str: ...

    @abstractmethod
    def get_run_metadata(self, name: str) -> Optional["RunMetadata"]: ...

    @abstractmethod
    def update_run_config(self, name: str, config: dict) -> None: ...

    @abstractmethod
    def set_run_config_value(self, run_name: str, key: str, value) -> None: ...

    @abstractmethod
    def set_study_config_value(self, study_id: str, key: str, value) -> None: ...

    @abstractmethod
    def delete_run(self, run_name: str) -> None: ...

    @abstractmethod
    def list_runs(self) -> List["RunMetadata"]: ...

    @abstractmethod
    def count_by_status(self, run_name: str) -> dict[str, int]: ...

    @abstractmethod
    def claim_run(self, run_name: str, worker_id: str) -> bool: ...

    @abstractmethod
    def release_run(self, run_name: str, final_status: str) -> None: ...

    @abstractmethod
    def update_heartbeat(self, run_name: str, worker_id: str) -> None:
        """Does nothing unless ``worker_id`` currently owns the run."""

    @abstractmethod
    def set_run_status(self, run_name: str, status: str) -> None: ...

    @abstractmethod
    def reclaim_stale_runs(self, timeout_minutes: int = 10) -> list[str]: ...

    @abstractmethod
    def update_structure(
        self,
        struct_id: str,
        status: str,
        atoms: Optional[Atoms] = None,
        metadata: Optional[dict] = None,
    ) -> None: ...

    @abstractmethod
    def clear_run_phonon_results(self, run_name: str) -> None: ...

    @abstractmethod
    def clear_run_duplicates(self, run_name: str) -> None: ...

    @abstractmethod
    def save_phonon_result(
        self,
        structure_id: str,
        run_name: str,
        min_phonon_freq: Optional[float],
        params_gz: Optional[str] = None,
        settings: Optional[dict] = None,
    ) -> None: ...

    @abstractmethod
    def get_phonon_result(self, structure_id: str) -> Optional["PhononResult"]: ...

    @abstractmethod
    def mark_duplicates(
        self,
        dropped_ids: list[str],
        kept_ids: list[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None: ...

    @abstractmethod
    def set_structure_excluded(self, structure_id: str, excluded: bool) -> None: ...

    @abstractmethod
    def get_structures(
        self,
        run_name: str,
        *,
        status: Optional[str] = None,
        statuses: Optional[tuple[str, ...]] = None,
        symprec: float = 1e-3,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> List[Structure]: ...

    @abstractmethod
    def get_structures_for_analysis(
        self,
        run_id: str,
        statuses: tuple = (StructureStatus.RELAXED,),
    ) -> List[Structure]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def create_study(
        self,
        study_id: str,
        system: str,
        domain: str,
        calculator: str,
        config: Optional[dict] = None,
    ) -> str: ...

    @abstractmethod
    def get_study(self, study_id: str) -> Optional["Study"]: ...

    @abstractmethod
    def update_study(self, study_id: str, fields: dict) -> None: ...

    @abstractmethod
    def delete_study(self, study_id: str) -> None: ...

    @abstractmethod
    def list_studies(self) -> List["Study"]: ...

    @abstractmethod
    def get_study_runs(self, study_id: str) -> List["RunMetadata"]: ...

    @abstractmethod
    def add_evaluation(
        self,
        structure_id: str,
        run_name: str,
        calculator: str,
        config_json: str,
        energy_per_atom: float,
        energy_total: float,
        min_phonon_freq: Optional[float] = None,
    ) -> str: ...

    @abstractmethod
    def clear_evaluations(
        self, run_name: str, calculator: Optional[str] = None
    ) -> None: ...

    @abstractmethod
    def has_evaluation(
        self, structure_id: str, calculator: str, config_json: str
    ) -> bool: ...

    @abstractmethod
    def get_evaluations(
        self, run_name: str, calculator: Optional[str] = None
    ) -> List["Evaluation"]: ...

    @abstractmethod
    def close(self) -> None: ...

    def vacuum(self) -> None:
        """Reclaim free space if the backend supports it.
        """
        return None

    @abstractmethod
    def add_generation_placeholders(
        self,
        run_name: str,
        placeholders: List[Tuple[str, int, int]],
    ) -> int: ...

    @abstractmethod
    def get_pending_generation(self, run_name: str) -> List[Candidate]: ...

    @abstractmethod
    def get_unrelaxed_candidates(self, run_name: str) -> List[Candidate]: ...

    @abstractmethod
    def update_generated_structure(
        self,
        struct_id: str,
        atoms: Atoms,
    ) -> None: ...

    @abstractmethod
    def discard_generation_placeholder(self, struct_id: str) -> None: ...

    @abstractmethod
    def get_url(self) -> Optional[str]: ...
