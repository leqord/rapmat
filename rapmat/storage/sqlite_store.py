"""SQLAlchemy-based StructureStore.
"""

import hashlib
import json
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

from ase import Atoms
from filelock import FileLock, Timeout
from sqlalchemy import delete, func, insert, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from rapmat.core.config import SearchConfig, merge_config_dicts
from rapmat.core.entities import Candidate, PhononResult, RunMetadata
from rapmat.storage.base import StructureStore
from rapmat.storage.engine import make_engine, run_migrations
from rapmat.storage.models import (
    Evaluation,
    Phonon,
    PhononParams,
    Run,
    Structure,
    Study,
)
from rapmat.storage.status import RunStatus, StructureStatus
from rapmat.utils.console import get_logger


class SQLiteStore(StructureStore):
    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        db_file: str | Path,
        *,
        reclaim_stale_minutes: int | None = 10,
    ):
        self._db_file = str(db_file)
        self._lock = threading.RLock()
        self._engine = make_engine(self._db_file)
        run_migrations(self._engine)
        self._Session = sessionmaker(self._engine, expire_on_commit=False)

        if reclaim_stale_minutes is not None:
            self.reclaim_stale_runs(timeout_minutes=reclaim_stale_minutes)

    @classmethod
    def from_path(cls, db_path: Path, **kwargs) -> "SQLiteStore":
        db_path.mkdir(parents=True, exist_ok=True)

        lock_path = db_path / "rapmat.lock"
        file_lock = FileLock(lock_path, timeout=2.0)
        try:
            file_lock.acquire()
        except Timeout:
            raise RuntimeError(
                f"Another rapmat instance is using the database at {db_path}. "
                "Close it first."
            )

        instance = cls(db_path / "rapmat.sqlite", **kwargs)
        instance._file_lock = file_lock
        return instance

    def get_url(self) -> Optional[str]:
        return self._db_file

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with self._lock:
            with self._Session() as s:
                yield s

    # ------------------------------------------------------------------ #
    #  Run management
    # ------------------------------------------------------------------ #

    def create_run(
        self,
        name: str,
        study_id: str,
        config: Optional[dict] = None,
        worker_id: Optional[str] = None,
    ) -> str:
        if self.get_run_metadata(name) is not None:
            raise ValueError(
                f"Run '{name}' already exists. Use a different name or resume it."
            )

        study = self.get_study(study_id)
        if study is None:
            raise ValueError(f"Study '{study_id}' not found.")

        batch_cfg = config or {}
        run_elements = set(batch_cfg.get("formula", {}).keys())
        study_elements = set(study.system.split("-"))
        if not run_elements <= study_elements:
            extra = run_elements - study_elements
            raise ValueError(f"Elements {extra} not in study '{study.system}'.")

        now = datetime.now().isoformat()
        with self._session() as s:
            s.add(
                Run(
                    name=name,
                    batch_config=batch_cfg,
                    timestamp=now,
                    study_id=study_id,
                    run_status=str(RunStatus.GENERATING),
                    worker_id=worker_id,
                    heartbeat=now if worker_id else None,
                )
            )
            s.commit()
        return name

    def _run_to_metadata(self, run: Run, *, merged: bool) -> RunMetadata:
        batch_cfg = run.batch_config
        if merged:
            study_cfg: dict = {}
            system = domain = calculator = ""
            if run.study_id:
                study = self.get_study(run.study_id)
                if study:
                    study_cfg = study.config
                    domain = study.domain
                    system = study.system
                    calculator = study.calculator
            config = merge_config_dicts(
                study_cfg,
                batch_cfg,
                system=system,
                domain=domain,
                calculator=calculator,
            )
        else:
            domain = ""
            config = batch_cfg

        return RunMetadata(
            name=run.name,
            domain=domain,
            config=config,
            timestamp=run.timestamp,
            study_id=run.study_id or None,
            run_status=run.run_status,
            worker_id=run.worker_id,
        )

    def get_run_metadata(self, name: str) -> Optional[RunMetadata]:
        with self._session() as s:
            run = s.get(Run, name)
        if run is None:
            return None
        return self._run_to_metadata(run, merged=True)

    def update_run_config(self, name: str, config: dict) -> None:
        with self._session() as s:
            s.execute(
                update(Run).where(Run.name == name).values(batch_config=config)
            )
            s.commit()

    def set_run_config_value(self, run_name: str, key: str, value) -> None:
        with self._session() as s:
            run = s.get(Run, run_name)
            cfg = dict(run.batch_config) if run else {}
        cfg[key] = value
        self.update_run_config(run_name, cfg)

    def set_study_config_value(self, study_id: str, key: str, value) -> None:
        study = self.get_study(study_id)
        if study is None:
            return
        cfg = dict(study.config)
        cfg[key] = value
        self.update_study(study_id, {"config": cfg})

    def delete_run(self, run_name: str) -> None:
        with self._session() as s:
            s.execute(delete(Run).where(Run.name == run_name))
            s.commit()

    def list_runs(self) -> List[RunMetadata]:
        with self._session() as s:
            runs = s.scalars(select(Run)).all()
        return [self._run_to_metadata(r, merged=False) for r in runs]

    def count_by_status(self, run_name: str) -> dict[str, int]:
        with self._session() as s:
            rows = s.execute(
                select(Structure.status, func.count())
                .where(Structure.run == run_name)
                .group_by(Structure.status)
            ).all()
        return {status: int(cnt) for status, cnt in rows}

    # ------------------------------------------------------------------ #
    #  Run-level locking
    # ------------------------------------------------------------------ #

    def claim_run(self, run_name: str, worker_id: str) -> bool:
        claimable = [
            str(RunStatus.PENDING),
            str(RunStatus.GENERATING),
            str(RunStatus.FAILED),
            str(RunStatus.INTERRUPTED),
        ]
        with self._session() as s:
            result = s.execute(
                update(Run)
                .where(
                    Run.name == run_name,
                    or_(Run.run_status.in_(claimable), Run.run_status.is_(None)),
                )
                .values(
                    run_status=str(RunStatus.PROCESSING),
                    worker_id=worker_id,
                    heartbeat=datetime.now().isoformat(),
                )
            )
            s.commit()
        return result.rowcount > 0

    def release_run(self, run_name: str, final_status: str) -> None:
        with self._session() as s:
            s.execute(
                update(Run)
                .where(Run.name == run_name)
                .values(run_status=str(final_status), worker_id=None, heartbeat=None)
            )
            s.commit()

    def update_heartbeat(self, run_name: str, worker_id: str) -> None:
        with self._session() as s:
            s.execute(
                update(Run)
                .where(Run.name == run_name, Run.worker_id == worker_id)
                .values(heartbeat=datetime.now().isoformat())
            )
            s.commit()

    def set_run_status(self, run_name: str, status: str) -> None:
        with self._session() as s:
            s.execute(
                update(Run).where(Run.name == run_name).values(run_status=str(status))
            )
            s.commit()

    def reclaim_stale_runs(self, timeout_minutes: int = 10) -> list[str]:
        cutoff = datetime.now().timestamp() - timeout_minutes * 60
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()
        active = [str(RunStatus.PROCESSING), str(RunStatus.GENERATING)]
        with self._session() as s:
            names = s.scalars(
                update(Run)
                .where(
                    Run.run_status.in_(active),
                    Run.heartbeat.is_not(None),
                    Run.heartbeat < cutoff_iso,
                )
                .values(
                    run_status=str(RunStatus.PENDING),
                    worker_id=None,
                    heartbeat=None,
                )
                .returning(Run.name)
            ).all()
            s.commit()
        return list(names)

    # ------------------------------------------------------------------ #
    #  Study management
    # ------------------------------------------------------------------ #

    def create_study(
        self,
        study_id: str,
        system: str,
        domain: str,
        calculator: str,
        config: Optional[dict] = None,
    ) -> str:
        if self.get_study(study_id) is not None:
            raise ValueError(f"Study '{study_id}' already exists.")
        with self._session() as s:
            s.add(
                Study(
                    study_id=study_id,
                    system=system,
                    domain=domain,
                    calculator=calculator,
                    config=config or {},
                    timestamp=datetime.now().isoformat(),
                )
            )
            s.commit()
        return study_id

    def get_study(self, study_id: str) -> Optional[Study]:
        with self._session() as s:
            return s.get(Study, study_id)

    def update_study(self, study_id: str, fields: dict) -> None:
        if "config" not in fields:
            return
        with self._session() as s:
            s.execute(
                update(Study)
                .where(Study.study_id == study_id)
                .values(config=fields["config"])
            )
            s.commit()

    def delete_study(self, study_id: str) -> None:
        # Runs, structures, evaluations and phonon rows cascade.
        with self._session() as s:
            s.execute(delete(Study).where(Study.study_id == study_id))
            s.commit()

    def list_studies(self) -> List[Study]:
        with self._session() as s:
            return list(s.scalars(select(Study)).all())

    def get_study_runs(self, study_id: str) -> List[RunMetadata]:
        with self._session() as s:
            runs = s.scalars(select(Run).where(Run.study_id == study_id)).all()
        return [self._run_to_metadata(r, merged=False) for r in runs]

    # ------------------------------------------------------------------ #
    #  Generation placeholders and candidates
    # ------------------------------------------------------------------ #

    def get_unrelaxed_candidates(self, run_name: str) -> List[Candidate]:
        with self._session() as s:
            rows = s.execute(
                select(Structure.id, Structure.initial_atoms).where(
                    Structure.run == run_name,
                    Structure.status == str(StructureStatus.GENERATED),
                )
            ).all()
        return [Candidate(id=sid, atoms=atoms) for sid, atoms in rows]

    def add_generation_placeholders(
        self,
        run_name: str,
        placeholders: List[Tuple[str, int, int]],
    ) -> int:
        if not placeholders:
            return 0
        with self._session() as s:
            s.execute(
                insert(Structure),
                [
                    {
                        "id": cid,
                        "run": run_name,
                        "status": str(StructureStatus.GENERATING),
                        "gen_spg": spg,
                        "gen_fu": fu,
                        "energy_per_atom": 0.0,
                        "fmax": 0.0,
                        "converged": False,
                        "excluded": False,
                        "initial_atoms": None,
                        "final_atoms": None,
                    }
                    for cid, spg, fu in placeholders
                ],
            )
            s.commit()
        return len(placeholders)

    def get_pending_generation(self, run_name: str) -> List[Candidate]:
        with self._session() as s:
            rows = s.execute(
                select(Structure.id, Structure.gen_spg, Structure.gen_fu).where(
                    Structure.run == run_name,
                    Structure.status == str(StructureStatus.GENERATING),
                )
            ).all()
        return [
            Candidate(id=sid, gen_spg=spg, gen_fu=fu) for sid, spg, fu in rows
        ]

    def update_generated_structure(self, struct_id: str, atoms: Atoms) -> None:
        with self._session() as s:
            s.execute(
                update(Structure)
                .where(Structure.id == struct_id)
                .values(
                    status=str(StructureStatus.GENERATED),
                    initial_atoms=atoms,
                )
            )
            s.commit()

    def discard_generation_placeholder(self, struct_id: str) -> None:
        with self._session() as s:
            s.execute(
                update(Structure)
                .where(Structure.id == struct_id)
                .values(status=str(StructureStatus.DISCARDED))
            )
            s.commit()

    # ------------------------------------------------------------------ #
    #  Structure updates
    # ------------------------------------------------------------------ #

    def update_structure(
        self,
        struct_id: str,
        status: str,
        atoms: Optional[Atoms] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        meta = metadata or {}
        values: dict = {
            "status": str(status),
            "energy_per_atom": float(meta.get("energy_per_atom", 0.0)),
            "fmax": float(meta.get("fmax", 0.0)),
            "converged": bool(meta.get("converged", False)),
        }
        if atoms is not None:
            values["final_atoms"] = atoms
        with self._session() as s:
            s.execute(
                update(Structure).where(Structure.id == struct_id).values(**values)
            )
            s.commit()

    def clear_run_phonon_results(self, run_name: str) -> None:
        with self._session() as s:
            s.execute(delete(Phonon).where(Phonon.run == run_name))
            s.execute(delete(PhononParams).where(PhononParams.run == run_name))
            s.commit()

    def clear_run_duplicates(self, run_name: str) -> None:
        with self._session() as s:
            s.execute(
                update(Structure)
                .where(Structure.run == run_name)
                .values(duplicate=None)
            )
            s.commit()

    def set_structure_excluded(self, structure_id: str, excluded: bool) -> None:
        with self._session() as s:
            s.execute(
                update(Structure)
                .where(Structure.id == structure_id)
                .values(excluded=excluded)
            )
            s.commit()

    def save_phonon_result(
        self,
        structure_id: str,
        run_name: str,
        min_phonon_freq: Optional[float],
        params_gz: Optional[str] = None,
        settings: Optional[dict] = None,
    ) -> None:
        settings = settings or {}
        values = {
            "structure_id": structure_id,
            "run": run_name,
            "min_phonon_freq": _opt_float(min_phonon_freq),
            "supercell": _opt_json(settings.get("supercell")),
            "mesh": _opt_json(settings.get("mesh")),
            "displacement": _opt_float(settings.get("displacement")),
            "symprec": _opt_float(settings.get("symprec")),
            "calculator": settings.get("calculator"),
        }
        with self._session() as s:
            stmt = sqlite_insert(Phonon).values(**values)
            s.execute(
                stmt.on_conflict_do_update(
                    index_elements=["structure"],
                    set_=_excluded_set(stmt, Phonon, pk="structure"),
                )
            )
            if params_gz:
                blob_stmt = sqlite_insert(PhononParams).values(
                    structure_id=structure_id, run=run_name, params_gz=params_gz
                )
                s.execute(
                    blob_stmt.on_conflict_do_update(
                        index_elements=["structure"],
                        set_=_excluded_set(blob_stmt, PhononParams, pk="structure"),
                    )
                )
            s.commit()

    def get_phonon_result(self, structure_id: str) -> Optional[PhononResult]:
        with self._session() as s:
            row = s.get(Phonon, structure_id)
            blob = s.get(PhononParams, structure_id) if row else None
        if row is None:
            return None
        return PhononResult(
            min_phonon_freq=row.min_phonon_freq,
            params_gz=(blob.params_gz or "") if blob else "",
            supercell=row.supercell,
            mesh=row.mesh,
            displacement=row.displacement,
            symprec=row.symprec,
            calculator=row.calculator,
        )

    # ------------------------------------------------------------------ #
    #  Duplicate marking
    # ------------------------------------------------------------------ #

    def mark_duplicates(
        self,
        dropped_ids: list[str],
        kept_ids: list[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        total = len(dropped_ids) + len(kept_ids)
        done = 0

        def _on_batch(n: int) -> None:
            nonlocal done
            done += n
            if progress_callback is not None:
                progress_callback(done, total)

        if progress_callback is not None:
            progress_callback(0, total)
        self._batch_set_duplicate_flag(dropped_ids, True, on_batch=_on_batch)
        self._batch_set_duplicate_flag(kept_ids, False, on_batch=_on_batch)

    def _batch_set_duplicate_flag(
        self,
        ids: list[str],
        flag: bool,
        on_batch: Optional[Callable[[int], None]] = None,
    ) -> None:
        BATCH = 500
        for i in range(0, len(ids), BATCH):
            batch = ids[i : i + BATCH]
            with self._session() as s:
                s.execute(
                    update(Structure)
                    .where(Structure.id.in_(batch))
                    .values(duplicate=flag)
                )
                s.commit()
            if on_batch is not None:
                on_batch(len(batch))

    # ------------------------------------------------------------------ #
    #  Querying
    # ------------------------------------------------------------------ #

    def get_structures_for_analysis(
        self,
        run_id: str,
        statuses: tuple = (StructureStatus.RELAXED,),
    ) -> List[Structure]:
        status_vals = [str(s) for s in statuses]
        with self._session() as s:
            rows = s.execute(
                select(
                    Structure.id,
                    Structure.status,
                    Structure.energy_per_atom,
                    Structure.initial_atoms,
                    Structure.final_atoms,
                ).where(Structure.run == run_id, Structure.status.in_(status_vals))
            ).all()

        results: List[Structure] = []
        for sid, status, epa, initial_atoms, final_atoms in rows:
            relaxed = status == StructureStatus.RELAXED
            atoms = final_atoms if relaxed else initial_atoms
            if atoms is None:
                continue
            results.append(
                Structure(
                    id=sid,
                    status=status,
                    energy_per_atom=float(epa),
                    final_atoms=atoms if relaxed else None,
                    initial_atoms=None if relaxed else atoms,
                )
            )
        return results

    def get_structures(
        self,
        run_name: str,
        *,
        status: Optional[str] = None,
        statuses: Optional[tuple[str, ...]] = None,
        symprec: float = 1e-3,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> List[Structure]:
        effective = statuses or ((status,) if status else None)

        stmt = select(Structure).where(Structure.run == run_name)
        if effective:
            stmt = stmt.where(Structure.status.in_([str(s) for s in effective]))

        with self._session() as s:
            structs = list(s.scalars(stmt).all())
            phonon_freq = dict(
                s.execute(
                    select(Phonon.structure_id, Phonon.min_phonon_freq).where(
                        Phonon.run == run_name
                    )
                ).all()
            )

        try:
            meta = self.get_run_metadata(run_name)
            cfg = meta.search_config if meta else SearchConfig()
        except Exception as exc:
            get_logger("rapmat.storage").warning(
                "get_run_metadata(%s) failed, derived fields use defaults: %s",
                run_name, exc,
            )
            cfg = SearchConfig()

        total = len(structs)
        for i, st in enumerate(structs):
            if progress_callback is not None:
                progress_callback(i + 1, total, f"Processing structure {i + 1}/{total}...")

            st.min_phonon_freq = phonon_freq.get(st.id)
            st.pressure_gpa = cfg.pressure_gpa
            st.domain = cfg.domain
            st.symprec = symprec
        return structs

    def count(self) -> int:
        with self._session() as s:
            return int(s.scalar(select(func.count()).select_from(Structure)) or 0)

    # ------------------------------------------------------------------ #
    #  Evaluation records
    # ------------------------------------------------------------------ #

    def add_evaluation(
        self,
        structure_id: str,
        run_name: str,
        calculator: str,
        config_json: str,
        energy_per_atom: float,
        energy_total: float,
        min_phonon_freq: Optional[float] = None,
    ) -> str:
        eval_id = _eval_id(structure_id, calculator, config_json)
        values = {
            "id": eval_id,
            "structure_id": structure_id,
            "run": run_name,
            "calculator": calculator,
            "config_json": config_json,
            "energy_per_atom": float(energy_per_atom),
            "energy_total": float(energy_total),
            "min_phonon_freq": (
                float(min_phonon_freq) if min_phonon_freq is not None else None
            ),
        }
        with self._session() as s:
            stmt = sqlite_insert(Evaluation).values(**values)
            s.execute(
                stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_=_excluded_set(stmt, Evaluation, pk="id"),
                )
            )
            s.commit()
        return eval_id

    def has_evaluation(
        self, structure_id: str, calculator: str, config_json: str
    ) -> bool:
        eval_id = _eval_id(structure_id, calculator, config_json)
        with self._session() as s:
            return s.get(Evaluation, eval_id) is not None

    def clear_evaluations(
        self,
        run_name: str,
        calculator: Optional[str] = None,
    ) -> None:
        stmt = delete(Evaluation).where(Evaluation.run == run_name)
        if calculator is not None:
            stmt = stmt.where(Evaluation.calculator == calculator)
        with self._session() as s:
            s.execute(stmt)
            s.commit()

    def get_evaluations(
        self,
        run_name: str,
        calculator: Optional[str] = None,
    ) -> List[Evaluation]:
        stmt = select(Evaluation).where(Evaluation.run == run_name)
        if calculator is not None:
            stmt = stmt.where(Evaluation.calculator == calculator)
        with self._session() as s:
            return list(s.scalars(stmt).all())

    # ------------------------------------------------------------------ #
    #  Maintenance
    # ------------------------------------------------------------------ #

    def vacuum(self) -> None:
        """Full reclaim: rewrite the database, returning all free pages to the OS."""
        with self._lock:
            with self._engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                conn.exec_driver_sql("VACUUM")

    def close(self) -> None:
        try:
            with self._lock:
                with self._engine.connect() as conn:
                    conn.execution_options(isolation_level="AUTOCOMMIT")
                    conn.exec_driver_sql("PRAGMA incremental_vacuum")
                self._engine.dispose()
        except Exception:
            pass

        if hasattr(self, "_file_lock") and getattr(self._file_lock, "is_locked", False):
            try:
                self._file_lock.release()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Raw access
    # ------------------------------------------------------------------ #

    def _read(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            with self._engine.connect() as conn:
                result = conn.exec_driver_sql(sql, params)
                return [dict(r._mapping) for r in result]

    def _write(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            with self._engine.begin() as conn:
                conn.exec_driver_sql(sql, params)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _excluded_set(stmt, model, *, pk: str) -> dict:
    """INSERT-OR-REPLACE."""
    return {
        c.name: getattr(stmt.excluded, c.name)
        for c in model.__table__.columns
        if c.name != pk
    }


def _eval_id(structure_id: str, calculator: str, config_json: str) -> str:
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:12]
    return f"{structure_id}_{calculator}_{config_hash}"


def _opt_json(value) -> Optional[str]:
    return json.dumps(value) if value is not None else None


def _opt_float(value) -> Optional[float]:
    return float(value) if value is not None else None
