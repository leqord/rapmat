from typing import Collection, List, Optional

from pydantic import BaseModel

from rapmat.calculators import ProgressCalcCallback
from rapmat.calculators.factory import (CalculatorProvider, finalize_provider,
                                        set_provider_label)
from rapmat.core.entities import ResultRow
from rapmat.core.phonon import calculate_phonons_with_freq, serialize_phonons
from rapmat.core.phonon_settings import PhononSettings
from rapmat.storage.base import StructureStore
from rapmat.utils.common import workdir_context
from rapmat.utils.console import get_logger
from rapmat.utils.progress import ProgressCallback

_logger = get_logger("rapmat.phonon_stability")


class PhononRunSummary(BaseModel):
    targets: int = 0
    computed: int = 0
    skipped: int = 0
    failed: int = 0
    unsaved: int = 0

    def describe(self) -> str:
        parts = [f"{self.computed} computed"]
        if self.skipped:
            parts.append(f"{self.skipped} already stored")
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.unsaved:
            parts.append(f"{self.unsaved} not saved")
        noun = "structure" if self.targets == 1 else "structures"
        return f"Phonons for {self.targets} {noun}: {', '.join(parts)}."


def compute_dynamical_stability_for_results(
    results: List[ResultRow],
    phonon_top: int,
    settings: PhononSettings,
    store: Optional["StructureStore"] = None,
    progress_callback: ProgressCallback | None = None,
    *,
    calculator_command: str = "",
    skip_ids: Collection[str] = (),
    session_hint: str | None = None,
) -> PhononRunSummary:
    summary = PhononRunSummary()
    if phonon_top < 1:
        return summary

    targets = [r for r in results if r.converged and r.atoms is not None][:phonon_top]
    todo = [r for r in targets if r.structure_id not in skip_ids]
    summary.targets = len(targets)
    summary.skipped = len(targets) - len(todo)
    if not todo:
        return summary

    domains = {r.structure.domain for r in todo}
    if len(domains) > 1:
        raise ValueError(
            f"Phonon targets mix domains ({', '.join(sorted(domains))})"
        )
    monolayer = domains == {"monolayer"}

    stored_settings = settings.model_dump(mode="json")

    with workdir_context(None, session_hint=session_hint) as wdir:
        total = len(todo)

        _bar = {"current": 0}

        def _sub_progress(_current, _total, message, *args, **kwargs) -> None:
            if progress_callback is not None:
                progress_callback(_bar["current"], total, message)

        calculator_for = CalculatorProvider(
            settings.calculator,
            wdir,
            config=settings.run_config(calculator_command),
            callback=ProgressCalcCallback(_sub_progress),
            auto_settings=settings.calculator_settings == "auto",
            monolayer=monolayer,
            log_callback=lambda msg: _sub_progress(0, 0, msg),
        )

        def _persist(result: ResultRow, phonons, min_freq: float) -> bool:
            sid = result.structure_id
            run = result.structure.run
            if not run:
                _logger.warning("No run for %s, phonon result not saved.", sid)
                return False

            blob = None
            try:
                blob = serialize_phonons(phonons)
            except Exception as exc:
                _logger.warning(
                    "Could not serialize phonopy output for %s: %s", sid, exc
                )
            try:
                store.save_phonon_result(
                    sid, run, min_freq, params_gz=blob, settings=stored_settings
                )
            except Exception as exc:
                _logger.warning(
                    "Could not persist phonon result for %s: %s", sid, exc
                )
                return False
            return True

        def _process_one(result: ResultRow) -> None:
            set_provider_label(calculator_for, result.structure_id)

            try:
                phonons, min_freq = calculate_phonons_with_freq(
                    result.atoms,
                    calculator_for=calculator_for,
                    displacement=settings.displacement,
                    supercell=settings.supercell,
                    qpoint_mesh=settings.mesh,
                    reduce_primitive=settings.reduce_primitive,
                    symprec=settings.symprec,
                    progress_callback=_sub_progress,
                )
            except Exception as e:
                _logger.error(
                    "Phonon calc failed for %s: %s",
                    result.structure_id, e,
                    exc_info=True,
                )
                result.structure.min_phonon_freq = None
                summary.failed += 1
                return

            result.structure.min_phonon_freq = min_freq
            summary.computed += 1
            if store is not None and result.structure_id:
                if not _persist(result, phonons, min_freq):
                    summary.unsaved += 1

        try:
            for i, result in enumerate(todo):
                _bar["current"] = i
                msg = f"Structure {i + 1}/{total}: {result.formula}"
                if progress_callback is not None:
                    progress_callback(i, total, msg)
                _process_one(result)
        finally:
            finalize_provider(calculator_for)

        if progress_callback is not None:
            progress_callback(total, total, "Done")

    return summary
