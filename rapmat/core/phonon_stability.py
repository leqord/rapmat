from typing import List, Optional, Tuple

from rapmat.calculators import Calculators, ProgressCalcCallback
from rapmat.calculators.factory import CalculatorProvider
from rapmat.core.entities import ResultRow
from rapmat.core.phonon import calculate_phonons_with_freq, serialize_phonons
from rapmat.storage.base import StructureStore
from rapmat.utils.common import workdir_context
from rapmat.utils.console import get_logger
from rapmat.utils.progress import ProgressCallback

_logger = get_logger("rapmat.phonon_stability")


def compute_dynamical_stability_for_results(
    results: List[ResultRow],
    phonon_top: int,
    phonon_cutoff: float,
    phonon_supercell: Tuple[int, int, int],
    phonon_mesh: Tuple[int, int, int],
    phonon_displacement: float,
    phonon_calculator: Calculators,
    store: Optional["StructureStore"] = None,
    calculator_config: dict | None = None,
    progress_callback: ProgressCallback | None = None,
    symprec: float = 1e-3,
    reduce_primitive: bool = True,
    run_name: str | None = None,
    auto_settings: bool = False,
    monolayer: bool = False,
) -> bool:
    if phonon_top < 1:
        return False
    if not results:
        return False

    target_results: List[ResultRow] = []
    for result in results:
        if result.converged:
            target_results.append(result)
            if len(target_results) >= phonon_top:
                break

    if not target_results:
        return False

    with workdir_context(None) as wdir:
        updated = False
        total = len(target_results)

        _bar = {"current": 0}

        def _sub_progress(_current, _total, message, *args, **kwargs) -> None:
            if progress_callback is not None:
                progress_callback(_bar["current"], total, message)

        calculator_for = CalculatorProvider(
            phonon_calculator,
            wdir,
            config=calculator_config,
            callback=ProgressCalcCallback(_sub_progress),
            auto_settings=auto_settings,
            monolayer=monolayer,
            log_callback=lambda msg: _sub_progress(0, 0, msg),
        )

        calc_name = phonon_calculator.value

        def _persist(result: ResultRow, phonons, min_freq: float) -> None:
            sid = result.structure_id
            rname = result.run_name or run_name
            if rname is None:
                _logger.warning(
                    "No run name for %s; skipping phonon persistence.", sid
                )
                return

            blob = None
            try:
                blob = serialize_phonons(phonons)
            except Exception as exc:
                _logger.warning(
                    "Could not serialize phonopy output for %s: %s", sid, exc
                )
            try:
                store.save_phonon_result(
                    sid,
                    rname,
                    min_freq,
                    params_gz=blob,
                    settings={
                        "supercell": [int(x) for x in phonon_supercell],
                        "mesh": [int(x) for x in phonon_mesh],
                        "displacement": float(phonon_displacement),
                        "symprec": float(symprec),
                        "calculator": calc_name,
                    },
                )
            except Exception as exc:
                _logger.warning(
                    "Could not persist phonon result for %s: %s", sid, exc
                )

        def _process_one(result: ResultRow) -> None:
            nonlocal updated
            atoms = result.atoms
            if atoms is None:
                result.structure.min_phonon_freq = None
                result.dynamical_stability = None
                return

            try:
                phonons, min_freq = calculate_phonons_with_freq(
                    atoms,
                    calculator_for=calculator_for,
                    displacement=phonon_displacement,
                    supercell=phonon_supercell,
                    qpoint_mesh=phonon_mesh,
                    reduce_primitive=reduce_primitive,
                    symprec=symprec,
                    progress_callback=_sub_progress,
                )
                result.structure.min_phonon_freq = min_freq
                result.dynamical_stability = not (min_freq < phonon_cutoff)
                if store is not None and result.structure_id:
                    _persist(result, phonons, min_freq)
                updated = True
            except Exception as e:
                _logger.error(
                    "Phonon calc failed for ID %s: %s",
                    result.index, e,
                    exc_info=True,
                )
                result.structure.min_phonon_freq = None
                result.dynamical_stability = None
                updated = True

        for i, result in enumerate(target_results):
            _bar["current"] = i
            msg = f"Structure {i + 1}/{total}: {result.formula}"
            if progress_callback is not None:
                progress_callback(i, total, msg)
            _process_one(result)

        if progress_callback is not None:
            progress_callback(total, total, "Done")

        return updated
