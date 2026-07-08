import traceback
from pathlib import Path

from rapmat.storage.status import StructureStatus
from rapmat.utils.progress import ProgressCallback

# ------------------------------------------------------------------ #
#  Orchestration loops (used by TUI and tests)
# ------------------------------------------------------------------ #


from rapmat.core.generation_worker import \
    generate_one_structure as _generate_one_structure


def run_processing_loop(
    run_name: str,
    store,
    config: dict,
    workdir_path: Path,
    worker_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_flag: list[bool] | None = None,
):
    import numpy as np
    from ase.units import GPa as _GPa

    from rapmat.calculators import Calculators, ProgressCalcCallback
    from rapmat.calculators.factory import load_calculator
    from rapmat.core.config import SearchConfig
    from rapmat.core.relaxation import structure_relax
    from rapmat.core.sanity import check_sanity
    from rapmat.utils.common import free_cuda_memory
    from rapmat.utils.console import get_logger
    logger = get_logger("rapmat.csp")
    from rapmat.utils.structure import format_spg

    _calc_cb = ProgressCalcCallback(progress_callback)

    cfg = config if isinstance(config, SearchConfig) else SearchConfig.model_validate(config)

    calculator_name = cfg.calculator.upper()
    calculator_config = cfg.calculator_config
    domain_val = cfg.domain
    symprec = cfg.symprec
    pressure_gpa = cfg.pressure_gpa
    pressure_evA3 = pressure_gpa * _GPa
    min_dist = cfg.min_dist
    use_sanity_pymatgen = cfg.sanity_pymatgen
    sanity_pymatgen_tol = cfg.sanity_pymatgen_tol

    force_conv_crit = cfg.force_conv_crit
    steps_max = cfg.steps_max
    forces_break = cfg.forces_break

    calculator_workdir_path = workdir_path / Path("calculator")
    calculator_workdir_path.mkdir(parents=True, exist_ok=True)

    candidates = store.get_unrelaxed_candidates(run_name)

    n_relaxed = 0

    free_cuda_memory()

    _calc_cb.on_status(f"Loading calculator {calculator_name}...")
    calculator = load_calculator(
        Calculators(calculator_name),
        calculator_workdir_path,
        config=calculator_config,
        callback=_calc_cb,
    )

    counter: int = 0
    discarded_sanity = 0
    n_candidates = len(candidates)

    def _report(msg: str) -> None:
        if progress_callback:
            progress_callback(counter, n_candidates, msg)

    def _run_loop():
        nonlocal counter, discarded_sanity, n_relaxed
        nonlocal calculator

        for candidate in candidates:
            counter += 1
            if worker_id and counter % 10 == 0:
                store.update_heartbeat(run_name, worker_id)
            struct_id = candidate.id

            for attempt in range(3):
                structure = candidate.atoms.copy()
                from rapmat.calculators import cleanup_calculator_files

                cleanup_calculator_files(calculator)

                try:
                    structure.calc = calculator
                    structure.info["initial_spg"] = format_spg(
                        structure, symprec=symprec
                    )

                    store.update_generated_structure(
                        struct_id, structure
                    )

                    _report("Relaxing...")

                    def _optim_cb(step: int, max_steps: int, msg: str) -> None:
                        if progress_callback:
                            msg_fmt = f"Relaxing {struct_id}: {msg}"
                            progress_callback(
                                counter, n_candidates, msg_fmt, is_log=False
                            )

                    converged, relaxed_structure = structure_relax(
                        structure,
                        force_conv_crit=force_conv_crit,
                        steps_max=steps_max,
                        mask=[1, 1, 0, 0, 0, 1] if domain_val == "monolayer" else None,
                        opt_logfile=str(
                            calculator_workdir_path
                            / Path(f"opt_{struct_id.replace('/', '_')}.log")
                        ),
                        scalar_pressure=pressure_evA3,
                        forces_break=forces_break,
                        cancel_flag=cancel_flag,
                        progress_callback=_optim_cb,
                    )

                    if cancel_flag and cancel_flag[0]:
                        break

                    _report("Data collection...")
                    relaxed_structure.info["energy"] = (
                        relaxed_structure.get_potential_energy()
                    )
                    forces = relaxed_structure.get_forces()
                    relaxed_structure.info["forces"] = forces
                    relaxed_structure.info["fmax"] = np.max(
                        np.linalg.norm(forces, axis=1)
                    )
                    relaxed_structure.info["converged"] = converged
                    relaxed_structure.info["initial_spg"] = structure.info[
                        "initial_spg"
                    ]

                    relaxed_structure.info["final_spg"] = format_spg(
                        relaxed_structure, symprec=symprec
                    )

                    _report("Metadata preparation...")
                    energy = relaxed_structure.info["energy"]

                    meta = {
                        "energy_per_atom": energy / len(relaxed_structure),
                        "fmax": relaxed_structure.info["fmax"],
                        "converged": relaxed_structure.info["converged"],
                    }

                    _report("Checking sanity...")
                    if not check_sanity(
                        relaxed_structure,
                        min_dist=min_dist,
                        use_pymatgen=use_sanity_pymatgen,
                        pymatgen_tol=sanity_pymatgen_tol,
                    ):
                        discarded_sanity += 1
                        _report(f"Discarded {struct_id}: failed sanity check")
                        store.update_structure(
                            struct_id,
                            status=StructureStatus.DISCARDED,
                            atoms=relaxed_structure,
                            metadata=meta,
                        )
                        break

                    _report("Saving to database...")
                    store.update_structure(
                        struct_id,
                        status=StructureStatus.RELAXED,
                        atoms=relaxed_structure,
                        metadata=meta,
                    )
                    n_relaxed += 1
                    break

                except Exception as ex:
                    tb = traceback.format_exc()
                    _report(f"ERROR on {struct_id} (attempt {attempt + 1}/3): {ex}")
                    _report(f"Traceback:\n{tb}")
                    if attempt == 2:
                        logger.error(
                            "Failed to relax structure %s: %s",
                            struct_id, ex, exc_info=True,
                        )
                        store.update_structure(
                            struct_id, status=StructureStatus.ERROR
                        )
                        break

                    try:
                        del calculator
                    except Exception:
                        pass
                    free_cuda_memory()

                    _report(
                        f"Reloading calculator {calculator_name} after error (attempt {attempt + 1})..."
                    )
                    try:
                        calculator = load_calculator(
                            Calculators(calculator_name),
                            calculator_workdir_path,
                            config=calculator_config,
                            callback=_calc_cb,
                        )
                        _report(f"Calculator {calculator_name} reloaded successfully.")
                    except Exception as reload_ex:
                        logger.error(
                            "Calculator reload failed: %s",
                            reload_ex, exc_info=True,
                        )
                        _report(f"CRITICAL ERROR: Reload failed: {reload_ex}")
                        _report(f"Reload traceback:\n{traceback.format_exc()}")
                        store.update_structure(
                            struct_id, status=StructureStatus.ERROR
                        )
                        break

            if progress_callback:
                progress_callback(
                    counter, n_candidates, f"Processed {counter}/{n_candidates}"
                )

    _run_loop()

    pressure_msg = f" | Pressure: {pressure_gpa} GPa" if pressure_gpa > 0 else ""
    logger.info(
        "Done. Run: %s | Storage: %s%s | Relaxed: %d | Discarded (sanity): %d",
        run_name, store.get_url(), pressure_msg, n_relaxed, discarded_sanity,
    )

    return None


def run_generation_loop(
    run_name: str,
    store,
    config: dict,
    worker_id: str | None = None,
    workers: int = 1,
    progress_callback: ProgressCallback | None = None,
    cancel_flag: list[bool] | None = None,
    log_callback=None,
) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from rapmat.core.config import SearchConfig
    from rapmat.utils.console import get_logger
    logger = get_logger("rapmat.csp")

    cfg = config if isinstance(config, SearchConfig) else SearchConfig.model_validate(config)

    domain_val = cfg.domain
    search_dim = 3 if domain_val == "bulk" else 2
    formula = cfg.formula
    thickness_cutoff = cfg.thickness_cutoff
    run_seed = cfg.seed  # int | None
    max_count = cfg.max_count

    elements = list(formula.keys())
    formula_values = list(formula.values())

    placeholders = store.get_pending_generation(run_name)
    if not placeholders:
        if progress_callback is None:
            logger.info("No structures left to generate.")
        return 0

    n_placeholders = len(placeholders)
    if progress_callback is None:
        logger.info(
            "Generating structures for run: %s - %d placeholders to generate.",
            run_name, n_placeholders,
        )

    if log_callback:
        log_callback(f"PyXtal max_count={max_count}, workers={workers}")

    generated = 0
    discarded = 0
    errors = 0


    def _log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    def _is_cancelled() -> bool:
        return cancel_flag is not None and cancel_flag[0]

    def _handle_result(status, struct_id, atoms, spg, fu):
        nonlocal generated, discarded, errors
        match status:
            case StructureStatus.GENERATED:
                store.update_generated_structure(struct_id, atoms)
                generated += 1
            case StructureStatus.DISCARDED:
                store.discard_generation_placeholder(struct_id)
                discarded += 1
            case StructureStatus.ERROR:
                logger.error(
                    "Structure for group %s / fu %s failed", spg, fu,
                )
                store.discard_generation_placeholder(struct_id)
                errors += 1

    if workers <= 1:
        for counter, ph in enumerate(placeholders, start=1):
            if _is_cancelled():
                _log("Generation cancelled by user")
                break

            if worker_id and counter % 20 == 0:
                store.update_heartbeat(run_name, worker_id)

            spg = ph.gen_spg
            fu = ph.gen_fu
            _log(f"[{counter}/{n_placeholders}] spg={spg} fu={fu}")
            struct_seed = (
                (run_seed + counter) % (2**32) if run_seed is not None else None
            )
            status, struct_id, atoms = _generate_one_structure(
                ph.id,
                spg,
                fu,
                elements,
                formula_values,
                search_dim,
                thickness_cutoff,
                seed=struct_seed,
                max_count=max_count,
            )
            _handle_result(status, struct_id, atoms, spg, fu)

    else:
        pool = ProcessPoolExecutor(max_workers=workers)

        with pool:
            futures = {
                pool.submit(
                    _generate_one_structure,
                    ph.id,
                    ph.gen_spg,
                    ph.gen_fu,
                    elements,
                    formula_values,
                    search_dim,
                    thickness_cutoff,
                    seed=((run_seed + idx) % (2**32) if run_seed is not None else None),
                    max_count=max_count,
                ): ph
                for idx, ph in enumerate(placeholders, start=1)
            }

            for counter, future in enumerate(as_completed(futures), start=1):
                if _is_cancelled():
                    _log("Generation cancelled - cancelling pending futures...")
                    for f in futures:
                        f.cancel()
                    break

                if worker_id and counter % 20 == 0:
                    store.update_heartbeat(run_name, worker_id)

                status, struct_id, atoms = future.result()
                ph = futures[future]
                _handle_result(
                    status, struct_id, atoms, ph.gen_spg, ph.gen_fu
                )
                
                if counter % 100 == 0:
                    _log(
                        f"Generated {counter}/{n_placeholders} (ok={generated}, disc={discarded}, err={errors})"
                    )

    _log(f"Generation finished: {generated} ok, {discarded} discarded, {errors} errors")
    return generated


def execute_run(
    run_name: str,
    store,
    config,
    *,
    worker_id: str,
    workers: int = 1,
    progress_callback: ProgressCallback | None = None,
    log_callback=None,
    cancel_flag: list[bool] | None = None,
) -> None:
    """Execute a claimed run to the finish.
    """
    import time

    from rapmat.storage.status import RunStatus
    from rapmat.utils.common import workdir_context

    flag = cancel_flag if cancel_flag is not None else [False]

    def _log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    def _cb(current: int, total: int, message: str = "", is_log: bool = True) -> None:
        if progress_callback is None:
            if is_log and message:
                _log(message)
            return
        try:
            progress_callback(current, total, message, is_log=is_log)
        except KeyboardInterrupt:
            flag[0] = True
            raise

    try:
        with workdir_context(None) as workdir_path:
            _log(f"Working directory: {workdir_path}")

            pending = store.get_pending_generation(run_name)
            if pending:
                _log(f"Generation phase: {len(pending)} placeholders pending...")
                store.set_run_status(run_name, RunStatus.GENERATING)
                run_generation_loop(
                    run_name=run_name,
                    store=store,
                    config=config,
                    worker_id=worker_id,
                    workers=workers,
                    progress_callback=_cb,
                    cancel_flag=flag,
                    log_callback=log_callback,
                )
                if flag[0]:
                    raise KeyboardInterrupt("Cancelled by user")
                _cb(
                    0, 0,
                    "Generation complete. Initializing calculator...",
                )

            store.set_run_status(run_name, RunStatus.PROCESSING)
            _log(f"Starting processing phase for {run_name}...")

            t0 = time.monotonic()
            run_processing_loop(
                run_name=run_name,
                store=store,
                config=config,
                workdir_path=workdir_path,
                worker_id=worker_id,
                progress_callback=_cb,
                cancel_flag=flag,
            )
            _log(
                f"Run '{run_name}' computation finished in "
                f"{time.monotonic() - t0:.2f} seconds."
            )

            store.release_run(run_name, RunStatus.COMPLETED)
    except KeyboardInterrupt:
        store.release_run(run_name, RunStatus.INTERRUPTED)
        raise
    except Exception:
        store.release_run(run_name, RunStatus.FAILED)
        raise
