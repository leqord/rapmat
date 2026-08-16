import json
from typing import Sequence

from pydantic import BaseModel

from rapmat.core.entities import Structure
from rapmat.utils.progress import ProgressCallback


def evaluation_config_key(
    *,
    calculator_config: dict,
    calculator_settings: str = "toml",
    run_phonons: bool = False,
    phonon_supercell=(3, 3, 3),
    phonon_mesh=(20, 20, 20),
    phonon_displacement: float = 1e-2,
) -> str:
    config: dict = {
        "run_phonons": run_phonons,
        "calculator_config": calculator_config,
    }
    if calculator_settings != "toml":
        config["calculator_settings"] = calculator_settings
    if run_phonons:
        config["phonon_supercell"] = phonon_supercell
        config["phonon_mesh"] = phonon_mesh
        config["phonon_displacement"] = phonon_displacement

    return json.dumps(config, sort_keys=True)


class ComparisonRow(BaseModel):
    """One structure compared between the MLIP and a reference calculator."""

    id: str = ""
    formula: str = ""
    mlip_epa: float = 0.0
    ref_epa: float = 0.0
    mlip_phonon_freq: float | None = None
    ref_phonon_freq: float | None = None
    mlip_rank: int | None = None
    ref_rank: int | None = None


# ------------------------------------------------------------------ #
#  Evaluation loop (used by TUI and tests)
# ------------------------------------------------------------------ #


def run_eval_loop(
    pending: list[Structure],
    store,
    run_name: str,
    calculator_for,
    calculator_name: str,
    config_json: str,
    *,
    run_phonons: bool = False,
    phonon_displacement: float = 1e-2,
    phonon_supercell: tuple = (3, 3, 3),
    phonon_mesh: tuple = (20, 20, 20),
    progress_callback: ProgressCallback | None = None,
    log_callback=None,
    reduce_to_primitive: bool = True,
    symprec: float = 1e-3,
) -> None:
    from rapmat.calculators import cleanup_calculator_files
    from rapmat.calculators.vasp import preflight_potcars
    from rapmat.core.phonon import calculate_min_phonon_freq
    from rapmat.utils.console import get_logger
    logger = get_logger("rapmat.evaluation")

    if pending:
        preflight_potcars(calculator_for(pending[0].atoms), pending[0].atoms)

    n_total = len(pending)
    for i, rec in enumerate(pending, 1):
        atoms = rec.atoms.copy()
        atoms.pbc = True
        calculator = None

        try:
            calculator = calculator_for(atoms)
            cleanup_calculator_files(calculator)
            atoms.calc = calculator

            ref_energy = atoms.get_potential_energy()
            ref_epa = ref_energy / len(atoms)

            ref_phonon_freq = None
            if run_phonons:
                ref_phonon_freq = calculate_min_phonon_freq(
                    atoms,
                    calculator_for=calculator_for,
                    displacement=phonon_displacement,
                    supercell=phonon_supercell,
                    qpoint_mesh=phonon_mesh,
                    reduce_primitive=reduce_to_primitive,
                    symprec=symprec,
                    progress_callback=progress_callback,
                    log_label=rec.id,
                    log_callback=log_callback,
                )

            store.add_evaluation(
                structure_id=rec.id,
                run_name=run_name,
                calculator=calculator_name,
                config_json=config_json,
                energy_per_atom=ref_epa,
                energy_total=ref_energy,
                min_phonon_freq=ref_phonon_freq,
            )
        except Exception as e:
            import traceback
            import os
            from pathlib import Path

            err_msg = f"Failed to evaluate structure {rec.id}: {e}"
            if log_callback:
                log_callback(err_msg)
                log_callback(traceback.format_exc())

                calc_dir = getattr(calculator, "directory", None)
                if calc_dir and os.path.exists(calc_dir):
                    for out_file in ["vasp.out", "OUTCAR"]:
                        fpath = Path(calc_dir) / out_file
                        if fpath.exists():
                            log_callback(f"--- START OF {out_file} ---")
                            try:
                                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                                    log_callback(f.read())
                            except Exception as read_e:
                                log_callback(f"Failed to read {out_file}: {read_e}")
                            log_callback(f"--- END OF {out_file} ---")

            else:
                logger.error("%s", err_msg, exc_info=True)

        if progress_callback:
            progress_callback(i, n_total, f"Evaluated {i}/{n_total}")


# ------------------------------------------------------------------ #
#  Pure metric helpers
# ------------------------------------------------------------------ #


def compute_ranking_metrics(
    results: Sequence[ComparisonRow],
    phonon_cutoff: float = -0.15,
    stable_only: bool = True,
) -> dict:
    from scipy.stats import kendalltau

    subset = list(results)
    stable_only_applied = False

    if stable_only:
        has_phonon = all(
            r.mlip_phonon_freq is not None
            and r.ref_phonon_freq is not None
            for r in subset
        )
        if has_phonon:
            subset = [
                r
                for r in subset
                if r.mlip_phonon_freq >= phonon_cutoff
                and r.ref_phonon_freq >= phonon_cutoff
            ]
            stable_only_applied = True

    n = len(subset)
    if n < 2:
        return {
            "kendall_tau": None,
            "p_value": None,
            "mae_epa": None,
            "n_structures": n,
            "stable_only_applied": stable_only_applied,
        }

    mlip_vals = [r.mlip_epa for r in subset]
    ref_vals = [r.ref_epa for r in subset]

    tau, p_value = kendalltau(mlip_vals, ref_vals)
    mae = sum(abs(m - r) for m, r in zip(mlip_vals, ref_vals)) / n

    return {
        "kendall_tau": float(tau),
        "p_value": float(p_value),
        "mae_epa": float(mae),
        "n_structures": n,
        "stable_only_applied": stable_only_applied,
    }


def compute_stability_metrics(
    results: Sequence[ComparisonRow],
    phonon_cutoff: float = -0.15,
) -> dict | None:
    valid = [
        r
        for r in results
        if r.mlip_phonon_freq is not None
        and r.ref_phonon_freq is not None
    ]
    if not valid:
        return None

    tp = fp = fn = tn = 0
    for r in valid:
        ref_stable = r.ref_phonon_freq >= phonon_cutoff
        mlip_stable = r.mlip_phonon_freq >= phonon_cutoff
        if ref_stable and mlip_stable:
            tp += 1
        elif not ref_stable and mlip_stable:
            fp += 1
        elif ref_stable and not mlip_stable:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_total": len(valid),
        "n_stable_ref": tp + fn,
        "n_stable_mlip": tp + fp,
    }


# ------------------------------------------------------------------ #
#  Evaluation helpers
# ------------------------------------------------------------------ #


def select_eval_records(records: Sequence, top_n: int) -> list:
    """Order candidate structures by MLIP energy and keep the lowest ``top_n``.
    """
    ordered = sorted(records, key=lambda r: r.energy_per_atom)
    if top_n and top_n > 0:
        ordered = ordered[:top_n]
    return ordered


def eval_rows_from_cache(records: Sequence, eval_map: dict, run_name: str) -> list:
    """Build :class:`ResultRow`s for records that have a cached evaluation.
    """
    from rapmat.core.entities import ResultRow

    rows: list = []
    for idx, rec in enumerate(records, 1):
        ev = eval_map.get(rec.id)
        if ev is None:
            continue
        rows.append(
            ResultRow(
                structure=rec,
                index=idx,
                run_name=run_name,
                ref_energy_per_atom=ev.energy_per_atom,
                ref_phonon_freq=ev.min_phonon_freq,
            )
        )
    return rows


def comparison_from_result_rows(rows: Sequence) -> list[ComparisonRow]:
    """Map evaluated :class:`ResultRow`s to :class:`ComparisonRow`s.
    """
    out: list[ComparisonRow] = []
    for r in rows:
        if r.ref_energy_per_atom is None:
            continue
        out.append(
            ComparisonRow(
                id=r.structure_id,
                formula=r.formula,
                mlip_epa=r.energy_per_atom,
                ref_epa=r.ref_energy_per_atom,
                mlip_phonon_freq=r.min_phonon_freq,
                ref_phonon_freq=r.ref_phonon_freq,
            )
        )
    return out
