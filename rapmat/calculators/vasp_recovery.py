from __future__ import annotations

from pathlib import Path

RECOVERY_KEYS = frozenset({"ismear", "isym", "symprec"})

TETRAHEDRON = "tetrahedron"
SYMMETRY = "symmetry"
PATHOLOGICAL = "pathological"

_SIGNATURES = (
    ("BZINTS: Tetrahedron", TETRAHEDRON),
    ("IBZKPT: not all point group", SYMMETRY),
    ("POSMAP:", SYMMETRY),
    ("PRICELV:", SYMMETRY),
    ("INVGRP:", SYMMETRY),
    ("RHOSYG:", SYMMETRY),
    ("internal error in: us.F", PATHOLOGICAL),
    ("SETYLM_AUG", PATHOLOGICAL),
)

_LADDER: dict[str, tuple[dict, ...]] = {
    SYMMETRY: ({"isym": 0},),
    TETRAHEDRON: ({"isym": 0}, {"ismear": 0}),
    PATHOLOGICAL: (),
}

_COMMAND_MISSING = ("returned an error: 127", ": not found", "No such file or directory")

WARN_PREFIX = "WARNING: "

DEVIATION_NOTE = "deviates from the OMat24 protocol"


def classify_vasp_abort(text: str) -> str | None:
    if not text:
        return None
    for needle, kind in _SIGNATURES:
        if needle in text:
            return kind
    return None


def is_command_missing(text: str) -> bool:
    return bool(text) and any(needle in text for needle in _COMMAND_MISSING)


def read_vasp_out(directory, *, tail_bytes: int = 64_000) -> str:
    if directory is None:
        return ""

    path = Path(directory) / "vasp.out"
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - tail_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def next_override(abort: str | None, applied: dict) -> dict | None:
    for patch in _LADDER.get(abort or "", ()):
        if any(applied.get(key) != value for key, value in patch.items()):
            return dict(patch)
    return None


def format_deviations(applied: dict) -> str:
    return ", ".join(f"{key.upper()}={value}" for key, value in sorted(applied.items()))


def warn(log_callback, message: str) -> None:
    if log_callback is not None:
        log_callback(f"{WARN_PREFIX}{message}")
        return

    from rapmat.utils.console import get_logger

    get_logger("rapmat.task").warning("%s", message)


def evaluate_with_recovery(
    atoms,
    calculator_for,
    compute,
    *,
    label: str = "",
    log_callback=None,
    max_recoveries: int = 2,
):
    from rapmat.calculators import cleanup_calculator_files
    from rapmat.calculators.factory import (apply_provider_overrides,
                                            provider_overrides,
                                            provider_pinned_keys)

    tag = f"{label}: " if label else ""

    for attempt in range(max_recoveries + 1):
        calculator = calculator_for(atoms)
        cleanup_calculator_files(calculator)
        atoms.calc = calculator

        try:
            return compute(atoms)
        except Exception as exc:
            # NOTE: read before retrying
            text = read_vasp_out(getattr(calculator, "directory", None))
            kind = classify_vasp_abort(text) or classify_vasp_abort(str(exc))

            if kind is None:
                if is_command_missing(text) or is_command_missing(str(exc)):
                    warn(
                        log_callback,
                        f"{tag}the VASP command is not runnable",
                    )
                raise

            patch = next_override(kind, provider_overrides(calculator_for))
            if patch is None or attempt == max_recoveries:
                raise

            pinned = provider_pinned_keys(calculator_for) & set(patch)
            if pinned:
                names = ", ".join(sorted(key.upper() for key in pinned))
                warn(
                    log_callback,
                    f"{tag}{kind} abort, but {names} is pinned in your "
                    "calculator settings",
                )
                raise

            if not apply_provider_overrides(calculator_for, patch):
                raise

            warn(
                log_callback,
                f"{tag}{kind} abort, retrying with "
                f"{format_deviations(patch)} ({DEVIATION_NOTE})",
            )
