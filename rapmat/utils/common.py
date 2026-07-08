import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import chemparse
from ase.data import atomic_numbers

from rapmat.config import APP_TMPDIR_SUFFIX


def parse_formula(formula: str) -> dict[str, int]:
    raw = chemparse.parse_formula(formula)
    counts: dict[str, int] = {}
    for elem, val in raw.items():
        if val != int(val) or val < 1:
            raise ValueError(
                f"Formula must have integer stoichiometry, got {elem}{val} in '{formula}'."
            )
        counts[elem] = int(val)
    return counts


def parse_system(system: str) -> list[str]:
    elements = [e.strip() for e in system.split("-") if e.strip()]
    if not elements:
        raise ValueError(f"Invalid system string: '{system}'.")

    for e in elements:
        if e not in atomic_numbers:
            raise ValueError(f"Invalid element symbol: '{e}' in system '{system}'.")

    return sorted(set(elements))


def format_system(elements: list[str]) -> str:
    return "-".join(sorted(set(elements)))


def format_formula(formula: dict[str, int]) -> str:
    return "".join(f"{el}{n}" if n > 1 else el for el, n in formula.items())


# TODO: use DateTime?
def format_timestamp(ts: str) -> str:
    return ts[:16].replace("T", " ")


def free_cuda_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@contextmanager
def workdir_context(workdir: str | None) -> Generator[Path, None, None]:
    if workdir is None:
        with tempfile.TemporaryDirectory(suffix=APP_TMPDIR_SUFFIX) as td:
            yield Path(td)
    else:
        path = Path(workdir)
        path.mkdir(parents=True, exist_ok=True)
        yield path.resolve()
