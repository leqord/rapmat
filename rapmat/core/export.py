"""Structure and results-table export helpers.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from ase import Atoms


def save_structure(
    atoms: Atoms,
    directory: str | Path,
    ident: str,
    fmt: str,
    *,
    cell_mode: str = "As-is",
    symprec: float = 1e-3,
    prefix: str = "",
) -> Path:
    """Write one structure as ``{prefix}structure_{ident}.{fmt}`` in ``directory``.

    ``cell_mode``: "As-is" | "Conventional" | "Primitive". 
    Returns the written path.
    """
    from ase.io import write as write_ase_structure

    from rapmat.utils.structure import standardize_atoms

    if cell_mode == "Conventional":
        atoms = standardize_atoms(atoms, symprec=symprec)
    elif cell_mode == "Primitive":
        atoms = standardize_atoms(atoms, symprec=symprec, to_primitive=True)

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}structure_{ident}.{fmt}"
    write_ase_structure(str(out_path), atoms)
    return out_path


def export_results_table(
    headers: list[str],
    rows: list[list],
    directory: str | Path,
    fmt: str,
) -> Path:
    """Write ``results_table.{fmt}`` from precomputed headers/rows.

    ``fmt`` is "txt" or "csv". Returns the path.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_table.{fmt}"

    if fmt == "txt":
        from tabulate import tabulate

        out_path.write_text(
            tabulate(rows, headers=headers, tablefmt="simple"), encoding="utf-8"
        )
    elif fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(headers)
        writer.writerows(rows)
        out_path.write_text(buf.getvalue(), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported results-table format: {fmt!r}")
    return out_path
