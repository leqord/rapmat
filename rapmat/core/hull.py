from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

from rapmat.core.entities import ResultRow, Structure
from rapmat.storage.base import StructureStore
from rapmat.utils.common import parse_system

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def get_composition_fraction(formula: dict[str, int], element: str) -> float:
    total = sum(formula.values())
    return formula.get(element, 0) / total if total else 0.0


# ------------------------------------------------------------------ #
#  Fetching and filtering
# ------------------------------------------------------------------ #


def _effective_epa(s: Structure, use_enthalpy: bool) -> float:
    if use_enthalpy and s.enthalpy_per_atom is not None:
        return s.enthalpy_per_atom
    return s.energy_per_atom


def collect_study_structures(
    store: StructureStore,
    study_id: str,
    *,
    symprec: float = 1e-3,
) -> tuple[list[Structure], str, bool]:
    """Gather every relaxed structure across all study's runs.
    """
    study = store.get_study(study_id)
    if study is None:
        raise ValueError(f"Study '{study_id}' not found.")

    use_enthalpy = study.search_config.pressure_gpa > 0
    structures: list[Structure] = []
    for run in store.get_study_runs(study_id):
        structures.extend(
            store.get_structures(run.name, status="relaxed", symprec=symprec)
        )
    return structures, study.system, use_enthalpy


def hull_input(
    structures: list[Structure],
    *,
    hide_unconverged: bool = True,
    hide_duplicates: bool = False,
    hide_excluded: bool = True,
) -> list[Structure]:
    """The set that shapes the hull is what the table shows, except the view filters.

    View filters (e.g. thickness, EAH cutoff) never reach here.
    """
    out: list[Structure] = []
    for s in structures:
        if hide_unconverged and not s.converged:
            continue
        if hide_duplicates and s.duplicate is True:
            continue
        if hide_excluded and s.excluded:
            continue
        out.append(s)
    return out


# ------------------------------------------------------------------ #
#  Reference energies
# ------------------------------------------------------------------ #


def _is_pure(comp: Composition, element: str) -> bool:
    return len(comp.elements) == 1 and comp.elements[0].symbol == element


def _reference_structures(
    structures: list[Structure],
    elements: list[str],
    *,
    use_enthalpy: bool = False,
) -> dict[str, tuple[float, str]]:
    """For each element, the lowest-energy pure structure.
    """
    refs: dict[str, tuple[float, str]] = {}
    for el in elements:
        best: tuple[float, str] | None = None
        for s in structures:
            if not _is_pure(Composition(s.formula), el):
                continue
            epa = _effective_epa(s, use_enthalpy)
            if best is None or epa < best[0]:
                best = (epa, s.id)
        if best is None:
            raise ValueError(f"No relaxed pure-{el} structures found.")
        refs[el] = best
    return refs


def get_reference_energies(
    structures: list[Structure],
    system: str,
    *,
    use_enthalpy: bool = False,
) -> dict[str, float]:
    elements = parse_system(system)
    return {
        el: epa
        for el, (epa, _sid) in _reference_structures(
            structures, elements, use_enthalpy=use_enthalpy
        ).items()
    }


# ------------------------------------------------------------------ #
#  Phase diagram construction
# ------------------------------------------------------------------ #


def build_phase_diagram(
    structures: list[Structure],
    system: str,
    *,
    use_enthalpy: bool = False,
    show_all: bool = True,
    hull_cutoff: float = 0.0,
) -> tuple[PhaseDiagram, list[ResultRow]]:
    """Build the convex hull from the given structures.
    """
    elements = parse_system(system)
    if len(elements) < 2:
        raise ValueError(
            "Phase diagram requires a binary or larger system (2+ elements). "
            "Use build_energy_ranking() for single-element studies."
        )

    refs = _reference_structures(structures, elements, use_enthalpy=use_enthalpy)
    ref_energies = {el: epa for el, (epa, _sid) in refs.items()}
    ref_ids = {sid for _epa, sid in refs.values()}

    entries: list[PDEntry] = []
    structure_data: list[ResultRow] = []
    compositions_seen: set[str] = set()

    for s in structures:
        comp = Composition(s.formula)
        epa = _effective_epa(s, use_enthalpy)
        n_atoms = int(comp.num_atoms)
        total_value = epa * n_atoms
        entries.append(PDEntry(comp, total_value))

        e_ref = sum(comp[el_str] * ref_energies[el_str] for el_str in ref_energies)
        formation_energy = (total_value - e_ref) / n_atoms

        reduced = comp.reduced_formula
        compositions_seen.add(reduced)

        structure_data.append(
            ResultRow(
                structure=s,
                effective_per_atom=epa,
                formation_energy=formation_energy,
                reduced_formula=reduced,
                is_reference=s.id in ref_ids,
                composition_frac=(
                    comp.get_atomic_fraction(elements[1])
                    if len(elements) == 2
                    else None
                ),
            )
        )

    if not (compositions_seen - {el for el in elements}):
        raise ValueError(
            "Need at least one intermediate composition between pure endpoints. "
            "Only pure-element structures found."
        )

    pd = PhaseDiagram(entries)

    for sd in structure_data:
        comp = Composition(sd.formula)
        entry = PDEntry(comp, sd.effective_per_atom * comp.num_atoms)
        sd.energy_above_hull = float(pd.get_e_above_hull(entry))
        sd.is_stable = sd.energy_above_hull < 1e-6

    if not show_all:
        structure_data = [
            sd for sd in structure_data if sd.energy_above_hull <= hull_cutoff + 1e-9
        ]

    structure_data.sort(key=lambda d: d.composition_frac or 0.0)
    return pd, structure_data


def build_energy_ranking(
    structures: list[Structure],
    *,
    use_enthalpy: bool = False,
    show_all: bool = True,
    hull_cutoff: float = 0.0,
) -> list[ResultRow]:
    """Energy ranking for single-element systems, from the given structures.
    """
    structure_data = [
        ResultRow(
            structure=s,
            effective_per_atom=_effective_epa(s, use_enthalpy),
            reduced_formula=Composition(s.formula).reduced_formula,
        )
        for s in structures
    ]
    if not structure_data:
        return []

    structure_data.sort(key=lambda d: d.effective_per_atom)
    min_e = structure_data[0].effective_per_atom
    ref_id = structure_data[0].structure_id

    for sd in structure_data:
        sd.energy_above_hull = sd.effective_per_atom - min_e
        sd.is_stable = sd.energy_above_hull < 1e-6
        sd.is_reference = sd.structure_id == ref_id

    if not show_all:
        structure_data = [
            sd for sd in structure_data if sd.energy_above_hull <= hull_cutoff + 1e-9
        ]

    return structure_data


# ------------------------------------------------------------------ #
#  Binary hull plotting
# ------------------------------------------------------------------ #


def plot_binary_hull(
    structure_data: list[ResultRow],
    system: str,
    *,
    save_path: Optional[Path] = None,
    show: bool = True,
    use_enthalpy: bool = False,
) -> Figure:
    elements = parse_system(system)
    if len(elements) != 2:
        raise ValueError("plot_binary_hull only supports binary systems.")

    fig, ax = plt.subplots(figsize=(8, 5))

    xs = np.array([d.composition_frac for d in structure_data])
    ys = np.array([d.formation_energy for d in structure_data])
    stable = np.array([d.is_stable for d in structure_data])

    if (~stable).any():
        ax.scatter(
            xs[~stable],
            ys[~stable],
            marker="o",
            s=50,
            facecolors="none",
            edgecolors="#999999",
            linewidths=1.0,
            zorder=3,
            label="Unstable",
        )
        for x, y, eah in zip(
            xs[~stable],
            ys[~stable],
            [d.energy_above_hull for d, s in zip(structure_data, stable) if not s],
        ):
            hull_y = y - eah
            ax.plot(
                [x, x],
                [hull_y, y],
                color="#cccccc",
                linewidth=0.8,
                linestyle="--",
                zorder=1,
            )

    if stable.any():
        hull_xs = np.concatenate([[0.0], xs[stable], [1.0]])
        hull_ys = np.concatenate([[0.0], ys[stable], [0.0]])
        order = np.argsort(hull_xs)
        ax.plot(
            hull_xs[order],
            hull_ys[order],
            color="#2176ff",
            linewidth=1.5,
            zorder=2,
            label="Hull",
        )
        ax.scatter(
            xs[stable],
            ys[stable],
            marker="o",
            s=70,
            color="#2176ff",
            zorder=4,
            label="Stable",
        )
        for sd in (d for d, s in zip(structure_data, stable) if s):
            ax.annotate(
                sd.reduced_formula,
                (sd.composition_frac, sd.formation_energy),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )
    else:
        ax.plot([0, 1], [0, 0], color="#2176ff", linewidth=1.5, zorder=2)

    quantity = "enthalpy" if use_enthalpy else "energy"
    ax.axhline(0, color="black", linewidth=0.5, zorder=0)
    ax.set_xlabel(f"$x$ in {elements[0]}$_{{1-x}}${elements[1]}$_x$")
    ax.set_ylabel(f"Formation {quantity} (eV/A)")
    ax.set_title(f"Convex hull - {elements[0]}-{elements[1]}")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
