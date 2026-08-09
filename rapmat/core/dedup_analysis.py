import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.spatial.distance import pdist, squareform

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
except ImportError:
    StructureMatcher = None  # type: ignore[assignment,misc]

from rapmat.core.entities import Structure
from rapmat.utils.console import get_logger
from rapmat.utils.progress import ProgressCallback


def _to_pymatgen(atoms: Atoms):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AseAtomsAdaptor.get_structure(atoms)


def forces_cosine_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    a = f1.ravel()
    b = f2.ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0 if (norm_a < 1e-12 and norm_b < 1e-12) else 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class DedupSimulationResult:

    total: int = 0
    kept: int = 0

    dropped_by_vector: int = 0
    rescued_by_pymatgen: int = 0
    rescued_by_forces: int = 0
    final_dropped: int = 0

    pymatgen_comparisons: int = 0
    pymatgen_mismatches: int = 0
    force_comparisons: int = 0
    force_mismatches: int = 0

    kept_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Shared pipeline defaults
# --------------------------------------------------------------------------- #
DEFAULT_SOAP_R_CUT = 6.0
DEFAULT_SOAP_N_MAX = 8
DEFAULT_SOAP_L_MAX = 6
DEFAULT_SOAP_SIGMA = 1.0
DEFAULT_SURVIVAL_TARGETS = [95, 90, 75, 50, 25, 10, 5]

@dataclass(frozen=True)
class MetricSpec:
    choice: str
    short: str
    axis: str
    hint: str
    default_threshold: float


METRICS: dict[str, MetricSpec] = {
    "euclidean": MetricSpec(
        choice="L2",
        short="L2",
        axis="L2 Distance",
        hint="Using scale-dependent L2 distance",
        default_threshold=1e-2,
    ),
    "cosine": MetricSpec(
        choice="L2+norm",
        short="L2+norm",
        axis="Cosine Distance (1-cos)",
        hint="Using scale-independent L2 (1 - cos) on the normalized set",
        default_threshold=1e-7,
    ),
}

METRIC_BY_CHOICE = {spec.choice: key for key, spec in METRICS.items()}


class DedupAnalysisError(Exception):
    """A run cannot be analysed."""


def compute_pairwise_distances(
    vectors: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    return pdist(vectors, metric=metric)


@dataclass
class PreparedDistances:
    with_vec: list["Structure"]
    dist_sq: np.ndarray
    condensed: np.ndarray


def _sorted_with_vectors(structures: list) -> list:
    with_vec = [s for s in structures if s.descriptor is not None]
    with_vec.sort(key=lambda s: s.energy_per_atom)
    return with_vec


def prepare_distances(
    structures: list, metric: str = "euclidean"
) -> PreparedDistances:
    with_vec = _sorted_with_vectors(structures)
    if not with_vec:
        return PreparedDistances(
            with_vec=[], dist_sq=np.empty((0, 0)), condensed=np.empty((0,))
        )
    mat = np.vstack([s.descriptor for s in with_vec])
    condensed = pdist(mat, metric=metric)
    return PreparedDistances(with_vec, squareform(condensed), condensed)


def simulate_deduplication(
    structures: list,
    *,
    threshold: float = 1e-2,
    use_pymatgen: bool = False,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
    use_forces: bool = False,
    force_cosine_threshold: float = 0.95,
    metric: str = "euclidean",
    dist_sq: np.ndarray | None = None,
    progress_callback: ProgressCallback | None = None,
) -> DedupSimulationResult:
    result = DedupSimulationResult(total=len(structures))

    with_vec = _sorted_with_vectors(structures)
    if not with_vec:
        result.kept = result.total
        result.kept_ids = [s.id for s in structures]
        return result

    N = len(with_vec)
    if dist_sq is None:
        mat = np.vstack([s.descriptor for s in with_vec])
        dist_sq = squareform(pdist(mat, metric=metric))

    matcher = None
    if use_pymatgen and StructureMatcher is not None:
        matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)

    dropped = set()

    for i in range(N):
        if progress_callback:
            progress_callback(i, N, f"Dedup: {i}/{N}", is_log=False)

        if i in dropped:
            continue

        mask = dist_sq[i] < threshold
        mask[i] = False
        neighbours_idx = np.where(mask)[0]

        for j in neighbours_idx:
            if j in dropped or j <= i:
                continue

            result.dropped_by_vector += 1
            confirmed = True

            if matcher is not None:
                result.pymatgen_comparisons += 1
                try:
                    pmg_i = _to_pymatgen(with_vec[i].atoms)
                    pmg_j = _to_pymatgen(with_vec[j].atoms)
                    if not matcher.fit(pmg_i, pmg_j):
                        result.pymatgen_mismatches += 1
                        result.rescued_by_pymatgen += 1
                        result.dropped_by_vector -= 1
                        confirmed = False
                except Exception as exc:
                    get_logger("rapmat.dedup").warning(
                        "pymatgen comparison failed for pair (%s, %s), "
                        "treating as non-duplicate: %s",
                        with_vec[i].id, with_vec[j].id, exc,
                    )
                    result.pymatgen_mismatches += 1
                    result.rescued_by_pymatgen += 1
                    result.dropped_by_vector -= 1
                    confirmed = False

            if confirmed and use_forces:
                f_i = with_vec[i].forces
                f_j = with_vec[j].forces
                if (
                    f_i is not None
                    and f_j is not None
                    and np.shape(f_i) == np.shape(f_j)
                ):
                    result.force_comparisons += 1
                    cos_sim = forces_cosine_similarity(f_i, f_j)
                    if cos_sim < force_cosine_threshold:
                        result.force_mismatches += 1
                        result.rescued_by_forces += 1
                        result.dropped_by_vector -= 1
                        confirmed = False
                else:
                    result.rescued_by_forces += 1
                    result.dropped_by_vector -= 1
                    confirmed = False

            if confirmed:
                dropped.add(j)

    if progress_callback:
        progress_callback(N, N, f"Dedup: {N}/{N}", is_log=False)

    result.final_dropped = len(dropped)
    result.kept = N - len(dropped)
    result.kept_ids = [with_vec[i].id for i in range(N) if i not in dropped]
    result.dropped_ids = [with_vec[i].id for i in sorted(dropped)]

    no_vec = [s for s in structures if s.descriptor is None]
    result.kept += len(no_vec)
    result.kept_ids.extend(s.id for s in no_vec)
    result.total = len(structures)

    return result


def _greedy_dedup_count(dist_sq: np.ndarray, threshold: float) -> int:
    N = dist_sq.shape[0]
    alive = np.ones(N, dtype=bool)
    for i in range(N):
        if not alive[i]:
            continue

        neigh = dist_sq[i] < threshold
        neigh[: i + 1] = False
        alive[neigh] = False
    return int(alive.sum())


def survival_thresholds(
    dist_sq: np.ndarray,
    condensed: np.ndarray,
    ratios: list[float],
) -> list[tuple[float, int]]:
    """For each ratio, the distance threshold whose greedy deduplication keeps ~that
    fraction of structures (binary search over the observed distances)."""
    N = int(dist_sq.shape[0])
    if N == 0:
        return [(0.0, 0) for _ in ratios]

    candidates = np.unique(condensed[condensed > 0.0])
    M = len(candidates)

    cache: dict[int, int] = {}

    def kept_at(k: int) -> int:
        v = cache.get(k)
        if v is None:
            v = _greedy_dedup_count(dist_sq, float(candidates[k]))
            cache[k] = v
        return v

    eps_kept = _greedy_dedup_count(dist_sq, 1e-12) if M == 0 else None

    results: list[tuple[float, int]] = []
    for ratio in ratios:
        if ratio >= 1.0:
            results.append((0.0, N))
            continue
        if M == 0:
            results.append((1e-12, int(eps_kept)))
            continue

        target = max(1, int(round(N * ratio)))

        if kept_at(M - 1) > target:
            results.append((float(candidates[M - 1]), kept_at(M - 1)))
            continue

        lo, hi = 0, M - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if kept_at(mid) <= target:
                hi = mid
            else:
                lo = mid + 1

        k = lo
        kept_k = kept_at(k)
        if k > 0:
            kept_prev = kept_at(k - 1)
            if abs(kept_prev - target) <= abs(kept_k - target):
                k, kept_k = k - 1, kept_prev
        results.append((float(candidates[k]), kept_k))

    return results


def find_threshold_for_survival(
    structures: list,
    target_survival_ratio: float,
) -> tuple[float, int]:
    with_vec = [s for s in structures if s.descriptor is not None]
    if not with_vec:
        return 0.0, len(structures)
    if target_survival_ratio >= 1.0:
        return 0.0, len(with_vec)

    prep = prepare_distances(structures)
    return survival_thresholds(
        prep.dist_sq, prep.condensed, [target_survival_ratio]
    )[0]


@dataclass
class DedupAnalysis:
    """Outcome of a full dedup analysis for one run.
    """

    run_name: str
    stage: str
    threshold: float
    metric: str
    structures: list
    prep: PreparedDistances
    sim: DedupSimulationResult
    distances: np.ndarray
    percentiles: list[tuple[int, float, int]]

    n_structs: int
    n_pairs: int
    min_dist: float
    max_dist: float
    mean_dist: float
    median_dist: float
    std_dist: float
    below_thresh: int

    use_pymatgen: bool
    use_forces: bool
    ltol: float
    stol: float
    angle_tol: float
    force_cosine_threshold: float

    def simulate_at(
        self,
        threshold: float,
        progress_callback: ProgressCallback | None = None,
    ) -> DedupSimulationResult:
        """Re-run the dedup simulation at ``threshold``, reusing descriptors."""
        return simulate_deduplication(
            self.structures,
            threshold=threshold,
            use_pymatgen=self.use_pymatgen,
            ltol=self.ltol,
            stol=self.stol,
            angle_tol=self.angle_tol,
            use_forces=self.use_forces,
            force_cosine_threshold=self.force_cosine_threshold,
            metric=self.metric,
            dist_sq=self.prep.dist_sq,
            progress_callback=progress_callback,
        )

    def to_result_dict(self) -> dict:
        """For the TUI dedup screen."""
        return {
            "n_structs": self.n_structs,
            "n_pairs": self.n_pairs,
            "min_dist": self.min_dist,
            "max_dist": self.max_dist,
            "mean_dist": self.mean_dist,
            "median_dist": self.median_dist,
            "std_dist": self.std_dist,
            "below_thresh": self.below_thresh,
            "threshold": self.threshold,
            "metric": self.metric,
            "sim": self.sim,
            "percentiles": self.percentiles,
            "distances": self.distances,
            "stage": self.stage,
            "run_name": self.run_name,
            "use_pymatgen": self.use_pymatgen,
            "use_forces": self.use_forces,
        }


def run_dedup_analysis(
    store,
    run_name: str,
    *,
    stage: str = "relaxed",
    threshold: float = 1e-2,
    metric: str = "euclidean",
    soap_r_cut: float = DEFAULT_SOAP_R_CUT,
    soap_n_max: int = DEFAULT_SOAP_N_MAX,
    soap_l_max: int = DEFAULT_SOAP_L_MAX,
    soap_sigma: float = DEFAULT_SOAP_SIGMA,
    use_pymatgen: bool = False,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
    use_forces: bool = False,
    force_cosine_threshold: float = 0.95,
    survival_targets: Optional[list[int]] = None,
    progress_callback: ProgressCallback | None = None,
) -> "DedupAnalysis":
    """Run the full dedup pipeline for one run.
    Raises :class:`DedupAnalysisError` when the run cannot be processed.
    """
    from rapmat.storage import SOAPDescriptor
    from rapmat.storage.status import StructureStatus

    if survival_targets is None:
        survival_targets = DEFAULT_SURVIVAL_TARGETS

    if metric not in METRICS:
        raise DedupAnalysisError(f"Unknown distance metric '{metric}'")

    def _emit(current: int, total: int, message: str, is_log: bool = False) -> None:
        if progress_callback:
            progress_callback(current, total, message, is_log=is_log)

    meta = store.get_run_metadata(run_name)
    if meta is None:
        raise DedupAnalysisError(f"Run '{run_name}' not found")

    elements = list(meta.search_config.formula.keys())
    if not elements:
        raise DedupAnalysisError("Cannot determine species from run config")

    descriptor = SOAPDescriptor(
        species=elements,
        r_cut=soap_r_cut,
        n_max=int(soap_n_max),
        l_max=int(soap_l_max),
        sigma=soap_sigma,
    )

    statuses = (
        (StructureStatus.GENERATED,)
        if stage == "candidates"
        else (StructureStatus.RELAXED,)
    )

    _emit(0, 5, f"Loading {stage} structures...", is_log=True)
    structures = store.get_structures_for_analysis(run_name, statuses=statuses)
    if not structures:
        raise DedupAnalysisError(f"No {stage} structures found")
    if len(structures) < 2:
        raise DedupAnalysisError("Need >= 2 structures for analysis")

    _emit(1, 5, f"Loaded {len(structures)} structures, computing vectors...",
          is_log=True)
    for s in structures:
        s.descriptor = descriptor.compute(s.atoms)

    _emit(2, 5, "Computing pairwise distances")
    prep = prepare_distances(structures, metric=metric)
    distances = prep.condensed

    _emit(3, 5, "Simulating deduplication", is_log=True)
    sim = simulate_deduplication(
        structures,
        threshold=threshold,
        use_pymatgen=use_pymatgen,
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        use_forces=use_forces,
        force_cosine_threshold=force_cosine_threshold,
        metric=metric,
        dist_sq=prep.dist_sq,
        progress_callback=progress_callback,
    )

    _emit(4, 5, "Computing survival thresholds")
    surv = survival_thresholds(
        prep.dist_sq, prep.condensed, [p / 100.0 for p in survival_targets]
    )
    percentiles = [(p, t, k) for p, (t, k) in zip(survival_targets, surv)]

    n_pairs = len(distances)
    below = int(np.sum(distances < threshold)) if n_pairs else 0

    return DedupAnalysis(
        run_name=run_name,
        stage=stage,
        threshold=threshold,
        metric=metric,
        structures=structures,
        prep=prep,
        sim=sim,
        distances=distances,
        percentiles=percentiles,
        n_structs=len(structures),
        n_pairs=n_pairs,
        min_dist=float(np.min(distances)) if n_pairs else 0.0,
        max_dist=float(np.max(distances)) if n_pairs else 0.0,
        mean_dist=float(np.mean(distances)) if n_pairs else 0.0,
        median_dist=float(np.median(distances)) if n_pairs else 0.0,
        std_dist=float(np.std(distances)) if n_pairs else 0.0,
        below_thresh=below,
        use_pymatgen=use_pymatgen,
        use_forces=use_forces,
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        force_cosine_threshold=force_cosine_threshold,
    )


def plot_distance_histogram(
    distances: np.ndarray,
    *,
    threshold: Optional[float] = None,
    save_path: Path | str | None = None,
    title: str = "Pairwise Descriptor Distance Distribution",
    axis_label: str = "L2 Distance",
    bins: int = 200,
) -> None:
    import matplotlib

    if save_path is not None:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    dist_sq = squareform(distances)
    np.fill_diagonal(dist_sq, np.inf)
    closest_neighbor = np.min(dist_sq, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.hist(distances, bins=bins, edgecolor="none", alpha=0.75)
    ax1.set_xlabel(axis_label)
    ax1.set_ylabel("Pair Count")
    ax1.set_title(title)

    if threshold is not None:
        ax1.axvline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"threshold = {threshold}",
        )
        ax1.legend()

    ax2.hist(
        closest_neighbor,
        bins=min(bins, len(closest_neighbor) // 2 or 1),
        edgecolor="none",
        alpha=0.75,
    )
    ax2.set_xlabel(f"{axis_label} to Closest Neighbor")
    ax2.set_ylabel("Structure Count")
    ax2.set_title("Closest Neighbor Distance Distribution")

    if threshold is not None:
        ax2.axvline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"threshold = {threshold}",
        )
        ax2.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
    else:
        plt.show()
