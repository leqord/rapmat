"""Tests for the dedup analysis / simulation module."""

import numpy as np
import pytest
from ase.build import bulk
from conftest import add_generated_candidate

from rapmat.core.dedup_analysis import (DedupAnalysisError,
                                        _greedy_dedup_count,
                                        compute_pairwise_distances,
                                        energy_merge_mask,
                                        find_threshold_for_survival,
                                        plot_distance_histogram,
                                        prepare_distances,
                                        run_dedup_analysis,
                                        simulate_deduplication,
                                        survival_thresholds)
from rapmat.core.entities import Structure
from rapmat.storage import SOAPDescriptor, SQLiteStore


def _make_entry(struct_id, atoms, energy, vector, forces=None):
    a = atoms.copy()
    if forces is not None:
        a.info["forces"] = forces
    return Structure(
        id=struct_id,
        status="relaxed",
        final_atoms=a,
        energy_per_atom=energy,
        descriptor=vector,
    )


# ------------------------------------------------------------------ #
#  compute_pairwise_distances
# ------------------------------------------------------------------ #


class TestPairwiseDistances:
    def test_identical_vectors(self):
        vecs = np.array([[1.0, 0.0], [1.0, 0.0]])
        d = compute_pairwise_distances(vecs, metric="euclidean")
        assert len(d) == 1
        assert d[0] == pytest.approx(0.0)

    def test_known_distance(self):
        vecs = np.array([[0.0, 0.0], [3.0, 4.0]])
        d = compute_pairwise_distances(vecs, metric="euclidean")
        assert d[0] == pytest.approx(5.0)

    def test_three_vectors(self):
        vecs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        d = compute_pairwise_distances(vecs, metric="euclidean")
        assert len(d) == 3

    def test_default_metric_is_cosine(self):
        vecs = np.array([[1.0, 0.0], [2.0, 0.0]])   # same direction, 2x length
        assert compute_pairwise_distances(vecs)[0] == pytest.approx(0.0)
        assert compute_pairwise_distances(
            vecs, metric="euclidean")[0] == pytest.approx(1.0)


# ------------------------------------------------------------------ #
#  simulate_deduplication
# ------------------------------------------------------------------ #


class TestSimulateDedup:
    @pytest.fixture
    def soap(self):
        return SOAPDescriptor(species=["Cu"], n_max=4, l_max=3)

    def test_empty_input(self):
        result = simulate_deduplication([], threshold=0.1)
        assert result.total == 0
        assert result.kept == 0

    def test_no_vectors(self):
        cu = bulk("Cu", "fcc", a=3.615)
        entries = [_make_entry("s/1", cu, -3.0, None)]
        result = simulate_deduplication(entries, threshold=0.1)
        assert result.kept == 1
        assert result.final_dropped == 0

    def test_identical_structures_dropped(self, soap):
        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        entries = [
            _make_entry("s/1", cu, -3.0, vec.copy()),
            _make_entry("s/2", cu, -2.0, vec.copy()),
        ]
        result = simulate_deduplication(entries, threshold=1.0)
        assert result.kept == 1
        assert result.final_dropped == 1
        assert "s/1" in result.kept_ids

    def test_distinct_structures_kept(self, soap):
        cu_fcc = bulk("Cu", "fcc", a=3.615)
        cu_bcc = bulk("Cu", "bcc", a=2.87)
        vec_fcc = soap.compute(cu_fcc)
        vec_bcc = soap.compute(cu_bcc)
        entries = [
            _make_entry("s/1", cu_fcc, -3.0, vec_fcc),
            _make_entry("s/2", cu_bcc, -2.5, vec_bcc),
        ]
        result = simulate_deduplication(
            entries, threshold=1e-5, metric="euclidean"
        )
        assert result.kept == 2
        assert result.final_dropped == 0

    def test_pymatgen_rescues_collision(self, soap):
        cu_fcc = bulk("Cu", "fcc", a=3.615)
        cu_bcc = bulk("Cu", "bcc", a=2.87)
        vec = soap.compute(cu_fcc)
        entries = [
            _make_entry("s/1", cu_fcc, -3.0, vec.copy()),
            _make_entry("s/2", cu_bcc, -2.5, vec.copy()),
        ]
        result = simulate_deduplication(
            entries,
            threshold=999.0,
            use_pymatgen=True,
        )
        assert result.kept == 2
        assert result.rescued_by_pymatgen >= 1

    def test_forces_rescue_different_gradients(self, soap):
        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        f1 = np.array([[1.0, 0.0, 0.0]])
        f2 = np.array([[0.0, 1.0, 0.0]])
        entries = [
            _make_entry("s/1", cu, -3.0, vec.copy(), forces=f1),
            _make_entry("s/2", cu, -2.5, vec.copy(), forces=f2),
        ]
        result = simulate_deduplication(
            entries,
            threshold=999.0,
            use_forces=True,
            force_cosine_threshold=0.95,
        )
        assert result.kept == 2
        assert result.rescued_by_forces >= 1

    def test_progress_callback(self, soap):
        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        entries = [_make_entry("s/1", cu, -3.0, vec)]
        calls = []
        simulate_deduplication(
            entries,
            threshold=0.1,
            progress_callback=lambda c, t, m, is_log=False: calls.append((c, t)),
        )
        assert len(calls) > 0

    def test_lower_energy_kept(self, soap):
        """The structure with lower energy is kept when two are identical."""
        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        entries = [
            _make_entry("s/high", cu, -1.0, vec.copy()),
            _make_entry("s/low", cu, -5.0, vec.copy()),
        ]
        result = simulate_deduplication(entries, threshold=1.0)
        assert "s/low" in result.kept_ids
        assert "s/high" in result.dropped_ids


# ------------------------------------------------------------------ #
#  Distance metric and SOAP sigma
# ------------------------------------------------------------------ #


class TestMetricAndSigma:
    def test_sigma_changes_descriptor(self):
        cu = bulk("Cu", "fcc", a=3.615)
        broad = SOAPDescriptor(species=["Cu"], n_max=4, l_max=3).compute(cu)
        narrow = SOAPDescriptor(
            species=["Cu"], n_max=4, l_max=3, sigma=0.5
        ).compute(cu)
        assert broad.shape == narrow.shape
        assert np.all(np.isfinite(broad))
        assert np.all(np.isfinite(narrow))
        assert not np.allclose(broad, narrow)

    def test_cosine_is_scale_invariant(self):
        cu = bulk("Cu", "fcc", a=3.615)
        v = np.array([3.0, 4.0])
        entries = [
            _make_entry("s/1", cu, -3.0, v),
            _make_entry("s/2", cu, -2.0, 2.0 * v),
        ]
        cos = simulate_deduplication(entries, threshold=1e-6, metric="cosine")
        assert cos.kept == 1
        l2 = simulate_deduplication(entries, threshold=1e-6, metric="euclidean")
        assert l2.kept == 2

    def test_cosine_distances_bounded(self):
        cu = bulk("Cu", "fcc", a=3.615)
        vecs = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([-1.0, 0.0]),
        ]
        entries = [
            _make_entry(f"s/{i}", cu, -3.0 + i, v) for i, v in enumerate(vecs)
        ]
        prep = prepare_distances(entries, metric="cosine")
        assert prep.condensed.min() >= 0.0
        assert prep.condensed.max() <= 2.0 + 1e-12
        assert np.max(prep.condensed) == pytest.approx(2.0)

    def test_identical_vectors_merge_under_both_metrics(self):
        cu = bulk("Cu", "fcc", a=3.615)
        v = np.array([1.0, 2.0, 3.0])
        for metric in ("euclidean", "cosine"):
            entries = [
                _make_entry("s/1", cu, -3.0, v.copy()),
                _make_entry("s/2", cu, -2.0, v.copy()),
            ]
            result = simulate_deduplication(
                entries, threshold=1e-6, metric=metric
            )
            assert result.kept == 1, metric

    def test_unknown_metric_rejected(self):
        with pytest.raises(DedupAnalysisError):
            run_dedup_analysis(None, "whatever", metric="chebyshev")


# ------------------------------------------------------------------ #
#  Energy window
# ------------------------------------------------------------------ #


class TestEnergyWindow:
    @pytest.fixture
    def identical_pair(self):
        cu = bulk("Cu", "fcc", a=3.615)
        v = np.array([1.0, 2.0, 3.0])
        return cu, v

    def test_window_blocks_merge_of_distant_energies(self, identical_pair):
        cu, v = identical_pair
        entries = [
            _make_entry("s/1", cu, -3.0, v.copy()),
            _make_entry("s/2", cu, -2.0, v.copy()),
        ]
        merged = simulate_deduplication(entries, threshold=1.0)
        assert merged.kept == 1

        kept = simulate_deduplication(entries, threshold=1.0, energy_window=1e-3)
        assert kept.kept == 2
        assert kept.rescued_by_energy == 1
        assert kept.energy_mismatches == 1
        assert kept.energy_comparisons == 1

    def test_window_allows_merge_within_window(self, identical_pair):
        cu, v = identical_pair
        entries = [
            _make_entry("s/1", cu, -3.0, v.copy()),
            _make_entry("s/2", cu, -3.0 + 1e-6, v.copy()),
        ]
        res = simulate_deduplication(entries, threshold=1.0, energy_window=1e-3)
        assert res.kept == 1
        assert res.rescued_by_energy == 0
        assert res.energy_comparisons == 1

    def test_disabled_window_matches_current_behaviour(self, identical_pair):
        cu, v = identical_pair
        entries = [
            _make_entry("s/1", cu, -3.0, v.copy()),
            _make_entry("s/2", cu, -2.0, v.copy()),
        ]
        base = simulate_deduplication(entries, threshold=1.0)
        same = simulate_deduplication(entries, threshold=1.0, energy_window=None)
        assert (base.kept, base.final_dropped) == (same.kept, same.final_dropped)
        assert same.energy_comparisons == 0
        assert same.rescued_by_energy == 0

    def test_waterfall_arithmetic_balances(self, identical_pair):
        cu, v = identical_pair
        entries = [
            _make_entry("s/1", cu, -3.0, v.copy()),
            _make_entry("s/2", cu, -2.0, v.copy()),
        ]
        r = simulate_deduplication(entries, threshold=1.0, energy_window=1e-3)
        assert r.total - r.dropped_by_vector == r.kept

    def test_energy_runs_before_pymatgen(self, identical_pair):
        cu, v = identical_pair
        entries = [
            _make_entry("s/1", cu, -3.0, v.copy()),
            _make_entry("s/2", cu, -2.0, v.copy()),
        ]
        r = simulate_deduplication(
            entries, threshold=1.0, energy_window=1e-3, use_pymatgen=True
        )
        assert r.energy_comparisons == 1
        assert r.pymatgen_comparisons == 0


class TestEnergyMergeMask:
    def test_symmetric_with_true_diagonal(self):
        m = energy_merge_mask([0.0, 0.5, 1.0], 0.6)
        assert np.array_equal(m, m.T)
        assert m.diagonal().all()

    def test_boundary_is_inclusive(self):
        m = energy_merge_mask([0.0, 1.0], 1.0)
        assert m[0, 1]
        assert not energy_merge_mask([0.0, 1.0], 0.999999)[0, 1]

    def test_greedy_count_respects_mask(self):
        dist = np.zeros((4, 4))
        assert _greedy_dedup_count(dist, 1.0) == 1
        forbid = np.eye(4, dtype=bool)
        assert _greedy_dedup_count(dist, 1.0, forbid) == 4

    def test_survival_thresholds_keep_more_with_mask(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 4))
        from scipy.spatial.distance import pdist, squareform
        cond = pdist(X)
        sq = squareform(cond)
        energies = np.linspace(0, 1, 30)
        mask = energy_merge_mask(energies, 0.01)

        (_t0, k0), = survival_thresholds(sq, cond, [0.25])
        (_t1, k1), = survival_thresholds(sq, cond, [0.25], merge_ok=mask)
        assert k1 >= k0


# ------------------------------------------------------------------ #
#  find_threshold_for_survival
# ------------------------------------------------------------------ #


class TestFindThresholdForSurvival:
    @pytest.fixture
    def soap(self):
        return SOAPDescriptor(species=["Cu"], n_max=4, l_max=3)

    def test_full_survival(self, soap):
        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        entries = [_make_entry("s/1", cu, -3.0, vec)]
        thresh, kept = find_threshold_for_survival(entries, 1.0)
        assert kept == 1
        assert thresh == 0.0

    def test_no_vectors(self):
        cu = bulk("Cu", "fcc", a=3.615)
        entries = [_make_entry("s/1", cu, -3.0, None)]
        thresh, kept = find_threshold_for_survival(entries, 0.5)
        assert kept == 1

    def test_identical_structures_low_survival(self, soap):
        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        entries = [
            _make_entry(f"s/{i}", cu, -3.0 + i * 0.1, vec.copy()) for i in range(10)
        ]
        thresh, kept = find_threshold_for_survival(entries, 0.1)
        assert kept <= 3

    def test_distinct_structures_high_survival(self, soap):
        cu_fcc = bulk("Cu", "fcc", a=3.615)
        cu_bcc = bulk("Cu", "bcc", a=2.87)
        vec_fcc = soap.compute(cu_fcc)
        vec_bcc = soap.compute(cu_bcc)
        entries = [
            _make_entry("s/fcc", cu_fcc, -3.0, vec_fcc),
            _make_entry("s/bcc", cu_bcc, -2.5, vec_bcc),
        ]
        thresh, kept = find_threshold_for_survival(entries, 0.95)
        assert kept == 2

    def test_monotonic_survival(self, soap):
        """Lower survival target requires a larger (or equal) threshold."""
        entries = []
        for i in range(30):
            cu = bulk("Cu", "fcc", a=3.615 + i * 0.05)
            vec = soap.compute(cu)
            entries.append(_make_entry(f"s/{i}", cu, -3.0 + i * 0.1, vec))
        prev_thresh = -1.0
        for target in [0.9, 0.5, 0.1]:
            thresh, _ = find_threshold_for_survival(entries, target)
            assert thresh >= prev_thresh - 1e-6
            prev_thresh = thresh


# ------------------------------------------------------------------ #
#  plot_distance_histogram
# ------------------------------------------------------------------ #


class TestPlotHistogram:
    def test_saves_png(self, tmp_path):
        distances = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 0.8])
        out = tmp_path / "hist.png"
        plot_distance_histogram(distances, threshold=0.5, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_saves_svg(self, tmp_path):
        distances = np.array([0.1, 0.5, 1.0])
        out = tmp_path / "hist.svg"
        plot_distance_histogram(distances, save_path=out)
        assert out.exists()

    def test_axis_label(self, tmp_path):
        distances = np.array([0.1, 0.5, 1.0])
        out = tmp_path / "hist_cos.png"
        plot_distance_histogram(
            distances, save_path=out, axis_label="Cosine Distance (1-cos)"
        )
        assert out.exists()
        assert out.stat().st_size > 0


# ------------------------------------------------------------------ #
#  Store integration: get_structures_for_analysis
# ------------------------------------------------------------------ #


class TestStoreAnalysis:
    @pytest.fixture
    def soap(self):
        return SOAPDescriptor(species=["Cu"], n_max=4, l_max=3)

    def test_returns_relaxed_structures(self, tmp_path, soap):
        store = SQLiteStore.from_path(tmp_path / "analysis_db")
        store.create_study(
            study_id="run1",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="run1", study_id="run1")

        cu = bulk("Cu", "fcc", a=3.615)
        add_generated_candidate(store, "run1", "run1/1", cu)
        store.update_structure(
            "run1/1",
            "relaxed",
            atoms=cu,
            metadata={"energy_per_atom": -3.0, "converged": True},
        )

        results = store.get_structures_for_analysis("run1", statuses=("relaxed",))
        assert len(results) == 1
        assert results[0].energy_per_atom == pytest.approx(-3.0)
        assert results[0].descriptor is None
        assert results[0].atoms is not None

    def test_returns_generated_structures(self, tmp_path, soap):
        store = SQLiteStore.from_path(tmp_path / "analysis_gen_db")
        store.create_study(
            study_id="run1",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="run1", study_id="run1")

        cu = bulk("Cu", "fcc", a=3.615)
        add_generated_candidate(store, "run1", "run1/1", cu)

        results = store.get_structures_for_analysis("run1", statuses=("generated",))
        assert len(results) == 1
        assert results[0].id == "run1/1"

    def test_filters_by_status(self, tmp_path, soap):
        store = SQLiteStore.from_path(tmp_path / "analysis_filter_db")
        store.create_study(
            study_id="run1",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="run1", study_id="run1")

        cu = bulk("Cu", "fcc", a=3.615)
        add_generated_candidate(store, "run1", "run1/1", cu)
        store.update_structure(
            "run1/1",
            "relaxed",
            atoms=cu,
            metadata={"energy_per_atom": -3.0, "converged": True},
        )

        cu2 = bulk("Cu", "fcc", a=3.62)
        add_generated_candidate(store, "run1", "run1/2", cu2)

        relaxed = store.get_structures_for_analysis("run1", statuses=("relaxed",))
        assert len(relaxed) == 1

        generated = store.get_structures_for_analysis("run1", statuses=("generated",))
        assert len(generated) == 1

        both = store.get_structures_for_analysis(
            "run1", statuses=("generated", "relaxed")
        )
        assert len(both) == 2

    def _store_two_relaxed_with_forces(self, tmp_path, soap, f1, f2):
        """Persist two same-vector relaxed structures carrying the given forces,
        reload them via the store, and attach SOAP vectors.

        Returns the reloaded structure dicts ready for ``simulate_deduplication``.
        """
        store = SQLiteStore.from_path(tmp_path / "forces_db")
        store.create_study(
            study_id="run1",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="run1", study_id="run1")

        cu = bulk("Cu", "fcc", a=3.615)
        for sid, energy, forces in (("run1/1", -3.0, f1), ("run1/2", -2.5, f2)):
            atoms = cu.copy()
            atoms.info["forces"] = forces
            add_generated_candidate(store, "run1", sid, atoms)
            store.update_structure(
                sid,
                "relaxed",
                atoms=atoms,
                metadata={"energy_per_atom": energy, "converged": True},
            )

        results = store.get_structures_for_analysis("run1", statuses=("relaxed",))
        assert all(r.forces is not None for r in results)
        for r in results:
            r.descriptor = soap.compute(r.atoms)
        return results

    def test_forces_confirm_drop_via_store(self, tmp_path, soap):
        """Same-vector structures with parallel forces are confirmed duplicates."""
        cu = bulk("Cu", "fcc", a=3.615)
        parallel = np.tile([0.1, 0.0, 0.0], (len(cu), 1))
        results = self._store_two_relaxed_with_forces(
            tmp_path, soap, parallel.copy(), parallel.copy()
        )
        sim = simulate_deduplication(
            results, threshold=1.0, use_forces=True, force_cosine_threshold=0.95
        )
        assert sim.force_comparisons >= 1
        assert sim.kept == 1

    def test_forces_rescue_via_store(self, tmp_path, soap):
        """Same-vector structures with opposing forces are rescued (kept)."""
        cu = bulk("Cu", "fcc", a=3.615)
        forward = np.tile([0.1, 0.0, 0.0], (len(cu), 1))
        backward = np.tile([-0.1, 0.0, 0.0], (len(cu), 1))
        results = self._store_two_relaxed_with_forces(
            tmp_path, soap, forward, backward
        )
        sim = simulate_deduplication(
            results, threshold=1.0, use_forces=True, force_cosine_threshold=0.95
        )
        assert sim.force_comparisons >= 1
        assert sim.rescued_by_forces >= 1
        assert sim.kept == 2


# ------------------------------------------------------------------ #
#  Deduplication screen
# ------------------------------------------------------------------ #


class TestDedupScreenValidation:
    def _screen(self):
        from rapmat.storage import SQLiteStore
        from rapmat.tui.app import RapmatApp
        from rapmat.tui.screens.dedup import DedupScreen
        from rapmat.tui.state import AppState

        store = SQLiteStore(":memory:")
        state = AppState(store=store)
        app = RapmatApp(state)
        screen = DedupScreen(state, app._router)
        return screen, screen.build()

    def test_renders_with_new_fields(self):
        screen, widget = self._screen()
        canv = widget.render((121, 40), focus=True)
        assert canv is not None

    def test_defaults_pass_validation(self):
        screen, _ = self._screen()
        vals = screen._form.get_values()
        vals["metric"] = "euclidean"
        assert screen._validate(vals) == []

    def test_cosine_threshold_out_of_range(self):
        screen, _ = self._screen()
        vals = screen._form.get_values()
        vals["dedup_threshold"] = 2.5

        vals["metric"] = "cosine"
        assert screen._validate(vals)

        vals["metric"] = "euclidean"
        assert screen._validate(vals) == []

    def test_form_opens_on_the_default_metric(self):
        from rapmat.core.dedup_analysis import DEFAULT_METRIC, METRICS

        screen, _ = self._screen()
        vals = screen._form.get_values()
        assert vals["metric"] == METRICS[DEFAULT_METRIC].choice
        assert (vals["dedup_threshold"]
                == pytest.approx(METRICS[DEFAULT_METRIC].default_threshold))

    def test_metric_switch_resets_threshold_to_its_default(self):
        from rapmat.core.dedup_analysis import DEFAULT_METRIC, METRICS

        other = next(k for k in METRICS if k != DEFAULT_METRIC)
        screen, _ = self._screen()
        dd = screen._form.get_widget("metric")

        for key in (other, DEFAULT_METRIC):
            dd._pick(None, dd.options.index(METRICS[key].choice))
            assert (screen._form.get_values()["dedup_threshold"]
                    == pytest.approx(METRICS[key].default_threshold)), key

    def test_each_metric_default_passes_validation(self):
        from rapmat.core.dedup_analysis import METRICS

        screen, _ = self._screen()
        for key, spec in METRICS.items():
            vals = screen._form.get_values()
            vals["metric"] = key
            vals["dedup_threshold"] = spec.default_threshold
            assert screen._validate(vals) == [], key

    def test_energy_window_validated_only_when_enabled(self):
        screen, _ = self._screen()
        vals = screen._form.get_values()
        vals["metric"] = "euclidean"
        vals["energy_window"] = 0.0

        vals["energy_dedup"] = False
        assert screen._validate(vals) == []

        vals["energy_dedup"] = True
        assert any("Energy window" in e for e in screen._validate(vals))

    def test_threshold_must_be_positive(self):
        screen, _ = self._screen()
        vals = screen._form.get_values()
        vals["metric"] = "euclidean"
        vals["dedup_threshold"] = 0.0
        assert screen._validate(vals)
