"""Unit tests for rapmat.core.evaluation metric functions."""

import pytest

from rapmat.core.entities import Evaluation, ResultRow, Structure
from rapmat.core.evaluation import (ComparisonRow, comparison_from_result_rows,
                                    compute_ranking_metrics,
                                    compute_stability_metrics,
                                    eval_rows_from_cache, select_eval_records)


def _rows(dicts):
    return [ComparisonRow(**d) for d in dicts]

# ------------------------------------------------------------------ #
#  compute_ranking_metrics
# ------------------------------------------------------------------ #


class TestRankingMetrics:
    def test_perfect_agreement(self):
        results = [
            {"mlip_epa": -5.0, "ref_epa": -5.0},
            {"mlip_epa": -4.0, "ref_epa": -4.0},
            {"mlip_epa": -3.0, "ref_epa": -3.0},
        ]
        m = compute_ranking_metrics(_rows(results), stable_only=False)
        assert m["kendall_tau"] == pytest.approx(1.0)
        assert m["mae_epa"] == pytest.approx(0.0)
        assert m["n_structures"] == 3
        assert m["stable_only_applied"] is False

    def test_reversed_ranking(self):
        results = [
            {"mlip_epa": -5.0, "ref_epa": -3.0},
            {"mlip_epa": -4.0, "ref_epa": -4.0},
            {"mlip_epa": -3.0, "ref_epa": -5.0},
        ]
        m = compute_ranking_metrics(_rows(results), stable_only=False)
        assert m["kendall_tau"] == pytest.approx(-1.0)
        assert m["n_structures"] == 3

    def test_mae_calculation(self):
        results = [
            {"mlip_epa": -5.0, "ref_epa": -5.1},
            {"mlip_epa": -4.0, "ref_epa": -3.8},
        ]
        m = compute_ranking_metrics(_rows(results), stable_only=False)
        assert m["mae_epa"] == pytest.approx(0.15)

    def test_insufficient_data_returns_none(self):
        m = compute_ranking_metrics(_rows([]), stable_only=False)
        assert m["kendall_tau"] is None
        assert m["p_value"] is None
        assert m["mae_epa"] is None
        assert m["n_structures"] == 0

    def test_single_structure_returns_none(self):
        results = [{"mlip_epa": -5.0, "ref_epa": -5.0}]
        m = compute_ranking_metrics(_rows(results), stable_only=False)
        assert m["kendall_tau"] is None
        assert m["n_structures"] == 1

    def test_stable_only_filters_when_phonon_data_present(self):
        results = [
            {
                "mlip_epa": -5.0,
                "ref_epa": -5.0,
                "mlip_phonon_freq": 1.0,
                "ref_phonon_freq": 1.0,
            },
            {
                "mlip_epa": -4.0,
                "ref_epa": -4.0,
                "mlip_phonon_freq": -1.0,
                "ref_phonon_freq": 1.0,
            },
            {
                "mlip_epa": -3.0,
                "ref_epa": -3.0,
                "mlip_phonon_freq": 1.0,
                "ref_phonon_freq": -1.0,
            },
            {
                "mlip_epa": -2.0,
                "ref_epa": -2.0,
                "mlip_phonon_freq": 1.0,
                "ref_phonon_freq": 1.0,
            },
        ]
        m = compute_ranking_metrics(_rows(results), phonon_cutoff=-0.15, stable_only=True)
        assert m["stable_only_applied"] is True
        assert m["n_structures"] == 2

    def test_stable_only_skipped_without_phonon_data(self):
        results = [
            {"mlip_epa": -5.0, "ref_epa": -5.0},
            {"mlip_epa": -4.0, "ref_epa": -4.0},
        ]
        m = compute_ranking_metrics(_rows(results), stable_only=True)
        assert m["stable_only_applied"] is False
        assert m["n_structures"] == 2

    def test_stable_only_skipped_with_partial_phonon_data(self):
        results = [
            {
                "mlip_epa": -5.0,
                "ref_epa": -5.0,
                "mlip_phonon_freq": 1.0,
                "ref_phonon_freq": None,
            },
            {
                "mlip_epa": -4.0,
                "ref_epa": -4.0,
                "mlip_phonon_freq": 1.0,
                "ref_phonon_freq": 1.0,
            },
        ]
        m = compute_ranking_metrics(_rows(results), stable_only=True)
        assert m["stable_only_applied"] is False
        assert m["n_structures"] == 2

    def test_all_filtered_out_returns_none(self):
        results = [
            {
                "mlip_epa": -5.0,
                "ref_epa": -5.0,
                "mlip_phonon_freq": -1.0,
                "ref_phonon_freq": -1.0,
            },
            {
                "mlip_epa": -4.0,
                "ref_epa": -4.0,
                "mlip_phonon_freq": -1.0,
                "ref_phonon_freq": -1.0,
            },
        ]
        m = compute_ranking_metrics(_rows(results), phonon_cutoff=-0.15, stable_only=True)
        assert m["stable_only_applied"] is True
        assert m["kendall_tau"] is None
        assert m["n_structures"] == 0


# ------------------------------------------------------------------ #
#  compute_stability_metrics
# ------------------------------------------------------------------ #


class TestStabilityMetrics:
    def test_perfect_classification(self):
        results = [
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": 1.0},
            {"mlip_phonon_freq": -1.0, "ref_phonon_freq": -1.0},
        ]
        m = compute_stability_metrics(_rows(results), phonon_cutoff=-0.15)
        assert m is not None
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["n_total"] == 2
        assert m["n_stable_ref"] == 1
        assert m["n_stable_mlip"] == 1

    def test_all_false_positives(self):
        results = [
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": -1.0},
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": -1.0},
        ]
        m = compute_stability_metrics(_rows(results), phonon_cutoff=-0.15)
        assert m is not None
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)
        assert m["f1"] == pytest.approx(0.0)

    def test_all_false_negatives(self):
        results = [
            {"mlip_phonon_freq": -1.0, "ref_phonon_freq": 1.0},
            {"mlip_phonon_freq": -1.0, "ref_phonon_freq": 1.0},
        ]
        m = compute_stability_metrics(_rows(results), phonon_cutoff=-0.15)
        assert m is not None
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)

    def test_mixed_classification(self):
        results = [
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": 1.0},  # TP
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": -1.0},  # FP
            {"mlip_phonon_freq": -1.0, "ref_phonon_freq": 1.0},  # FN
            {"mlip_phonon_freq": -1.0, "ref_phonon_freq": -1.0},  # TN
        ]
        m = compute_stability_metrics(_rows(results), phonon_cutoff=-0.15)
        assert m is not None
        assert m["precision"] == pytest.approx(0.5)  # 1/(1+1)
        assert m["recall"] == pytest.approx(0.5)  # 1/(1+1)
        assert m["f1"] == pytest.approx(0.5)  # 2*0.5*0.5/(0.5+0.5)
        assert m["n_stable_ref"] == 2
        assert m["n_stable_mlip"] == 2

    def test_returns_none_for_empty(self):
        assert compute_stability_metrics(_rows([]), phonon_cutoff=-0.15) is None

    def test_returns_none_for_missing_data(self):
        results = [
            {"mlip_phonon_freq": None, "ref_phonon_freq": 1.0},
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": None},
        ]
        assert compute_stability_metrics(_rows(results), phonon_cutoff=-0.15) is None

    def test_partial_data_uses_valid_only(self):
        results = [
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": 1.0},
            {"mlip_phonon_freq": None, "ref_phonon_freq": 1.0},
        ]
        m = compute_stability_metrics(_rows(results), phonon_cutoff=-0.15)
        assert m is not None
        assert m["n_total"] == 1

    def test_custom_cutoff(self):
        results = [
            {"mlip_phonon_freq": -0.2, "ref_phonon_freq": -0.2},
        ]
        m_loose = compute_stability_metrics(_rows(results), phonon_cutoff=-0.5)
        assert m_loose is not None
        assert m_loose["n_stable_ref"] == 1

        m_strict = compute_stability_metrics(_rows(results), phonon_cutoff=-0.15)
        assert m_strict is not None
        assert m_strict["n_stable_ref"] == 0


# ------------------------------------------------------------------ #
#  Evaluation helpers
# ------------------------------------------------------------------ #


def _struct(sid, epa):
    return Structure(id=sid, status="relaxed", energy_per_atom=epa, converged=True)


def _eval(sid, epa, freq=None):
    return Evaluation(
        id=f"{sid}_e",
        structure_id=sid,
        calculator="VASP",
        config_json="{}",
        energy_per_atom=epa,
        energy_total=epa,
        min_phonon_freq=freq,
    )


class TestSelectEvalRecords:
    def test_sorts_by_energy_and_slices(self):
        recs = [_struct("a", -3.0), _struct("b", -5.0), _struct("c", -4.0)]
        out = select_eval_records(recs, top_n=2)
        assert [r.id for r in out] == ["b", "c"]

    def test_top_n_zero_keeps_all(self):
        recs = [_struct("a", -3.0), _struct("b", -5.0)]
        out = select_eval_records(recs, top_n=0)
        assert [r.id for r in out] == ["b", "a"]

    def test_does_not_mutate_input_order(self):
        recs = [_struct("a", -3.0), _struct("b", -5.0)]
        select_eval_records(recs, top_n=0)
        assert [r.id for r in recs] == ["a", "b"]


class TestEvalRowsFromCache:
    def test_only_cached_rows_carry_ref(self):
        recs = [_struct("a", -5.0), _struct("b", -4.0), _struct("c", -3.0)]
        eval_map = {"a": _eval("a", -5.1, freq=2.0), "c": _eval("c", -3.2)}
        rows = eval_rows_from_cache(recs, eval_map, run_name="r1")
        assert [r.structure_id for r in rows] == ["a", "c"]
        assert rows[0].ref_energy_per_atom == pytest.approx(-5.1)
        assert rows[0].ref_phonon_freq == pytest.approx(2.0)
        assert rows[1].ref_phonon_freq is None

        assert len(recs) - len(rows) == 1

    def test_empty_when_nothing_cached(self):
        assert eval_rows_from_cache([_struct("a", -5.0)], {}, run_name="r1") == []


class TestComparisonFromResultRows:
    def test_round_trip_into_ranking_metrics(self):
        recs = [_struct("a", -5.0), _struct("b", -4.0), _struct("c", -3.0)]
        eval_map = {
            "a": _eval("a", -5.0),
            "b": _eval("b", -4.0),
            "c": _eval("c", -3.0),
        }
        rows = eval_rows_from_cache(recs, eval_map, run_name="r1")
        comparison = comparison_from_result_rows(rows)
        assert len(comparison) == 3
        m = compute_ranking_metrics(comparison, stable_only=False)
        assert m["kendall_tau"] == pytest.approx(1.0)
        assert m["n_structures"] == 3

    def test_drops_rows_without_ref(self):
        row = ResultRow(structure=_struct("a", -5.0), ref_energy_per_atom=None)
        assert comparison_from_result_rows([row]) == []
